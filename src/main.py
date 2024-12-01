import os
import logging
from typing import List, Dict, Optional

import uvicorn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import Document

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')

# Consts
EMBENDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # Показала себя с хорошей стороны при работе с Русским языком
EMBENDING_FUNCTION = HuggingFaceEmbeddings(
    model_name=EMBENDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}  # Используем CPU, но можно также использовать GPU
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
YAZZH_GEO_API_BASE = "https://geo.hack-it.yazzh.ru/api/v2"
YAZZH_MAIN_API_BASE = "https://hack-it.yazzh.ru/api/v1"

# _______________________


class MissingEnvironmentVariableError(Exception):
    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.message = f"Переменная окружения {self.variable_name} не установлена!"
        super().__init__(self.message)


# _______________________

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    context: Optional[Dict[str, str]] = Field(None, description="Optional context for query")


class QueryResponse(BaseModel):
    response: str
    # sources: Optional[List[Dict[str, str]]] = Field(None, description="Optional context for query")
    # confidence: Optional[float] = Field(None, description="Optional confidence of query")


class DataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.contacts_df = None
        self.questions_df = None
        self.load_datasets()
        
    def load_datasets(self):  # Todo: Ref [Magic]
        try:
            # Load contacts DataFrame
            contacts_path = os.path.join(self.data_dir, "contacts.xlsx")
            self.contacts_df = pd.read_excel(contacts_path)
            
            # Create a synthetic 'description' column
            self.contacts_df['description'] = self.contacts_df.apply(
                lambda row: f"{row['name']} - {row['category']} - Телефон: {row['phones']}", 
                axis=1
            )
            
            # Load questions DataFrame
            questions_path = os.path.join(self.data_dir, "questions.xlsx")
            self.questions_df = pd.read_excel(questions_path)
            
            # Rename the first column to 'question' and use it
            self.questions_df.columns = [
                'question' if i == 0 else f'category_{i}' 
                for i in range(len(self.questions_df.columns))
            ]
            
            # Fill NaN values
            self.contacts_df['description'] = self.contacts_df['description'].fillna('')
            self.questions_df['question'] = self.questions_df['question'].fillna('')

            logger.info(f"Contacts DataFrame columns: {list(self.contacts_df.columns)}")
            logger.info(f"Questions DataFrame columns: {list(self.questions_df.columns)}")
            logger.info("Datasets loaded and preprocessed successfully")
        
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise


class MultiModalRetriever:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.data_dir = self.data_processor.data_dir
        
        self.vectorstore = self._create_vectorstore()
        self.bm25_contacts = self.__create_bm25_index(
            self.data_processor.contacts_df['description'].tolist()  # Todo: Ref [Magic]
        )
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.data_processor.contacts_df['description'].tolist()  # Todo: Ref [Magic]
        )

    def _create_vectorstore(self):  # Only contacts using
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Todo: Ref [Magic]
        
        contacts_docs = [
            Document(page_content=desc, metadata={'source': 'contacts'})  # Todo: Ref [Magic]
            for desc in self.data_processor.contacts_df['description'].tolist()  # TODO: Ref [Magic]
        ]
        
        split_docs = text_splitter.split_documents(contacts_docs)
        
        return Chroma.from_documents(
            documents=split_docs, 
            embedding=EMBENDING_FUNCTION,
            persist_directory=os.path.join(self.data_dir, "db")
        )

    def __create_bm25_index(self, documents):
        tokenized_docs = [doc.lower().split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def hybrid_search(self, query: str, k: int = 5) -> List[str]:
        vector_results = self.vectorstore.similarity_search(query, k=k//2)
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_contacts.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k//2]
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        tfidf_top_indices = np.argsort(cosine_scores)[::-1][:k//2]
        
        combined_indices = list(set(bm25_top_indices) | set(tfidf_top_indices))
        combined_results = [
            self.data_processor.contacts_df.iloc[idx]['description']  # TODO: Ref [Magic]
            for idx in combined_indices
        ]
        
        final_results = list(set(
            [doc.page_content for doc in vector_results] + combined_results
        ))
        
        return final_results[:k]


class CityAssistant:
    def __init__(self, data_path: str) -> None:
        self.data_processor = DataProcessor(data_path)
        self.retriever = MultiModalRetriever(self.data_processor)

        # Загрузка данных из CSV
        data = pd.read_csv(os.path.join(data_path, 'question_classification.csv'))

        # Разделяем данные на признаки (X) и метки (y)
        X = data['question']
        y = data['label']

        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Преобразование текста в числовые признаки с помощью TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Обучение модели
        self.model = LogisticRegression()
        self.model.fit(X_train_tfidf, y_train)

        # Предсказание на тестовых данных
        y_pred = self.model.predict(X_test_tfidf)

        # Оценка модели
        logger.info(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        logger.info(classification_report(y_test, y_pred))

    def process_query(self, query: str) -> QueryResponse:
        try:
            new_question_tfidf = self.vectorizer.transform([query])
            prediction = self.model.predict(new_question_tfidf)
            logger.info(f'Предсказанный ответ: {prediction[0]}')
            if prediction[0] == "template":
                return QueryResponse(
                    response=f"Запрос: {query}\n\nПредоставь исчерпывающий и полезный ответ.",
                )

            # Perform hybrid search
            search_results = self.retriever.hybrid_search(query)

            # Prepare context for LLM
            context = "\n".join(search_results)

            # Calculate confidence
            confidence = len(search_results) / 5.0  # TODO: Ref [Simplified]

            return QueryResponse(
                response=f"Контекст: {context}\n\nЗапрос: {query}\n\nПредоставь исчерпывающий и полезный ответ.",
                # sources=[{"text": result} for result in search_results],
                # confidence=confidence
            )
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResponse(
                response="Извините, произошла ошибка при обработке запроса.",
                # confidence=0.0
            )


# FastAPI Application
app = FastAPI(
    title="Я Здесь Живу - Городской Помощник",
    description="Интеллектуальный AI-ассистент для Санкт-Петербурга",
    version="1.0.0"
)

data_dir_path_env_var = "DATA_DIR_PATH"
data_dir_path = os.getenv(data_dir_path_env_var)
if not data_dir_path:
    raise MissingEnvironmentVariableError(data_dir_path_env_var)

assistant = CityAssistant(data_dir_path)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query and return comprehensive response
    """
    try:
        response = assistant.process_query(request.query)
        return response
    except Exception as e:
        logger.error(f"API query error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}


def start_app():
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="debug"
    )


if __name__ == "__main__":
    start_app()
