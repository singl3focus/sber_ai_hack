import os
import logging
from typing import List, Dict, Optional

import uvicorn
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import Document

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
YAZZH_GEO_API_BASE = "https://geo.hack-it.yazzh.ru/api/v2"
YAZZH_MAIN_API_BASE = "https://hack-it.yazzh.ru/api/v1"


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    context: Optional[Dict[str, str]] = Field(None, description="Optional context for query")


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = Field(default=0.0)


class DataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.contacts_df = None
        self.questions_df = None
        self.load_datasets()
        
    def load_datasets(self):
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
        self.vectorstore = self._create_vectorstore()
        self.bm25_contacts = self.__create_bm25_index(
            self.data_processor.contacts_df['description'].tolist()
        )
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.data_processor.contacts_df['description'].tolist()
        )

    def _create_vectorstore(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        contacts_docs = [
            Document(page_content=desc, metadata={'source': 'contacts'}) 
            for desc in self.data_processor.contacts_df['description'].tolist()
        ]
        
        split_docs = text_splitter.split_documents(contacts_docs)
        
        return Chroma.from_documents(
            documents=split_docs, 
            embedding=OpenAIEmbeddings(),
            persist_directory="C:\\Users\\Дом\\Documents\\1GithubProjects\\sber_ai_hack\\data\\contacts"
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
            self.data_processor.contacts_df.iloc[idx]['description'] 
            for idx in combined_indices
        ]
        
        final_results = list(set(
            [doc.page_content for doc in vector_results] + combined_results
        ))
        
        return final_results[:k]


class CityAssistant:
    def __init__(self):
        self.data_processor = DataProcessor("C:\\Users\\Дом\\Documents\\1GithubProjects\\sber_ai_hack\\data")
        self.retriever = MultiModalRetriever(self.data_processor)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def process_query(self, query: str) -> QueryResponse:
        try:
            # Perform hybrid search
            search_results = self.retriever.hybrid_search(query)

            # Prepare context for LLM
            context = "\n".join(search_results)

            # Generate response using LLM
            response = self.llm.predict(
                f"Context: {context}\n\nQuery: {query}\n\nProvide a comprehensive and helpful response."
            )

            # Calculate confidence (simplified)
            confidence = len(search_results) / 5.0

            return QueryResponse(
                response=response,
                sources=[{"text": result} for result in search_results],
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResponse(
                response="Извините, произошла ошибка при обработке запроса.",
                confidence=0.0
            )

# FastAPI Application
app = FastAPI(
    title="Я Здесь Живу - Городской Помощник",
    description="Интеллектуальный AI-ассистент для Санкт-Петербурга",
    version="1.0.0"
)

assistant = CityAssistant()


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


def app():
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="debug"
    )

if __name__ == "__main__":
    app()
