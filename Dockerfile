FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV DATA_DIR_PATH=/app/data
ENV TZ="Europe/Moscow"

CMD ["uvicorn", "src.main:app", "--reload"]
