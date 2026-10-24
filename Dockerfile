FROM python:3.11-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8

# Install system dependencies required for some python packages (like chromadb)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensuring FastAPI, Uvicorn, and Multipart are installed for the API
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
