FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN echo "=== requirements.txt ===" && cat requirements.txt && \
    pip install -v -r requirements.txt

COPY . .

CMD ["python", "main.py"]
