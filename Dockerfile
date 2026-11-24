FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY inference ./inference
COPY dashboard ./dashboard
COPY results ./results

EXPOSE 8080
ENTRYPOINT ["python", "inference/benchmark.py"]
