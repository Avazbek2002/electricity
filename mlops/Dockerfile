# The pipeline image: every Airflow task and every CI job runs this (one image per commit).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "-m", "elecsim.clock", "--status"]
