# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# copy app code
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# create models dir and ensure /mnt/data exists (mounted at runtime)
RUN mkdir -p /mnt/data/models

# copy only the FastAPI app
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
