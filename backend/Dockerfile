# Build context is the PROJECT ROOT (see docker-compose.yml: context: .,
# dockerfile: backend/Dockerfile) so that both backend/ and ml/ can be
# copied into the image -- the API imports shared preprocessing/predict
# code directly from ml/src, and loads trained artifacts from ml/models.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY ml/src ./ml/src
COPY ml/models ./ml/models

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
