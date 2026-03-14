FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ecko_web.py .
COPY core ./core
COPY web ./web
COPY extras ./extras
COPY safety ./safety

# Runtime directories
RUN mkdir -p \
    /app/characters \
    /app/characters_web \
    /app/memories \
    /app/rag/extra \
    /app/rag/conversations \
    /app/safety \
    /app/ssl \
    /app/static \
    /app/logs \
    /app/ascii_art

EXPOSE 5050

CMD ["python", "ecko_web.py"]
