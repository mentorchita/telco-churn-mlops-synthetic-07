FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Persistent volume mount point for ChromaDB
RUN mkdir -p /data/chroma

RUN useradd -m appuser && chown -R appuser /app /data
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
