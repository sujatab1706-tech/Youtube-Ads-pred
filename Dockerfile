# Small, production-style image for the click prediction API.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what's needed to serve: feature code, API code and the trained model.
COPY src/features.py src/features.py
COPY app/ app/
COPY models/model.joblib models/metrics.json models/

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /srv/logs && chown appuser /srv/logs
USER appuser

# Prediction and feedback logs. Mount a volume here to keep them: -v $(pwd)/logs:/srv/logs
VOLUME /srv/logs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
