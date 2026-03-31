FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcairo2 \
        libffi-dev \
        potrace \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "2"]
