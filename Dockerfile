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
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8000", "--threads=4", "--channel-timeout=120", "--recv-bytes=65536", "app:app"]
