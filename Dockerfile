FROM python:3.12-slim

# Keeps Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Disable pip cache to save disk space
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Single worker, no reload — safe for 512 MB / 0.1 CPU
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "30"]
