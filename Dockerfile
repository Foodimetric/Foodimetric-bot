# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install UV
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using UV 
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080
EXPOSE 8080

# Command to run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"] 