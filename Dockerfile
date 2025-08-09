
# Useing an official Python runtime as a base image
FROM python:3.9-slim

# working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "api.py"]