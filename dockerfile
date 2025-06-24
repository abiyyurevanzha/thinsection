# Gunakan Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy semua file
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
