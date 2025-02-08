# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/backend/readtime_model.py .
COPY src/backend/main.py .
COPY src/frontend/index.html /var/www/html/

# Configure nginx
COPY nginx.conf /etc/nginx/sites-available/default

# Train the model and generate the pickle file
RUN python readtime_model.py

# Expose ports for both nginx and FastAPI
EXPOSE 80 8000

# Copy the startup script
COPY start.sh .
RUN chmod +x start.sh

# Command to run both services
CMD ["./start.sh"]