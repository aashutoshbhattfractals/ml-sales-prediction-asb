# Use Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .
COPY model.pkl .

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
