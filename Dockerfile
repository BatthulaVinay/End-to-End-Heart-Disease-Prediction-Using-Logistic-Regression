# Base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get upgrade -y && \
	pip install --no-cache-dir -r requirements.txt && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy code & model
COPY . .

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

