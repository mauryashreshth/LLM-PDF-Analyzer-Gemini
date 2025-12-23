# Use a lightweight Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some PDF tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only PyTorch first. 
# This prevents sentence-transformers from downloading the huge GPU version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]