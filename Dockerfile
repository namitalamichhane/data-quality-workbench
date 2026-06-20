FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for data science packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the default Streamlit hosting port
EXPOSE 7860

# Configure health checks for the container
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the Streamlit execution hub on container startup
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]