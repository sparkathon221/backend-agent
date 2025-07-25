# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app/backend-agent

# Install system dependencies (optional: add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY backend-agent /app/backend-agent

# Create and activate virtual environment
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Set entrypoint to always use the virtual environment
ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && exec \"$@\"", "--"]

# Default command (can be overridden)
CMD ["python"]
