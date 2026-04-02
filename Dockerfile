FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Environment variables (set at runtime)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# Default: expose a simple health-check HTTP server + env API
# The environment is accessed via Python directly (not HTTP),
# but we expose port 7860 for HuggingFace Spaces compatibility.
EXPOSE 7860

# Run the FastAPI server for HF Spaces (reset/step/state endpoints)
CMD ["python", "server.py"]
