# ==========================
# 1. Base Image
# ==========================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ==========================
# 2. Install System Dependencies
# ==========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================
# 3. Copy and Install Backend Dependencies
# ==========================
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# ==========================
# 4. Copy and Install Frontend Dependencies
# ==========================
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install -r /app/frontend/requirements.txt

# ==========================
# 5. Copy Source Code
# ==========================
COPY backend /app/backend
COPY frontend /app/frontend

# ==========================
# 6. Expose Port and Define Startup Command
# ==========================
# Example: Backend runs on 8000, frontend on 8500
EXPOSE 8000 8500

# Modify this depending on how you start the app
# (e.g. FastAPI for backend, Streamlit/Flask for frontend)
CMD ["/bin/bash", "-c", "python backend/dti_backend.py & python frontend/Home.py"]
