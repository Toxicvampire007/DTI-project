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
# 3. Install Python Dependencies
# ==========================
# Add all packages you actually use below
RUN pip install --upgrade pip && \
    pip install deeppurpose torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn flask streamlit

# ==========================
# 4. Copy Source Code
# ==========================
COPY backend /app/backend
COPY frontend /app/frontend

# ==========================
# 5. Expose Ports
# ==========================
EXPOSE 8000 8500

# ==========================
# 6. Start Backend and Frontend
# ==========================
# Runs backend on port 8000 and frontend (e.g. Streamlit) on port 8500
CMD ["/bin/bash", "-c", "python backend/dti_backend.py & python frontend/Home.py"]
