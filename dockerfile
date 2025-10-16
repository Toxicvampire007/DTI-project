# ==========================
# 1. Base Image
# ==========================
FROM python:3.10-slim

# ==========================
# 2. Environment Setup
# ==========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ==========================
# 3. System Dependencies
# ==========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==========================
# 4. Preinstall Core Libraries (Layer caching optimization)
# ==========================
# This layer rarely changes — ensures faster rebuilds
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn flask streamlit rdkit git+https://github.com/bp-kelley/descriptastorus pandas-flavor 

# ==========================
# 5. Install DeepPurpose + DGL (Graph support)
# ==========================
RUN pip install deeppurpose && \
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.0/repo.html

# ==========================
# 6. Copy Source Code
# ==========================
COPY Backend /app/Backend
COPY Frontend /app/Frontend

# ==========================
# 7. Expose Ports
# ==========================
EXPOSE 8000 8500

# ==========================
# 8. Start Backend & Frontend
# ==========================
# Backend (Flask) -> port 8000
# Frontend (Streamlit) -> port 8500
CMD ["/bin/bash", "-c", "python Backend/dti_backend.py & streamlit run Frontend/Home.py --server.port=8500 --server.address=0.0.0.0"]

