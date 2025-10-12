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
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==========================
# 3. Install Python Dependencies
# ==========================
RUN pip install --upgrade pip && \
    pip install deeppurpose torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn flask streamlit nest_asyncio rdkit git+https://github.com/bp-kelley/descriptastorus pandas-flavor

# ==========================
# 4. Copy Source Code
# ==========================
COPY Backend /app/Backend
COPY Frontend /app/Frontend

# ==========================
# 5. Expose Ports
# ==========================
EXPOSE 8000 8500

# ==========================
# 6. Start Backend and Frontend
# ==========================
CMD ["/bin/bash", "-c", "python Backend/dti_backend.py & python Frontend/Home.py --server.port=8500 --server.address=0.0.0.0"]

