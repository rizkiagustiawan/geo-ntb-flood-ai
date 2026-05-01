# ============================================================
# 🛰 Sumbawa-A.E.C.O — Production Dockerfile
# Multi-stage: build Rust engine, then deploy lightweight Python image
# ============================================================

# --- Stage 1: Build the Rust/PyO3 engine ---
FROM python:3.11-slim AS builder

# Install system dependencies (Termasuk Clang untuk bindgen, GDAL, dan patchelf)
RUN apt-get update && apt-get install -y \
    python3-dev \
    libssl-dev \
    pkg-config \
    curl \
    build-essential \
    libgdal-dev \
    gdal-bin \
    clang \
    libclang-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN pip install --no-cache-dir maturin

WORKDIR /build

# Copy Rust engine source and build the wheel
COPY rust_engine/ ./rust_engine/
RUN cd rust_engine && maturin build --release --out /build/wheels/

# --- Stage 2: Runtime image ---
FROM python:3.11-slim

# Install runtime dependencies (GDAL & G++ untuk library python)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the pre-built Rust wheel dari stage builder
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy the rest of the source code
COPY . .

# Create output directories (Penting untuk persistence data NTB)
RUN mkdir -p outputs/predictions outputs/models data/processed outputs/web

# Environment configuration
ENV PYTHONPATH="/app/src:/app/api:${PYTHONPATH}"

# Expose FastAPI port
EXPOSE 8000

# Launch the application
CMD ["python", "api/main.py"]
