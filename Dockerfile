# ============================================================
# 🛰️ Sumbawa-A.E.C.O — Production Dockerfile
# Multi-stage: build Rust engine, then deploy lightweight Python image
# ============================================================

# --- Stage 1: Build the Rust/PyO3 engine ---
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgdal-dev \
    gdal-bin \
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

# Install system dependencies (GDAL for rasterio)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set GDAL environment variables to prevent rasterio installation errors
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the pre-built Rust wheel from the builder stage
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy the rest of the source code
COPY . .

# Create output directories to prevent permission issues during runtime
RUN mkdir -p outputs/predictions outputs/models data/processed

# RTK binary discovery: container should set RTK_BIN or have rtk on PATH
# No hardcoded local paths
ENV PYTHONPATH="/app/src:/app/api:${PYTHONPATH}"

# Expose FastAPI port
EXPOSE 8000

# Launch the application
CMD ["python", "api/main.py"]