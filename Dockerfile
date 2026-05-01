# Base Image
FROM python:3.11-slim

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# System Dependencies & Rust Toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python Dependencies (Layer Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt maturin

# Rust Engine Compilation
COPY rust_engine/ ./rust_engine/
RUN cd rust_engine && maturin build --release --out dist && pip install dist/*.whl

# App Code (Tanpa folder data/)
COPY api/ ./api/
COPY assets/ ./assets/
COPY index.html .

# Execution
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]