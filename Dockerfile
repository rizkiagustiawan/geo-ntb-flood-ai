# Use a stable base image
FROM python:3.11-slim

# Install system dependencies (Including gdal-bin for GDAL CLI support)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (Optional: Uncomment if you need to compile rtk within the container)
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Set GDAL environment variables to prevent rasterio installation errors
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Create output directories to prevent permission issues during runtime
RUN mkdir -p outputs/predictions outputs/models data/processed

# Expose FastAPI port
EXPOSE 8000

# Launch the application
CMD ["python", "api/main.py"]