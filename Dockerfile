# FastAPI + rasterio/xarray stack in a slim Python base
FROM python:3.11-slim

# Useful defaults for stability and performance
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/home/app/.config/matplotlib \
    GDAL_CACHEMAX=512 \
    GDAL_NUM_THREADS=ALL_CPUS \
    GDAL_TIFF_OVR_BLOCKSIZE=128

# System packages required by matplotlib (Agg), font rendering, and runtime basics
# Note: rasterio/rio-cogeo wheels bundle GDAL/PROJ, so we avoid pinning system GDAL here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    tini \
    libfreetype6 \
    libpng16-16 \
    fonts-dejavu-core \
    fontconfig \
  && rm -rf /var/lib/apt/lists/*

# App directory and non-root user
WORKDIR /app
RUN useradd -ms /bin/bash app \
  && mkdir -p /app/cache "$MPLCONFIGDIR" \
  && chown -R app:app /app "$MPLCONFIGDIR"

# Install Python deps first (leverage Docker layer cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r /tmp/requirements.txt

# Copy application code
COPY . /app

# Switch to non-root
USER app

EXPOSE 8000

# Tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]