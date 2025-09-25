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

# Configure app directory and non-root user with configurable UID/GID
WORKDIR /app

ARG APP_UID=1000
ARG APP_GID=1000

# Create group and user with provided IDs (if they don't already exist)
RUN groupadd -g "${APP_GID}" app || true \
  && useradd -m -u "${APP_UID}" -g "${APP_GID}" -s /bin/bash app || true \
  && mkdir -p /app/cache "${MPLCONFIGDIR}" \
  && chown -R "${APP_UID}:${APP_GID}" /app "${MPLCONFIGDIR}"

# Install Python deps first (leverage Docker layer cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r /tmp/requirements.txt

# Copy application code and fix ownership
COPY . /app
RUN chown -R "${APP_UID}:${APP_GID}" /app

# Switch to non-root
USER app

EXPOSE 8000

# Tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]