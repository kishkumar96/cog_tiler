#!/usr/bin/env python3
"""
COG Configuration and Optimization Settings
Centralized configuration for Cloud Optimized GeoTIFF generation and serving
"""

import os
from typing import Dict, Any

# =============================================================================
# COG Generation Profiles
# =============================================================================

# Optimized COG profile for marine forecast data
OPTIMIZED_COG_PROFILE: Dict[str, Any] = {
    'driver': 'GTiff',
    'compress': 'DEFLATE',        # Good compression for imagery
    'blockxsize': 256,            # Optimal tile size for web serving
    'blockysize': 256,
    'tiled': True,                # Essential for COG
    'interleave': 'pixel',        # Better for RGBA data
    'predictor': 2,               # Horizontal differencing for better compression
    'zlevel': 6,                  # Balanced compression vs speed
    'bigtiff': 'IF_SAFER'         # Handle large files automatically
}

# High compression profile for archival storage
HIGH_COMPRESSION_PROFILE: Dict[str, Any] = {
    'driver': 'GTiff',
    'compress': 'DEFLATE',
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'interleave': 'pixel',
    'predictor': 2,
    'zlevel': 9,                  # Maximum compression
    'bigtiff': 'IF_SAFER'
}

# Fast generation profile for development/testing
FAST_PROFILE: Dict[str, Any] = {
    'driver': 'GTiff',
    'compress': 'LZW',            # Faster than DEFLATE
    'blockxsize': 256,
    'blockysize': 256,
    'tiled': True,
    'interleave': 'pixel',
    'bigtiff': 'IF_SAFER'
}

# =============================================================================
# GDAL Environment Optimization
# =============================================================================

def configure_gdal_environment():
    """
    Configure GDAL environment variables for optimal COG performance.
    Call this once at application startup.
    """
    gdal_settings = {
        # Memory and threading
        "GDAL_CACHEMAX": "512",                    # 512MB cache
        "GDAL_NUM_THREADS": "ALL_CPUS",            # Use all available CPUs
        
        # COG-specific optimizations
        "GDAL_TIFF_OVR_BLOCKSIZE": "256",          # Match COG tile size
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", # Skip directory scanning
        "VSI_CACHE": "TRUE",                       # Enable VSI caching
        "VSI_CACHE_SIZE": "25000000",              # 25MB VSI cache
        
        # Performance optimizations
        "GDAL_HTTP_TIMEOUT": "30",                 # 30 second HTTP timeout
        "GDAL_HTTP_RETRY_DELAY": "1",              # 1 second retry delay
        "GDAL_HTTP_MAX_RETRY": "3",                # Max 3 retries
        "CPL_VSIL_CURL_CACHE_SIZE": "200000000",   # 200MB curl cache
        
        # Compression optimizations
        "COMPRESS_OVERVIEW": "DEFLATE",            # Use DEFLATE for overviews
        "PREDICTOR_OVERVIEW": "2",                 # Predictor for overviews
        "ZLEVEL_OVERVIEW": "6",                    # Compression level for overviews
        
        # Reduce I/O operations
        "GDAL_TIFF_INTERNAL_MASK": "TRUE",         # Use internal masks
        "GDAL_TIFF_OVR_BLOCKSIZE": "256"           # Overview block size
    }
    
    for key, value in gdal_settings.items():
        os.environ.setdefault(key, value)
    
    return gdal_settings

# =============================================================================
# COG Validation Settings
# =============================================================================

COG_VALIDATION_CONFIG = {
    "strict": True,                    # Strict COG compliance checking
    "overview_level": 512,             # Minimum overview level
    "overview_count": 3,               # Minimum number of overview levels
    "blocksize": [256, 512],           # Acceptable block sizes
    "compress": ["DEFLATE", "LZW"],    # Acceptable compression methods
}

# =============================================================================
# Cache Management Settings
# =============================================================================

CACHE_CONFIG = {
    "max_age_hours": 24,               # Maximum age before cleanup
    "max_cache_size_gb": 10,           # Maximum total cache size
    "cleanup_threshold": 0.8,          # Cleanup when 80% full
    "min_free_space_gb": 1,            # Minimum free space to maintain
}

# =============================================================================
# Tile Generation Settings
# =============================================================================

TILE_CONFIG = {
    "default_tile_size": 256,          # Default tile size in pixels
    "max_zoom": 18,                    # Maximum zoom level
    "min_zoom": 0,                     # Minimum zoom level
    "resampling": "nearest",           # Default resampling method
    "web_optimized": True,             # Generate web-optimized COGs
    "overview_resampling": "nearest",  # Overview resampling method
}

# =============================================================================
# Data Type Optimization
# =============================================================================

# Optimize data types based on variable type
VARIABLE_DTYPE_MAP = {
    "hs": "float32",          # Wave height - float precision needed
    "tpeak": "float32",       # Wave period - float precision needed  
    "dir": "uint16",          # Direction - integer degrees (0-360)
    "u10": "float32",         # Wind U component - float precision needed
    "v10": "float32",         # Wind V component - float precision needed
    "inundation_depth": "float32",  # Inundation depth - float precision needed
}

# Color scaling optimization for different variables
VARIABLE_SCALING = {
    "hs": {"vmin": 0.0, "vmax": 6.0, "colormap": "plasma"},
    "tpeak": {"vmin": 2.0, "vmax": 20.0, "colormap": "viridis"},
    "dir": {"vmin": 0.0, "vmax": 360.0, "colormap": "hsv"},
    "u10": {"vmin": -20.0, "vmax": 20.0, "colormap": "RdBu_r"},
    "v10": {"vmin": -20.0, "vmax": 20.0, "colormap": "RdBu_r"},
    "inundation_depth": {"vmin": 0.0, "vmax": 3.0, "colormap": "Blues"},
}

if __name__ == "__main__":
    # Test configuration
    print("🔧 COG Configuration Test")
    print("=" * 50)
    
    # Configure GDAL environment
    settings = configure_gdal_environment()
    print("✅ GDAL Environment Configured:")
    for key, value in settings.items():
        print(f"   {key} = {value}")
    
    print(f"\n📊 Available COG Profiles:")
    print(f"   • Optimized: {len(OPTIMIZED_COG_PROFILE)} settings")
    print(f"   • High Compression: {len(HIGH_COMPRESSION_PROFILE)} settings")
    print(f"   • Fast: {len(FAST_PROFILE)} settings")
    
    print(f"\n🎨 Variable Configurations:")
    for var, config in VARIABLE_SCALING.items():
        print(f"   • {var}: {config['colormap']} [{config['vmin']}-{config['vmax']}]")