#!/usr/bin/env python3
"""
COG Generation and Serving System
Generate and serve Cloud Optimized GeoTIFFs for wave forecast data
"""
import pandas as pd
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict
import xarray as xr
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles
import tempfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

class COGGenerator:
    """
    A Cloud Optimized GeoTIFF (COG) generator for ocean wave data from THREDDS servers.
    This class provides functionality to generate, cache, and manage COG files from wave data
    served via THREDDS Data Server. It includes features for file locking, metadata management,
    and automatic cleanup of old files.
    Attributes:
        cache_dir (Path): Directory where COG files and metadata are stored
        metadata_file (Path): Path to the JSON file storing COG metadata
        metadata (Dict): In-memory metadata dictionary for tracking generated COGs
    Methods:
        load_metadata(): Load COG metadata from the JSON file
        save_metadata(): Save current metadata to the JSON file
        generate_cog_name(): Create a consistent filename for COG files based on parameters
        cog_exists(): Check if a specific COG file already exists in the cache
        generate_wave_cog(): Main method to generate COG files from THREDDS wave data
        list_available_cogs(): Get information about all available COG files
        cleanup_old_cogs(): Remove COG files older than specified age
    The class handles:
    - Data retrieval from THREDDS servers (specifically SPC ocean model data)
    - Interpolation of unstructured mesh data to regular grids
    - Colormap application and RGBA conversion
    - Coordinate system transformation from WGS84 to Web Mercator
    - COG optimization with tiling and compression
    - Concurrent access protection via file locking
    - Metadata tracking and file lifecycle management
    Example:
        generator = COGGenerator(cache_dir="cache/cogs")
        cog_path = generator.generate_wave_cog(
            variable="hs", 
            time_index=0,
            vmin=0.0, 
            vmax=4.0, 
            colormap="plasma"
    """
    def __init__(self, cache_dir: str = "cache/cogs"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cog_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load COG metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save COG metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_cog_name(self, variable: str, time_index: int, 
                         vmin: float, vmax: float, colormap: str) -> str:
        """Generate a consistent COG filename"""
        params = f"{variable}_{time_index}_{vmin}_{vmax}_{colormap}"
        hash_obj = hashlib.md5(params.encode())
        return f"{variable}_{time_index}_{hash_obj.hexdigest()[:8]}.tif"
    
    def cog_exists(self, cog_name: str) -> bool:
        """Check if COG file already exists"""
        cog_path = self.cache_dir / cog_name
        return cog_path.exists()
    
    def generate_wave_cog(self, variable: str = "hs", time_index: int = 0,
                         vmin: float = 0.0, vmax: float = 4.0, 
                         colormap: str = "plasma") -> Path:
        """Generate a COG file for wave data with file locking"""
        from filelock import FileLock
        
        cog_name = self.generate_cog_name(variable, time_index, vmin, vmax, colormap)
        cog_path = self.cache_dir / cog_name
        
        # Use file locking to prevent concurrent generation
        lock_path = str(cog_path) + ".lock"
        
        with FileLock(lock_path, timeout=300):
            # Return existing COG if it exists and is recent (after acquiring lock)
            if self.cog_exists(cog_name):
                print(f"Using existing COG: {cog_name}")
                return cog_path
            
            print(f"Generating new COG: {cog_name}")
            
            # Load data from THREDDS
            url = os.getenv("THREDDS_UGRID_URL", "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc")            
            print(f"Using THREDDS URL: {url}")
            
        try:
            print(f"Loading data from THREDDS for {variable} at time {time_index}...")
            with xr.open_dataset(url) as ds:
                # Get coordinates and data
                lon = ds['mesh_node_lon'].values
                lat = ds['mesh_node_lat'].values
                data = ds[variable].isel(time=time_index).values
                
                # Clean data
                valid_mask = ~np.isnan(data)
                if np.sum(valid_mask) < 10:
                    raise ValueError("Insufficient valid data")
                
                data_clean = data[valid_mask]
                lon_clean = lon[valid_mask]
                lat_clean = lat[valid_mask]
                
                # Create regular grid for COG
                # Use actual data bounds with small buffer
                lon_min, lon_max = lon_clean.min(), lon_clean.max()
                lat_min, lat_max = lat_clean.min(), lat_clean.max()
                
                # Add small buffer (2% of range)
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                buffer_lon = lon_range * 0.02
                buffer_lat = lat_range * 0.02
                
                lon_min -= buffer_lon
                lon_max += buffer_lon
                lat_min -= buffer_lat
                lat_max += buffer_lat
                
                # High resolution grid
                grid_size = 1024
                lon_grid = np.linspace(lon_min, lon_max, grid_size)
                lat_grid = np.linspace(lat_min, lat_max, grid_size)
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                
                # Interpolate data to regular grid
                points = np.column_stack((lon_clean, lat_clean))
                grid_points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))
                
                grid_data = griddata(
                    points, data_clean, grid_points, 
                    method='linear', fill_value=np.nan
                ).reshape(grid_size, grid_size)
                
                # Apply colormap
                cmap = plt.get_cmap(colormap)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                
                # Convert to RGBA
                rgba_data = cmap(norm(grid_data))
                rgba_data = (rgba_data * 255).astype(np.uint8)
                
                # Handle transparency for NaN values
                alpha_mask = ~np.isnan(grid_data)
                rgba_data[:, :, 3] = alpha_mask.astype(np.uint8) * 255
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                # Convert to Web Mercator for tile serving
                from rasterio.warp import transform_bounds, calculate_default_transform
                
                # Transform bounds to Web Mercator
                # Transform bounds to Web Mercator (for reference)
                transform_bounds(
                    CRS.from_epsg(4326), CRS.from_epsg(3857),
                    lon_min, lat_min, lon_max, lat_max
                )
                # Calculate transform for Web Mercator
                transform, width, height = calculate_default_transform(
                    CRS.from_epsg(4326), CRS.from_epsg(3857),
                    grid_size, grid_size,
                    left=lon_min, bottom=lat_min, right=lon_max, top=lat_max
                )
                
                # Create profile for GeoTIFF in Web Mercator
                profile = {
                    'driver': 'GTiff',
                    'height': height,
                    'width': width,
                    'count': 4,  # RGBA
                    'dtype': 'uint8',
                    'crs': CRS.from_epsg(3857),  # Web Mercator for tiles
                    'transform': transform,
                    'compress': 'deflate',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256,
                    'interleave': 'pixel'
                }
                
                # First create a temporary file in WGS84
                with tempfile.NamedTemporaryFile(suffix='_4326.tif', delete=False) as tmp_4326:
                    tmp_4326_path = tmp_4326.name
                
                # Create WGS84 profile first
                transform_4326 = from_bounds(lon_min, lat_min, lon_max, lat_max, 
                                           grid_size, grid_size)
                profile_4326 = {
                    'driver': 'GTiff',
                    'height': grid_size,
                    'width': grid_size,
                    'count': 4,  # RGBA
                    'dtype': 'uint8', 
                    'crs': CRS.from_epsg(4326),
                    'transform': transform_4326,
                    'compress': 'deflate'
                }
                
                # Write WGS84 GeoTIFF
                with rasterio.open(tmp_4326_path, 'w', **profile_4326) as dst_4326:
                    for i in range(4):  # Write RGBA bands
                        dst_4326.write(rgba_data[:, :, i], i + 1)
                
                # Reproject to Web Mercator
                with rasterio.open(tmp_4326_path) as src:
                    with rasterio.open(tmp_path, 'w', **profile) as dst:
                        for i in range(1, 5):  # Reproject each band
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst.transform, 
                                dst_crs=dst.crs,
                                resampling=Resampling.bilinear
                            )
                
                # Clean up WGS84 temp file
                os.unlink(tmp_4326_path)
                
                # Add metadata to the Web Mercator COG
                with rasterio.open(tmp_path, 'r+') as dst:
                    dst.update_tags(
                        variable=variable,
                        time_index=str(time_index),
                        vmin=str(vmin),
                        vmax=str(vmax),
                        colormap=colormap,
                        data_min=str(np.nanmin(grid_data)),
                        data_max=str(np.nanmax(grid_data)),
                        created=str(pd.Timestamp.now()),
                        source="SPC THREDDS"
                    )
                
                # Convert to COG with optimized profile
                cog_profile = {
                    'driver': 'GTiff',
                    'compress': 'DEFLATE',
                    'blockxsize': 256,
                    'blockysize': 256,
                    'tiled': True,
                    'interleave': 'pixel',
                    'predictor': 2,  # Horizontal differencing for better compression
                    'zlevel': 6      # Balanced compression vs speed
                }
                
                # Add retry logic for COG generation
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        cog_translate(
                            tmp_path,
                            str(cog_path),
                            cog_profile,
                            quiet=True
                        )
                        
                        # Validate the generated COG
                        is_valid_cog, cog_errors, cog_warnings = cog_validate(str(cog_path))
                        if not is_valid_cog:
                            print(f"⚠️ Generated COG failed validation: {cog_errors}")
                            # Still do basic read test as fallback
                            with rasterio.open(str(cog_path)) as test_src:
                                test_src.read(1, window=Window(0, 0, 10, 10))
                        else:
                            print(f"✅ COG validation passed for {variable} t={time_index}")
                            if cog_warnings:
                                print(f"   Warnings: {cog_warnings}")
                        
                        break  # Success, exit retry loop
                        
                    except Exception as cog_error:
                        print(f"COG generation attempt {attempt + 1} failed: {cog_error}")
                        if attempt == max_retries - 1:
                            raise cog_error
                        
                        # Clean up partial file before retry
                        if cog_path.exists():
                            cog_path.unlink()
                        time.sleep(0.5)  # Brief pause before retry
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Update metadata
                self.metadata[cog_name] = {
                    'variable': variable,
                    'time_index': time_index,
                    'vmin': vmin,
                    'vmax': vmax,
                    'colormap': colormap,
                    'created': str(pd.Timestamp.now()),
                    'data_range': [float(np.nanmin(grid_data)), float(np.nanmax(grid_data))],
                    'file_size': os.path.getsize(cog_path)
                }
                self.save_metadata()
                
                print(f"✅ COG generated successfully: {cog_path}")
                return cog_path
                
        except Exception as e:
            print(f"❌ COG generation failed: {e}")
            raise
    
    def list_available_cogs(self) -> Dict:
        """List all available COG files"""
        available = {}
        for cog_file in self.cache_dir.glob("*.tif"):
            name = cog_file.name
            if name in self.metadata:
                available[name] = self.metadata[name]
        return available
    
    def cleanup_old_cogs(self, max_age_hours: int = 24):
        """Remove COG files older than specified hours"""
        current_time = time.time()
        removed_count = 0
        
        for cog_file in self.cache_dir.glob("*.tif"):
            file_age = current_time - cog_file.stat().st_mtime
            if file_age > (max_age_hours * 3600):
                cog_file.unlink()
                # Remove from metadata
                if cog_file.name in self.metadata:
                    del self.metadata[cog_file.name]
                removed_count += 1
        
        if removed_count > 0:
            self.save_metadata()
            print(f"🗑️ Cleaned up {removed_count} old COG files")

if __name__ == "__main__":
    import pandas as pd
    
    # Test COG generation
    generator = COGGenerator()
    
    print("🌊 Testing COG Generation")
    print("=" * 50)
    
    # Generate test COGs for different variables
    test_params = [
        {"variable": "hs", "time_index": 0, "vmin": 0.0, "vmax": 4.0, "colormap": "plasma"},
        {"variable": "hs", "time_index": 6, "vmin": 0.0, "vmax": 4.0, "colormap": "plasma"},
        {"variable": "tpeak", "time_index": 0, "vmin": 2.0, "vmax": 20.0, "colormap": "viridis"}
    ]
    
    for params in test_params:
        try:
            cog_path = generator.generate_wave_cog(**params)
            print(f"Generated: {cog_path}")
        except Exception as e:
            print(f"Failed: {params} - {e}")
    
    # List available COGs
    # List available COGs
    print("\n📋 Available COGs:")
    available = generator.list_available_cogs()
    for name, info in available.items():
        print(f"  {name}: {info.get('variable', 'N/A')} t={info.get('time_index', 'N/A')} [{info.get('vmin', 'N/A')}-{info.get('vmax', 'N/A')}]")
    
    import rasterio
    with rasterio.open("cache/cogs/tpeak_175_60798a45.tif") as src:
        print(src.bounds)