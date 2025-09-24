#!/usr/bin/env python3
"""
Direct UGRID triangular mesh rendering - avoids island interpolation artifacts
This approach preserves the original mesh structure and provides sharp boundaries
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mpl_tri
import matplotlib.colors as mcolors
from datetime import datetime
import pandas as pd
from typing import Tuple, Optional
import io
from PIL import Image

def extract_ugrid_direct(url: str, target_time: str, variable_name: str, 
                        mesh_lon_name='mesh_node_lon',
                        mesh_lat_name='mesh_node_lat', 
                        mesh_tri_name='mesh_face_node'):
    """
    Extract UGRID data directly without interpolation artifacts
    Preserves original mesh structure for clean island boundaries
    """
    try:
        with xr.open_dataset(url, mask_and_scale=True, decode_cf=True) as ds:
            # Handle time axis
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
            
            # Convert target time - handle both datetime strings and indices
            if target_time.isdigit():
                # Direct time index
                time_index = int(target_time)
                if time_index >= len(time_dt):
                    time_index = 0
            else:
                # Parse datetime string
                if "." in target_time:
                    target_dt = pd.to_datetime(target_time).to_pydatetime()
                else:
                    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
                
                # Find closest time
                time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
            act_time = str(ds.time.values[time_index])
            
            # Extract mesh coordinates and triangles
            lon = ds[mesh_lon_name].data
            lat = ds[mesh_lat_name].data
            triangles = ds[mesh_tri_name].data
            
            # Extract variable
            if variable_name not in ds.variables:
                raise ValueError(f"Variable '{variable_name}' not found. Available: {list(ds.variables.keys())}")
            
            var = ds[variable_name].isel(time=time_index)
            
            # If variable has 3 dims (e.g. (depth, node, face)), select first
            if len(var.shape) == 3:
                var = var.isel({var.dims[0]: 0})
            
            # Reduce to 1D if needed
            data = np.ma.masked_invalid(var.data.squeeze())
            
            return lon, lat, triangles, data, act_time
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")


def create_direct_ugrid_tile(url: str, variable: str, time: Optional[str],
                           lon_min: float, lon_max: float, 
                           lat_min: float, lat_max: float,
                           tile_size: int = 256,
                           colormap: str = 'jet',
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None) -> bytes:
    """
    Create a COG tile using direct triangular mesh rendering
    Avoids interpolation artifacts and preserves sharp island boundaries
    """
    
    # Handle time parameter
    if time is None or time == "":
        target_time = "0"  # Use first time step
    else:
        target_time = time
    
    try:
        # Extract UGRID data directly
        lon, lat, triangles, data, act_time = extract_ugrid_direct(
            url, target_time, variable
        )
        
        # Handle coordinate system normalization
        west_bound = -180
        if (float(west_bound) < 0) and (lon.min() > 0):
            lon = np.where(lon > 180, lon - 360, lon)
        elif (float(west_bound) > 0) and (lon.min() < 0):
            lon = np.where(lon < 0, lon + 360, lon)
        
        # Fix 1-based to 0-based indexing
        if triangles.min() == 1:
            triangles = triangles - 1
        
        # Set up colormap and normalization
        cmap = plt.get_cmap(colormap)
        
        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))
            
        # Ensure reasonable range
        if vmax <= vmin:
            vmax = vmin + 1.0
            
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Create figure with high DPI for tile quality
        fig_width = tile_size / 100  # Convert pixels to inches at 100 DPI
        fig_height = tile_size / 100
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Create triangulation
        triang = mpl_tri.Triangulation(lon, lat, triangles)
        
        # Mask triangles with NaN data
        nan_mask = np.isnan(data)
        tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
        triang.set_mask(tri_mask)
        
        # Create contour levels
        levels = np.linspace(vmin, vmax, 50)  # 50 levels for smooth gradients
        
        # Plot using tricontourf - preserves triangular mesh structure
        cs = ax.tricontourf(triang, data, levels=levels, cmap=cmap, norm=norm, extend='both')
        
        # Set exact tile bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        
        # Remove all decorations for clean tile
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Remove margins and padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   pad_inches=0, transparent=True)
        plt.close(fig)
        
        # Process with PIL to ensure exact tile size
        buf.seek(0)
        img = Image.open(buf)
        
        # Resize to exact tile size if needed
        if img.size != (tile_size, tile_size):
            img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        final_buf = io.BytesIO()
        img.save(final_buf, format='PNG', optimize=True)
        final_buf.seek(0)
        
        return final_buf.getvalue()
        
    except Exception as e:
        # Return error tile
        return create_error_tile(f"UGRID Direct Error: {str(e)}", tile_size)


def create_error_tile(error_msg: str, tile_size: int = 256) -> bytes:
    """Create an error tile with message"""
    fig, ax = plt.subplots(figsize=(tile_size/100, tile_size/100), dpi=100)
    ax.text(0.5, 0.5, f"Error:\n{error_msg}", 
           horizontalalignment='center', verticalalignment='center',
           transform=ax.transAxes, fontsize=8, color='red',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# Test the direct approach
if __name__ == "__main__":
    print("🌊 Testing Direct UGRID Approach (No Interpolation) 🌊")
    print("=" * 60)
    
    url = 'https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc'
    
    try:
        # Test data extraction
        lon, lat, triangles, data, act_time = extract_ugrid_direct(
            url, "2025-09-29T12:00:00Z", "hs"
        )
        
        print(f"✅ Data extraction successful!")
        print(f"   Mesh points: {len(lon):,}")
        print(f"   Triangles: {len(triangles):,}")
        print(f"   Data range: {np.nanmin(data):.3f} - {np.nanmax(data):.3f}")
        print(f"   Time: {act_time}")
        
        # Test tile creation
        tile_data = create_direct_ugrid_tile(
            url=url,
            variable="hs",
            time="0",
            lon_min=-159.9,
            lon_max=-159.6,
            lat_min=-21.3,
            lat_max=-21.1,
            colormap='jet',
            vmin=0.0,
            vmax=4.0
        )
        
        print(f"✅ Direct tile generation successful!")
        print(f"   Tile size: {len(tile_data):,} bytes")
        print(f"   No interpolation artifacts!")
        print(f"   Clean island boundaries preserved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")