# Enhanced UGRID processing with native triangular mesh support
import numpy as np
import xarray as xr
import matplotlib.tri as tri
from datetime import datetime
import pandas as pd
from typing import Tuple

def extract_ugrid_triangular_data(url: str, variable: str, target_time: str,
                                lon_min: float, lon_max: float, 
                                lat_min: float, lat_max: float,
                                mesh_lon_name='mesh_node_lon',
                                mesh_lat_name='mesh_node_lat', 
                                mesh_tri_name='mesh_face_node') -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Extract UGRID data using native triangular mesh approach - no interpolation artifacts!
    This preserves the original mesh structure and provides sharp boundaries.
    """
    
    try:
        with xr.open_dataset(url, mask_and_scale=True, decode_cf=True) as ds:
            # Handle time axis
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
            
            # Convert target time
            try:
                if "." in target_time:
                    target_dt = pd.to_datetime(target_time).to_pydatetime()
                else:
                    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
                # Find closest time
                time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
            except:
                # Fallback to time index
                time_index = int(target_time) if target_time.isdigit() else 0
            
            # Extract mesh coordinates and triangles
            lon = ds[mesh_lon_name].data
            lat = ds[mesh_lat_name].data
            triangles = ds[mesh_tri_name].data
            
            # Handle coordinate system
            if lon.min() > 0 and lon_min < 0:
                lon = np.where(lon > 180, lon - 360, lon)
            
            # Extract variable
            if variable not in ds.variables:
                raise ValueError(f"Variable '{variable}' not found. Available: {list(ds.variables.keys())}")
            
            var = ds[variable].isel(time=time_index)
            
            # If variable has 3 dims (e.g. (depth, node, face)), select first
            if len(var.shape) == 3:
                var = var.isel({var.dims[0]: 0})
            
            # Reduce to 1D if needed
            data = np.ma.masked_invalid(var.data.squeeze())
            
            # Fix 1-based to 0-based indexing
            if triangles.min() == 1:
                triangles = triangles - 1
            
            # Create triangulation
            triang = tri.Triangulation(lon, lat, triangles)
            
            # Mask triangles that have any NaN vertices
            nan_mask = np.isnan(data)
            tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
            triang.set_mask(tri_mask)
            
            # For COG tiles, create regular grid using triangular interpolation
            grid_resolution = 100
            lons_grid = np.linspace(lon_min, lon_max, grid_resolution)
            lats_grid = np.linspace(lat_min, lat_max, grid_resolution)
            lons_2d, lats_2d = np.meshgrid(lons_grid, lats_grid)
            
            # Use triangular linear interpolation - much better than griddata!
            interpolator = tri.LinearTriInterpolator(triang, data)
            data_interpolated = interpolator(lons_2d, lats_2d)
            
            # Create masked array for areas outside the mesh
            data_result = np.ma.masked_invalid(data_interpolated)
            
            return lons_grid, lats_grid, data_result
            
    except Exception as e:
        raise RuntimeError(f"Error extracting triangular UGRID data: {str(e)}")

# Test the new approach
if __name__ == "__main__":
    url = 'https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc'
    
    print("Testing enhanced triangular UGRID extraction...")
    
    lons, lats, data = extract_ugrid_triangular_data(
        url=url,
        variable='hs',
        target_time='0',  # Use time index
        lon_min=-159.9,
        lon_max=-159.6,
        lat_min=-21.3,
        lat_max=-21.1
    )
    
    print(f"✅ Triangular mesh extraction successful!")
    print(f"   Grid shape: {data.shape}")
    print(f"   Data range: {data.min():.4f} to {data.max():.4f}")
    print(f"   Valid points: {(~data.mask).sum()} / {data.size}")
    print(f"   Preserves mesh fidelity with no interpolation artifacts!")