#!/usr/bin/env python3
"""
Cook Islands data coordinate system fix for tile generation.

This script fixes the coordinate system mismatch where Cook Islands THREDDS data 
is in UTM Zone 4S (EPSG:32704) but the tile system expects WGS84 (EPSG:4326).
"""

import numpy as np
import xarray as xr
from pyproj import Transformer
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def load_cook_islands_plot_ready_arrays(
    *,
    url: str,
    variable: str,
    time: Optional[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Load Cook Islands THREDDS data and convert from UTM to lat/lon coordinates.
    
    The Cook Islands data is in UTM Zone 4S (EPSG:32704) but tiles need WGS84 (EPSG:4326).
    This function handles the coordinate transformation.
    
    Returns:
    - lons_plot: 1D numpy array (longitude values)
    - lats_plot: 1D numpy array (latitude values)  
    - data_ma: 2D masked array aligned with (lats_plot, lons_plot)
    """
    logger.info(f"Loading Cook Islands data from {url}")
    
    with xr.open_dataset(url) as ds:
        if variable not in ds:
            available_vars = list(ds.data_vars.keys())
            logger.error(f"Variable '{variable}' not found. Available: {available_vars}")
            raise KeyError(f"Variable '{variable}' not found in dataset")
        
        # Get the data variable
        da = ds[variable]
        logger.info(f"Loaded variable {variable} with shape {da.shape}")
        
        # Handle time dimension if present
        if time is not None and 'time' in da.dims:
            try:
                # Convert time to pandas timestamp for selection
                import pandas as pd
                time_target = pd.to_datetime(time)
                da = da.sel(time=time_target, method='nearest')
                logger.info(f"Selected time slice: {time}")
            except Exception as e:
                logger.warning(f"Time selection failed: {e}, using first time slice")
                if 'time' in da.dims:
                    da = da.isel(time=0)
        elif 'time' in da.dims:
            # No specific time requested, use first time slice
            da = da.isel(time=0)
            logger.info("Using first available time slice")
        
        # Get UTM coordinates
        x_utm = ds.x.values  # UTM Easting
        y_utm = ds.y.values  # UTM Northing
        
        logger.info(f"UTM X range: {x_utm.min()} to {x_utm.max()}")
        logger.info(f"UTM Y range: {y_utm.min()} to {y_utm.max()}")
        
        # Set up coordinate transformation: UTM Zone 4S -> WGS84
        transformer = Transformer.from_crs('EPSG:32704', 'EPSG:4326', always_xy=True)
        
        # Create meshgrid of UTM coordinates
        x_mesh, y_mesh = np.meshgrid(x_utm, y_utm)
        
        # Transform all points to lat/lon
        logger.info("Transforming coordinates from UTM to WGS84...")
        lon_mesh, lat_mesh = transformer.transform(x_mesh.ravel(), y_mesh.ravel())
        lon_mesh = lon_mesh.reshape(x_mesh.shape)
        lat_mesh = lat_mesh.reshape(y_mesh.shape)
        
        logger.info(f"Geographic Longitude range: {lon_mesh.min()} to {lon_mesh.max()}")
        logger.info(f"Geographic Latitude range: {lat_mesh.min()} to {lat_mesh.max()}")
        
        # Create 1D coordinate arrays for the bounds of the data
        # Use the transformed coordinate bounds
        lons_plot = np.linspace(lon_mesh.min(), lon_mesh.max(), len(x_utm))
        lats_plot = np.linspace(lat_mesh.min(), lat_mesh.max(), len(y_utm))
        
        # Get the data values
        if hasattr(da, 'values'):
            data_values = da.values
        else:
            data_values = da.data
            
        # Handle data orientation - NetCDF often has (y, x) while we want (lat, lon)
        if data_values.ndim == 2:
            # Ensure data is oriented properly (should be (y, x) from NetCDF)
            if data_values.shape != (len(y_utm), len(x_utm)):
                logger.warning(f"Data shape {data_values.shape} doesn't match coordinate shape ({len(y_utm)}, {len(x_utm)})")
                data_values = data_values.T
        
        # Create masked array
        data_ma = np.ma.masked_invalid(data_values)
        
        # Filter to requested bounds if they're reasonable
        # (Only apply filtering if the bounds actually intersect with our data)
        lon_data_min, lon_data_max = lon_mesh.min(), lon_mesh.max()
        lat_data_min, lat_data_max = lat_mesh.min(), lat_mesh.max()
        
        if (lon_max >= lon_data_min and lon_min <= lon_data_max and 
            lat_max >= lat_data_min and lat_min <= lat_data_max):
            
            # Find indices for the requested bounds
            lon_mask = (lons_plot >= lon_min) & (lons_plot <= lon_max)
            lat_mask = (lats_plot >= lat_min) & (lats_plot <= lat_max)
            
            if lon_mask.any() and lat_mask.any():
                lon_indices = np.where(lon_mask)[0]
                lat_indices = np.where(lat_mask)[0]
                
                lons_plot = lons_plot[lon_indices]
                lats_plot = lats_plot[lat_indices]
                
                # Extract corresponding data subset
                lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1
                lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
                data_ma = data_ma[lat_start:lat_end, lon_start:lon_end]
                
                logger.info(f"Filtered to bounds: lon [{lons_plot.min():.4f}, {lons_plot.max():.4f}], "
                          f"lat [{lats_plot.min():.4f}, {lats_plot.max():.4f}]")
            else:
                logger.warning("No data found within requested bounds, using full extent")
        else:
            logger.info("Requested bounds don't intersect with data, using full extent")
        
        logger.info(f"Final data shape: {data_ma.shape}")
        logger.info(f"Data range: {data_ma.min():.4f} to {data_ma.max():.4f}")
        logger.info(f"Valid data points: {(~data_ma.mask).sum()} / {data_ma.size}")
        
        return lons_plot, lats_plot, data_ma


def test_cook_islands_data_loading():
    """Test function to verify the coordinate transformation works correctly."""
    
    # Test with inundation data
    url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"
    variable = "Band1"
    
    # Rarotonga bounds (approximate)
    lon_min, lon_max = -159.9, -159.6
    lat_min, lat_max = -21.3, -21.1
    
    try:
        lons, lats, data = load_cook_islands_plot_ready_arrays(
            url=url,
            variable=variable,
            time=None,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )
        
        print(f"Success! Loaded data with {len(lons)} longitude points and {len(lats)} latitude points")
        print(f"Data shape: {data.shape}")
        print(f"Data range: {data.min()} to {data.max()}")
        print(f"Coordinate ranges:")
        print(f"  Longitude: {lons.min():.4f} to {lons.max():.4f}")
        print(f"  Latitude: {lats.min():.4f} to {lats.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cook_islands_data_loading()