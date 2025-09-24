#!/usr/bin/env python3
"""
Cook Islands Local Test - Generate synthetic inundation data for testing
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import os

def create_cook_islands_test_data():
    """Create synthetic inundation depth data for Rarotonga"""
    
    # Rarotonga bounds
    lon_min, lon_max = -159.9, -159.6
    lat_min, lat_max = -21.3, -21.1
    
    # Create coordinate arrays
    lons = np.linspace(lon_min, lon_max, 100)
    lats = np.linspace(lat_min, lat_max, 80)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create realistic inundation pattern
    # Simulate coastal inundation (higher near coastline)
    center_lon, center_lat = -159.75, -21.2  # Approximate Rarotonga center
    
    # Distance from center (simulating island shape)
    dist_from_center = np.sqrt((lon_grid - center_lon)**2 + (lat_grid - center_lat)**2)
    
    # Create inundation pattern:
    # - Higher inundation near coast (certain distance from center)
    # - Lower inundation inland and offshore
    coastal_distance = 0.05  # Degrees from center where coast would be
    inundation = np.where(
        (dist_from_center > coastal_distance * 0.8) & 
        (dist_from_center < coastal_distance * 1.5),
        # Coastal areas - high inundation risk
        2.0 + np.random.normal(0, 0.5, lon_grid.shape),
        # Inland/offshore - lower risk
        0.1 + np.maximum(0, np.random.normal(0, 0.3, lon_grid.shape))
    )
    
    # Add some hotspots of high inundation
    hotspot1_lon, hotspot1_lat = -159.8, -21.15
    hotspot2_lon, hotspot2_lat = -159.7, -21.25
    
    for hs_lon, hs_lat in [(hotspot1_lon, hotspot1_lat), (hotspot2_lon, hotspot2_lat)]:
        hs_dist = np.sqrt((lon_grid - hs_lon)**2 + (lat_grid - hs_lat)**2)
        hotspot_mask = hs_dist < 0.02
        inundation[hotspot_mask] += 3.0 * np.exp(-hs_dist[hotspot_mask] / 0.01)
    
    # Ensure non-negative and cap at reasonable maximum
    inundation = np.clip(inundation, 0, 8.0)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'inundation_depth': (['lat', 'lon'], inundation),
        },
        coords={
            'lon': lons,
            'lat': lats,
            'time': datetime.now()
        },
        attrs={
            'title': 'Synthetic Rarotonga Inundation Depth Data',
            'description': 'Test dataset for Cook Islands COG visualization',
            'source': 'Generated for testing purposes',
            'units': 'meters'
        }
    )
    
    # Add variable attributes
    ds['inundation_depth'].attrs = {
        'long_name': 'Sea Level Inundation Depth',
        'units': 'meters',
        'description': 'Predicted sea level inundation depth above ground level'
    }
    
    return ds

def save_and_test_dataset():
    """Save test dataset and verify it works with COG tiler"""
    
    print("=== Creating Cook Islands Test Dataset ===")
    ds = create_cook_islands_test_data()
    
    # Save to NetCDF
    output_file = 'cook_islands_test_data.nc'
    ds.to_netcdf(output_file)
    print(f"✅ Saved test dataset: {output_file}")
    
    # Display info
    print(f"\nDataset Info:")
    print(f"  Variables: {list(ds.data_vars.keys())}")
    print(f"  Coordinates: {list(ds.coords)}")
    print(f"  Inundation range: {ds['inundation_depth'].min().values:.2f} - {ds['inundation_depth'].max().values:.2f} m")
    print(f"  Spatial extent: lon({ds.lon.min().values:.3f}, {ds.lon.max().values:.3f}), lat({ds.lat.min().values:.3f}, {ds.lat.max().values:.3f})")
    
    # Create a quick visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(ds.lon, ds.lat, ds.inundation_depth, 
                    levels=[0, 0.5, 1, 2, 3, 5, 8],
                    cmap='viridis', 
                    antialiased=True)
    plt.colorbar(im, ax=ax, label='Inundation Depth (m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude') 
    ax.set_title('Cook Islands (Rarotonga) - Synthetic Inundation Depth')
    ax.grid(True, alpha=0.3)
    
    viz_file = 'cook_islands_test_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {viz_file}")
    plt.close()
    
    return output_file

if __name__ == "__main__":
    test_file = save_and_test_dataset()
    print(f"\n=== Next Steps ===")
    print(f"1. Use this test file: file://{os.path.abspath(test_file)}")
    print(f"2. Test with COG tiler: /cog/cook-islands/10/57/573.png")
    print(f"3. Once network access is fixed, switch back to: gemthredsshpc.spc.int")