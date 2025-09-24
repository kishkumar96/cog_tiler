#!/usr/bin/env python3
"""
Test real THREDDS data visualization with correct coordinates
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pyproj import Transformer
import requests

def test_real_thredds_visualization():
    """Test visualization of real Cook Islands data"""
    
    print("🌊 Testing Real Cook Islands THREDDS Data Visualization 🌊")
    print("=" * 60)
    
    url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"
    
    try:
        # Load the dataset
        print("📡 Loading dataset from THREDDS...")
        ds = xr.open_dataset(url)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Variables: {list(ds.data_vars.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        
        # Get the data
        data = ds.Band1
        print(f"Data shape: {data.shape}")
        print(f"Data range: {data.min().values:.3f} to {data.max().values:.3f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🏝️ Cook Islands Real THREDDS Data - Multiple Views 🌊', fontsize=16)
        
        # 1. Raw data plot
        ax1 = axes[0, 0]
        im1 = ax1.imshow(data.values, origin='upper', cmap='viridis')
        ax1.set_title('Raw Data (UTM Coordinates)')
        ax1.set_xlabel('X (UTM)')
        ax1.set_ylabel('Y (UTM)')
        plt.colorbar(im1, ax=ax1, label='Band1 Values')
        
        # 2. Data with valid values highlighted
        ax2 = axes[0, 1]
        valid_data = data.where(~np.isnan(data.values) & (data.values != 0))
        im2 = ax2.imshow(valid_data.values, origin='upper', cmap='plasma')
        ax2.set_title('Valid Data Only (Non-zero, Non-NaN)')
        ax2.set_xlabel('X (UTM)')
        ax2.set_ylabel('Y (UTM)')
        plt.colorbar(im2, ax=ax2, label='Valid Band1 Values')
        
        # 3. Geographic projection
        ax3 = axes[1, 0]
        # Convert UTM to lat/lon for display
        transformer = Transformer.from_crs('EPSG:32704', 'EPSG:4326', always_xy=True)
        
        x_utm = ds.x.values
        y_utm = ds.y.values
        
        # Sample conversion for corners
        x_min, x_max = x_utm.min(), x_utm.max()
        y_min, y_max = y_utm.min(), y_utm.max()
        
        lon_min, lat_max = transformer.transform(x_min, y_max)
        lon_max, lat_min = transformer.transform(x_max, y_min)
        
        extent = [lon_min, lon_max, lat_min, lat_max]
        
        im3 = ax3.imshow(data.values, origin='upper', cmap='ocean', extent=extent)
        ax3.set_title('Geographic View (Lat/Lon)')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(im3, ax=ax3, label='Band1 Values')
        
        # 4. Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        total_pixels = data.size
        valid_pixels = np.count_nonzero(~np.isnan(data.values))
        nonzero_pixels = np.count_nonzero(data.values)
        
        stats_text = f"""
📊 Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━
🌐 Dataset URL: 
   {url.split('/')[-1]}

📍 Spatial Info:
   UTM Zone 4S coordinates
   X: {x_min:.0f} to {x_max:.0f} m
   Y: {y_min:.0f} to {y_max:.0f} m
   
   Lat: {lat_min:.6f}° to {lat_max:.6f}°
   Lon: {lon_min:.6f}° to {lon_max:.6f}°

📈 Data Info:
   Total pixels: {total_pixels:,}
   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)
   Non-zero pixels: {nonzero_pixels:,} ({nonzero_pixels/total_pixels*100:.1f}%)
   
   Min value: {data.min().values:.3f}
   Max value: {data.max().values:.3f}
   Mean value: {data.mean().values:.3f}

✅ THREDDS Connection: SUCCESS
✅ Data Loading: SUCCESS  
✅ Visualization: SUCCESS
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = 'real_thredds_data_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved analysis: {output_file}")
        plt.show()
        
        # Also test a simple tile generation with correct bounds
        print(f"\\n🎯 Testing Tile Generation with Correct Bounds:")
        print(f"   Using bounds: Lat {lat_min:.6f} to {lat_max:.6f}")
        print(f"                 Lon {lon_min:.6f} to {lon_max:.6f}")
        
        # Calculate correct tile for zoom 8
        import math
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        def deg2num(lat_deg, lon_deg, zoom):
            lat_rad = math.radians(lat_deg)
            n = 2.0 ** zoom
            xtile = int((lon_deg + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (xtile, ytile)
        
        for zoom in [6, 7, 8, 9, 10]:
            x, y = deg2num(center_lat, center_lon, zoom)
            print(f"   Zoom {zoom}: x={x}, y={y}")
            
        return {
            'success': True,
            'bounds': {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max},
            'center': {'lat': center_lat, 'lon': center_lon},
            'data_stats': {
                'total_pixels': int(total_pixels),
                'valid_pixels': int(valid_pixels), 
                'nonzero_pixels': int(nonzero_pixels),
                'min': float(data.min().values),
                'max': float(data.max().values),
                'mean': float(data.mean().values)
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = test_real_thredds_visualization()
    
    if result['success']:
        print("\\n🎉 SUCCESS: Real THREDDS data analysis complete!")
        print("The dataset is working and contains valid data.")
    else:
        print(f"\\n❌ FAILED: {result['error']}")