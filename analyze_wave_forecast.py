#!/usr/bin/env python3
"""
Analyze and visualize Cook Islands wave forecast data from UGRID
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.tri import Triangulation
import warnings
warnings.filterwarnings('ignore')

def analyze_wave_forecast_data():
    """Comprehensive analysis of Cook Islands wave forecast data"""
    
    print("🌊 Analyzing Cook Islands Wave Forecast Data (UGRID) 🌊")
    print("=" * 60)
    
    url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
    
    try:
        # Load the dataset
        print("📡 Loading UGRID wave dataset from THREDDS...")
        ds = xr.open_dataset(url)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Variables: {len(ds.data_vars)} total")
        print(f"Coordinates: {list(ds.coords.keys())}")
        print(f"Dimensions: {dict(ds.sizes)}")
        
        # Get coordinate data
        lons = ds.mesh_node_lon.values
        lats = ds.mesh_node_lat.values
        times = ds.time.values
        
        print(f"\\n📍 Spatial Coverage:")
        print(f"   Longitude: {lons.min():.6f}° to {lons.max():.6f}°")
        print(f"   Latitude: {lats.min():.6f}° to {lats.max():.6f}°")
        print(f"   Grid points: {len(lons):,}")
        print(f"   Time steps: {len(times)}")
        print(f"   Time range: {str(times[0])[:19]} to {str(times[-1])[:19]}")
        
        # Wave variables analysis
        wave_vars = {
            'hs': 'Significant Wave Height',
            'tpeak': 'Peak Wave Period', 
            'tm02': 'Mean Wave Period',
            'dirm': 'Mean Wave Direction',
            'dirp': 'Peak Wave Direction',
            'u10': 'Wind Speed (East)',
            'v10': 'Wind Speed (North)'
        }
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Main title
        fig.suptitle('🌊 Cook Islands Wave Forecast Data Analysis 🏝️', fontsize=20, y=0.95)
        
        # Get latest time step data
        latest_time = -1
        
        # 1. Wave Height Map
        ax1 = plt.subplot(3, 3, 1)
        hs_data = ds.hs.isel(time=latest_time).values
        scatter = ax1.scatter(lons, lats, c=hs_data, s=2, cmap='viridis', vmin=0, vmax=np.nanmax(hs_data))
        ax1.set_title(f'Significant Wave Height\\n(Latest: {str(times[latest_time])[:16]})')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax1, label='Wave Height (m)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Wave Period Map  
        ax2 = plt.subplot(3, 3, 2)
        tp_data = ds.tpeak.isel(time=latest_time).values
        scatter2 = ax2.scatter(lons, lats, c=tp_data, s=2, cmap='plasma', vmin=0, vmax=np.nanmax(tp_data))
        ax2.set_title('Peak Wave Period')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(scatter2, ax=ax2, label='Period (s)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Wind Speed
        ax3 = plt.subplot(3, 3, 3)
        u10_data = ds.u10.isel(time=latest_time).values
        v10_data = ds.v10.isel(time=latest_time).values
        wind_speed = np.sqrt(u10_data**2 + v10_data**2)
        scatter3 = ax3.scatter(lons, lats, c=wind_speed, s=2, cmap='coolwarm', vmin=0, vmax=np.nanmax(wind_speed))
        ax3.set_title('Wind Speed (10m)')
        ax3.set_xlabel('Longitude') 
        ax3.set_ylabel('Latitude')
        plt.colorbar(scatter3, ax=ax3, label='Speed (m/s)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Time series at center point
        ax4 = plt.subplot(3, 3, 4)
        center_idx = len(lons) // 2  # Middle point
        hs_timeseries = ds.hs.isel(mesh_node=center_idx).values
        ax4.plot(range(len(hs_timeseries)), hs_timeseries, 'b-', linewidth=2)
        ax4.set_title(f'Wave Height Time Series\\n(Center Point: {lats[center_idx]:.3f}°, {lons[center_idx]:.3f}°)')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Wave Height (m)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Wave Direction (if available)
        ax5 = plt.subplot(3, 3, 5)
        if 'dirp' in ds:
            dirp_data = ds.dirp.isel(time=latest_time).values
            scatter5 = ax5.scatter(lons, lats, c=dirp_data, s=2, cmap='hsv', vmin=0, vmax=360)
            ax5.set_title('Peak Wave Direction')
            ax5.set_xlabel('Longitude')
            ax5.set_ylabel('Latitude')
            plt.colorbar(scatter5, ax=ax5, label='Direction (°)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Data Statistics
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
📊 Dataset Statistics
━━━━━━━━━━━━━━━━━━━━
🌐 Dataset: Rarotonga_UGRID.nc
📍 Grid Type: Unstructured (UGRID)
🗓️ Forecast Length: {len(times)} hours
⚓ Grid Points: {len(lons):,}

🌊 Wave Height (hs):
   Range: {np.nanmin(ds.hs.values):.2f} - {np.nanmax(ds.hs.values):.2f} m
   Mean: {np.nanmean(ds.hs.values):.2f} m

⏱️ Wave Period (tpeak):  
   Range: {np.nanmin(ds.tpeak.values):.1f} - {np.nanmax(ds.tpeak.values):.1f} s
   Mean: {np.nanmean(ds.tpeak.values):.1f} s

💨 Wind Speed:
   Range: {np.nanmin(wind_speed):.1f} - {np.nanmax(wind_speed):.1f} m/s
   Mean: {np.nanmean(wind_speed):.1f} m/s

✅ UGRID Connection: SUCCESS
✅ Wave Data: AVAILABLE
✅ Multi-variable: {len(wave_vars)} variables
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # 7-9. Variable comparison plots
        plot_idx = 7
        comparison_vars = ['hs', 'tpeak', 'u10']
        for var in comparison_vars:
            if var in ds and plot_idx <= 9:
                ax = plt.subplot(3, 3, plot_idx)
                data = ds[var].isel(time=latest_time).values
                
                # Histogram
                ax.hist(data[~np.isnan(data)], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{wave_vars.get(var, var)} Distribution')
                ax.set_xlabel(f'{ds[var].units}' if hasattr(ds[var], 'units') else 'Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        plt.tight_layout()
        
        # Save the analysis
        output_file = 'cook_islands_wave_forecast_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\\n💾 Saved wave analysis: {output_file}")
        plt.show()
        
        # Return useful information
        return {
            'success': True,
            'wave_variables': list(wave_vars.keys()),
            'spatial_bounds': {
                'lat_min': float(lats.min()),
                'lat_max': float(lats.max()),
                'lon_min': float(lons.min()), 
                'lon_max': float(lons.max())
            },
            'grid_info': {
                'grid_points': len(lons),
                'time_steps': len(times),
                'grid_type': 'UGRID_unstructured'
            },
            'data_ranges': {
                'wave_height': [float(np.nanmin(ds.hs.values)), float(np.nanmax(ds.hs.values))],
                'wave_period': [float(np.nanmin(ds.tpeak.values)), float(np.nanmax(ds.tpeak.values))],
                'wind_speed': [float(np.nanmin(wind_speed)), float(np.nanmax(wind_speed))]
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = analyze_wave_forecast_data()
    
    if result['success']:
        print("\\n🎉 SUCCESS: Wave forecast data analysis complete!")
        print("✅ Multiple wave variables available for COG tiling")
        print(f"✅ {result['grid_info']['grid_points']:,} spatial points")
        print(f"✅ {result['grid_info']['time_steps']} time steps of forecast")
        
        # Show recommended tile coordinates
        bounds = result['spatial_bounds']
        center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
        center_lon = (bounds['lon_min'] + bounds['lon_max']) / 2
        
        import math
        def deg2num(lat_deg, lon_deg, zoom):
            lat_rad = math.radians(lat_deg)
            n = 2.0 ** zoom
            xtile = int((lon_deg + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return (xtile, ytile)
        
        print(f"\\n🎯 Recommended tile coordinates for wave data:")
        print(f"   Center: {center_lat:.6f}°, {center_lon:.6f}°")
        for zoom in [7, 8, 9]:
            x, y = deg2num(center_lat, center_lon, zoom)
            print(f"   Zoom {zoom}: x={x}, y={y}")
            
    else:
        print(f"\\n❌ FAILED: {result['error']}")