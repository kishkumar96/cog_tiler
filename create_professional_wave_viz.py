#!/usr/bin/env python3
"""
Create proper continuous field visualization for Cook Islands wave forecast data
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

def create_professional_wave_viz():
    """Create professional continuous wave field visualization"""
    
    print("🌊 Creating Professional Wave Field Visualization 🌊")
    print("=" * 60)
    
    url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
    
    try:
        print("📡 Loading UGRID dataset...")
        ds = xr.open_dataset(url)
        
        # Get coordinates and latest data
        lons = ds.mesh_node_lon.values
        lats = ds.mesh_node_lat.values
        
        # Get latest time step
        latest_time = -1
        hs_data = ds.hs.isel(time=latest_time).values
        u10_data = ds.u10.isel(time=latest_time).values  # Wind eastward
        v10_data = ds.v10.isel(time=latest_time).values  # Wind northward
        
        # Calculate wind speed and direction
        wind_speed = np.sqrt(u10_data**2 + v10_data**2)
        wind_dir = np.arctan2(v10_data, u10_data) * 180 / np.pi
        
        print(f"✅ Data loaded: {len(lons)} mesh points")
        print(f"   Wave height: {np.nanmin(hs_data):.2f} - {np.nanmax(hs_data):.2f} m")
        print(f"   Wind speed: {np.nanmin(wind_speed):.1f} - {np.nanmax(wind_speed):.1f} m/s")
        
        # Create high-resolution grid for interpolation
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        # Create regular grid
        grid_res = 100  # Grid resolution
        lon_grid = np.linspace(lon_min, lon_max, grid_res)
        lat_grid = np.linspace(lat_min, lat_max, grid_res)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        
        # Remove NaN values for interpolation
        valid_mask = ~(np.isnan(hs_data) | np.isnan(wind_speed) | np.isnan(lons) | np.isnan(lats))
        
        if np.sum(valid_mask) > 10:  # Need at least 10 valid points
            lons_valid = lons[valid_mask]
            lats_valid = lats[valid_mask]
            hs_valid = hs_data[valid_mask]
            wind_speed_valid = wind_speed[valid_mask]
            u10_valid = u10_data[valid_mask]
            v10_valid = v10_data[valid_mask]
            
            print(f"   Interpolating {np.sum(valid_mask)} valid points...")
            
            # Interpolate to regular grid
            hs_grid = griddata((lons_valid, lats_valid), hs_valid, (LON, LAT), method='linear', fill_value=np.nan)
            wind_speed_grid = griddata((lons_valid, lats_valid), wind_speed_valid, (LON, LAT), method='linear', fill_value=np.nan)
            u10_grid = griddata((lons_valid, lats_valid), u10_valid, (LON, LAT), method='linear', fill_value=np.nan)
            v10_grid = griddata((lons_valid, lats_valid), v10_valid, (LON, LAT), method='linear', fill_value=np.nan)
            
            # Create the visualization
            fig = plt.figure(figsize=(16, 12))
            
            # Main plot - Wave Height with Wind Vectors (like your reference)
            ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
            
            # Add map features
            ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
            ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            # Plot wave height as continuous field
            levels = np.linspace(0, np.nanmax(hs_valid), 20)
            contourf = ax1.contourf(LON, LAT, hs_grid, levels=levels, cmap='viridis', 
                                  transform=ccrs.PlateCarree(), alpha=0.8, extend='max')
            
            # Add wind vectors (subsample for clarity)
            skip = 8  # Every 8th point
            ax1.quiver(LON[::skip, ::skip], LAT[::skip, ::skip], 
                      u10_grid[::skip, ::skip], v10_grid[::skip, ::skip],
                      transform=ccrs.PlateCarree(), scale=50, alpha=0.7,
                      color='black', width=0.003)
            
            # Add colorbar
            plt.colorbar(contourf, ax=ax1, label='Significant Wave Height (m)', shrink=0.8)
            
            ax1.set_title('🌊 Wave Height + Wind Vectors\\n(Professional Field Display)', fontsize=12)
            ax1.set_extent([lon_min, lon_max, lat_min, lat_max])
            ax1.gridlines(draw_labels=True, alpha=0.3)
            
            # 2. Wind Speed Field
            ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
            ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
            ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            wind_levels = np.linspace(0, np.nanmax(wind_speed_valid), 15)
            wind_contourf = ax2.contourf(LON, LAT, wind_speed_grid, levels=wind_levels, 
                                       cmap='coolwarm', transform=ccrs.PlateCarree(), alpha=0.8)
            plt.colorbar(wind_contourf, ax=ax2, label='Wind Speed (m/s)', shrink=0.8)
            
            ax2.set_title('💨 Wind Speed Field', fontsize=12)
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])
            ax2.gridlines(draw_labels=True, alpha=0.3)
            
            # 3. Original scatter plot for comparison
            ax3 = plt.subplot(2, 2, 3)
            scatter = ax3.scatter(lons_valid, lats_valid, c=hs_valid, s=20, cmap='viridis', alpha=0.7)
            ax3.set_title('Original UGRID Mesh Points\\n(Raw Data Visualization)')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            plt.colorbar(scatter, ax=ax3, label='Wave Height (m)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Technical information
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')
            
            info_text = f"""
🌊 PROFESSIONAL WAVE FIELD VISUALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CONTINUOUS FIELD DISPLAY:
   • Interpolated to {grid_res}×{grid_res} regular grid
   • Smooth contour fills (not scattered dots)
   • Vector arrows for wind direction
   • Proper cartographic projection

📊 DATA PROCESSING:
   • Source: UGRID unstructured mesh
   • Valid points: {np.sum(valid_mask):,}
   • Interpolation: Linear gridding
   • Visualization: Matplotlib + Cartopy

🌊 CURRENT CONDITIONS:
   • Wave height: {np.nanmin(hs_valid):.2f} - {np.nanmax(hs_valid):.2f} m
   • Wind speed: {np.nanmin(wind_speed_valid):.1f} - {np.nanmax(wind_speed_valid):.1f} m/s
   • Coverage: Cook Islands region
   • Resolution: High-resolution interpolated

🎯 WHY CONTINUOUS FIELDS?
   • Matches professional weather displays
   • Shows spatial patterns clearly  
   • Better for decision-making
   • Standard oceanographic practice

💡 FOR COG TILES:
   Need to implement similar interpolation
   in the tile generation pipeline
            """
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
            
            plt.tight_layout()
            
            # Save the professional visualization
            output_file = 'professional_wave_field_display.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved professional visualization: {output_file}")
            plt.show()
            
            print("\\n🎉 PROFESSIONAL WAVE FIELD VISUALIZATION COMPLETE!")
            print("✅ Continuous field display (not dots)")
            print("✅ Wind vector arrows")
            print("✅ Proper cartographic projection") 
            print("✅ Smooth interpolated contours")
            
            return True
            
        else:
            print("❌ Not enough valid data points for interpolation")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_professional_wave_viz()
    
    if success:
        print("\\n" + "="*60)
        print("🌊 SOLUTION: CONTINUOUS FIELD VISUALIZATION! 🌊") 
        print("="*60)
        print("The issue was using scattered points instead of")
        print("interpolated continuous fields. The new visualization")
        print("matches professional oceanographic displays with:")
        print("• Smooth contour-filled areas")
        print("• Vector arrows for wind/wave direction") 
        print("• Proper cartographic projection")
        print("• No more dots - continuous fields!")
    else:
        print("\\n❌ Failed to create professional visualization")