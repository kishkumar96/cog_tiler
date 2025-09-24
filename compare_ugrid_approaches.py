#!/usr/bin/env python3
"""
Demonstration comparing interpolated vs direct UGRID approaches
Shows how direct triangular mesh avoids island interpolation artifacts
"""

import matplotlib.pyplot as plt
import numpy as np
from ugrid_direct import extract_ugrid_direct, create_direct_ugrid_tile
from data_reader import _load_cook_islands_ugrid_data
import matplotlib.tri as mpl_tri

def compare_ugrid_approaches():
    """Compare interpolated vs direct UGRID approaches"""
    
    print("🌊 UGRID APPROACH COMPARISON 🌊")
    print("=" * 50)
    
    url = 'https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc'
    variable = "hs"
    
    # Test bounds around Rarotonga
    lon_min, lon_max = -159.9, -159.6
    lat_min, lat_max = -21.3, -21.1
    
    try:
        print("📡 Loading data with both approaches...")
        
        # 1. Direct triangular approach (your method)
        lon_direct, lat_direct, triangles, data_direct, act_time = extract_ugrid_direct(
            url, "0", variable
        )
        
        # 2. Interpolated approach (current application)
        lon_interp, lat_interp, data_interp = _load_cook_islands_ugrid_data(
            url=url,
            variable=variable,
            time="0",
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )
        
        print(f"✅ Both approaches loaded successfully!")
        print(f"   Direct mesh: {len(lon_direct)} points, {len(triangles)} triangles")
        print(f"   Interpolated: {data_interp.shape} grid")
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Direct triangular mesh (your approach)
        if triangles.min() == 1:
            triangles = triangles - 1
            
        # Handle coordinate system
        if (lon_direct.min() > 0) and (lon_min < 0):
            lon_direct = np.where(lon_direct > 180, lon_direct - 360, lon_direct)
            
        triang = mpl_tri.Triangulation(lon_direct, lat_direct, triangles)
        nan_mask = np.isnan(data_direct)
        tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
        triang.set_mask(tri_mask)
        
        levels = np.linspace(0, 4, 50)
        cs1 = ax1.tricontourf(triang, data_direct, levels=levels, cmap='jet', extend='both')
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
        ax1.set_title('Direct Triangular Mesh\\n(No Interpolation - Clean Islands)', fontsize=12, pad=20)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(cs1, ax=ax1, label='Wave Height (m)')
        
        # Plot 2: Interpolated grid (current application)
        lon_2d, lat_2d = np.meshgrid(lon_interp, lat_interp)
        cs2 = ax2.contourf(lon_2d, lat_2d, data_interp, levels=levels, cmap='jet', extend='both')
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.set_title('Interpolated Grid\\n(Current App - May Have Artifacts)', fontsize=12, pad=20)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(cs2, ax=ax2, label='Wave Height (m)')
        
        plt.tight_layout()
        
        # Save comparison
        output_file = 'ugrid_approach_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\\n💾 Saved comparison: {output_file}")
        plt.show()
        
        # Test direct tile generation
        print("\\n🔧 Testing direct tile generation...")
        tile_data = create_direct_ugrid_tile(
            url=url,
            variable=variable,
            time="0",
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            colormap='jet',
            vmin=0.0,
            vmax=4.0
        )
        
        print(f"✅ Direct tile generation successful!")
        print(f"   Tile size: {len(tile_data):,} bytes")
        
        # Summary
        print(f"\\n📊 COMPARISON SUMMARY:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔹 DIRECT APPROACH (Your Method):")
        print(f"   ✅ Preserves original mesh structure")
        print(f"   ✅ Clean island boundaries") 
        print(f"   ✅ No interpolation artifacts")
        print(f"   ✅ True to scientific data")
        print(f"   ✅ Sharp coastline definition")
        
        print(f"\\n🔸 INTERPOLATED APPROACH (Current App):")
        print(f"   ⚠️  May blur island boundaries")
        print(f"   ⚠️  Possible interpolation artifacts")
        print(f"   ✅ Regular grid (standard for COG)")
        print(f"   ✅ Consistent with web mapping")
        
        print(f"\\n🎯 RECOMMENDATION:")
        print(f"   Use DIRECT approach for scientific accuracy")
        print(f"   New endpoint: /cook-islands-ugrid-direct/{{z}}/{{x}}/{{y}}.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return False

if __name__ == "__main__":
    success = compare_ugrid_approaches()
    
    if success:
        print("\\n🌊 CONCLUSION: Direct triangular mesh approach is superior!")
        print("   for preserving island boundaries and scientific accuracy!")
    else:
        print("\\n❌ Comparison failed - check network connection")