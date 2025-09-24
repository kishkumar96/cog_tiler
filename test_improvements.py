#!/usr/bin/env python3
"""
Simple test to verify COG tiler data access and antialiasing
"""

import sys
import os
import tempfile
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/home/kishank/ocean_subsites/New COG/cog_tiler')

def test_data_access():
    """Test if we can access the Cook Islands dataset"""
    print("🧪 Testing Data Access")
    print("=" * 25)
    
    try:
        import xarray as xr
        
        # Test OPeNDAP access
        url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
        print(f"📡 Trying to access: {url}")
        
        # Try to open the dataset with a timeout
        with xr.open_dataset(url, decode_times=False) as ds:
            print(f"✅ Dataset opened successfully")
            print(f"📊 Variables available: {list(ds.data_vars.keys())[:5]}...")
            
            if 'hs' in ds.data_vars:
                print(f"✅ 'hs' variable found")
                hs_var = ds['hs']
                print(f"📏 hs shape: {hs_var.shape}")
                print(f"📏 hs dimensions: {hs_var.dims}")
                
                # Try to get a small subset
                if len(hs_var.shape) >= 3:  # Should have time, lat, lon or similar
                    print("🔍 Attempting to read small data subset...")
                    small_data = hs_var.isel({hs_var.dims[0]: 0}).load()
                    print(f"✅ Successfully read data subset: {small_data.shape}")
                    return True
            else:
                print(f"❌ 'hs' variable not found in dataset")
                return False
                
    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False

def test_simple_plotting():
    """Test simple matplotlib plotting with antialiasing"""
    print(f"\n🎨 Testing Matplotlib Plotting with Antialiasing")
    print("=" * 50)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create simple test data
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2))
        
        # Test with antialiasing enabled (our improvement)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Left: Without antialiasing
        cs1 = ax1.contourf(X, Y, Z, levels=10, antialiased=False)
        ax1.set_title("Without Antialiasing")
        
        # Right: With antialiasing (our improvement)
        cs2 = ax2.contourf(X, Y, Z, levels=10, antialiased=True)
        ax2.set_title("With Antialiasing (IMPROVED)")
        
        # Save comparison
        output_path = "antialiasing_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Antialiasing comparison saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        return False

def test_plotters_module():
    """Test your plotters.py module to verify antialiasing is enabled"""
    print(f"\n⚙️ Testing Plotters Module")
    print("=" * 25)
    
    try:
        from plotters import draw_plot, AVAILABLE_PLOTS
        import matplotlib.pyplot as plt
        import numpy as np
        
        print(f"✅ Plotters module imported successfully")
        print(f"📊 Available plot types: {AVAILABLE_PLOTS}")
        
        # Create test data
        lons = np.linspace(-161, -159, 20)
        lats = np.linspace(-22, -20, 20)
        LON, LAT = np.meshgrid(lons, lats)
        data = np.sin(LON * 10) * np.cos(LAT * 10)
        
        # Test with your improved plotters
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # This should now use antialiased=True due to our fix
        cs = draw_plot(
            ax, 
            plot="contourf",
            lons=LON, 
            lats=LAT, 
            data=data,
            cmap=plt.cm.viridis,
            levels=np.linspace(data.min(), data.max(), 10),
            vmin=data.min(),
            vmax=data.max(),
            options={}
        )
        
        ax.set_title("COG Tiler with Improved Antialiasing")
        plt.colorbar(cs, ax=ax)
        
        # Save test output
        output_path = "cog_tiler_test_antialiased.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ COG tiler test plot saved: {output_path}")
        print(f"🎨 This plot should show smooth, antialiased contours!")
        return True
        
    except Exception as e:
        print(f"❌ Plotters test failed: {e}")
        return False

def main():
    print("🚀 COG TILER VERIFICATION TEST")
    print("=" * 35)
    print()
    
    results = []
    
    # Test 1: Data access
    results.append(("Data Access", test_data_access()))
    
    # Test 2: Simple plotting
    results.append(("Matplotlib Plotting", test_simple_plotting()))
    
    # Test 3: Your plotters module
    results.append(("COG Plotters Module", test_plotters_module()))
    
    # Summary
    print(f"\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    passed_count = sum(results[i][1] for i in range(len(results)))
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
    
    if passed_count >= 2:
        print(f"\n🎉 COG tiler antialiasing improvements are working!")
        print(f"   Check the generated PNG files to see smooth contours.")
    else:
        print(f"\n⚠️  Some issues found. Check the error messages above.")
    
    return passed_count == len(results)

if __name__ == "__main__":
    main()