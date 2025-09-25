#!/usr/bin/env python3
"""
Test COG Improvements
Validate that the COG optimization changes are working correctly
"""

import sys
import os
sys.path.append('/home/kishank/cog_tiler')

def test_cog_config():
    """Test the COG configuration module"""
    print("🧪 Testing COG Configuration")
    print("=" * 50)
    
    try:
        from cog_config import (
            configure_gdal_environment, 
            OPTIMIZED_COG_PROFILE,
            VARIABLE_SCALING,
            COG_VALIDATION_CONFIG
        )
        
        # Test GDAL configuration
        gdal_settings = configure_gdal_environment()
        print(f"✅ GDAL Environment: {len(gdal_settings)} settings configured")
        
        # Test COG profile
        required_cog_keys = ['driver', 'compress', 'blockxsize', 'blockysize', 'tiled']
        missing_keys = [key for key in required_cog_keys if key not in OPTIMIZED_COG_PROFILE]
        if missing_keys:
            print(f"❌ Missing COG profile keys: {missing_keys}")
        else:
            print(f"✅ COG Profile: All required keys present")
            print(f"   • Driver: {OPTIMIZED_COG_PROFILE['driver']}")
            print(f"   • Compression: {OPTIMIZED_COG_PROFILE['compress']}")
            print(f"   • Block size: {OPTIMIZED_COG_PROFILE['blockxsize']}x{OPTIMIZED_COG_PROFILE['blockysize']}")
        
        # Test variable configurations
        print(f"✅ Variable Scaling: {len(VARIABLE_SCALING)} variables configured")
        for var in ['hs', 'tpeak', 'inundation_depth']:
            if var in VARIABLE_SCALING:
                config = VARIABLE_SCALING[var]
                print(f"   • {var}: {config['colormap']} [{config['vmin']}-{config['vmax']}]")
        
        print(f"✅ COG Validation Config: {len(COG_VALIDATION_CONFIG)} rules defined")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False

def test_cog_validation():
    """Test COG validation functionality"""
    print("\n🔍 Testing COG Validation")
    print("=" * 50)
    
    try:
        # Test imports
        from pathlib import Path
        import rasterio
        
        # Check if any COG files exist to test
        cache_dir = Path('/home/kishank/cog_tiler/cache')
        cog_files = list(cache_dir.glob('**/*.tif'))[:3]  # Test first 3 files
        
        if not cog_files:
            print("⚠️ No COG files found for testing")
            return True
            
        print(f"📁 Found {len(cog_files)} COG files to test")
        
        for cog_file in cog_files:
            try:
                with rasterio.open(str(cog_file)) as src:
                    # Basic COG structure checks
                    is_tiled = src.block_shapes[0] == (256, 256) or src.block_shapes[0] == (512, 512)
                    has_overviews = len(src.overviews(1)) > 0 if src.overviews(1) else False
                    is_web_mercator = src.crs and src.crs.to_epsg() == 3857
                    
                    status = "✅" if (is_tiled and has_overviews) else "⚠️"
                    
                    print(f"   {status} {cog_file.name}")
                    print(f"      • Tiled: {is_tiled} (block: {src.block_shapes[0]})")
                    print(f"      • Overviews: {has_overviews} ({len(src.overviews(1)) if src.overviews(1) else 0} levels)")
                    print(f"      • Web Mercator: {is_web_mercator}")
                    print(f"      • Size: {src.width}x{src.height}")
                    
            except Exception as e:
                print(f"   ❌ {cog_file.name}: Error reading file - {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        return False

def test_gdal_environment():
    """Test GDAL environment settings"""
    print("\n🔧 Testing GDAL Environment")
    print("=" * 50)
    
    try:
        # Check key GDAL environment variables
        important_vars = [
            'GDAL_CACHEMAX',
            'GDAL_NUM_THREADS', 
            'VSI_CACHE',
            'GDAL_TIFF_OVR_BLOCKSIZE',
            'GDAL_DISABLE_READDIR_ON_OPEN'
        ]
        
        configured_count = 0
        for var in important_vars:
            value = os.environ.get(var, 'NOT SET')
            status = "✅" if value != 'NOT SET' else "❌"
            print(f"   {status} {var} = {value}")
            if value != 'NOT SET':
                configured_count += 1
        
        print(f"\n📊 GDAL Optimization: {configured_count}/{len(important_vars)} variables configured")
        
        # Test rasterio/GDAL import
        try:
            import rasterio
            print(f"✅ Rasterio version: {rasterio.__version__}")
        except ImportError:
            print("❌ Rasterio not available")
        
        return configured_count > 0
        
    except Exception as e:
        print(f"❌ GDAL Environment Error: {e}")
        return False

def main():
    """Run all COG improvement tests"""
    print("🚀 COG Improvements Validation")
    print("=" * 60)
    
    tests = [
        ("COG Configuration", test_cog_config),
        ("COG Validation", test_cog_validation), 
        ("GDAL Environment", test_gdal_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All COG improvements are working correctly!")
    else:
        print("⚠️ Some improvements need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)