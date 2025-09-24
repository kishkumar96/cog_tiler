#!/usr/bin/env python3
"""
Test script to verify OPeNDAP access with proxy configuration
"""
import os
import xarray as xr
import requests
from urllib.parse import urlparse

# Set proxy environment variables
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:9000'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:9000'
os.environ['http_proxy'] = 'http://127.0.0.1:9000'
os.environ['https_proxy'] = 'http://127.0.0.1:9000'

print("=== Proxy Configuration Test ===")
print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")

# Test URLs
test_urls = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW_monthly.html",
    "http://apdrc.soest.hawaii.edu:80/dods/public_data/satellite_product/NOAA/nearsurface_wind/daily_10m_wind/uv10m.info",
    "https://thredds.jpl.nasa.gov/thredds/dodsC/OceanTemperature/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/2020/001/20200101090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc.html"
]

print("\n=== Testing URL Access ===")
for url in test_urls:
    try:
        print(f"\nTesting: {url}")
        response = requests.head(url, timeout=15, allow_redirects=True)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Success")
        else:
            print(f"⚠️  Got status {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n=== Testing xarray with OPeNDAP ===")
# Try a simple OPeNDAP dataset
opendap_url = "http://apdrc.soest.hawaii.edu:80/dods/public_data/satellite_product/NOAA/nearsurface_wind/daily_10m_wind/uv10m"

try:
    print(f"Attempting to open: {opendap_url}")
    
    # Configure xarray/dask to use proxy
    import dask
    dask.config.set({"array.slicing.split_large_chunks": False})
    
    # Try to open dataset
    ds = xr.open_dataset(opendap_url, engine='netcdf4')
    print("✅ Dataset opened successfully!")
    print("Variables:", list(ds.data_vars.keys())[:5])  # Show first 5 variables
    print("Dimensions:", dict(ds.dims))
    ds.close()
    
except Exception as e:
    print(f"❌ xarray failed: {e}")
    print("This might be due to:")
    print("1. Proxy configuration not properly passed to netcdf4/libcurl")
    print("2. Corporate firewall blocking OPeNDAP protocols")
    print("3. SSL certificate issues with proxy")

print("\n=== Test Complete ===")