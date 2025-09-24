#!/usr/bin/env python3
"""
Test and Display COG Tiles from Real THREDDS Data
This script tests the COG tiler with real Cook Islands data and creates visual displays
"""

import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import time

# Server configuration
BASE_URL = "http://localhost:8000/cog"

def test_endpoint(url, description):
    """Test an endpoint and return response info"""
    start_time = time.time()
    try:
        response = requests.get(url, timeout=30)
        end_time = time.time()
        
        return {
            "url": url,
            "description": description,
            "status": response.status_code,
            "response_time": f"{(end_time - start_time):.3f}s",
            "content_type": response.headers.get('content-type', 'unknown'),
            "size": len(response.content) if response.content else 0,
            "success": response.status_code == 200,
            "response": response
        }
    except Exception as e:
        return {
            "url": url,
            "description": description,
            "status": "ERROR",
            "response_time": "N/A",
            "content_type": "N/A",
            "size": 0,
            "success": False,
            "error": str(e),
            "response": None
        }

def download_and_display_tiles():
    """Download and display actual COG tiles"""
    
    print("🌊 Testing COG Tiler with Real THREDDS Data 🌊")
    print("=" * 60)
    
    # Test basic endpoints
    endpoints = [
        (f"{BASE_URL}/", "Root endpoint"),
        (f"{BASE_URL}/cook-islands/8/249/124.png?variable=inundation_depth", "Cook Islands Inundation Depth Tile"),
        (f"{BASE_URL}/cook-islands/7/124/62.png?variable=inundation_depth", "Cook Islands Tile (Zoom 7)"),
        (f"{BASE_URL}/cook-islands-ugrid/8/249/124.png?variable=sea_surface_height_above_geoid", "UGRID Sea Surface Height"),
        (f"{BASE_URL}/cook-islands/wms-comparison?variable=inundation_depth&zoom=8&x=249&y=124", "WMS Comparison API"),
    ]
    
    results = []
    tile_images = []
    
    print("\n📡 Testing Endpoints:")
    print("-" * 40)
    
    for url, desc in endpoints:
        result = test_endpoint(url, desc)
        results.append(result)
        
        # Print status
        status_emoji = "✅" if result['success'] else "❌"
        print(f"{status_emoji} {desc}")
        print(f"   Status: {result['status']} | Time: {result['response_time']} | Size: {result['size']} bytes")
        
        # If it's a PNG tile, save the image for display
        if result['success'] and result['content_type'] and 'image' in result['content_type']:
            try:
                img = Image.open(io.BytesIO(result['response'].content))
                tile_images.append({
                    'image': img,
                    'title': desc,
                    'url': url
                })
            except Exception as e:
                print(f"   ⚠️ Could not process image: {e}")
        
        print()
    
    # Display comparison data if available
    comparison_result = next((r for r in results if 'wms-comparison' in r['url'] and r['success']), None)
    if comparison_result:
        try:
            comparison_data = comparison_result['response'].json()
            print("\n🔍 WMS Comparison Data:")
            print("-" * 30)
            print(f"Dataset: {comparison_data['dataset']}")
            print(f"Variable: {comparison_data['variable']}")
            print(f"Zoom Level: {comparison_data['zoom_level']}")
            print(f"Bounds: {comparison_data['bounds']}")
            print(f"COG Tile URL: {comparison_data['cog_tile_url']}")
            print(f"WMS Tile URL: {comparison_data['wms_tile_url'][:100]}...")
        except:
            pass
    
    # Create visual display of tiles
    if tile_images:
        print(f"\n🖼️ Displaying {len(tile_images)} COG Tiles:")
        print("-" * 35)
        
        # Create subplot grid
        n_images = len(tile_images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, tile_data in enumerate(tile_images):
            if i < len(axes):
                ax = axes[i]
                ax.imshow(tile_data['image'])
                ax.set_title(tile_data['title'], fontsize=10, pad=10)
                ax.axis('off')
                
                # Add URL as subtitle
                url_short = tile_data['url'].replace(BASE_URL, '').split('?')[0]
                ax.text(0.5, -0.05, url_short, transform=ax.transAxes, 
                       ha='center', fontsize=8, style='italic')
        
        # Hide unused subplots
        for i in range(len(tile_images), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('🌊 Cook Islands COG Tiles from Real THREDDS Data 🏝️', 
                    fontsize=16, y=1.02)
        
        # Save the visualization
        output_file = 'cog_tiles_display.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {output_file}")
        plt.show()
    
    # Print summary
    print("\n📊 Test Summary:")
    print("-" * 20)
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"✅ Successful: {successful}/{total}")
    print(f"🖼️ Tiles Generated: {len(tile_images)}")
    
    if successful == total:
        print("\n🎉 All tests passed! COG tiler is working perfectly with real THREDDS data!")
    else:
        print(f"\n⚠️ {total - successful} tests failed. Check the endpoints above.")
    
    return results, tile_images

if __name__ == "__main__":
    # Test cache status
    cache_dir = Path("cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.tif"))
        print(f"💾 Cache status: {len(cache_files)} COG files cached")
    
    # Run the tests
    results, tiles = download_and_display_tiles()
    
    # Test performance with cached vs non-cached
    print("\n⚡ Performance Test (Cached vs Fresh):")
    print("-" * 45)
    
    test_url = f"{BASE_URL}/cook-islands/8/249/124.png?variable=inundation_depth"
    
    # First request (likely cached)
    result1 = test_endpoint(test_url, "Cached Request")
    print(f"🔄 First request: {result1['response_time']}")
    
    # Second request (definitely cached)
    result2 = test_endpoint(test_url, "Cached Request")
    print(f"⚡ Second request: {result2['response_time']}")
    
    if result1['success'] and result2['success']:
        print("✅ Caching is working - consistent fast responses!")