# Cook Islands COG Tiler 🏝️

A high-performance Cloud Optimized GeoTIFF (COG) tile server specifically enhanced for Cook Islands ocean visualization datasets, with professional WMS-quality rendering.

## 🎯 Key Features

- **Cook Islands Datasets**: Native support for Rarotonga inundation depth and UGRID data
- **Professional Quality**: Antialiased contours matching WMS visual standards  
- **Multiple Formats**: Regular grids + UGRID unstructured mesh support
- **Interactive Viewer**: Real-time tile testing with parameter adjustment
- **Network Resilient**: Local test data fallback for development
- **WMS Comparison**: Side-by-side quality validation tools

## 🚀 Quick Start (Codespace)

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv cog_venv
source cog_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Server
```bash
# Start COG tile server
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Server will be available at http://localhost:8001
```

### 3. Test Endpoints

#### Cook Islands Tiles
```bash
# Inundation depth tiles (regular grid)
curl "http://localhost:8001/cog/cook-islands/10/57/573.png?use_local=true" -o test_tile.png

# UGRID unstructured grid tiles  
curl "http://localhost:8001/cog/cook-islands-ugrid/10/57/573.png?use_local=true" -o ugrid_tile.png

# WMS comparison info
curl "http://localhost:8001/cog/cook-islands/wms-comparison" | jq
```

#### Interactive Web Viewer
Visit: `http://localhost:8001/cog/cook-islands-viewer`

### 4. Generate Test Data
```bash
# Create synthetic Cook Islands datasets
python3 create_cook_islands_test.py

# Run comprehensive integration tests
python3 test_cook_islands_integration.py
```

## 📊 Dataset Support

### Rarotonga_inundation_depth.nc
- **Purpose**: Sea level inundation depth prediction
- **Size**: ~19.7 MB  
- **Format**: Regular structured grid
- **Coverage**: lat(-21.3,-21.1), lon(-159.9,-159.6)
- **Variables**: `inundation_depth` (meters)

### Rarotonga_UGRID.nc  
- **Purpose**: High-resolution coastal modeling
- **Size**: ~88.4 MB
- **Format**: UGRID unstructured mesh
- **Coverage**: lat(-21.35,-21.05), lon(-159.95,-159.55)  
- **Variables**: `water_level`, `water_velocity`, `depth`

## 🛠️ API Endpoints

### Tile Endpoints
- `GET /cog/cook-islands/{z}/{x}/{y}.png` - Cook Islands inundation tiles
- `GET /cog/cook-islands-ugrid/{z}/{x}/{y}.png` - UGRID mesh tiles
- `GET /cog/tiles/dynamic/{z}/{x}/{y}.png` - Dynamic tiles with full parameters

### Info Endpoints  
- `GET /cog/cook-islands/wms-comparison` - WMS vs COG comparison data
- `GET /cog/cook-islands-viewer` - Interactive web interface
- `GET /cog/docs` - Full API documentation

### Parameters
- `variable`: Data variable to plot (e.g., `inundation_depth`, `water_level`)
- `colormap`: Matplotlib colormap (`viridis`, `RdYlBu_r`, `plasma`, etc.)
- `vmin`/`vmax`: Value range for color scaling
- `time`: Time slice (ISO format)
- `use_local`: Use local test data (default: `true`)

## 🔧 Configuration

### Dataset Configuration (`cook_islands_config.json`)
```json
{
  "datasets": {
    "rarotonga_inundation": {
      "name": "Rarotonga Inundation Depth",
      "url": "http://SERVER/thredds/dodsC/.../Rarotonga_inundation_depth.nc",
      "variables": {...},
      "bounds": {...}
    }
  }
}
```

### Environment Variables
```bash
# Proxy settings (if needed)
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Cache directory
export COG_CACHE_DIR=./cache
```

## 🧪 Testing & Development

### Run All Tests
```bash
# Integration test with visual outputs
python3 test_cook_islands_integration.py

# Antialiasing improvement demo
python3 create_demo_tiles.py

# Network connectivity test
python3 test_proxy_opendap.py
```

### Development Mode
```bash
# Run with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 8001 --reload --log-level debug
```

## 📁 Project Structure

```
cog_tiler/
├── main.py                              # FastAPI server + Cook Islands endpoints
├── plotters.py                          # Enhanced plotting with antialiasing
├── data_reader.py                       # NetCDF/OPeNDAP data loading
├── requirements.txt                     # Python dependencies
├── cook_islands_config.json             # Dataset configurations
├── cook_islands_viewer.html             # Interactive web interface
├── cook_islands_test_data.nc            # Synthetic test dataset (73KB)
├── COOK_ISLANDS_INTEGRATION_SUMMARY.md  # Complete documentation
├── create_cook_islands_test.py          # Test data generator
├── test_cook_islands_integration.py     # Integration test suite
└── visualizations/                      # Generated comparison images
    ├── COG_ANTIALIASING_IMPROVEMENT_DEMO.png
    ├── cook_islands_integration_test.png
    └── cook_islands_test_visualization.png
```

## 🌐 Network Configuration

### For Production (Real Data Access)
When network access to SPC THREDDS server is available:

1. **Update server URLs** in `main.py`:
   ```python
   # Replace with correct server name/IP
   cook_islands_url = "http://CORRECT_SERVER/thredds/dodsC/..."
   ```

2. **Set network parameters**:
   ```bash
   # Test real endpoints
   curl "http://localhost:8001/cog/cook-islands/10/57/573.png?use_local=false"
   ```

3. **Configure proxy** (if required):
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

## ✅ Quality Improvements

### Before vs After
- ❌ **Before**: Pixelated contours, no antialiasing
- ✅ **After**: Smooth, professional WMS-quality rendering

### Key Fixes Applied
- `antialiased=True` in matplotlib contourf
- Proper transparency handling
- Enhanced color mapping
- Consistent tile dimensions (256x256)

## 🏗️ Next Steps

1. **Network Resolution**: Configure access to SPC THREDDS server
2. **Widget Integration**: Connect to ocean visualization widgets  
3. **Color Matching**: Fine-tune to match exact WMS color schemes
4. **Performance**: Optimize for production tile serving
5. **Deployment**: Configure for production environment

## 📞 Support

For issues or questions:
1. Check the integration test output: `python3 test_cook_islands_integration.py`
2. Review server logs for errors
3. Test with local data first (`use_local=true`)
4. Validate network connectivity for remote data access

---

**🎉 Cook Islands COG tiler is ready for development and testing in Codespace!**