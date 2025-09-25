# Cook Islands COG Tiler 🏝️

A high-performance, on-the-fly Cloud Optimized GeoTIFF (COG) tile server. It is specifically enhanced for visualizing Cook Islands oceanographic datasets, including UGRID and gridded NetCDF data, with professional, WMS-quality rendering.

## 🎯 Key Features

- **Cook Islands Datasets**: Native support for Rarotonga inundation depth and UGRID data
- **On-the-Fly COG Generation**: Creates and caches COGs from raw NetCDF data upon first request.
- **Professional Quality**: Antialiased contours and direct triangular mesh rendering that matches WMS visual standards.
- **Multiple Formats**: Supports both regular grids and UGRID unstructured meshes, preserving scientific accuracy.
- **Multiple Interactive Viewers**:
    - `forecast-app`: A full-featured, world-class forecasting application.
    - `ugrid-comparison`: A tool to compare rendering techniques for UGRID data.
- **Dockerized**: Production-ready Docker and Docker Compose setup with health checks and persistent caching.
- **Network Resilient**: Local test data fallback for development.
- **WMS Comparison**: Side-by-side quality validation tools

## 🚀 Quick Start (Docker)

This is the recommended method for running the application.

### 1. Prerequisites
- Docker
- Docker Compose

### 2. Start the Server
```bash
# Build the image and start the service in the background
docker-compose up --build -d
```
This will start the server on `http://localhost:8082`. The `./cache` directory is mounted into the container, so your generated COGs will persist across restarts.

### 3. Access the Applications
Visit the main index page to access all applications:
**`http://localhost:8082/ncWMS/index`**

### 4. Stop the Server
```bash
docker-compose down
```

## 🚀 Quick Start (Local Development)

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Server
```bash
# Start COG tile server (runs on http://localhost:8082)
uvicorn main:app --host 0.0.0.0 --port 8082 --reload
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
├── cog_generator.py                     # Core on-the-fly COG generation logic
├── plotters.py                          # Enhanced plotting with antialiasing
├── data_reader.py                       # NetCDF/OPeNDAP data loading
├── requirements.txt                     # Python dependencies
├── forecast_app.html                    # Main forecast application
├── ugrid_comparison_app.html            # UGRID rendering comparison app
├── cook_islands_test_data.nc            # Synthetic test dataset (73KB)
├── Dockerfile                           # Container definition
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
   # Replace with correct server name/IP if needed
   cook_islands_url = "http://CORRECT_SERVER/thredds/dodsC/..."
   ```

2. **Test with remote data**:
   ```bash
   # Test real endpoints
   curl "http://localhost:8082/ncWMS/cook-islands/10/57/573.png?use_local=false"
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

1. **Widget Integration**: Connect the tile server to other ocean visualization widgets.
2. **Color Matching**: Fine-tune colormaps to match exact WMS color schemes if required.
3. **Performance**: Implement proactive COG pre-generation for even faster tile serving.
4. **Deployment**: Deploy using the provided Docker configuration to a production environment.

## 📞 Support

For issues or questions, please check the following:
1. Check the integration test output: `python3 test_cook_islands_integration.py`
2. Review server logs for errors
3. Test with local data first (`use_local=true`)
4. Validate network connectivity for remote data access

---

**🎉 Cook Islands COG tiler is ready for development and testing in Codespace!**