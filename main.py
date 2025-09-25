import os
import sys
import time
import uuid
import json
import hashlib
import tempfile
import shutil
import logging
from logging.config import dictConfig
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any, Union
from urllib.parse import unquote, urlencode
from uuid import uuid4
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

from fastapi import FastAPI, Response, HTTPException, Query, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from rio_tiler.io import Reader
from PIL import Image
from pydantic import BaseModel
from filelock import FileLock
import xarray as xr

from plotters import draw_plot, AVAILABLE_PLOTS
from data_reader import load_plot_ready_arrays
from cog_generator import COGGenerator

# -----------------------------------------------------------------------------
# Logging (structured, uvicorn-friendly)
# -----------------------------------------------------------------------------
dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": (
                '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
                '"msg":"%(message)s","module":"%(module)s","func":"%(funcName)s",'
                '"lineno":%(lineno)d,"request_id":"%(request_id)s"}'
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "console": {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "stream": sys.stderr,
            "formatter": "console",  # switch to "json" in prod
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "tile-server": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "cog-tiles": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["stderr"], "level": "INFO"},
})

logger = logging.getLogger("tile-server")
cog_logger = logging.getLogger("cog-tiles")

# -----------------------------------------------------------------------------
# Optimized GDAL/RasterIO settings for COG performance
# -----------------------------------------------------------------------------
try:
    from cog_config import configure_gdal_environment, OPTIMIZED_COG_PROFILE
    gdal_settings = configure_gdal_environment()
    logger.info(f"✅ Configured {len(gdal_settings)} GDAL optimization settings")
except ImportError:
    # Fallback configuration if cog_config not available
    os.environ.setdefault("GDAL_CACHEMAX", "512")
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_TIFF_OVR_BLOCKSIZE", "256")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", "25000000")
    logger.info("⚠️ Using fallback GDAL configuration")

# -----------------------------------------------------------------------------
# App Lifespan (Startup/Shutdown Events)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- On Startup ---
    # Use local cache directory for development, Docker path for production
    global CACHE_DIR
    CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Cache directory ensured at: {CACHE_DIR.resolve()}")
    yield
    # --- On Shutdown ---
    logger.info("Server is shutting down.")

# -----------------------------------------------------------------------------
# FastAPI app + middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    lifespan=lifespan,
    docs_url="/ncWMS/docs",
    openapi_url="/ncWMS/openapi.json",
    redoc_url="/ncWMS/redoc"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/ncWMS")

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # include request_id in all subsequent logs for this request via 'extra'
        try:
            response: StarletteResponse = await call_next(request)
        except Exception:
            logger.exception(
                f"Unhandled error ({rid}) on {request.method} {request.url.path}",
                extra={"request_id": rid},
            )
            raise
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                f"{request.method} {request.url.path} -> {dur_ms}ms",
                extra={"request_id": rid},
            )

        response.headers["X-Request-ID"] = rid
        return response

app.add_middleware(RequestContextMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class CogGenerateRequest(BaseModel):
    layer_id: str
    url: str
    variable: str
    time: Optional[str] = None
    colormap: str = "RdBu_r"
    vmin: float
    vmax: float
    step: float
    lon_min: float = 100.0
    lon_max: float = 300.0
    lat_min: float = -45.0
    lat_max: float = 45.0
    plot: str = "contourf"
    plot_options: Optional[Dict[str, Any]] = None

class CogGenerateResponse(BaseModel):
    job_id: str
    cog_path: str
    exists: bool
    size_bytes: int
    bounds_3857: Tuple[float, float, float, float]
    bounds_4326: Tuple[float, float, float, float]
    crs: str
    tile_querystring: str

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _hash_params(data: dict) -> str:
    # ensure 'time' is string for consistent hashing
    if data.get("time") is not None:
        data["time"] = str(data["time"])
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _normalize_url_param(u: str) -> str:
    decoded = unquote(u)
    # double-unquote if still encoded
    if "%3A" in decoded or "%2F" in decoded:
        decoded = unquote(decoded)
    return decoded

def canonicalize_params(
    *,
    layer_id: str,
    url: str,
    variable: str,
    time: Optional[str],
    colormap: str,
    vmin: float,
    vmax: float,
    step: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    plot: str = "contourf",
    plot_options: Optional[Union[str, Dict[str, Any]]] = None,
) -> dict:
    """
    Normalize all parameters for consistent cache-keying, including plot_options as canonical JSON string.
    """
    if plot_options:
        if isinstance(plot_options, dict):
            plot_options_str = json.dumps(plot_options, sort_keys=True, separators=(",", ":"))
        elif isinstance(plot_options, str):
            try:
                loaded = json.loads(plot_options)
                plot_options_str = json.dumps(loaded, sort_keys=True, separators=(",", ":"))
            except Exception:
                plot_options_str = plot_options
        else:
            plot_options_str = str(plot_options)
    else:
        plot_options_str = None

    return dict(
        layer_id=str(layer_id),
        url=_normalize_url_param(url),
        variable=variable,
        time=str(time) if time is not None else None,
        colormap=colormap,
        vmin=float(vmin),
        vmax=float(vmax),
        step=float(step),
        lon_min=float(lon_min),
        lon_max=float(lon_max),
        lat_min=float(lat_min),
        lat_max=float(lat_max),
        plot=str(plot or "contourf").lower(),
        plot_options=plot_options_str
    )

def _safe_unlink(p: Path):
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _get_cmap(name: str):
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap("RdBu_r")

def _png_response(img: Image.Image) -> Response:
    buf = BytesIO()
    img.save(buf, format="PNG")
    headers = {"Cache-Control": "public, max-age=3600"} # Cache for 1 hour
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

def _png_response_from_bytes(content: bytes) -> Response:
    """Create a PNG response directly from bytes, adding cache headers."""
    headers = {"Cache-Control": "public, max-age=86400"}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

def _parse_plot_options(plot_options: Optional[str]) -> Dict[str, Any]:
    if not plot_options:
        return {}
    try:
        if isinstance(plot_options, str):
            return json.loads(plot_options)
        return dict(plot_options)
    except Exception:
        return {}

def _tile_to_bounds(x: int, y: int, z: int) -> dict:
    """Convert XYZ tile to geographic bounds (WGS84)."""
    import math
    def tile_to_lat_lon(tx, ty, tz):
        n = 2.0 ** tz
        lon_deg = tx / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    lat_north, lon_west = tile_to_lat_lon(x, y, z)
    lat_south, lon_east = tile_to_lat_lon(x + 1, y + 1, z)

    return {"lon_min": lon_west, "lon_max": lon_east, "lat_min": lat_south, "lat_max": lat_north}

def _validate_cog_file(cog_path: Path) -> tuple[bool, list, list]:
    """
    Validate that a file is a proper Cloud Optimized GeoTIFF.
    
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    try:
        from rio_cogeo.cogeo import cog_validate
        return cog_validate(str(cog_path))
    except ImportError:
        # Fallback validation if rio-cogeo not available
        try:
            with rasterio.open(str(cog_path)) as src:
                # Basic checks for COG-like structure
                has_overviews = src.overviews(1) is not None and len(src.overviews(1)) > 0
                is_tiled = src.block_shapes[0] == (256, 256) or src.block_shapes[0] == (512, 512)
                return has_overviews and is_tiled, [], ["Using basic validation - install rio-cogeo for full validation"]
        except Exception as e:
            return False, [str(e)], []

def _tile_intersects_cog_bounds(reader: Reader, x: int, y: int, z: int) -> bool:
    """Check if a tile intersects with the COG bounds before extraction."""
    try:
        # Get COG bounds in WGS84
        bounds = reader.bounds
        
        # Convert tile to lat/lon bounds
        tile_bounds = _tile_to_bounds(x, y, z)
        
        # Check intersection
        return not (
            tile_bounds["lon_max"] < bounds[0] or  # tile is west of COG
            tile_bounds["lon_min"] > bounds[2] or  # tile is east of COG
            tile_bounds["lat_max"] < bounds[1] or  # tile is south of COG
            tile_bounds["lat_min"] > bounds[3]     # tile is north of COG
        )
    except Exception:
        return False

def _read_tile(reader: Reader, x: int, y: int, z: int) -> Image.Image:
    """Read a tile from COG, handling out-of-bounds gracefully by returning transparent tiles."""
    try:
        t0 = time.perf_counter()
        tile_data, mask = reader.tile(x, y, z)
        dur = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"rio-tiler tile z={z} x={x} y={y} took {dur}ms")
        tile_data = tile_data.astype(np.uint8)
        mask = mask.astype(np.uint8)
        arr = np.transpose(tile_data, (1, 2, 0))
        if arr.shape[2] == 4:
            rgba = arr.copy()
            rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
        else:
            rgba = np.dstack([arr, mask])
        return Image.fromarray(rgba, "RGBA")
    except Exception as e:
        # Handle out-of-bounds tiles gracefully by returning transparent tile
        if "TileOutsideBounds" in str(e) or "outside bounds" in str(e).lower():
            logger.debug(f"Tile {z}/{x}/{y} outside bounds, returning transparent tile")
            # Return a transparent 256x256 RGBA tile
            transparent_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
            return Image.fromarray(transparent_rgba, "RGBA")
        else:
            # Re-raise other exceptions
            logger.error(f"Unexpected tile error for {z}/{x}/{y}: {e}")
            raise

# -----------------------------------------------------------------------------
# Core: ensure COG exists/build
# -----------------------------------------------------------------------------
def ensure_cog_from_params(
    *,
    layer_id: str,
    url: str,
    variable: str,
    time: Optional[str],
    colormap: str,
    vmin: float,
    vmax: float,
    step: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    plot: str = "contourf",
    plot_options: Optional[Union[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Ensures a COG exists for the given parameters, using canonical cache keys.
    Returns the path to the COG (pre-existing or newly generated).
    """
    params = canonicalize_params(
        layer_id=layer_id,
        url=url, variable=variable, time=time, colormap=colormap,
        vmin=vmin, vmax=vmax, step=step,
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
        plot=plot, plot_options=plot_options
    )
    options = _parse_plot_options(params["plot_options"])
    job_id = _hash_params(params)
    out_cog = CACHE_DIR / f"{job_id}.tif"
    lock_path = str(out_cog) + ".lock"
    tmp_out = Path(str(out_cog) + ".tmp")

    # Fast-path cache check
    if out_cog.exists() and out_cog.stat().st_size > 0:
        logger.info(f"COG cache HIT (fast-path) job_id={job_id}")
        return out_cog

    with FileLock(lock_path, timeout=300):
        # Re-check under lock
        if out_cog.exists() and out_cog.stat().st_size > 0:
            logger.info(f"COG cache HIT (post-lock) job_id={job_id}")
            return out_cog

        logger.info(f"COG cache MISS job_id={job_id} -> building")

        tmp_dir = Path(tempfile.gettempdir())
        uid = uuid4().hex
        temp_tif_4326 = tmp_dir / f"rgba_4326_{uid}.tif"
        temp_tif_3857 = tmp_dir / f"rgba_3857_{uid}.tif"

        try:
            # Load arrays ready for plotting
            lons_plot, lats_plot, data_ma, data_bounds = load_plot_ready_arrays(
                layer_id=params["layer_id"],
                url=params["url"],
                variable=params["variable"],
                time=params["time"],
                lon_min=params["lon_min"],
                lon_max=params["lon_max"],
                lat_min=params["lat_min"],
                lat_max=params["lat_max"],
                plot=params["plot"],
                options=options,
            )

            # Levels/checks
            if step <= 0:
                raise ValueError("step must be > 0")
            if vmax <= vmin:
                raise ValueError("vmax must be > vmin")
            n_steps = int(np.ceil((vmax - vmin) / step))
            levels = np.linspace(vmin, vmax, n_steps + 1)

            cmap = _get_cmap(colormap)
            try:
                cmap.set_bad((0, 0, 0, 0))
            except Exception:
                pass

            # Render figure
            # Higher resolution for smaller pixels: ~2048px wide
            fig, ax = plt.subplots(figsize=(12, 8), dpi=170)
            fig.patch.set_alpha(0)
            ax.set_facecolor((0, 0, 0, 0))
            ax.axis("off")

            try:
                _ = draw_plot(
                    ax,
                    plot=plot,
                    lons=lons_plot,
                    lats=lats_plot,
                    data=data_ma,
                    cmap=cmap,
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    options=options,
                )
            except Exception:
                logger.exception("draw_plot failed for plot=%s with options=%s", plot, options)
                raise

            # Clamp axes to subset extent (avoid global stretch)
            ax.set_xlim(data_bounds['lon_min'], data_bounds['lon_max'])
            ax.set_ylim(data_bounds['lat_min'], data_bounds['lat_max'])

            fig.tight_layout(pad=0)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            image = np.array(fig.canvas.renderer.buffer_rgba())
        finally:
            plt.close("all")

        # Prepare RGBA for GeoTIFF
        image = image.transpose((2, 0, 1)).astype("uint8")

        # Geo-reference to subset bounds
        left, right = data_bounds['lon_min'], data_bounds['lon_max']
        bottom, top = data_bounds['lat_min'], data_bounds['lat_max']
        transform = from_bounds(left, bottom, right, top, w, h)

        profile4326 = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 4,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": transform,
            "photometric": "RGB",
            "interleave": "pixel",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
            "predictor": 2,
        }
        _safe_unlink(temp_tif_4326)
        with rasterio.open(temp_tif_4326, "w", **profile4326) as dst:
            dst.write(image)

        _safe_unlink(temp_tif_3857)
        with rasterio.open(temp_tif_4326) as src:
            transform_3857, width_3857, height_3857 = calculate_default_transform(
                src.crs, "EPSG:3857", src.width, src.height, *src.bounds
            )
            profile_3857 = src.profile.copy()
            profile_3857.update({
                "crs": "EPSG:3857",
                "transform": transform_3857,
                "width": width_3857,
                "height": height_3857,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "deflate",
                "predictor": 2,
                "interleave": "pixel",
            })
            with rasterio.open(temp_tif_3857, "w", **profile_3857) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform_3857,
                        dst_crs="EPSG:3857",
                        resampling=Resampling.nearest,
                        num_threads=0,
                    )

        from rio_cogeo.cogeo import cog_translate, cog_validate
        from rio_cogeo.profiles import cog_profiles

        # Optimized COG profile (consistent with cog_generator.py)
        cog_profile = {
            'driver': 'GTiff',
            'compress': 'DEFLATE',
            'blockxsize': 256,
            'blockysize': 256,
            'tiled': True,
            'interleave': 'pixel',
            'predictor': 2,  # Horizontal differencing for better compression
            'zlevel': 6      # Balanced compression vs speed
        }
        _safe_unlink(tmp_out)
        # Add overview generation to the COG creation process
        cog_translate(
            str(temp_tif_3857),
            str(tmp_out),
            cog_profile,
            overview_resampling="bilinear",  # Use bilinear for smoother overviews
            overview_level=5,                # Generate 5 levels of overviews
            web_optimized=True,
            in_memory=False,
            config={"GDAL_NUM_THREADS": "ALL_CPUS", "GDAL_TIFF_OVR_BLOCKSIZE": "256"},
        )
        
        # Validate the generated COG
        is_valid_cog, cog_errors, cog_warnings = cog_validate(str(tmp_out))
        if not is_valid_cog:
            cog_logger.warning(f"[COG] Generated COG failed validation: {cog_errors}")
        elif cog_warnings:
            cog_logger.info(f"[COG] COG validation warnings: {cog_warnings}")
        else:
            cog_logger.info(f"[COG] COG validation passed ✅")
            
        os.replace(tmp_out, out_cog)
        cog_logger.info(f"[COG] build COMPLETE job_id={job_id} size={out_cog.stat().st_size}")
        return out_cog

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@router.get("/")
def read_root():
    return {
        "message": "🌊 Cook Islands Wave Forecast API",
        "applications": {
            "forecast_app": "/ncWMS/forecast-app",
            "ugrid_comparison": "/ncWMS/ugrid-comparison",
            "index": "/ncWMS/index"
        },
        "endpoints": {
            "cog_status": "/ncWMS/cog-status",
            "health": "/ncWMS/health"
        },
        "status": "running"
    }

@router.get("/cog-status")
def cog_status():
    """Check the health and status of cached COG files."""
    try:
        # Check cache directories
        cache_stats = {
            "cache_dir": str(CACHE_DIR),
            "cache_exists": CACHE_DIR.exists(),
            "total_cogs": 0,
            "valid_cogs": 0,
            "invalid_cogs": 0,
            "total_size_mb": 0,
            "cog_files": []
        }
        
        if CACHE_DIR.exists():
            for cog_file in CACHE_DIR.glob("**/*.tif"):
                if cog_file.is_file():
                    cache_stats["total_cogs"] += 1
                    file_size_mb = cog_file.stat().st_size / (1024 * 1024)
                    cache_stats["total_size_mb"] += file_size_mb
                    
                    # Validate COG
                    is_valid, errors, warnings = _validate_cog_file(cog_file)
                    if is_valid:
                        cache_stats["valid_cogs"] += 1
                        status = "✅ Valid COG"
                    else:
                        cache_stats["invalid_cogs"] += 1
                        status = f"❌ Invalid: {errors[:2]}"  # Show first 2 errors
                    
                    cache_stats["cog_files"].append({
                        "filename": cog_file.name,
                        "path": str(cog_file.relative_to(CACHE_DIR)),
                        "size_mb": round(file_size_mb, 2),
                        "status": status,
                        "warnings": warnings[:2] if warnings else []
                    })
        
        cache_stats["total_size_mb"] = round(cache_stats["total_size_mb"], 2)
        
        return {
            "status": "healthy" if cache_stats["invalid_cogs"] == 0 else "warning",
            "cache_stats": cache_stats,
            "gdal_config": {
                "GDAL_CACHEMAX": os.environ.get("GDAL_CACHEMAX", "not set"),
                "GDAL_NUM_THREADS": os.environ.get("GDAL_NUM_THREADS", "not set"),
                "VSI_CACHE": os.environ.get("VSI_CACHE", "not set"),
                "COG_optimizations": "✅ Enabled"
            }
        }
    except Exception as e:
        logger.exception("COG status check failed")
        return {"status": "error", "error": str(e)}

@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "COG Tile Server",
        "cache_dir_exists": CACHE_DIR.exists(),
        "timestamp": str(time.time())
    }

# Root redirect outside of router
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ncWMS/index")

DEFAULT_URL = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
DEFAULT_VAR = "hs"
DEFAULT_PARAMS = dict(
    layer_id="4",
    url=DEFAULT_URL, variable=DEFAULT_VAR, time="2025-09-24T00:00:00",
    colormap="viridis", vmin=0.0, vmax=4.0, step=0.5,
    lon_min=-162.0, lon_max=-158.0, lat_min=-22.0, lat_max=-19.0,
    plot="contourf", plot_options=None,
)
DEFAULT_COG: Optional[Path] = None

@router.get("/cook-islands/{z}/{x}/{y}.png")
def cook_islands_tiles(
    z: int, x: int, y: int,
    variable: str = Query("inundation_depth", description="Variable to plot"),
    time: Optional[str] = Query(None, description="Time slice (ISO format)"),
    colormap: str = Query("viridis", description="Matplotlib colormap"),
    vmin: float = Query(0.0, description="Min value for colormap"),
    vmax: float = Query(5.0, description="Max value for colormap"),
    use_local: bool = Query(False, description="Use local test data (fallback for network issues)"),
):
    """Cook Islands specific tile endpoint with defaults for Rarotonga inundation data."""
    headers = {"Cache-Control": "public, max-age=3600"}

    """Cook Islands specific tile endpoint with defaults for Rarotonga inundation data."""
    logger.info(f"Cook Islands tile request: variable={variable}, z={z}, x={x}, y={y}, time={time}")
    
    cook_islands_url = (
        "/workspaces/cog_tiler/cook_islands_test_data.nc"
        if use_local else
        "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"
    )

    # World-class styling for inundation data
    if variable == 'Band1':
        plot_type = "discrete_cmap"
        plot_opts = {
            "ranges": ["0.0-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", "2.0-5.0"],
            "cmap": "Blues",
            "mask_out_of_range": False, # Show values above max in the top color
            "transparent_below": 0.01
        }
    else:
        plot_type = "contourf"
        plot_opts = {"antialiased": True, "alpha": 0.8}

    try:
        cog_path = ensure_cog_from_params(
            layer_id="cook_islands_rarotonga",
            url=cook_islands_url,
            variable=variable,
            time=time,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            step=0.1,
            lon_min=-180, lon_max=180, lat_min=-90, lat_max=90,  # Use full extent
            plot=plot_type,
            plot_options=plot_opts
        )
        with Reader(str(cog_path)) as reader:
            # Let _read_tile handle bounds checking gracefully
            img = _read_tile(reader, x, y, z)
    except Exception:
        logger.exception("COG generation failed for Cook Islands")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    buf = BytesIO()
    img.save(buf, format="PNG")
    tile_bytes = buf.getvalue()

    return Response(content=tile_bytes, media_type="image/png", headers=headers)

@router.get("/cook-islands-ugrid/{z}/{x}/{y}.png")
def cook_islands_ugrid_tiles(
    z: int, x: int, y: int,
    variable: str = Query("hs", description="UGRID variable to plot"),
    time: Optional[str] = Query(None, description="Time slice (ISO format or index)"),
    colormap: str = Query("jet", description="Matplotlib colormap"),
    vmin: Optional[float] = Query(None, description="Min value for colormap"),
    vmax: Optional[float] = Query(None, description="Max value for colormap"),
):
    """
    Cook Islands UGRID tiles.
    This endpoint always generates a COG from the UGRID data first,
    then serves tiles from that COG.
    """
    headers = {"Cache-Control": "public, max-age=3600"}
    ugrid_url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"

    # --- COG-based approach ---
    
    # Define sensible defaults for vmin/vmax per variable
    variable_defaults = {
        'hs': {'vmin': 0.0, 'vmax': 4.0},
        'tpeak': {'vmin': 2.0, 'vmax': 20.0},
        'tm02': {'vmin': 2.0, 'vmax': 15.0},
        'dirp': {'vmin': 0.0, 'vmax': 360.0},
        'u10': {'vmin': -15.0, 'vmax': 15.0},
        'v10': {'vmin': -15.0, 'vmax': 15.0},
        'water_level': {'vmin': -2.0, 'vmax': 3.0}
    }
    defaults = variable_defaults.get(variable, {'vmin': 0.0, 'vmax': 1.0})
    final_vmin = vmin if vmin is not None else defaults['vmin']
    final_vmax = vmax if vmax is not None else defaults['vmax']

    # Fallback to interpolated approach
    try:
        cog_path = ensure_cog_from_params(
            layer_id="cook_islands_ugrid",
            url=ugrid_url,
            variable=variable,
            time=time,
            colormap=colormap,
            vmin=final_vmin,
            vmax=final_vmax,
            step=0.1,
            lon_min=-180, lon_max=180, lat_min=-90, lat_max=90, # Use full extent to load all data
            plot="contourf",
            plot_options={"antialiased": True, "alpha": 0.9}
        )
        with Reader(str(cog_path)) as reader:
            try:
                img = _read_tile(reader, x, y, z)
            except Exception:
                logger.exception("UGRID tile generation error (interpolated)")
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    except Exception:
        logger.exception("UGRID COG generation failed (interpolated)")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    buf = BytesIO()
    img.save(buf, format="PNG")
    tile_bytes = buf.getvalue()
    return Response(content=tile_bytes, media_type="image/png", headers=headers)

# Initialize COG generator
cog_generator = COGGenerator()

@router.get("/cogs/list")
def list_available_cogs():
    """List all available COG files with metadata."""
    try:
        available_cogs = cog_generator.list_available_cogs()
        return {"status": "success", "cogs": available_cogs, "count": len(available_cogs)}
    except Exception as e:
        logger.exception("list_available_cogs error")
        return {"status": "error", "message": str(e)}

@router.post("/cogs/generate")
def generate_cog(  # Changed to 'def' to run in a thread pool
    variable: str = Query("hs", description="Variable to generate COG for"),
    time_index: int = Query(0, description="Time index"),
    vmin: float = Query(0.0, description="Min value"),
    vmax: float = Query(4.0, description="Max value"),
    colormap: str = Query("plasma", description="Colormap")
):
    """Manually trigger COG generation."""
    try:
        cog_path = cog_generator.generate_wave_cog(
            variable=variable, time_index=time_index, vmin=vmin, vmax=vmax, colormap=colormap
        )
        return {
            "status": "success",
            "cog_path": str(cog_path),
            "message": f"COG generated successfully for {variable} at time {time_index}"
        }
    except Exception as e:
        logger.exception("generate_cog error")
        return {"status": "error", "message": str(e)}

@router.delete("/cogs/cleanup")
def cleanup_old_cogs(max_age_hours: int = Query(24, description="Max age in hours")):
    """Clean up old COG files."""
    try:
        cog_generator.cleanup_old_cogs(max_age_hours)
        return {"status": "success", "message": f"Cleaned up COGs older than {max_age_hours} hours"}
    except Exception as e:
        logger.exception("cleanup_old_cogs error")
        return {"status": "error", "message": str(e)}

@router.get("/cook-islands-cog/{z}/{x}/{y}.png")
def cook_islands_cog_tiles(
    z: int, x: int, y: int,
    variable: str = Query("hs", description="Variable to plot"),
    time: int = Query(0, description="Time index"),
    colormap: str = Query("plasma", description="Matplotlib colormap"),
    vmin: Optional[float] = Query(0.0, description="Min value for colormap"),
    vmax: Optional[float] = Query(4.0, description="Max value for colormap"),
):
    """
    Cook Islands wave forecast tiles served from pre-generated COG files.
    Much faster than real-time processing!
    """
    try:
        logger.info(f"COG request: {variable} t={time} z={z} x={x} y={y}")
        cog_path = cog_generator.generate_wave_cog(
            variable=variable, time_index=time, vmin=vmin, vmax=vmax, colormap=colormap
        )
        logger.info(f"COG path: {cog_path}")
        with Reader(str(cog_path)) as reader:
            # Use the new _read_tile function that handles bounds gracefully
            img = _read_tile(reader, x, y, z)
            return _png_response(img)
    
    except Exception:
        logger.exception("COG tile generation failed")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        return _png_response(img)

@router.get("/cook-islands-ugrid-direct/{z}/{x}/{y}.png")
def cook_islands_ugrid_direct_tiles(
    z: int, x: int, y: int,
    variable: str = Query("hs", description="UGRID variable to plot"),
    time: Optional[str] = Query(None, description="Time slice (ISO format or index)"),
    colormap: str = Query("jet", description="Matplotlib colormap"),
    vmin: Optional[float] = Query(None, description="Min value for colormap"),
    vmax: Optional[float] = Query(None, description="Max value for colormap"),
):
    """
    Cook Islands UGRID direct triangular mesh tiles - NO INTERPOLATION ARTIFACTS.
    Preserves original mesh structure for clean island boundaries.
    """
    from ugrid_direct import create_direct_ugrid_tile
    bounds = _tile_to_bounds(x, y, z)
    ugrid_url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"

    try:
        tile_bytes = create_direct_ugrid_tile(
            url=ugrid_url,
            variable=variable,
            time=time,
            lon_min=bounds["lon_min"],
            lon_max=bounds["lon_max"],
            lat_min=bounds["lat_min"],
            lat_max=bounds["lat_max"],
            tile_size=256,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        return Response(content=tile_bytes, media_type="image/png")
    except Exception:
        logger.exception("Direct UGRID tile error")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        return _png_response(img)

@router.get("/tiles/{z}/{x}/{y}.png")
def tiles_static(z: int, x: int, y: int):
    global DEFAULT_COG
    if DEFAULT_COG is None or not DEFAULT_COG.exists():
        try:
            DEFAULT_COG = ensure_cog_from_params(**DEFAULT_PARAMS)
        except Exception:
            logger.exception("Failed to build DEFAULT_COG")
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            return _png_response(img)

    with Reader(str(DEFAULT_COG)) as reader:
        try:
            img = _read_tile(reader, x, y, z)
        except Exception:
            logger.exception("tiles_static read error")
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    return _png_response(img)

@router.get("/tiles/dynamic/{z}/{x}/{y}.png")
def tiles_dynamic(
    z: int, x: int, y: int,
    layer_id: str = Query(..., description="Layer id"),
    url: str = Query(..., description="OPeNDAP URL (URL-encoded or plain)"),
    variable: str = Query(..., description="Variable name"),
    time: Optional[str] = Query(None, description="Time index or ISO string"),
    colormap: str = Query("RdBu_r"),
    vmin: float = Query(...),
    vmax: float = Query(...),
    step: float = Query(...),
    lon_min: float = Query(100.0),
    lon_max: float = Query(300.0),
    lat_min: float = Query(-45.0),
    lat_max: float = Query(45.0),
    plot: str = Query("contourf", description=f"Plot type: {'|'.join(AVAILABLE_PLOTS)}"),
    plot_options: Optional[str] = Query(
        None,
        description="JSON dict of matplotlib kwargs; for 'discrete', include ranges (list) and colors (list).",
    ),
):
    headers = {"Cache-Control": "public, max-age=3600"}
    params = canonicalize_params(
        layer_id=layer_id,
        url=url,
        variable=variable,
        time=time,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        step=step,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        plot=plot,
        plot_options=plot_options,
    )
    try:
        cog_path = ensure_cog_from_params(**params)
    except Exception as e:
        logger.exception("tiles_dynamic ensure_cog_from_params error")
        raise HTTPException(status_code=400, detail=str(e))

    with Reader(str(cog_path)) as reader:
        try:
            img = _read_tile(reader, x, y, z)
        except Exception:
            logger.exception("tiles_dynamic tile read error")
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    buf = BytesIO()
    img.save(buf, format="PNG")
    tile_bytes = buf.getvalue()
    return Response(content=tile_bytes, media_type="image/png", headers=headers)

@router.get("/cook-islands/wms-comparison")
def cook_islands_wms_comparison(
    variable: str = Query("inundation_depth", description="Variable to compare"),
    time: Optional[str] = Query(None, description="Time slice"),
    zoom: int = Query(10, description="Zoom level for comparison"),
):
    """
    Compare Cook Islands COG tiles with the original WMS service.
    """
    wms_url = "https://gemthreddshpc.spc.int/thredds/wms/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"
    import math
    lat, lon = -21.2, -159.75
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    cog_tile_url = f"/ncWMS/cook-islands/{zoom}/{x}/{y}.png?variable={variable}"
    if time:
        cog_tile_url += f"&time={time}"

    wms_params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": variable,
        "styles": "boxfill/rainbow",
        "crs": "EPSG:4326",
        "bbox": "-159.9,-21.3,-159.6,-21.1",
        "width": "256",
        "height": "256",
        "format": "image/png",
        "transparent": "true"
    }
    if time:
        wms_params["time"] = time
    wms_tile_url = wms_url + "?" + urlencode(wms_params)

    return {
        "comparison": "Cook Islands COG vs WMS",
        "dataset": "Rarotonga_inundation_depth.nc",
        "variable": variable,
        "time": time,
        "zoom_level": zoom,
        "center_tile": {"x": x, "y": y, "z": zoom},
        "cog_tile_url": cog_tile_url,
        "wms_tile_url": wms_tile_url,
        "wms_params": wms_params,
        "bounds": {"lat_min": -21.3, "lat_max": -21.1, "lon_min": -159.9, "lon_max": -159.6}
    }

# -----------------------------------------------------------------------------
# Forecast Application Endpoints
# -----------------------------------------------------------------------------
@router.get("/forecast/metadata")
def get_forecast_metadata():
    """Get metadata for the forecast application."""
    return JSONResponse({
        "application": {
            "name": "Cook Islands Wave Forecast",
            "version": "1.0.0",
            "description": "World-class marine forecasting application using SPC THREDDS data"
        },
        "data_source": {
            "provider": "Pacific Community (SPC)",
            "server": "gemthreddshpc.spc.int",
            "protocol": "OPeNDAP",
            "model": "SCHISM + WaveWatch III"
        },
        "coverage": {
            "region": "Cook Islands",
            "center": [-21.2367, -159.7777],
            "bounds": {"lat_min": -21.4, "lat_max": -21.1, "lon_min": -159.9, "lon_max": -159.6}
        },
        "variables": {
            "hs": {
                "name": "Significant Wave Height", "units": "meters",
                "description": "Height of waves as measured from trough to crest",
                "range": [0, 8], "colormap": "plasma", "source": "ugrid"
            },
            "tpeak": {
                "name": "Peak Wave Period", "units": "seconds",
                "description": "Time interval between successive wave crests",
                "range": [0, 20], "colormap": "viridis", "source": "ugrid"
            },
            "dirp": {
                "name": "Peak Wave Direction", "units": "degrees",
                "description": "Direction waves are coming from (meteorological convention)",
                "range": [0, 360], "colormap": "hsv", "source": "ugrid"
            },
            "dirm": {
                "name": "Mean Wave Direction", "units": "degrees",
                "description": "Mean direction waves are coming from (meteorological convention)",
                "range": [0, 360], "colormap": "hsv", "source": "ugrid"
            },
            "u10": {
                "name": "Wind U Component", "units": "m/s",
                "description": "Eastward wind speed at 10m above surface",
                "range": [-25, 25], "colormap": "RdBu_r", "source": "ugrid"
            },
            "v10": {
                "name": "Wind V Component", "units": "m/s",
                "description": "Northward wind speed at 10m above surface",
                "range": [-25, 25], "colormap": "RdBu_r", "source": "ugrid"
            },
            "tm02": {
                "name": "Mean Wave Period", "units": "seconds",
                "description": "Mean time interval between successive wave crests",
                "range": [2, 15], "colormap": "coolwarm", "source": "ugrid"
            },
            "Band1": {
                "name": "Inundation Depth", "units": "meters",
                "description": "Coastal flooding depth above mean sea level",
                "range": [0, 5], "colormap": "Blues", "source": "gridded"
            }
        },
        "temporal": {
            "forecast_hours": 229,
            "forecast_days": 9.5,
            "time_step": "1 hour",
            "update_frequency": "4x daily"
        },
        "last_updated": datetime.utcnow().isoformat() + "Z"
    })

@router.get("/forecast/status")
def get_forecast_status():
    """Get current forecast system status."""
    datasets = {
        "inundation": "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc",
        "ugrid": "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
    }
    status = {
        "system": "operational",
        "last_check": datetime.utcnow().isoformat() + "Z",
        "data_sources": {},
        "services": {"cog_server": "running", "tile_generation": "operational", "data_access": "operational"}
    }

    for name, url in datasets.items():
        try:
            with xr.open_dataset(url) as ds:
                time_dim = None
                if name != "inundation" and 'time' in ds.dims:
                    time_dim = len(ds.time)
                variables = list(ds.data_vars.keys())
                status["data_sources"][name] = {
                    "status": "available",
                    "url": url,
                    "variables": len(variables),
                    "time_steps": time_dim,
                    "last_accessed": datetime.utcnow().isoformat() + "Z"
                }
        except Exception as e:
            status["data_sources"][name] = {
                "status": "error",
                "url": url,
                "error": str(e),
                "last_accessed": datetime.utcnow().isoformat() + "Z"
            }

    all_sources_ok = all(source["status"] == "available" for source in status["data_sources"].values())
    status["system"] = "operational" if all_sources_ok else "degraded"
    return JSONResponse(status)

@router.get("/forecast/current/{variable}")
def get_current_forecast_value(variable: str, lat: float = -21.2367, lon: float = -159.7777, time_index: int = 0):
    """Get current forecast value for a specific location and variable (simplified nearest logic)."""
    
    # Determine the correct dataset URL and access method based on the variable
    if variable == 'Band1':
        url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"
        is_ugrid = False
    else:
        url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
        is_ugrid = True

    try:
        with xr.open_dataset(url) as ds:
            da = ds[variable]
            if 'time' in da.dims and time_index < len(da.time):
                da = da.isel(time=time_index)

            if is_ugrid:
                value = float(da.isel(mesh_node=500).values)  # Simplified logic for UGRID
            else:
                # Inundation data is in UTM, so we need to transform the requested lat/lon
                from pyproj import Transformer
                transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32704', always_xy=True)
                x_utm, y_utm = transformer.transform(lon, lat)
                
                # Select using the dataset's native x/y coordinates
                value = float(da.sel(x=x_utm, y=y_utm, method='nearest').values)

        if np.isnan(value):
            value = None

        return JSONResponse({
            "variable": variable,
            "value": value,
            "location": {"lat": lat, "lon": lon},
            "time_index": time_index,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success"
        })
    except Exception as e:
        logger.exception("get_current_forecast_value error")
        return JSONResponse({
            "variable": variable,
            "value": None,
            "location": {"lat": lat, "lon": lon},
            "time_index": time_index,
            "error": str(e),
            "status": "error"
        }, status_code=500)

@router.get("/cook-islands-viewer", response_class=HTMLResponse)
def cook_islands_viewer():
    """Serve the Cook Islands tile viewer interface."""
    html_path = Path("cook_islands_viewer.html")
    if not html_path.exists():
        logger.error("cook_islands_viewer.html not found")
        return HTMLResponse(content="<h1>Cook Islands Viewer not found</h1>", status_code=404)
    try:
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read cook_islands_viewer.html")
        return HTMLResponse(content="<h1>Error loading Cook Islands Viewer</h1>", status_code=500)

@router.get("/forecast-app", response_class=HTMLResponse)
def forecast_app():
    """Serve the forecast application."""
    html_path = Path("forecast_app.html")
    if not html_path.exists():
        logger.error("forecast_app.html not found")
        return HTMLResponse("<h1>Forecast App not found</h1>", status_code=404)
    try:
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read forecast_app.html")
        return HTMLResponse("<h1>Error loading Forecast App</h1>", status_code=500)

@router.get("/ugrid-comparison", response_class=HTMLResponse)
def ugrid_comparison_app():
    """Serve the UGRID approach comparison application."""
    html_path = Path("ugrid_comparison_app.html")
    if not html_path.exists():
        logger.error("ugrid_comparison_app.html not found")
        return HTMLResponse("<h1>UGRID Comparison Application not found</h1>", status_code=404)
    try:
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read ugrid_comparison_app.html")
        return HTMLResponse("<h1>Error loading UGRID Comparison Application</h1>", status_code=500)

@router.get("/index", response_class=HTMLResponse)
def index_page():
    """Serve a simple index page with links to all applications."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>🌊 Cook Islands Wave Forecast Applications</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; margin: 0; padding: 40px; min-height: 100vh;
                display: flex; align-items: center; justify-content: center;
            }
            .container {
                background: rgba(0, 0, 0, 0.8);
                padding: 40px; border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                text-align: center; max-width: 600px;
            }
            h1 { color: #00ff87; margin-bottom: 30px; font-size: 2.5em; }
            .app-links { display: grid; gap: 20px; margin: 30px 0; }
            .app-link {
                background: rgba(0, 255, 135, 0.1);
                border: 2px solid #00ff87;
                padding: 20px; border-radius: 10px; text-decoration: none; color: white;
                transition: all 0.3s ease; display: block;
            }
            .app-link:hover {
                background: rgba(0, 255, 135, 0.2);
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 255, 135, 0.3);
            }
            .app-title { font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: #00ff87; }
            .app-desc { font-size: 0.9em; opacity: 0.8; }
            .new-badge {
                background: #ff6b6b; color: white; padding: 4px 8px;
                border-radius: 15px; font-size: 0.7em; margin-left: 10px;
            }
            .footer { margin-top: 30px; font-size: 0.8em; opacity: 0.7; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌊 Cook Islands Wave Forecast</h1>
            <p>Advanced UGRID oceanographic visualization applications</p>
            <div class="app-links">
                <a href="/ncWMS/forecast-app" class="app-link">
                    <div class="app-title">🌊 Original Forecast Application</div>
                    <div class="app-desc">
                        Full-featured wave forecast with multiple variables,
                        time controls, and interactive mapping
                    </div>
                </a>
                <a href="/ncWMS/ugrid-comparison" class="app-link">
                    <div class="app-title">📐 UGRID Approach Comparison <span class="new-badge">NEW</span></div>
                    <div class="app-desc">
                        Compare Direct Triangular Mesh vs Interpolated approaches.
                        See the difference in island boundary preservation!
                    </div>
                </a>
                <a href="/ncWMS/docs" class="app-link">
                    <div class="app-title">📚 API Documentation</div>
                    <div class="app-desc">Complete API reference with interactive testing interface</div>
                </a>
            </div>
            <div class="footer">
                <p>🔬 <strong>Scientific Innovation:</strong> Direct triangular mesh rendering preserves
                original UGRID structure and eliminates interpolation artifacts around islands.</p>
                <p>Data Source: Pacific Community (SPC) THREDDS Server</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.delete("/cache/clear")
def clear_cache():
    """
    Deletes all files and subdirectories within the cache directory.

    This is a destructive operation intended for development and debugging.
    It will remove all generated COG files and lock files.
    """
    global CACHE_DIR
    if not CACHE_DIR.exists() or not CACHE_DIR.is_dir():
        logger.warning("Cache clear request: directory not found.")
        return JSONResponse(
            content={"status": "not_found", "message": "Cache directory does not exist."},
            status_code=404,
        )

    deleted_files = 0
    deleted_dirs = 0
    errors = []

    logger.info(f"Clearing cache directory: {CACHE_DIR}")
    for item in CACHE_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
                deleted_dirs += 1
            else:
                item.unlink()
                deleted_files += 1
        except Exception as e:
            error_detail = {"item": str(item.relative_to(CACHE_DIR)), "error": str(e)}
            errors.append(error_detail)
            logger.error(f"Failed to delete cache item {item}: {e}")

    if errors:
        return JSONResponse(content={"status": "partial_error", "message": "Cache partially cleared.", "deleted_files": deleted_files, "deleted_dirs": deleted_dirs, "errors": errors}, status_code=500)

    return JSONResponse(content={"status": "success", "message": "Cache directory has been cleared.", "deleted_files": deleted_files, "deleted_dirs": deleted_dirs})

# Mount router first
app.include_router(router)

# Mount static files under /ncWMS for consistency (after router to avoid conflicts)
app.mount("/ncWMS", StaticFiles(directory=".", html=True), name="static")
