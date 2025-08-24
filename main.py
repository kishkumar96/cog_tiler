import logging
import os
import tempfile
import json
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from urllib.parse import unquote, urlencode
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from rio_tiler.io import Reader
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter
from filelock import FileLock

from plotters import draw_plot, AVAILABLE_PLOTS
from data_reader import load_plot_ready_arrays

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tile-server")
logger22 = logging.getLogger("cog-tiles")

# Optional GDAL/RasterIO tuning
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("GDAL_TIFF_OVR_BLOCKSIZE", "128")

app = FastAPI(
    docs_url="/cog/docs",
    redoc_url="/cog/redoc",
    openapi_url="/cog/openapi.json",
    favicon_url="/cog/favicon.ico"
)
router = APIRouter(prefix="/cog")

@router.get("/")
def read_root():
    return {"Message": "On-demand OPeNDAP → COG Tile Server (Pluggable Plotting)"}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/app/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
#CACHE_DIR = Path("cache")
#CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Models ----------

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

# ---------- Helpers ----------

def _hash_params(data: dict) -> str:
    if data.get("time") is not None:
        data["time"] = str(data["time"])
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _normalize_url_param(u: str) -> str:
    decoded = unquote(u)
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
        return plt.get_cmap(name)  # avoid .copy() for older Matplotlib
    except Exception:
        return plt.get_cmap("RdBu_r")

def _png_response(img: Image.Image) -> Response:
    buf = BytesIO()
    img.save(buf, format="PNG")
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

    # Fast-path: return if COG exists and nonzero
    # DEV: Disable cache read post-lock
    if out_cog.exists() and out_cog.stat().st_size > 0:
        logger22.info(f"COG cache HIT (fast-path) job_id={job_id}")
        return out_cog

    with FileLock(lock_path, timeout=300):
        # DEV: Disable cache read post-lock
        if out_cog.exists() and out_cog.stat().st_size > 0:
            logger22.info(f"COG cache HIT (post-lock) job_id={job_id}")
            return out_cog

        tmp_dir = Path(tempfile.gettempdir())
        uid = uuid4().hex
        temp_tif_4326 = tmp_dir / f"rgba_4326_{uid}.tif"
        temp_tif_3857 = tmp_dir / f"rgba_3857_{uid}.tif"

        try:
            # Load arrays ready for plotting from the data reader
            lons_plot, lats_plot, data_ma = load_plot_ready_arrays(
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

            # Plot only the subset (prevents "stretching" across the full global axis)
            fig, ax = plt.subplots(figsize=(10, 6), dpi=256 / 2.54)
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

            # Clamp axes to subset extent
            ax.set_xlim(float(lons_plot.min()), float(lons_plot.max()))
            ax.set_ylim(float(lats_plot.min()), float(lats_plot.max()))

            fig.tight_layout(pad=0)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            image = np.array(fig.canvas.renderer.buffer_rgba())
        finally:
            plt.close("all")

        # Prepare RGBA for GeoTIFF
        image = image.transpose((2, 0, 1)).astype("uint8")

        # Geo-reference using the same subset bounds (fixes longitude stretching)
        left, right = float(lons_plot.min()), float(lons_plot.max())
        bottom, top = float(lats_plot.min()), float(lats_plot.max())
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

        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        cog_profile = cog_profiles.get("deflate")
        cog_profile.update({
            "blockxsize": 256,
            "blockysize": 256,
            "zlevel": 9,
            "predictor": 2,
            "interleave": "pixel",
            "tiled": True,
        })
        _safe_unlink(tmp_out)
        cog_translate(
            str(temp_tif_3857),
            str(tmp_out),
            cog_profile,
            overview_resampling="nearest",
            web_optimized=True,
            in_memory=False,
            config={"GDAL_NUM_THREADS": "ALL_CPUS", "GDAL_TIFF_OVR_BLOCKSIZE": "128"},
        )
        os.replace(tmp_out, out_cog)
        logger22.info(f"[COG] build COMPLETE ")
        return out_cog

# ---------- Routes ----------

DEFAULT_URL = "https://ocean-thredds01.spc.int/thredds/dodsC/POP/model/regional/noaa/nrt/daily/sst_anomalies/oisst-avhrr-v02r01.20250815_preliminary.nc"
DEFAULT_VAR = "anom"
DEFAULT_PARAMS = dict(
    layer_id="4",
    url=DEFAULT_URL, variable=DEFAULT_VAR, time=None,
    colormap="RdBu_r", vmin=-2.0, vmax=2.0, step=0.5,
    lon_min=100.0, lon_max=300.0, lat_min=-45.0, lat_max=45.0,
    plot="contourf", plot_options=None,
)

# Lazy default COG: don't fetch at import time
DEFAULT_COG: Optional[Path] = None

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
            tile_data, mask = reader.tile(x, y, z)
            tile_data = tile_data.astype(np.uint8)
            mask = mask.astype(np.uint8)
            arr = np.transpose(tile_data, (1, 2, 0))
            if arr.shape[2] == 4:
                rgba = arr.copy()
                rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
            else:
                rgba = np.dstack([arr, mask])
            img = Image.fromarray(rgba, "RGBA")
        except Exception:
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
        raise HTTPException(status_code=400, detail=str(e))

    with Reader(str(cog_path)) as reader:
        try:
            tile_data, mask = reader.tile(x, y, z)
            tile_data = tile_data.astype(np.uint8)
            mask = mask.astype(np.uint8)
            arr = np.transpose(tile_data, (1, 2, 0))
            if arr.shape[2] == 4:
                rgba = arr.copy()
                rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
            else:
                rgba = np.dstack([arr, mask])
            img = Image.fromarray(rgba, "RGBA")
        except Exception:
            img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    return _png_response(img)


@router.post("/generate", response_model=CogGenerateResponse)
def cog_generate(payload: CogGenerateRequest):
    options = payload.plot_options or {}
    params = canonicalize_params(
        layer_id=payload.layer_id,
        url=payload.url,
        variable=payload.variable,
        time=payload.time,
        colormap=payload.colormap,
        vmin=payload.vmin,
        vmax=payload.vmax,
        step=payload.step,
        lon_min=payload.lon_min,
        lon_max=payload.lon_max,
        lat_min=payload.lat_min,
        lat_max=payload.lat_max,
        plot=payload.plot,
        plot_options=options,
    )
    job_id = _hash_params(params)
    cog_path = ensure_cog_from_params(**params)

    exists = cog_path.exists()
    size_bytes = cog_path.stat().st_size if exists else 0

    bounds_3857 = (0.0, 0.0, 0.0, 0.0)
    bounds_4326 = (0.0, 0.0, 0.0, 0.0)
    crs = "EPSG:3857"
    if exists:
        with rasterio.open(cog_path) as src:
            crs = src.crs.to_string() if src.crs else "EPSG:3857"
            b = src.bounds
            bounds_3857 = (b.left, b.bottom, b.right, b.top)
            try:
                bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *bounds_3857, densify_pts=21)
            except Exception:
                pass

    tile_params = dict(params)
    return CogGenerateResponse(
        job_id=job_id,
        cog_path=str(cog_path),
        exists=exists,
        size_bytes=size_bytes,
        bounds_3857=bounds_3857,
        bounds_4326=bounds_4326,
        crs=crs,
        tile_querystring=urlencode(tile_params),
    )


app.include_router(router)