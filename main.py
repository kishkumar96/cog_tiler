import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from rio_tiler.io import Reader
from io import BytesIO
from PIL import Image
from pathlib import Path
import os
import tempfile
from uuid import uuid4
import hashlib
import json
from typing import Optional, Tuple, Dict, Any
from urllib.parse import unquote, urlencode
from pydantic import BaseModel
import logging
from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse

# COG tools
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from filelock import FileLock

# Plot dispatchers (make sure plotters.py is in the same folder or on PYTHONPATH)
from plotters import draw_plot, AVAILABLE_PLOTS

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

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Models ----------

class CogGenerateRequest(BaseModel):
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

def _safe_unlink(p: Path):
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _normalize_url_param(u: str) -> str:
    decoded = unquote(u)
    if "%3A" in decoded or "%2F" in decoded:
        decoded = unquote(decoded)
    return decoded

def _select_time(da: xr.DataArray, time_param: Optional[str]) -> xr.DataArray:
    if "time" not in da.dims:
        return da.squeeze()
    if time_param is not None:
        try:
            idx = int(str(time_param).strip())
            return da.isel(time=idx).squeeze()
        except Exception:
            pass
    if time_param is None:
        return da.isel(time=0).squeeze()
    try:
        import pandas as pd
        times_pd = pd.to_datetime(da["time"].values, utc=True)
        target_pd = pd.to_datetime(str(time_param), utc=True, errors="coerce")
        if pd.isna(target_pd):
            return da.isel(time=0).squeeze()
        i = int((times_pd - target_pd).abs().argmin())
        return da.isel(time=i).squeeze()
    except Exception:
        return da.isel(time=0).squeeze()

def _get_cmap(name: str):
    try:
        return plt.get_cmap(name)  # avoid .copy() for older Matplotlib
    except Exception:
        return plt.get_cmap("RdBu_r")

def _find_coords(da: xr.DataArray) -> Tuple[str, str]:
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lon_name = next((n for n in lon_candidates if n in da.coords or n in da.dims), None)
    lat_name = next((n for n in lat_candidates if n in da.coords or n in da.dims), None)
    if lon_name is None or lat_name is None:
        raise ValueError("Could not determine lon/lat coordinate names in the dataset.")
    return lon_name, lat_name

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
    plot_options: Optional[str] = None,
) -> Path:
    url = _normalize_url_param(url)
    options = _parse_plot_options(plot_options)

    params = dict(
        url=url, variable=variable, time=time,
        colormap=colormap, vmin=float(vmin), vmax=float(vmax), step=float(step),
        lon_min=float(lon_min), lon_max=float(lon_max), lat_min=float(lat_min), lat_max=float(lat_max),
        plot=str(plot or "contourf").lower(), plot_options=options,
    )
    job_id = _hash_params(params)
    out_cog = CACHE_DIR / f"{job_id}.tif"
    lock_path = str(out_cog) + ".lock"
    tmp_out = Path(str(out_cog) + ".tmp")

    if out_cog.exists() and out_cog.stat().st_size > 0:
        logger22.info(f"COG cache HIT (fast-path) job_id={job_id}")
        return out_cog

    with FileLock(lock_path, timeout=300):
        if out_cog.exists() and out_cog.stat().st_size > 0:
            logger22.info(f"COG cache HIT (post-lock) job_id={job_id}")
            return out_cog

        tmp_dir = Path(tempfile.gettempdir())
        uid = uuid4().hex
        temp_tif_4326 = tmp_dir / f"rgba_4326_{uid}.tif"
        temp_tif_3857 = tmp_dir / f"rgba_3857_{uid}.tif"

        try:
            ds = xr.open_dataset(url)
            try:
                da = ds[variable]
                da2d = _select_time(da, time)

                lon_name, lat_name = _find_coords(da2d)
                if list(da2d.dims) != [lat_name, lon_name]:
                    da2d = da2d.transpose(lat_name, lon_name)

                lons = da2d.coords[lon_name].values
                lats = da2d.coords[lat_name].values

                lons_0360 = (lons + 360) % 360
                lon_mask = (lons_0360 >= lon_min) & (lons_0360 <= lon_max)
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                mask = lat_mask[:, None] & lon_mask[None, :]
                masked_data = np.where(mask, da2d.values, np.nan)

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

                fig, ax = plt.subplots(figsize=(10, 6), dpi=256 / 2.54)
                fig.patch.set_alpha(0)
                ax.set_facecolor((0, 0, 0, 0))
                ax.axis("off")

                _ = draw_plot(
                    ax,
                    plot=plot,
                    lons=lons,
                    lats=lats,
                    data=masked_data,
                    cmap=cmap,
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    options=options,
                )

                fig.tight_layout(pad=0)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                image = np.array(fig.canvas.renderer.buffer_rgba())
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
            plt.close(fig)

            image = image.transpose((2, 0, 1)).astype("uint8")

            left, right = float(np.min(lons)), float(np.max(lons))
            bottom, top = float(np.min(lats)), float(np.max(lats))
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
            logger22.info(f"####################################")
            logger22.info(f"####################################")
            logger22.info(f"[COG] build COMPLETE ")
            return out_cog
        finally:
            _safe_unlink(Path(temp_tif_4326))
            _safe_unlink(Path(temp_tif_3857))
            _safe_unlink(tmp_out)

# ---------- Routes ----------

DEFAULT_URL = "https://ocean-thredds01.spc.int/thredds/dodsC/POP/model/regional/noaa/nrt/daily/sst_anomalies/oisst-avhrr-v02r01.20250815_preliminary.nc"
DEFAULT_VAR = "anom"
DEFAULT_PARAMS = dict(
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
        except Exception as e:
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
    try:
        cog_path = ensure_cog_from_params(
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
    hash_input = dict(
        url=_normalize_url_param(payload.url),
        variable=payload.variable,
        time=payload.time,
        colormap=payload.colormap,
        vmin=float(payload.vmin),
        vmax=float(payload.vmax),
        step=float(payload.step),
        lon_min=float(payload.lon_min),
        lon_max=float(payload.lon_max),
        lat_min=float(payload.lat_min),
        lat_max=float(payload.lat_max),
        plot=str(payload.plot or "contourf").lower(),
        plot_options=options,
    )
    job_id = _hash_params(hash_input)

    cog_path = ensure_cog_from_params(
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
        plot_options=json.dumps(options) if options else None,
    )

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

    tile_params = {
        "url": payload.url,
        "variable": payload.variable,
        "time": payload.time or "",
        "colormap": payload.colormap,
        "vmin": str(payload.vmin),
        "vmax": str(payload.vmax),
        "step": str(payload.step),
        "lon_min": str(payload.lon_min),
        "lon_max": str(payload.lon_max),
        "lat_min": str(payload.lat_min),
        "lat_max": str(payload.lat_max),
        "plot": str(payload.plot or "contourf").lower(),
    }
    if options:
        tile_params["plot_options"] = json.dumps(options)

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