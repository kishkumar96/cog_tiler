import io
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from starlette.responses import StreamingResponse

app = FastAPI()

# Preload dataset
url = "https://ocean-thredds01.spc.int/thredds/dodsC/POP/model/regional/noaa/nrt/daily/sst_anomalies/oisst-avhrr-v02r01.20250815_preliminary.nc"
ds = xr.open_dataset(url)
if "time" in ds["anom"].dims:
    data2d = ds["anom"].isel(time=0).squeeze()
else:
    data2d = ds["anom"].squeeze()

# Tile to bounds, EPSG:4326 (lat/lon)
def tile_bounds(x, y, z):
    n = 2.0 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    lat_bottom = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top

@app.get("/tiles/{z}/{x}/{y}.png")
def get_tile(z: int, x: int, y: int):
    min_lon, min_lat, max_lon, max_lat = tile_bounds(x, y, z)
    # Create a grid for the tile (256x256)
    tile_lons = np.linspace(min_lon, max_lon, 256)
    tile_lats = np.linspace(min_lat, max_lat, 256)
    Lon, Lat = np.meshgrid(tile_lons, tile_lats)
    # Interpolate data to tile grid
    interp = data2d.interp(lon=(Lon[0]), lat=(Lat[:,0]), method="linear")
    tile_data = interp.values

    # If all data is nan, return transparent tile
    if np.isnan(tile_data).all():
        buf = io.BytesIO()
        plt.imsave(buf, np.zeros((256,256,4), dtype=np.uint8), format="png")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    # Plot tile as image for speed and seamless edges
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.imshow(tile_data, origin="lower", extent=[min_lon, max_lon, min_lat, max_lat], cmap="RdBu_r", vmin=-2, vmax=2)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")