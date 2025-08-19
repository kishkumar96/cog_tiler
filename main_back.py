from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WMS_URL = "https://ocean-thredds01.spc.int/thredds/wms/POP/model/regional/noaa/nrt/daily/sst_anomalies/latest.ncml"  # Use the THREDDS WMS endpoint

def tile_xyz_to_bbox_3857(x, y, z):
    # WebMercator tile to bbox in EPSG:3857
    tile_size = 256
    initial_resolution = 2 * math.pi * 6378137 / tile_size
    origin_shift = 2 * math.pi * 6378137 / 2.0
    n = 2.0 ** z
    x_min = x / n
    y_min = y / n
    x_max = (x + 1) / n
    y_max = (y + 1) / n

    def lonlat_to_mercator(lon, lat):
        mx = lon * origin_shift / 180.0
        my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        my = my * origin_shift / 180.0
        return mx, my

    # Leaflet requests tiles in EPSG:3857, so let's compute the bbox directly
    def tile_to_mercator_bounds(x, y, z):
        n = 2.0 ** z
        lon_left = x / n * 360.0 - 180.0
        lon_right = (x + 1) / n * 360.0 - 180.0

        lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

        mx_left, my_top = lonlat_to_mercator(lon_left, lat_top)
        mx_right, my_bottom = lonlat_to_mercator(lon_right, lat_bottom)
        return mx_left, my_bottom, mx_right, my_top

    return tile_to_mercator_bounds(x, y, z)

@app.get("/tiles/{z}/{x}/{y}.png")
async def proxy_tile(z: int, x: int, y: int):
    try:
        # Calculate bbox in EPSG:3857
        minx, miny, maxx, maxy = tile_xyz_to_bbox_3857(x, y, z)
        bbox = f"{minx},{miny},{maxx},{maxy}"

        # Compose WMS request
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": "anom",  # adjust layer name as needed!
            "STYLES": "raster/div-RdBu-inv",
            "CRS": "EPSG:3857",
            "BBOX": bbox,
            "WIDTH": 256,
            "HEIGHT": 256,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "colorscalerange":"-2,2"
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(WMS_URL, params=params)
            if resp.status_code == 200:
                return Response(content=resp.content, media_type="image/png")
            else:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))