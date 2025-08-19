import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

url = "https://ocean-thredds01.spc.int/thredds/dodsC/POP/model/regional/noaa/nrt/daily/sst_anomalies/oisst-avhrr-v02r01.20250815_preliminary.nc"
var_name = "anom"
vmin, vmax, step = -2, 2, 0.5
levels = np.arange(vmin, vmax + step, step)
cmap = "RdBu_r"

# Open dataset
ds = xr.open_dataset(url)
da = ds[var_name]

# Get the first timestamp (2D)
if "time" in da.dims:
    data2d = da.isel(time=0).squeeze()
else:
    data2d = da.squeeze()

lons = data2d["lon"].values
lats = data2d["lat"].values

# Meshgrid if coordinates are 1D
if lats.ndim == 1 and lons.ndim == 1:
    Lon, Lat = np.meshgrid(lons, lats)
else:
    Lon, Lat = lons, lats

# Plot without basemap
fig, ax = plt.subplots(figsize=(10, 6))
cf = ax.contourf(Lon, Lat, data2d.values, levels=levels, cmap=cmap, extend="both")
plt.colorbar(cf, ax=ax, orientation="vertical", label="SST Anomaly (°C)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_ylim(np.min(lats), np.max(lats))
timestamp = str(data2d.coords.get("time", ""))
ax.set_title(f"SST Anomaly")
plt.tight_layout()
plt.show()