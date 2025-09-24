# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:15:39 2025

@author: anujd
"""

from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import NearestNDInterpolator
import xarray as xr
import pandas as pd

def extract_from_dap_ugrid(url, target_time, variable_name, mesh_lon_name='mesh_node_lon',
                     mesh_lat_name='mesh_node_lat', mesh_tri_name='mesh_face_node'):

    try:
        with xr.open_dataset(url, mask_and_scale=True, decode_cf=True) as ds:
            # Handle time axis
            if isinstance(ds.time.values[0], bytes):
                time_str = [t.decode('utf-8') for t in ds.time.values]
                time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
            else:
                time_dt = [pd.to_datetime(t).to_pydatetime() for t in ds.time.values]
            # Convert target time
            if "." in target_time:
                target_dt = pd.to_datetime(target_time).to_pydatetime()
            else:
                target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
            # Find closest time
            time_index = np.argmin([abs((t - target_dt).total_seconds()) for t in time_dt])
            act_time = str(ds.time.values[time_index])
            # Extract mesh coordinates and triangles
            lon = ds[mesh_lon_name].data
            lat = ds[mesh_lat_name].data
            triangles = ds[mesh_tri_name].data
            # Extract variable
            if variable_name not in ds.variables:
                raise ValueError(f"Variable '{variable_name}' not found. Available: {list(ds.variables.keys())}")
            var = ds[variable_name].isel(time=time_index)
            # If variable has 3 dims (e.g. (depth, node, face)), select first
            if len(var.shape) == 3:
                var = var.isel({var.dims[0]: 0})
            # Reduce to 1D if needed
            data = np.ma.masked_invalid(var.data.squeeze())
            return lon, lat, triangles, data, act_time
    except Exception as e:
        raise RuntimeError(f"Error accessing OpenDAP data: {str(e)}")

url = 'https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc'
variable_name = "hs"
target_time = "2025-09-29T12:00:00Z"
lon, lat, triangles, data, act_time = extract_from_dap_ugrid(url, target_time, variable_name)

# --- Plotting logic starts here ---
def get_custom_colormap(nColors, vmin, vmax, cmap):
    bounds = np.linspace(vmin, vmax, nColors + 1)
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm, bounds


# Use standard jet colormap, min=0, max=4
min_color_plot = 0.0
max_color_plot = 4.0
steps = 0.5
west_bound = -180
title = 'Significant Wave Height'
unit = 'm'
is_direction = False

fig, ax2 = plt.subplots(figsize=(8, 6))
from mpl_toolkits.axes_grid1 import make_axes_locatable

hs_dir = None
if is_direction:
    _, _, _, hs_dir, _ = extract_from_dap_ugrid(url, target_time, 'dirm')
else:
    hs_dir = None

cmap = plt.get_cmap('jet')
levels = np.arange(min_color_plot, max_color_plot + steps, steps)
norm = mpl.colors.Normalize(vmin=min_color_plot, vmax=max_color_plot)
ticks = levels[::2] if len(levels) > 10 else levels

if triangles.min() == 1:
    triangles = triangles - 1

if (float(west_bound) < 0) and (lon.min() > 0):
    lon = np.where(lon > 180, lon - 360, lon)
elif (float(west_bound) > 0) and (lon.min() < 0):
    lon = np.where(lon < 0, lon + 360, lon)

lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
lon_margin = (lon_max - lon_min) * 0.01
lat_margin = (lat_max - lat_min) * 0.01

plot_west = lon_min - lon_margin + 0.011
plot_east = lon_max + lon_margin - 0.011
plot_south = lat_min - lat_margin + 0.011
plot_north = lat_max + lat_margin - 0.011

ax2.cla()
divider = make_axes_locatable(ax2)
ax_legend = divider.append_axes("right", size="5%", pad=0.12)

triang = mpl.tri.Triangulation(lon, lat, triangles)
nan_mask = np.isnan(data)
tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
triang.set_mask(tri_mask)

cs = ax2.tricontourf(triang, data, cmap=cmap, norm=norm, levels=levels)

ax2.set_xlim(plot_west, plot_east)
ax2.set_ylim(plot_south, plot_north)
ax2.set_aspect('auto')
ax2.tick_params(axis='both', labelsize=6)
ax2.set_title(title,pad=10, fontsize=8)
ax2.grid(
    which='both',
    color='lightgray',
    linestyle=':',
    linewidth=0.8,
    alpha=0.8
)
if hs_dir is not None:
    x_arr = np.linspace(lon.min(), lon.max(), 30)
    y_arr = np.linspace(lat.min(), lat.max(), 30)
    xlon, ylat = np.meshgrid(x_arr, y_arr)
    xlon = xlon.flatten()
    ylat = ylat.flatten()
    interp = NearestNDInterpolator(list(zip(lon, lat)), hs_dir)
    zdir = interp(xlon, ylat)
    zdir = 270 - zdir
    zdir[zdir < 0] += 360
    udir = np.cos(np.deg2rad(zdir))
    vdir = np.sin(np.deg2rad(zdir))
    ax2.quiver(xlon, ylat, udir, vdir, units='xy', zorder=2, color='k', width=0.003, headwidth=3, headlength=5, alpha=0.8)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = mpl.pyplot.colorbar(
    sm,
    cax=ax_legend,
    orientation='vertical',
    extend='max',
    format='{x:.2f}',
    drawedges=True,
    label=f'{unit}',
    norm=norm,
    ticks=ticks
)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(f'{unit}', fontsize=6)
ax_legend.set_title("")

plt.tight_layout()
#plt.show()
plt.savefig('professional_wave_viz.png', dpi=300)
print("✅ Saved professional wave visualization: professional_wave_viz.png")
 