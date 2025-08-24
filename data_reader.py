import json
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import xarray as xr
import sys
import os
import requests
from datetime import datetime

def get_root_path(config_file: str = "config.json") -> str:
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["root-path"]

def get_layer_dataset_download_info(layer_id, time=None, root_dir=None, mapper_filename='dataset_mapper.json'):
    """
    Given a layer_id, optional time, and optional root_dir, reads the mapper file for dataset_id,
    queries the dataset API, and returns:
        - path: local_directory_path (with {root-dir} replaced if root_dir is given)
        - file_name: download_file_prefix + download_file_infix + download_file_suffix
    If download_file_infix contains % (strftime), uses 'time' to fill it in.
    If layer_id does not exist in the mapping, returns 0 and does not execute further.
    """
    # Convert layer_id to string for mapping lookup
    layer_id_str = str(layer_id)
    
    # Read the mapping file from the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mapper_path = os.path.join(current_dir, mapper_filename)
    
    with open(mapper_path, 'r') as f:
        mapping = json.load(f)
    
    # Get the dataset_id for the given layer_id
    dataset_id = mapping.get(layer_id_str)
    if not dataset_id:
        return 0  # Immediately return 0 and DO NOT execute further
    
    # Query the API for this dataset_id
    url = f"https://ocean-middleware.spc.int/middleware/api/dataset/{dataset_id}/?format=json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    # Extract the required fields
    prefix = data["download_file_prefix"]
    infix = data["download_file_infix"]
    suffix = data["download_file_suffix"]
    local_directory_path = data["local_directory_path"]
    if layer_id == "26": 
        infix = "%Y%m_%Y%m"
    elif layer_id == "36":
        infix = "%Y%m_%Y%m"
        suffix = ".nc"
    elif layer_id == "37":
        infix = "decile.%Y%m"
        suffix = ".nc"
    elif layer_id == "35" or layer_id == "39":
        infix = "%Y%m"
        suffix = ".nc"

    

    # Prepare file name
    if "%" in infix:
        if "_" in infix and "AQUA" in infix:
            first_fmt, last_fmt = infix.split("_", 1)
            # Parse the base date
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            # First day of month
            first_day = dt.replace(day=1)
            # Last day of month: go to next month, subtract 1 day
            if dt.month == 12:
                next_month = dt.replace(year=dt.year + 1, month=1, day=1)
            else:
                next_month = dt.replace(month=dt.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
            # Format
            infix_formatted = f"{first_day.strftime(first_fmt)}_{last_day.strftime(last_fmt)}"
        elif not time:
            raise ValueError("Time must be provided for infix formatting.")
        # Parse time string like "2025-10-16T12:00:00Z"
        elif layer_id == "36": 
            first_fmt, last_fmt = infix.split("_", 1)
            # Parse the base date
            # Parse the base date
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            # First day of current month
            first_day = dt.replace(day=1)

            # Calculate the first day of the month two months ahead
            if first_day.month > 10:
                # December or November
                year = first_day.year + 1
                month = (first_day.month + 2) % 12
                if month == 0: month = 12
            else:
                year = first_day.year
                month = first_day.month + 2
            next2_month = first_day.replace(year=year, month=month, day=1)
            # Last day is the last day of that month (go to next month, subtract 1 day)
            if month == 12:
                month3 = 1
                year3 = year + 1
            else:
                month3 = month + 1
                year3 = year
            month3_first = next2_month.replace(year=year3, month=month3, day=1)
            last_day = month3_first - timedelta(days=1)

            # Format
            infix_formatted = f"{first_day.strftime(first_fmt)}_{last_day.strftime(last_fmt)}"
        else:
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            infix_formatted = dt.strftime(infix)
    elif "none" in infix:
        infix_formatted = ""
        suffix = ""
    else:
        infix_formatted = infix

    file_name = f"{prefix}{infix_formatted}{suffix}"
    if not file_name.endswith('.nc'):
        file_name += '.nc'
    if layer_id == "16":
        file_name = 'latest.nc'
    if layer_id == "2" or layer_id =="10" or layer_id =="11" or layer_id =="12" or layer_id =="14":
        file_name = 'latest_merged.nc'
    if layer_id == "19":
        file_name = 'latest_merged.nc'
    if layer_id == "47":
        file_name = 'sst_trend.nc'
    if layer_id == "41":
        def get_weekly_filename(time_str):
            """
            Given a time string, returns the AQUA_MODIS 8-day composite filename
            based on the custom start date 2025-05-25.
            """
            # Reference start and end date from your first dataset
            ref_start = datetime(2025, 5, 25)
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            days_since_ref = (dt - ref_start).days
            period_index = days_since_ref // 8
            # Handle dates before the reference period
            if days_since_ref < 0:
                raise ValueError("Date is before the first available dataset period.")
            start_dt = ref_start + timedelta(days=period_index * 8)
            end_dt = start_dt + timedelta(days=7)
            fname = f"AQUA_MODIS.{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.L3m.8D.CHL.chlor_a.4km.NRT.nc"
            return fname

        file_name = get_weekly_filename(time)
    if layer_id == "26":
        local_directory_path = "{root-dir}/model/regional/copernicus/hindcast/monthly/ssh"
    if layer_id == "35" or layer_id == "39":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/monthly/sst_anomalies"
    if layer_id == "36":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/3monthly/sst_anomalies"
    if layer_id == "37":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/decile/sst_anomalies"
    if layer_id == "47":
        local_directory_path = "{root-dir}/model/regional/noaa/hindcast/trend"
    if layer_id == "2" or layer_id == "10" or layer_id =="11" or layer_id =="12" or layer_id =="14":
        local_directory_path = "{root-dir}/model/regional/bom/forecast/hourly/wavewatch3_latest"
    # Replace {root-dir} if root_dir is supplied
    if root_dir:
        path = local_directory_path.replace("{root-dir}", root_dir)
    else:
        path = local_directory_path

    return {
        "path": path,
        "file_name": file_name
    }


def _select_time(da: xr.DataArray, time_param: Optional[str]) -> xr.DataArray:
    """
    Return a 2D slice (lat, lon) by selecting a time based on index or ISO datetime.
    Falls back to the first time if parsing fails or time is missing.
    """
    if "time" not in da.dims:
        return da.squeeze()

    if time_param is not None:
        # Try integer index
        try:
            idx = int(str(time_param).strip())
            return da.isel(time=idx).squeeze()
        except Exception:
            pass

    if time_param is None:
        return da.isel(time=0).squeeze()

    # Try datetime-nearest
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


def _find_coords(da: xr.DataArray) -> Tuple[str, str]:
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lon_name = next((n for n in lon_candidates if n in da.coords or n in da.dims), None)
    lat_name = next((n for n in lat_candidates if n in da.coords or n in da.dims), None)
    if lon_name is None or lat_name is None:
        raise ValueError("Could not determine lon/lat coordinate names in the dataset.")
    return lon_name, lat_name


def _parse_plot_options(plot_options: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if not plot_options:
        return {}
    try:
        if isinstance(plot_options, str):
            return json.loads(plot_options)
        return dict(plot_options)
    except Exception:
        return {}


def load_plot_ready_arrays(
    *,
    layer_id: str,
    url: str,
    variable: str,
    time: Optional[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    plot: str = "contourf",
    options: Optional[Union[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Open the remote dataset, extract the requested variable and time,
    subset to the lon/lat bbox (handling 0..360 or -180..180 domains and wrap),
    ensure increasing longitude and latitude order, and return arrays ready for plotting.

    Returns:
    - lons_plot: 1D numpy array (increasing)
    - lats_plot: 1D numpy array (increasing)
    - data_ma: 2D masked array aligned with (lats_plot, lons_plot)
    """
    plot = (plot or "contourf").lower()
    opts = _parse_plot_options(options)
    root_path = get_root_path()
    info = get_layer_dataset_download_info(str(layer_id),time,root_path)
    check_local = True
    local_file_name = ""
    if info == 0:
        check_local = False
        local_file_name = url
    else:
        local_file_name = "%s/%s" % (info['path'], info['file_name'])
        check_local = True

    print(check_local)
    print(local_file_name)
        

    with xr.open_dataset(local_file_name) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' not found in dataset")
        da = ds[variable]
        da2d = _select_time(da, time)

        lon_name, lat_name = _find_coords(da2d)
        if list(da2d.dims) != [lat_name, lon_name]:
            da2d = da2d.transpose(lat_name, lon_name)

        # Full coords as numpy arrays
        lons_all = np.asarray(da2d.coords[lon_name].values)
        lats_all = np.asarray(da2d.coords[lat_name].values)

        # Decide longitude domain from request:
        use_0360 = (0.0 <= lon_min <= 360.0) and (0.0 <= lon_max <= 360.0)
        lons_base = (lons_all + 360.0) % 360.0 if use_0360 else lons_all

        # Build longitude selection mask, supporting wrap (e.g., 350..20)
        if use_0360 and lon_min > lon_max:
            lon_mask = (lons_base >= lon_min) | (lons_base <= lon_max)
        else:
            lon_mask = (lons_base >= lon_min) & (lons_base <= lon_max)

        lat_mask = (lats_all >= lat_min) & (lats_all <= lat_max)

        lon_idx = np.where(lon_mask)[0]
        lat_idx = np.where(lat_mask)[0]
        if lon_idx.size == 0 or lat_idx.size == 0:
            raise ValueError("Requested lon/lat bbox selects no data from the dataset.")

        # Sort longitudes so X is strictly increasing within the chosen domain
        lons_sel = lons_base[lon_idx]
        order = np.argsort(lons_sel)
        lon_idx = lon_idx[order]
        lons_plot = lons_sel[order]  # 1D increasing

        # Subset data and latitudes
        da_sub = da2d.isel({lon_name: lon_idx, lat_name: lat_idx})
        lats_plot = np.asarray(da_sub.coords[lat_name].values)

        # Discrete plot special masking using optional 'mask' variable in dataset
        if plot == "discrete" and "mask" in ds.variables:
            mask_da = ds["mask"]
            # Select the same time index/value for mask
            mask_da = _select_time(mask_da, time)
            # Subset spatially
            mask_da = mask_da.isel({lon_name: lon_idx, lat_name: lat_idx})

            # Flip latitude if necessary (for both data and mask)
            if lats_plot[0] > lats_plot[-1]:
                lats_plot = lats_plot[::-1]
                da_sub = da_sub.isel({lat_name: slice(None, None, -1)})
                mask_da = mask_da.isel({lat_name: slice(None, None, -1)})

            data_vals = np.asarray(da_sub.values)
            mask_arr = np.asarray(mask_da.values)

            # Allow only classes 0..4 and where mask==0
            allowed_class = np.isin(data_vals, [0, 1, 2, 3, 4])
            allowed_mask = (mask_arr == 0)
            allowed = allowed_class & allowed_mask
            data_ma = np.ma.masked_where(~allowed, data_vals)
        else:
            # Ensure latitude increases (helps contouring and imshow consistency)
            if lats_plot[0] > lats_plot[-1]:
                lats_plot = lats_plot[::-1]
                da_sub = da_sub.isel({lat_name: slice(None, None, -1)})

            # Prepare data for plotting (masked where invalid)
            data_vals = np.asarray(da_sub.values)
            data_ma = np.ma.masked_invalid(data_vals)

    return lons_plot, lats_plot, data_ma