import json
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import xarray as xr
import sys
import os
import requests
from datetime import datetime, timedelta, timezone


def parse_time_to_naive_utc(time_str: Optional[str]) -> datetime:
    """
    Parse a wide range of ISO-8601 timestamp strings into a naive UTC datetime.
    Accepts:
      - 'YYYY-MM-DDTHH:MM:SSZ'
      - 'YYYY-MM-DDTHH:MM:SS.sssZ' (any fractional seconds)
      - 'YYYY-MM-DDTHH:MM:SS+00:00' (or other offsets)
    Returns a naive datetime in UTC (tzinfo=None).
    Raises ValueError if it cannot parse.
    """
    if time_str is None:
        raise ValueError("Time must be provided for time parsing.")

    t = str(time_str).strip()

    # 1) Try explicit common formats
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass

    # 2) Try ISO with offset via fromisoformat (replace 'Z' with '+00:00')
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    # 3) Try pandas if available
    try:
        import pandas as pd
        dt_pd = pd.to_datetime(t, utc=True, errors="raise")
        dt_py = dt_pd.to_pydatetime()
        # to_pydatetime can return a scalar or an array
        if hasattr(dt_py, "__iter__") and not isinstance(dt_py, datetime):
            dt_py = list(dt_py)[0]
        return dt_py.replace(tzinfo=None)
    except Exception:
        pass

    # 4) Try python-dateutil if available
    try:
        from dateutil import parser as du_parser
        dt = du_parser.parse(t)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    raise ValueError(f"Unrecognized time format: {time_str}")


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
            dt = parse_time_to_naive_utc(time)
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
            dt = parse_time_to_naive_utc(time)
            # First day of current month
            first_day = dt.replace(day=1)

            # Calculate the first day of the month two months ahead
            if first_day.month > 10:
                # December or November
                year = first_day.year + 1
                month = (first_day.month + 2) % 12
                if month == 0: 
                    month = 12
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
            dt = parse_time_to_naive_utc(time)
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
            dt = parse_time_to_naive_utc(time_str)
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
    
    # Special handling for Cook Islands THREDDS data
    if "gemthreddshpc.spc.int" in url and ("COK" in url or "cook" in layer_id.lower()):
        # UGRID unstructured mesh data (already has lat/lon coordinates)
        if "UGRID" in url or "ugrid" in layer_id.lower():
            return _load_cook_islands_ugrid_data(
                url=url,
                variable=variable,
                time=time,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max
            )
        # Inundation data (has UTM coordinates that need transformation)
        else:
            return _load_cook_islands_utm_data(
                url=url,
                variable=variable, 
                time=time,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max
            )
    plot = (plot or "contourf").lower()
    opts = _parse_plot_options(options)
    root_path = get_root_path()
    info = get_layer_dataset_download_info(str(layer_id), time, root_path)
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


def _load_cook_islands_utm_data(
    *,
    url: str,
    variable: str,
    time: Optional[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Special handler for Cook Islands THREDDS data which uses UTM Zone 4S coordinates.
    
    The Cook Islands data is in UTM Zone 4S (EPSG:32704) but tiles need WGS84 (EPSG:4326).
    This function handles the coordinate transformation.
    
    Returns:
    - lons_plot: 1D numpy array (longitude values)
    - lats_plot: 1D numpy array (latitude values)  
    - data_ma: 2D masked array aligned with (lats_plot, lons_plot)
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pyproj is required for Cook Islands UTM coordinate transformation")
    
    with xr.open_dataset(url) as ds:
        if variable not in ds:
            available_vars = list(ds.data_vars.keys())
            raise KeyError(f"Variable '{variable}' not found in dataset. Available: {available_vars}")
        
        # Get the data variable
        da = ds[variable]
        
        # Handle time dimension if present
        if time is not None and 'time' in da.dims:
            try:
                import pandas as pd
                time_target = pd.to_datetime(time)
                da = da.sel(time=time_target, method='nearest')
            except Exception:
                if 'time' in da.dims:
                    da = da.isel(time=0)
        elif 'time' in da.dims:
            da = da.isel(time=0)
        
        # Get UTM coordinates
        x_utm = ds.x.values  # UTM Easting
        y_utm = ds.y.values  # UTM Northing
        
        # Set up coordinate transformation: UTM Zone 4S -> WGS84
        transformer = Transformer.from_crs('EPSG:32704', 'EPSG:4326', always_xy=True)
        
        # Transform corner points to get geographic bounds
        x_min, x_max = x_utm.min(), x_utm.max()
        y_min, y_max = y_utm.min(), y_utm.max()
        
        lon_min_data, lat_max_data = transformer.transform(x_min, y_max)
        lon_max_data, lat_min_data = transformer.transform(x_max, y_min)
        
        # Create 1D coordinate arrays representing the geographic bounds
        lons_plot = np.linspace(lon_min_data, lon_max_data, len(x_utm))
        lats_plot = np.linspace(lat_min_data, lat_max_data, len(y_utm))
        
        # Get the data values
        if hasattr(da, 'values'):
            data_values = da.values
        else:
            data_values = da.data
            
        # Ensure proper data orientation and make it a masked array
        if data_values.ndim == 2:
            # NetCDF typically has (y, x) which matches (lat, lon)
            if data_values.shape != (len(y_utm), len(x_utm)):
                data_values = data_values.T
        
        # Create masked array
        data_ma = np.ma.masked_invalid(data_values)
        
        # Ensure latitude increases (standard for plotting)
        if lats_plot[0] > lats_plot[-1]:
            lats_plot = lats_plot[::-1]
            data_ma = data_ma[::-1, :]
        
        return lons_plot, lats_plot, data_ma


def _load_cook_islands_ugrid_data(
    *,
    url: str,
    variable: str,
    time: Optional[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    Load Cook Islands UGRID unstructured mesh data using native triangular mesh approach.
    This preserves the original mesh structure and provides sharp boundaries without interpolation artifacts.
    """
    import matplotlib.tri as tri
    
    with xr.open_dataset(url) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' not found in UGRID dataset")
        
        # Extract mesh coordinates and triangles
        lons_mesh = ds.mesh_node_lon.values  # 1D array of node longitudes
        lats_mesh = ds.mesh_node_lat.values  # 1D array of node latitudes
        triangles = ds.mesh_face_node.values  # Triangle connectivity
        
        # Handle coordinate system normalization for Pacific region
        if lons_mesh.min() > 0 and lon_min < 0:
            lons_mesh = np.where(lons_mesh > 180, lons_mesh - 360, lons_mesh)
        
        # Get variable data
        da = ds[variable]
        
        # Handle time selection
        if time is not None:
            # First try numeric time index (including string numbers like "0", "1", etc)
            try:
                if isinstance(time, str) and str(time).strip().isdigit():
                    time_idx = int(str(time).strip())
                    if 0 <= time_idx < len(ds.time):
                        da = da.isel(time=time_idx)
                    else:
                        da = da.isel(time=0)
                elif isinstance(time, int):
                    if 0 <= time < len(ds.time):
                        da = da.isel(time=time)
                    else:
                        da = da.isel(time=0)
                else:
                    # Try parsing as ISO datetime
                    try:
                        time_dt = parse_time_to_naive_utc(time)
                        da = da.sel(time=time_dt, method='nearest')
                    except (KeyError, TypeError, ValueError):
                        # Fall back to first time step
                        da = da.isel(time=0)
            except Exception:
                # If anything goes wrong with time selection, use first time step
                da = da.isel(time=0)
        else:
            da = da.isel(time=0)
            da = da.isel(time=0)
        
        # Get data values (should be 1D array matching mesh nodes)
        if hasattr(da, 'values'):
            data_values = da.values
        else:
            data_values = da.data
        
        # Handle 3D data (e.g., depth dimension) - take surface layer
        if len(data_values.shape) > 1:
            data_values = data_values.squeeze()
            if len(data_values.shape) > 1:
                data_values = data_values[0]  # First level
        
        # Create masked array for the data
        data_ma = np.ma.masked_invalid(data_values)
        
        # Fix 1-based to 0-based indexing for triangles
        if triangles.min() == 1:
            triangles = triangles - 1
        
        # Create triangulation object
        triang = tri.Triangulation(lons_mesh, lats_mesh, triangles)
        
        # Mask triangles that have any NaN vertices
        nan_mask = np.isnan(data_values)
        tri_mask = np.any(np.where(nan_mask[triang.triangles], True, False), axis=1)
        triang.set_mask(tri_mask)
        
        # Create regular grid for COG tile generation
        grid_resolution = 100  # Number of grid points in each dimension
        lons_grid = np.linspace(lon_min, lon_max, grid_resolution)
        lats_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lons_2d, lats_2d = np.meshgrid(lons_grid, lats_grid)
        
        # Use triangular linear interpolation - preserves mesh structure
        try:
            interpolator = tri.LinearTriInterpolator(triang, data_ma)
            data_interpolated = interpolator(lons_2d, lats_2d)
        except Exception as e:
            # Fallback to cubic interpolation if linear fails
            try:
                interpolator = tri.CubicTriInterpolator(triang, data_ma)
                data_interpolated = interpolator(lons_2d, lats_2d)
            except:
                # Final fallback to nearest neighbor
                from scipy.interpolate import griddata
                valid_mask = ~nan_mask
                if np.any(valid_mask):
                    lons_valid = lons_mesh[valid_mask]
                    lats_valid = lats_mesh[valid_mask]
                    data_valid = data_values[valid_mask]
                    points = np.column_stack([lons_valid, lats_valid])
                    data_interpolated = griddata(
                        points, data_valid, (lons_2d, lats_2d), 
                        method='nearest', fill_value=np.nan
                    )
                else:
                    data_interpolated = np.full_like(lons_2d, np.nan)
        
        # Create final masked array
        data_result = np.ma.masked_invalid(data_interpolated)
        
        return lons_grid, lats_grid, data_result