import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Dict, Any, List, Tuple

AVAILABLE_PLOTS = ("contourf", "contour", "pcolormesh", "imshow", "discrete", "discrete_cmap")


def _parse_discrete_segments(ranges: List[str]) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    for r in ranges:
        s = str(r).strip()
        if "-" in s:
            parts = s.split("-", 1)
            a = parts[0].strip()
            b = parts[1].strip()
            # If either side is empty, treat as single value
            if a == '' or b == '':
                try:
                    f = float(a or b)
                    start = end = f
                except Exception:
                    raise ValueError(f"Invalid range specifier: '{s}'")
            else:
                start, end = float(a), float(b)
                if end < start:
                    start, end = end, start
        else:
            start = end = float(s)
        segments.append((start, end))
    return segments


def draw_plot(
    ax: plt.Axes,
    *,
    plot: str,
    lons: np.ndarray,
    lats: np.ndarray,
    data: np.ndarray,
    cmap,
    levels: Optional[np.ndarray],
    vmin: Optional[float],
    vmax: Optional[float],
    options: Dict[str, Any],
):
    plot = (plot or "contourf").lower()

    if plot == "contourf":
        kw = {"extend": "both", "antialiased": False}
        kw.update(options or {})
        return ax.contourf(lons, lats, data, levels=levels, cmap=cmap, **kw)

    if plot == "contour":
        kw = {"linewidths": 0.7}
        kw.update(options or {})
        return ax.contour(lons, lats, data, levels=levels, cmap=cmap, **kw)

    if plot == "pcolormesh":
        kw = {"shading": "auto"}
        kw.update(options or {})
        return ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, **kw)

    if plot == "imshow":
        origin = "lower" if (lats[0] < lats[-1]) else "upper"
        extent = [float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))]
        kw = {"origin": origin}
        kw.update(options or {})
        return ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **kw)

    if plot == "discrete":
        ranges = (options or {}).get("ranges")
        colors = (options or {}).get("colors")
        transparent_below = (options or {}).get("transparent_below", None)
        mask_out_of_range = bool((options or {}).get("mask_out_of_range", True))

        if not isinstance(ranges, list) or not isinstance(colors, list):
            raise ValueError("discrete plot requires plot_options.ranges (list) and plot_options.colors (list)")
        segments = _parse_discrete_segments(ranges)
        if len(colors) != len(segments):
            raise ValueError("Number of colors must equal number of ranges for discrete plot")

        # Accept any input (masked or not), promote to float for NaN support
        if np.ma.isMaskedArray(data):
            base_mask = np.array(np.ma.getmaskarray(data), dtype=bool)
            data_f = data.astype(float, copy=False).filled(np.nan)
        else:
            data_f = np.asarray(data, dtype=float)
            base_mask = np.zeros_like(data_f, dtype=bool)

        # Only allow values exactly matching the segments' start values (e.g., 0,1,2,3,4)
        allowed_values = np.array([start for (start, end) in segments if start == end])
        allowed = np.isin(data_f, allowed_values)
        base_mask = base_mask | (~allowed)

        # Optionally, also apply threshold mask if requested
        threshold_mask = np.zeros_like(data_f, dtype=bool)
        if transparent_below is not None:
            threshold_mask = data_f < float(transparent_below)

        invalid = base_mask | (~np.isfinite(data_f)) | threshold_mask

        seg_idx = np.full(data_f.shape, -1, dtype=np.int16)
        valid_data = ~invalid
        if np.any(valid_data):
            vals = data_f
            for i, (start, end) in enumerate(segments):
                # For strict class coloring, only allow exact matches if start==end
                if start == end:
                    mask_i = valid_data & (vals == start)
                else:
                    mask_i = valid_data & (vals >= start) & (vals <= end)
                seg_idx[mask_i] = i

        seg_mask = invalid | ((seg_idx < 0) if mask_out_of_range else False)
        seg_idx_ma = np.ma.array(seg_idx, mask=seg_mask)

        listed = mcolors.ListedColormap(colors)
        listed.set_bad((0, 0, 0, 0))  # transparent

        N = len(segments)
        boundaries = np.arange(-0.5, N + 0.5, 1.0, dtype=float)
        norm = mcolors.BoundaryNorm(boundaries, N, clip=False)

        opt = dict(options or {})
        for k in ("ranges", "colors", "labels", "transparent_below", "mask_out_of_range"):
            opt.pop(k, None)

        kw = {"shading": "auto"}
        kw.update(opt)
        return ax.pcolormesh(lons, lats, seg_idx_ma, cmap=listed, norm=norm, **kw)

    if plot == "discrete_cmap":
        # options: ranges (list of str, e.g. ['0.01', '0.03-0.05', ...]), cmap (str or Colormap), transparent_below (optional)
        ranges = (options or {}).get("ranges") or (options or {}).get("values")
        cmap_name = (options or {}).get("cmap", "jet")
        transparent_below = (options or {}).get("transparent_below", None)
        mask_out_of_range = bool((options or {}).get("mask_out_of_range", True))

        if not isinstance(ranges, list) or len(ranges) == 0:
            raise ValueError("discrete_cmap plot requires plot_options.ranges (list of ranges or values to plot)")
        segments = _parse_discrete_segments(ranges)
        N = len(segments)

        # Accept any input (masked or not), promote to float for NaN support
        if np.ma.isMaskedArray(data):
            base_mask = np.array(np.ma.getmaskarray(data), dtype=bool)
            data_f = data.astype(float, copy=False).filled(np.nan)
        else:
            data_f = np.asarray(data, dtype=float)
            base_mask = np.zeros_like(data_f, dtype=bool)

        seg_idx = np.full(data_f.shape, -1, dtype=np.int16)
        valid_data = ~base_mask & np.isfinite(data_f)
        for i, (start, end) in enumerate(segments):
            # For strict class coloring, only allow exact matches if start==end (with tolerance for floats)
            if start == end:
                mask_i = valid_data & np.isclose(data_f, start, atol=1e-6)
            else:
                mask_i = valid_data & (data_f >= start) & (data_f <= end)
            seg_idx[mask_i] = i

        # Optionally, also apply threshold mask if requested
        threshold_mask = np.zeros_like(data_f, dtype=bool)
        if transparent_below is not None:
            threshold_mask = data_f < float(transparent_below)

        seg_mask = base_mask | (~np.isfinite(data_f)) | threshold_mask | ((seg_idx < 0) if mask_out_of_range else False)
        seg_idx_ma = np.ma.array(seg_idx, mask=seg_mask)

        cmap_obj = plt.get_cmap(cmap_name, N)
        boundaries = np.arange(-0.5, N + 0.5, 1.0, dtype=float)
        norm = mcolors.BoundaryNorm(boundaries, N, clip=False)
        cmap_obj.set_bad((0, 0, 0, 0))  # transparent

        opt = dict(options or {})
        for k in ("ranges", "values", "cmap", "labels", "transparent_below", "mask_out_of_range"):
            opt.pop(k, None)

        kw = {"shading": "auto"}
        kw.update(opt)
        return ax.pcolormesh(lons, lats, seg_idx_ma, cmap=cmap_obj, norm=norm, **kw)

    # Fallback to contourf if unknown
    kw = {"extend": "both", "antialiased": False}
    kw.update(options or {})
    return ax.contourf(lons, lats, data, levels=levels, cmap=cmap, **kw)