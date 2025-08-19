import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Dict, Any, List, Tuple

# Public: keep this in sync with main.py docstrings
AVAILABLE_PLOTS = ("contourf", "contour", "pcolormesh", "imshow", "discrete")


def _parse_discrete_segments(ranges: List[str]) -> List[Tuple[float, float]]:
    """
    Convert list like ["0-1","2-3","4"] into [(0,1),(2,3),(4,4)] with floats.
    """
    segments: List[Tuple[float, float]] = []
    for r in ranges:
        s = str(r).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            start, end = float(a), float(b)
        else:
            start = end = float(s)
        if end < start:
            start, end = end, start
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
    """
    Dispatch to different plot types.
    Supported:
      - contourf (default): uses 'levels'
      - contour: line contours, uses 'levels'
      - pcolormesh: uses vmin/vmax
      - imshow: uses vmin/vmax with extent
      - discrete: categorical ranges using options.ranges/colors/(optional)labels
    """
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
        # Expect: options = { "ranges": [...], "colors": [...], "labels": [...]? }
        ranges = (options or {}).get("ranges")
        colors = (options or {}).get("colors")
        if not isinstance(ranges, list) or not isinstance(colors, list):
            raise ValueError("discrete plot requires plot_options.ranges (list) and plot_options.colors (list)")
        segments = _parse_discrete_segments(ranges)
        if len(colors) != len(segments):
            raise ValueError("Number of colors must equal number of ranges for discrete plot")

        # Build categorical index for segments
        seg_idx = np.full(data.shape, -1, dtype=np.int16)
        for i, (start, end) in enumerate(segments):
            mask_i = (data >= start) & (data <= end)
            seg_idx[mask_i] = i

        # Mask nodata/out-of-range to transparent
        seg_mask = seg_idx < 0
        seg_idx_ma = np.ma.array(seg_idx, mask=seg_mask)

        listed = mcolors.ListedColormap(colors)
        listed.set_bad((0, 0, 0, 0))  # transparent

        # Boundaries at integer centers: [-0.5, 0.5, 1.5, ...]
        N = len(segments)
        boundaries = np.arange(-0.5, N + 0.5, 1.0, dtype=float)
        norm = mcolors.BoundaryNorm(boundaries, N)

        # Do not mutate caller options
        opt = dict(options or {})
        for k in ("ranges", "colors", "labels"):
            opt.pop(k, None)

        kw = {"shading": "auto"}
        kw.update(opt)
        return ax.pcolormesh(lons, lats, seg_idx_ma, cmap=listed, norm=norm, **kw)

    # Fallback to contourf if unknown
    kw = {"extend": "both", "antialiased": False}
    kw.update(options or {})
    return ax.contourf(lons, lats, data, levels=levels, cmap=cmap, **kw)