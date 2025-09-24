# plotters.py
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable

ColormapLike = Union[str, mcolors.Colormap]

AVAILABLE_PLOTS: Tuple[str, ...] = (
    "contourf", "contour", "pcolormesh", "imshow", "discrete", "discrete_cmap"
)

__all__ = [
    "AVAILABLE_PLOTS",
    "draw_plot",
    "make_colorbar",
]

# ----------------------
# Internal helpers
# ----------------------

def _parse_discrete_segments(ranges: List[str]) -> List[Tuple[float, float]]:
    """
    Parse range strings like '0.03-0.05' or '1' into [(start, end), ...].
    Accepts open-ended like '-2' or '5-' and treats them as singletons.
    """
    segments: List[Tuple[float, float]] = []
    for r in ranges:
        s = str(r).strip()
        if "-" in s:
            a, b = [p.strip() for p in s.split("-", 1)]
            if a == "" or b == "":
                # treat as singleton like "-2" or "5-"
                try:
                    f = float(a or b)
                    start = end = f
                except Exception as e:
                    raise ValueError(f"Invalid range specifier: '{s}'") from e
            else:
                start, end = float(a), float(b)
                if end < start:
                    start, end = end, start
        else:
            start = end = float(s)
        segments.append((start, end))
    return segments


def _validate_shapes(lons: np.ndarray, lats: np.ndarray, data: np.ndarray) -> None:
    """
    Ensure shapes are valid for contour/pcolor style functions.

    Valid cases:
        - X,Y are 2D and both match Z.shape
        - X is 1D with length Z.shape[1], Y is 1D with length Z.shape[0]
    """
    Z = np.asarray(data)
    X = np.asarray(lons)
    Y = np.asarray(lats)

    if Z.ndim != 2:
        raise ValueError(f"`data` must be 2D; got shape {Z.shape}")

    if X.ndim == Y.ndim == 2:
        if X.shape != Z.shape or Y.shape != Z.shape:
            raise ValueError(
                f"When lons,lats are 2D they must match data shape. "
                f"Got lons {X.shape}, lats {Y.shape}, data {Z.shape}"
            )
        return

    if X.ndim == Y.ndim == 1:
        if X.size != Z.shape[1] or Y.size != Z.shape[0]:
            raise ValueError(
                f"When lons,lats are 1D, len(lons)==data.shape[1] and len(lats)==data.shape[0]. "
                f"Got len(lons)={X.size}, len(lats)={Y.size}, data={Z.shape}"
            )
        return

    raise ValueError(
        "Invalid lons/lats shapes. Use either 2D arrays matching data or 1D lons (ncols) & 1D lats (nrows)."
    )


def _discrete_index_map(
    data: np.ndarray,
    segments: Sequence[Tuple[float, float]],
    *,
    mask_out_of_range: bool,
    transparent_below: Optional[float],
    atol: float = 1e-6,
) -> np.ma.MaskedArray:
    """
    Map data to integer bin indices (0..N-1) with masking behavior.

    Policy:
      - First matching segment wins (important for overlapping bins).
      - Singleton bins use np.isclose(value, start, atol).
      - Inclusive ranges for non-singletons (>= start and <= end).
    """
    # Promote to float; preserve existing mask
    if np.ma.isMaskedArray(data):
        base_mask = np.array(np.ma.getmaskarray(data), dtype=bool)
        vals = data.astype(float, copy=False).filled(np.nan)
    else:
        vals = np.asarray(data, dtype=float)
        base_mask = np.zeros_like(vals, dtype=bool)

    valid = ~base_mask & np.isfinite(vals)
    seg_idx = np.full(vals.shape, -1, dtype=np.int32)

    for i, (start, end) in enumerate(segments):
        if np.isclose(start, end, atol=atol):
            m = valid & np.isclose(vals, start, atol=atol)
        else:
            m = valid & (vals >= start) & (vals <= end)
        # Assign only where unassigned to preserve first-match precedence
        assignable = m & (seg_idx < 0)
        seg_idx[assignable] = i

    thr_mask = np.zeros_like(vals, dtype=bool)
    if transparent_below is not None:
        thr_mask = vals < float(transparent_below)

    oob_mask = (seg_idx < 0) if mask_out_of_range else np.zeros_like(vals, dtype=bool)
    final_mask = base_mask | ~np.isfinite(vals) | thr_mask | oob_mask
    return np.ma.array(seg_idx, mask=final_mask)


# ----------------------
# Public API
# ----------------------

def draw_plot(
    ax: Axes,
    *,
    plot: str,
    lons: np.ndarray,
    lats: np.ndarray,
    data: np.ndarray,
    cmap: Optional[ColormapLike] = None,
    levels: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Draw a georeferenced plot on `ax`.

    Args:
        ax: Matplotlib Axes.
        plot: One of AVAILABLE_PLOTS.
        lons, lats: 1D/2D grids (see _validate_shapes).
        data: 2D data array.
        cmap: Colormap name or object.
        levels: Required for contour/contourf.
        vmin, vmax: Optional scalar bounds for pcolormesh/imshow.
        options: Extra kwargs passed to Matplotlib plotting functions.
                 Discrete-specific keys are consumed here:
                   - ranges (or values): list[str] like ["0-1", "1-2", "3"]
                   - colors: list[str or rgba] (for plot="discrete" only)
                   - cmap: str (for plot="discrete_cmap")
                   - mask_out_of_range: bool (default True)
                   - transparent_below: float | None
                   - atol: float tolerance for singletons

    Returns:
        - For non-discrete plots: mappable (QuadMesh/ContourSet/Image).
        - For discrete/discrete_cmap: (mappable, extras) where
              extras = {"cmap", "norm", "tick_locs", "tick_labels"}.
          Use with `make_colorbar` or build your own colorbar.
    """
    options = dict(options or {})
    plot = (plot or "contourf").lower()

    # Validate shapes early for grid-based modes
    if plot in {"contourf", "contour", "pcolormesh", "discrete", "discrete_cmap"}:
        _validate_shapes(lons, lats, data)

    if plot == "contourf":
        if levels is None:
            raise ValueError("`levels` is required for contourf.")
        kw = {"extend": "both", "antialiased": True}
        kw.update(options)
        mappable = ax.contourf(lons, lats, data, levels=levels, cmap=cmap, **kw)
        return mappable

    if plot == "contour":
        if levels is None:
            raise ValueError("`levels` is required for contour.")
        kw = {"linewidths": 0.7}
        kw.update(options)
        mappable = ax.contour(lons, lats, data, levels=levels, cmap=cmap, **kw)
        return mappable

    if plot == "pcolormesh":
        kw = {"shading": "auto"}
        kw.update(options)
        mappable = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
        return mappable

    if plot == "imshow":
        Z = np.asarray(data)
        X = np.asarray(lons)
        Y = np.asarray(lats)

        # Compute extent safely regardless of 1D/2D arrays
        if X.ndim == 2:
            x_min, x_max = float(X.min()), float(X.max())
        else:
            x_min, x_max = (float(X[0]), float(X[-1])) if X[0] <= X[-1] else (float(X[-1]), float(X[0]))
        if Y.ndim == 2:
            y_min, y_max = float(Y.min()), float(Y.max())
        else:
            y_min, y_max = (float(Y[0]), float(Y[-1])) if Y[0] <= Y[-1] else (float(Y[-1]), float(Y[0]))

        extent = [x_min, x_max, y_min, y_max]
        origin = "lower" if y_min < y_max else "upper"

        kw = {"origin": origin}
        kw.update(options)
        mappable = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **kw)
        return mappable

    if plot in {"discrete", "discrete_cmap"}:
        ranges = options.pop("ranges", options.pop("values", None))
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("Discrete plots require plot_options.ranges (list of 'a-b' or 'v' strings).")

        segments = _parse_discrete_segments(ranges)
        mask_out_of_range = bool(options.pop("mask_out_of_range", True))
        transparent_below = options.pop("transparent_below", None)
        atol = float(options.pop("atol", 1e-6))

        idx_ma = _discrete_index_map(
            data, segments,
            mask_out_of_range=mask_out_of_range,
            transparent_below=transparent_below,
            atol=atol,
        )

        N = len(segments)
        boundaries = np.arange(-0.5, N + 0.5, 1.0, dtype=float)
        norm = mcolors.BoundaryNorm(boundaries, N, clip=False)

        if plot == "discrete":
            colors = options.pop("colors", None)
            if not isinstance(colors, list) or len(colors) != N:
                raise ValueError("plot='discrete' requires plot_options.colors list matching number of ranges.")
            cmap_obj = mcolors.ListedColormap(colors, name="discrete")
        else:  # discrete_cmap
            cmap_name = options.pop("cmap", "viridis")
            cmap_obj = plt.get_cmap(cmap_name, N)

        cmap_obj.set_bad((0, 0, 0, 0))  # transparent for masked

        kw = {"shading": "auto"}
        kw.update(options)
        mappable = ax.pcolormesh(lons, lats, idx_ma, cmap=cmap_obj, norm=norm, **kw)

        # Tick metadata for colorbar
        tick_locs = np.arange(N)
        tick_labels = []
        for (a, b) in segments:
            if np.isclose(a, b, atol=atol):
                tick_labels.append(f"{a:g}")
            else:
                tick_labels.append(f"{a:g}–{b:g}")

        extras = {
            "cmap": cmap_obj,
            "norm": norm,
            "tick_locs": tick_locs,
            "tick_labels": tick_labels,
        }
        return mappable, extras

    # Fallback: if levels given, use contourf; otherwise pcolormesh
    kw = {"extend": "both", "antialiased": True}
    kw.update(options)
    if levels is None:
        mappable = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        return mappable
    else:
        mappable = ax.contourf(lons, lats, data, levels=levels, cmap=cmap, **kw)
        return mappable


def make_colorbar(
    ax: Axes,
    mappable: ScalarMappable,
    *,
    extras: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    orientation: str = "vertical",
    **colorbar_kwargs: Any,
):
    """
    Convenience helper to add a colorbar that respects discrete extras.

    Args:
        ax: Axes to attach the colorbar to.
        mappable: Return value from draw_plot (QuadMesh/ContourSet/Image).
        extras: If provided (from discrete plots), should contain:
                {"tick_locs", "tick_labels"} to set ticks & labels.
        label: Optional colorbar label.
        orientation: "vertical" or "horizontal".
        **colorbar_kwargs: Forwarded to plt.colorbar.

    Returns:
        The created colorbar object.
    """
    cbar = plt.colorbar(mappable, ax=ax, orientation=orientation, **colorbar_kwargs)
    if extras:
        ticks = extras.get("tick_locs")
        labels = extras.get("tick_labels")
        if ticks is not None:
            cbar.set_ticks(ticks)
        if labels is not None:
            cbar.set_ticklabels(labels)
    if label:
        cbar.set_label(label)
    return cbar
