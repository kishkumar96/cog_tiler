#!/usr/bin/env python3
"""
Proactive COG Pre-generation Script

This script pre-generates COG files for the Cook Islands wave forecast,
populating the cache before users make requests. This transforms the API
from an on-demand generator to a high-speed file server.

Run this script periodically (e.g., via a cron job) when new forecast
data is available.
"""

import sys
import time
from pathlib import Path

# Ensure the main application's modules can be imported
sys.path.append(str(Path(__file__).parent.resolve()))

from main import ensure_cog_from_params, CACHE_DIR, logger


def pregenerate_forecast_cogs():
    """
    Loops through common variables and all time steps to pre-generate COGs.
    """
    logger.info("🚀 Starting proactive COG pre-generation...")

    # --- Configuration ---
    # Define the variables and time range to pre-generate
    variables_to_generate = {
        'hs': {'vmin': 0.0, 'vmax': 4.0, 'colormap': 'jet', 'plot': 'contourf'},
        'tpeak': {'vmin': 2.0, 'vmax': 20.0, 'colormap': 'viridis', 'plot': 'contourf'},
        'dirp': {'vmin': 0.0, 'vmax': 360.0, 'colormap': 'hsv', 'plot': 'contourf'},
        'inundation_depth': {'vmin': 0.0, 'vmax': 5.0, 'colormap': 'Blues', 'plot': 'discrete_cmap'},
    }
    total_time_steps = 229  # As per forecast metadata

    ugrid_url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_UGRID.nc"
    gridded_url = "https://gemthreddshpc.spc.int/thredds/dodsC/POP/model/country/spc/forecast/hourly/COK/Rarotonga_inundation_depth.nc"

    total_cogs_to_generate = len(variables_to_generate) * total_time_steps
    logger.info(f"Target: {total_cogs_to_generate} COGs to check/generate.")

    generated_count = 0
    skipped_count = 0
    start_time = time.perf_counter()

    for time_index in range(total_time_steps):
        for var, config in variables_to_generate.items():
            is_ugrid = var != 'inundation_depth'
            params = {
                "layer_id": "cook_islands_ugrid" if is_ugrid else "cook_islands_rarotonga",
                "url": ugrid_url if is_ugrid else gridded_url,
                "variable": var,
                "time": str(time_index),
                "colormap": config['colormap'],
                "vmin": config['vmin'],
                "vmax": config['vmax'],
                "step": 0.1,
                "lon_min": -180, "lon_max": 180, "lat_min": -90, "lat_max": 90,
                "plot": config['plot'],
                "plot_options": {"antialiased": True} if is_ugrid else None
            }

            try:
                # This will either generate the COG or return the existing path
                cog_path = ensure_cog_from_params(**params)
                if "MISS" in logger.handlers[0].stream.getvalue().splitlines()[-1]: # A bit of a hack to check logs
                    generated_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to generate COG for {var} at t={time_index}: {e}")

    end_time = time.perf_counter()
    logger.info("✅ COG pre-generation complete.")
    logger.info(f"   - Generated: {generated_count} new COGs")
    logger.info(f"   - Skipped:   {skipped_count} existing COGs")
    logger.info(f"   - Duration:  {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    pregenerate_forecast_cogs()