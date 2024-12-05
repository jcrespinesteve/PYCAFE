#!/usr/bin/env python
# coding: utf-8
from pycafe_model_def import pycafe

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import datetime
import time

# Time range
ts = pd.date_range("2003-01-01", "2003-12-31", freq="MS")
outdir = "/Net/Groups/BGI/work_2/jcrespin/cafe_outputs"
indir = "/Net/Groups/BGI/work_2/jcrespin/cafe_inputs"

# Chunk size
lat_chunk_size = 40
lon_chunk_size = 80
chunk_size = {"lat": lat_chunk_size, "lon": lon_chunk_size}

def run_cafe(date0, outdir, indir):
    # Setup
    date1 = date0 + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    dayofyear = (date0 + (date1 - date0) / 2).dayofyear
    outfile = f"{outdir}/cafe_out_mld03_{date0.strftime('%Y%m')}.nc"
    
    # Load input data
    cafe_dir = f"{indir}/cafe_in_mld03_{date0.strftime('%Y%m')}.nc"
    cafeindata = xr.open_dataset(cafe_dir, chunks=chunk_size)
    lat = cafeindata.lats.values
    lon = cafeindata.lons.values
    
    lat_chunks = np.arange(0, len(lat), lat_chunk_size)
    lon_chunks = np.arange(0, len(lon), lon_chunk_size)
    
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    lat_grid = dask.array.from_array(lat_grid, chunks=chunk_size)
    lon_grid = dask.array.from_array(lon_grid, chunks=chunk_size)

    # Define processing function
    @dask.delayed
    def process_chunk(par, chl, mld, lat_grid, aph_443, adg_443, bbp_443, bbp_s, sst, dayofyear, lat_chunks, lon_chunks):
        return pycafe(par, chl, mld, lat_grid, aph_443, adg_443, bbp_443, bbp_s, sst, dayofyear, lat_chunks, lon_chunks)

    # Prepare chunk processing
    chunks = []
    for lat_start in lat_chunks:
        for lon_start in lon_chunks:
            lat_end = min(lat_start + lat_chunk_size, len(lat))
            lon_end = min(lon_start + lon_chunk_size, len(lon))

            chunk = process_chunk(
                cafeindata.PAR.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.chl.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.mld.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                lat_grid[lat_start:lat_end, lon_start:lon_end],
                cafeindata.aph_443.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.adg_443.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.bbp_443.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.bbp_s.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                cafeindata.sst.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)),
                dayofyear,
                lat[lat_start:lat_end],
                lon[lon_start:lon_end],
            )
            chunks.append(chunk)

    # Compute chunks
    print("Running PyCAFE...")
    with ProgressBar():
        results = dask.compute(*chunks)

    combined_results = xr.merge(results)
    
    # Save output
    print("Saving output...")
    combined_results.to_netcdf(outfile)
    print(f"Saved to {outfile}")

# Loop over time steps
for date in ts:
    run_cafe(date, outdir, indir)
