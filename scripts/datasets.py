# -*- coding: utf-8 -*-
"""Functions to open and format datasets.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jan 11 19:37:25 2023

"""
import sys
import math
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from functools import wraps
from calendar import monthrange
from datetime import datetime, timedelta

import cfg


def get_plx_filename(exp, lon, r, v=1):
    """Get plx particle filename.

    Args:
        exp (int): Scenario.
        lon (int): Particle release longitude.
        r (int): Particle file repeat index.
        v (int, optional): Particle file version. Defaults to 1.

    Returns:
        filename (pathlib.Path): plx dataset filename (path to plx repo).

    """
    filename = 'data/plx/plx_{}_{}_v1r{:02d}.nc'.format(cfg.exp_abr[exp], lon, r)
    filename = cfg.plx / filename
    return filename


def get_felx_filename(exp, lon, r, v=1, subfolder=None):
    """Get felx particle filename and path.

    e.g., repo/felx/data/subfolder/felx_hist_165_vXrYY.nc

    Args:
        exp (int): Scenario.
        lon (int): Particle release longitude.
        r (int): Particle file repeat index.
        v (int, optional): Particle file version. Defaults to 1.
        subfolder (str, optional): subfolder. Defaults to None.

    Returns:
        filename (pathlib.Path): felx dataset filename.

    """
    filename = 'felx_{}_{}_v{}r{:02d}.nc'.format(cfg.exp_abr[exp], lon, v, r)
    if subfolder is not None:
        filename = subfolder + filename

    filename = cfg.data / filename
    return filename


def plx_particle_dataset(exp, lon, r, **kwargs):
    """Get plx particle dataset.

    Args:
        exp (int): Scenario {0, 1}.
        lon (int): Particle release longitude {165, 190, 220, 250}.
        r (int): Particle file repeat index {0-9}.

    Returns:
        ds (xarray.Dataset): Dataset particle trajectories.

    Todo:
        - chunk dataset
        - Filter spinup particles
        - Filter by available field times

    """
    file = get_plx_filename(exp, lon, r)
    kwargs.setdefault('decode_cf', True)
    ds = xr.open_dataset(str(file), **kwargs)
    # ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})

    # Convert these to float32 dtype (others have same dtype).
    for var in ['obs', 'traj', 'trajectory']:
        ds[var] = ds[var].astype('float32')
    return ds


def get_datetime_bounds(exp, test_months=2):
    """Get list of experiment start and end year datetimes."""
    y = cfg.years[exp]
    times = [datetime(y[0], 1, 1), datetime(y[1], 12, 31)]

    if cfg.home.drive == 'C:':
        times[0] = datetime(y[1], 1, 1)
        times[1] = datetime(y[1], test_months, 29)
    return times


def get_ofam_filenames(var, times):
    """Create OFAM3 file list based on selected times."""
    f = []
    for y in range(times[0].year, times[1].year + 1):
        for m in range(times[0].month, times[1].month + 1):
            f.append(str(cfg.ofam / 'ocean_{}_{}_{:02d}.nc'.format(var, y, m)))
    return f


def ofam3_datasets(exp, variables=cfg.bgc_name_map):
    """ Get OFAM3 BGC fields.

    Args:
        exp (int): Scenario.

    Returns:
        ds (xarray.Dataset): Dataset of BGC fields.

    Todo:
        - Chunk dataset
        - Change dtype
        - add constants
        - Add Kd490

    """
    dims_name_map = {'lon': ['xt_ocean', 'xu_ocean'],
                     'lat': ['yt_ocean', 'yu_ocean'],
                     'depth': ['st_ocean', 'sw_ocean']}

    mesh = xr.open_dataset(cfg.data / 'ofam_mesh_grid_part.nc')  # OFAM3 coords dataset.

    def rename_ofam3_coords(ds):
        rename_map = {'Time': 'time'}
        for k, v in dims_name_map.items():
            c = [i for i in v if i in ds.dims][0]
            rename_map[c] = k
            ds.coords[c] = mesh[v[0]].values
        ds = ds.rename(rename_map)
        return ds

    cs = 300
    chunks = {'Time': 1,
              'sw_ocean': 1, 'st_ocean': 1, 'depth': 1,
              'yt_ocean': cs, 'yu_ocean': cs, 'lat': cs,
              'xt_ocean': cs, 'xu_ocean': cs, 'lon': cs}

    nmonths = 2 if any([i for i in variables.keys() if i in cfg.bgc_name_map.keys()]) else 12
    # Dictionary mapping variables to file(s).
    files = []
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp, nmonths))
        files.append(f)
    files = np.concatenate(files)

    open_kwargs = dict(chunks=chunks, preprocess=rename_ofam3_coords,
                       compat='override', combine='by_coords', coords='minimal',
                       data_vars='minimal', combine_attrs='override',
                       parallel=False, decode_cf=None, decode_times=False)
    ds = xr.open_mfdataset(files, **open_kwargs)

    return ds


def format_Kd490_dataset():
    """Merge & add time dimension to Kd490 monthly files."""
    # Open kd490 dataset.
    files = [cfg.obs / 'GMIS_Kd490/GMIS_S_K490_{:02d}.nc'.format(m) for m in range(1, 13)]
    df = xr.open_mfdataset(files, combine='nested', concat_dim='time', decode_cf=False)

    # Create New Dataset.
    # Time bounds.
    time = np.array([np.datetime64('1981-{:02d}-16'.format(i+1)) for i in range(12)],
                    dtype='datetime64[ns]')
    time_bnds = np.array([['1981-{:02d}-01T12'.format(m),
                           '1981-{:02d}-{:02d}T12'.format(m, monthrange(1981, m)[-1])]
                          for m in range(1, 13)], dtype='datetime64[ns]')

    time_attrs = {'bounds': 'time_bnds',
                  # 'units': 'seconds since 1981-01-01 00:00:00',
                  # 'calendar': '365_day',
                  'axis': 'T',
                  'long_name': 'time',
                  'standard_name': 'time'
                  }
    coords = {'lat': df.lat, 'lon': df.lon, 'time': (['time'], time, time_attrs),
              'time_bnds': (['time', 'nv'], time_bnds), 'nv': [1., 2.]}

    attrs = {'Conventions': 'CF-1.7',
             'title': df.attrs['Title'],
             'history': df.attrs['history'],
             'institution': 'Joint Research Centre (IES)',
             'source': 'https://data.europa.eu/data/datasets/dfe3524f-7d57-486c-ac92-cfa730887c48?locale=en',
             'comment': 'Monthly climatology sea surface diffuse attenuation coefficient at 490nm in m^-1 (log10 scalling) derived from the SeaWiFS sensor.',
             'references': 'Werdell, P.J. (2005). Ocean color K490 algorithm evaluation. http://oceancolor.gsfc.nasa.gov/REPROCESSING/SeaWiFS/R5.1/k490_update.html'
             }

    ds = xr.Dataset({'Kd490': (['time', 'lat', 'lon'], df['Kd490'].values, df['Kd490'].attrs)},
                    coords=coords, attrs=attrs)

    ds.time.encoding = {'units': 'seconds since 1981-01-01 00:00:00'}

    # Save dataset.
    ds.to_netcdf(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc', format='NETCDF4',
                 encoding={'Kd490': {'shuffle': True, 'chunksizes': [1, 3503, 9577],
                                     'zlib': True, 'complevel': 5}})

    # # Interpolate lat and lon to OFAM3
    # # Open OFAM3 mesh coordinates.
    # mesh = xr.open_dataset(cfg.data / 'ofam_mesh_grid_part.nc')

    # Plot
    # ds.Kd490.plot(col='time', col_wrap=4)
    # plt.tight_layout()
    # plt.savefig(cfg.fig / 'particle_source_map.png', bbox_inches='tight',
    #             dpi=300)
    # files = cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc'
    # ds = xr.open_dataset(files)


def append_dataset_history(ds, msg):
    """Append dataset history with timestamp and message."""
    if 'history' not in ds.attrs:
        ds.attrs['history'] = ''
    else:
        ds.attrs['history'] += ' '
    ds.attrs['history'] += str(np.datetime64('now', 's')).replace('T', ' ')
    ds.attrs['history'] += ' ' + msg
    return ds


def save_dataset(ds, filename, msg=None):
    """Save dataset with history, message and encoding."""
    ds = ds.chunk()
    if msg is not None:
        ds = append_dataset_history(ds, msg)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(filename, encoding=encoding, compute=True)
