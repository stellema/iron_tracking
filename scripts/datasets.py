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


def get_plx_filename(exp, lon, r, v=1, folder='plx/'):
    """Get plx particle filename.

    Args:
        exp (int): Scenario.
        lon (int): Particle release longitude.
        r (int): Particle file repeat index.
        v (int, optional): Particle file version. Defaults to 1.

    Returns:
        filename (pathlib.Path): plx dataset filename (path to plx repo).

    """
    if r is not None:
        file = 'plx_{}_{}_v{}r{:02d}.nc'.format(cfg.exp_abr[exp], lon, v, r)
    else:
        file = 'plx_{}_{}_v{}.nc'.format(cfg.exp_abr[exp], lon, v)

    path = cfg.plx / 'data'
    if folder is not None:
        path = path / folder

    filename = path / file
    return filename


def get_felx_filename(exp, lon, r, v=1, folder=None):
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
    if folder is not None:
        filename = folder + filename

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
    # Dictionary mapping variables to file(s).
    if isinstance(variables, dict):
        variables = list(variables.values)

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

    nmonths = 2 if any([v for v in variables if v in ['fe', 'no3', 'temp']]) else 12

    files = []
    for var in variables:
        f = get_ofam_filenames(var, get_datetime_bounds(exp, nmonths))
        files.append(f)
    files = np.concatenate(files)

    open_kwargs = dict(chunks=chunks, preprocess=rename_ofam3_coords,
                       compat='override', combine='by_coords', coords='minimal',
                       data_vars='minimal', combine_attrs='override',
                       parallel=False, decode_cf=None, decode_times=False)
    ds = xr.open_mfdataset(files, **open_kwargs)

    return ds


def concat_exp_dimension(ds, add_diff=False):
    """Concatenate list of datasets along 'exp' dimension.

    Args:
        ds (list of xarray.Dataset): List of datasets.

    Returns:
        ds (xarray.Dataset): Concatenated dataset.

    """
    if add_diff:
        for var in ['time', 'rtime']:
            if var in ds[0].dims:
                if ds[0][var].size == ds[-1][var].size:
                    for i in range(len(ds)):
                        ds[i][var] = ds[0][var]

    if add_diff:
        # Calculate projected change (RCP-hist).
        ds = [*ds, ds[1] - ds[0]]

    ds = [ds[i].expand_dims(dict(exp=[i])) for i in range(len(ds))]
    ds = xr.concat(ds, 'exp')
    return ds


def combine_source_indexes(ds, z1, z2):
    """Merge the values of two sources in source dataset.

    Args:
        ds (xarray.Dataset): Particle source dataset.
        z1 (int): Source ID coordinate to merge (keep this index).
        z2 (int): Source ID coordinate to merge.

    Returns:
        ds_new (TYPE): Reduced Dataset.

    Notes:
        - Adds the values in 'zone' position z1 and z2.
        - IDs do not have to match positional indexes.

    """
    # Indexes of unchanged zones (concat to updated DataArray).
    z_keep = ds.zone.where((ds.zone != z1) & (ds.zone != z2), drop=True)

    # New dataset with zone z2 dropped (added zone goes to z1).
    ds_new = ds.sel(zone=ds.zone.where(ds.zone != z2, drop=True)).copy()

    # Add values in z2 and z2, then concat to ds_new.
    for var in ds.data_vars:
        dx = ds[var]
        if 'zone' in dx.dims:

            # Merge ragged arrays.
            if 'traj' in dx.dims:
                dx = dx.sel(zone=z1).combine_first(dx.sel(zone=z2, drop=1))

            # Sum arrays.
            elif 'rtime' in dx.dims:
                dx = dx.sel(zone=z1) + dx.sel(zone=z2, drop=1)

            # Sum elements.
            elif dx.ndim == 1:
                tmp = dx.sel(zone=z1).item() + dx.sel(zone=z2).item()
                dx = xr.DataArray(tmp, coords=dx.sel(zone=z1).coords)

            # Concat merged zone[z1] to zone[z_keep].
            ds_new[var] = xr.concat([dx, ds[var].sel(zone=z_keep)], dim='zone')

    return ds_new


def merge_interior_sources(ds):
    """Merge longitudes of source North/South Interior.

    Merge longitudes:
        - North interior: zone[6] = zone[6+7+8+9+10].
        - South interior lons: zone[8] = zone[12+13+14+15].
    Notes:
        - Modified to skip South interior (<165E).

    """
    nlons = [5, 5]  # Number of interior lons to merge [South, North].
    zi = [7, 12]   # Zone indexes to merge into [South, North].
    zf = [7, 8]  # New zone positions [South, North].

    # Iteratively combine next "source" index into first (z1).
    for i in range(2):  # [South, North].
        for a in range(1, nlons[i]):
            ds = combine_source_indexes(ds, zi[i], zi[i] + a)

    # Reset source name and colours.
    if set(['names', 'colors']).issubset(ds.data_vars):
        for i in zf:
            ds['names'][i] = cfg.zones.names[i]
            ds['colors'][i] = cfg.zones.colors[i]

    ds.coords['zone'] = np.arange(ds.zone.size, dtype=int)
    return ds


def format_Kd490_dataset():
    """Merge & add time dimension to Kd490 monthly files."""
    time_dim = ['time', 'month'][1]
    # Open kd490 dataset.
    files = [cfg.obs / 'GMIS_Kd490/GMIS_S_K490_{:02d}.nc'.format(m) for m in range(1, 13)]
    df = xr.open_mfdataset(files, combine='nested', concat_dim='time', decode_cf=0)

    # Create New Dataset.
    # Format longitudes.
    df = df.assign_coords(lon=df.lon % 360)
    df = df.sortby(df.lon)

    # Format time coord.
    # Keep as months.
    if time_dim == 'month':
        time = np.arange(1, 13, dtype=np.float32)
        time_bnds = np.array([[a, b] for a, b in zip(np.roll(time, 1), np.roll(time, -1))])
        time_attrs = {'axis': 'T', 'long_name': 'time', 'standard_name': 'time'}
    else:
        # Alt: Convert time to dates.
        y = 1981
        time = [np.datetime64('{}-{:02d}-16'.format(y, m)) for m in range(1, 13)]
        time = np.array(time, dtype='datetime64[ns]')

        time_bnds = [['{}-{:02d}-01T12'.format(y, m),
                      '{}-{:02d}-{:02d}T12'.format(y, m, monthrange(1981, m)[-1])]
                     for m in range(1, 13)]
        time_bnds = np.array(time_bnds, dtype='datetime64[ns]')

        time_attrs = {'bounds': 'time_bnds', 'axis': 'T',
                      # 'units': 'seconds since 1981-01-01 00:00:00',
                      # 'calendar': '365_day',
                      'long_name': 'time', 'standard_name': 'time'}

    coords = {'lat': df.lat, 'lon': df.lon, time_dim: ([time_dim], time, time_attrs),
              'time_bnds': ([time_dim, 'nv'], time_bnds), 'nv': [1., 2.]}

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
    ds.to_netcdf(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490_{}.nc'.format(time_dim), format='NETCDF4',
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
