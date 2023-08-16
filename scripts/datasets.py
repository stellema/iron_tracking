# -*- coding: utf-8 -*-
"""Functions to open and format datasets.

Notes:

Example:

Todo:
    * plx_particle_dataset ** remove lon & r kwargs
    * delete get_datetime_bounds & replace with exp.time_bnds

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jan 11 19:37:25 2023

"""
import calendar
import datetime
import matplotlib.pyplot as plt  # NOQA
from memory_profiler import profile
from netCDF4 import num2date, date2num
import numpy as np
import pandas as pd
import xarray as xr

import cfg
from cfg import paths, zones, bgc_vars


def plx_particle_dataset(file, **kwargs):
    """Get plx particle dataset.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.
        kwargs (dict): xarray.open_dataset keywords.


    Returns:
        ds (xarray.Dataset): Dataset particle trajectories.

    Todo:
        - chunk dataset
        - Filter spinup particles
        - Filter by available field times

    """
    kwargs.setdefault('decode_cf', True)
    ds = xr.open_dataset(str(file), **kwargs)
    # ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})

    # Convert these to float32 dtype (others have same dtype).
    for var in ['obs', 'traj', 'trajectory']:
        ds[var] = ds[var].astype('float32')
    return ds


def get_ofam_filenames(var, times):
    """Create OFAM3 file list based on selected times.

    Args:
        var (str): Variable name.
        times (list of datetime.datetime): Time bounds for field files.

    Returns:
        f (list of str): List of OFAM3 var filenames between given times.

    """
    files = []
    for t in pd.date_range(times[0], times[1] + pd.offsets.MonthEnd(0), freq='1M'):
        files.append('ocean_{}_{}_{:02d}.nc'.format(var, t.year, t.month))

    # Check for known missing files and replace with sub files.
    for v, f in [['zoo', 'ocean_zoo_2008_03.nc'], ['det', 'ocean_det_2008_01.nc']]:
        if v == var and f in files:
            files = np.where(np.array(files) == f, f[:-3] + '_sub.nc', files)

    files = [str(paths.ofam / f) for f in files]
    return files


def get_ofam3_coords():
    """Get OFAM3 coords."""
    return xr.open_dataset(paths.data / 'ofam_mesh_grid.nc')


def rename_ofam3_coords(ds):
    """Rename OFAM3 coords."""
    mesh = xr.open_dataset(paths.data / 'ofam_mesh_grid_part.nc')
    dims_name_map = {'lon': ['xt_ocean', 'xu_ocean'],
                     'lat': ['yt_ocean', 'yu_ocean'],
                     'depth': ['st_ocean', 'sw_ocean']}
    rename_map = {'Time': 'time'}
    for k, v in dims_name_map.items():
        c = [i for i in v if i in ds.dims][0]
        rename_map[c] = k
        ds.coords[c] = mesh[v[0]].values[:ds[c].size]
    ds = ds.rename(rename_map)
    return ds


def ofam3_datasets(exp, variables=bgc_vars, chunks=None, **kwargs):
    """Open OFAM3 fields for variables.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.
        variables (list of str, optional): OFAM3 variables. Defaults to bgc_vars.
        **kwargs (dict): xarray.open_mfdataset keywords.

    Returns:
        ds (xarray.Dataset): Dataset of OFAM3 fields.

    Todo:
        - Chunk dataset
        - Change dtype
        - add constants
        - Add Kd490

    """
    # Dictionary mapping variables to file(s).

    if chunks not in ['auto', False]:
        cs = [13, 22, 137, 816]
        chunks = {'Time': cs[0],
                  'sw_ocean': cs[1], 'st_ocean': cs[1], 'depth': cs[1],
                  'yt_ocean': cs[2], 'yu_ocean': cs[2], 'lat': cs[2],
                  'xt_ocean': cs[3], 'xu_ocean': cs[3], 'lon': cs[3]}

    open_kwargs = dict(chunks=chunks, compat='override', coords='minimal',
                       # preprocess=rename_ofam3_coords, combine='by_coords',
                       combine='nested', concat_dim='Time',
                       data_vars='minimal', combine_attrs='override',
                       decode_cf=None, decode_times=True)

    for k, v in kwargs.items():
        open_kwargs[k] = v

    time_bnds = exp.time_bnds if not exp.test else exp.test_time_bnds

    # files = np.concatenate(files)
    if isinstance(variables, list):
        dss = []
        files = []
        for i, var in enumerate(variables):
            files.append(get_ofam_filenames(var, time_bnds))
            dss.append(xr.open_mfdataset(files[i], **open_kwargs))
        ds = xr.merge(dss)
    else:
        files = get_ofam_filenames(variables, time_bnds)
        ds = xr.open_mfdataset(files, **open_kwargs)

    ds = rename_ofam3_coords(ds)
    ds = ds.drop_duplicates('time')
    return ds


def ofam_clim(var_list):
    """Open OFAM3 historical and RCP8.5 climatologies of a variable."""
    def preprocess(ds):
        ds = rename_ofam3_coords(ds)
        for v in ['average_DT', 'Time_bnds']:
            if v in ds:
                ds = ds.drop(v)
        return ds
    if not isinstance(var_list, list):
        var_list = list(var_list)

    ds = [xr.open_mfdataset([paths.ofam / 'clim/ocean_{}_{}_climo.nc'.format(v, y) for v in var_list],
                            preprocess=preprocess) for y in ['1981-2012', '2070-2101']]
    ds[1].coords['time'] = ds[0]['time'].copy()

    ds = concat_exp_dimension(ds, add_diff=True)
    return ds


def ofam_bgc_clim():
    """Open OFAM3 historical and RCP8.5 climatologies of BGC variables (and kd490 fields)."""
    def preprocess(ds):
        ds = rename_ofam3_coords(ds)
        for v in ['average_DT', 'Time_bnds']:
            if v in ds:
                ds = ds.drop(v)
        return ds

    var_list = ['phy', 'zoo', 'det', 'fe', 'no3', 'temp']
    ds = [xr.open_mfdataset([paths.ofam / 'clim/ocean_{}_{}_climo.nc'.format(v, y) for v in var_list],
                            preprocess=preprocess) for y in ['2000-2012', '2089-2101']]
    ds[1].coords['time'] = ds[0]['time'].copy()

    ds = concat_exp_dimension(ds, add_diff=True)
    ds = ds.where(~np.isnan(ds.no3.isel(exp=0)))

    fieldset = BGCFields(cfg.ExpData(scenario=0))
    kd = fieldset.kd490_dataset(chunks='auto')
    kd = kd.sel(lon=ds.lon.values, lat=ds.lat.values, method='nearest')
    kd.coords['lon'] = ds.lon.values
    kd.coords['lat'] = ds.lat.values
    kd['time'] = ds.time.copy()

    ds['kd'] = 10**kd.kd
    return ds


class BGCFields(object):
    """Biogeochemistry field datasets from OFAM3 and Kd490."""

    def __init__(self, exp):
        """Initialise & format felx particle dataset."""
        # self.filter_particles_by_start_time()
        # Select vairables
        self.exp = exp
        self.vars_ofam = ['phy', 'zoo', 'det', 'temp', 'fe', 'no3']
        self.vars_clim = ['kd']
        self.dim_map_ofam = dict(time='time', depth='z', lat='lat', lon='lon')
        self.dim_map_kd = dict(time='month', lat='lat', lon='lon')
        self.variables = np.concatenate((self.vars_ofam, self.vars_clim))

    def ofam_dataset(self, variables, **kwargs):
        """Get OFAM3 field(s) dataset."""
        # Set list of OFAM3 variables to open.
        if not isinstance(variables, list):
            ofam = ofam3_datasets(self.exp, variables=variables, **kwargs)
        else:
            ofam = ofam3_datasets(self.exp, variables=variables[0], **kwargs)
            for var in variables[1:]:
                ofam[var] = ofam3_datasets(self.exp, variables=var, **kwargs)[var]
        return ofam

    def kd490_dataset(self, chunks=None):
        """Get Kd490 climatology field."""
        if chunks not in ['auto', False]:
            chunks = {'month': 12, 'lat': 480, 'lon': 1980}
        kd = xr.open_dataset(paths.obs / 'GMIS_Kd490/GMIS_S_Kd490_month_interp.nc',
                             chunks=chunks, decode_times=True)

        # kd.coords['time'] = kd.time.dt.month
        kd = kd.rename(dict(month='time'))

        # Subset lat and lons to match ofam3 data.
        # kd = kd.sel(lat=slice(-10, 10), lon=slice(120, 285))
        kd = kd.rename({'Kd490': 'kd'})
        return kd


def concat_scenario_dimension(ds, add_diff=False):
    """Concatenate list of datasets along 'era' dimension.

    Args:
        ds (list of xarray.Dataset): List of datasets.
        add_diff (bool, optional): Add rcp minus hist coordinate. Defaults to False.

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

    ds = [ds[i].expand_dims(dict(era=[i])) for i in range(len(ds))]
    ds = xr.concat(ds, 'era')
    return ds


def combine_source_indexes(ds, z1, z2):
    """Merge the values of two sources in source dataset.

    Args:
        ds (xarray.Dataset): Particle source dataset.
        z1 (int): Source ID coordinate to merge (keep this index).
        z2 (int): Source ID coordinate to merge.

    Returns:
        ds_new (xarray.Dataset): Reduced Dataset.

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

    Note:
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
            ds['names'][i] = zones.names[i]
            ds['colors'][i] = zones.colors[i]

    ds.coords['zone'] = np.arange(ds.zone.size, dtype=int)
    return ds


def format_Kd490_dataset():
    """Merge & add time dimension to Kd490 monthly files."""
    time_dim = ['time', 'month'][1]
    interp = True
    # Open kd490 dataset.
    files = [paths.obs / 'GMIS_Kd490/GMIS_S_K490_{:02d}.nc'.format(m) for m in range(1, 13)]
    df = xr.open_mfdataset(files, combine='nested', concat_dim='time', decode_cf=0)

    # Create New Dataset.
    # Format longitudes.
    df = df.assign_coords(lon=df.lon % 360)
    df = df.sortby(df.lon)
    df = df.drop_duplicates('lon')
    if interp:
        df = df.sel(lat=slice(-10, 10), lon=slice(120, 285))  # , keep_attrs=True
        dx = 0.1
        df = df.interp(lon=np.arange(120, 285 + dx, dx), lat=np.arange(-10, 10 + dx, dx))

    # Format time coord.
    # Keep as months.
    if time_dim == 'month':
        time = np.arange(1, 13, dtype=np.float32)
        time_bnds = np.array([[a, b] for a, b in zip(np.roll(time, 1),
                                                     np.roll(time, -1))])
        time_attrs = {'axis': 'T', 'long_name': 'time', 'standard_name': 'time'}
    else:
        # Alt: Convert time to dates.
        y = 1981
        time = [np.datetime64('{}-{:02d}-16'.format(y, m)) for m in range(1, 13)]
        time = np.array(time, dtype='datetime64[ns]')

        time_bnds = [['{}-{:02d}-01T12'.format(y, m),
                      '{}-{:02d}-{:02d}T12'.format(y, m, calendar.monthrange(1981, m)[-1])]
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
             'source': ('https://data.europa.eu/data/datasets/'
                        'dfe3524f-7d57-486c-ac92-cfa730887c48?locale=en'),
             'comment': ('Monthly climatology sea surface diffuse attenuation coefficient at '
                         '490nm in m^-1 (log10 scalling) derived from the SeaWiFS sensor.'),
             'references': ('Werdell, P.J. (2005). Ocean color K490 algorithm evaluation. http://'
                            'oceancolor.gsfc.nasa.gov/REPROCESSING/SeaWiFS/R5.1/k490_update.html')
             }

    ds = xr.Dataset({'Kd490': (['time', 'lat', 'lon'], df['Kd490'].values, df['Kd490'].attrs)},
                    coords=coords, attrs=attrs)

    ds.encoding['unlimited_dims'] = {'time'}
    ds.time.encoding['units'] = 'days since 1850-01-01'
    ds.time.encoding['calendar'] = 'proleptic_gregorian'

    # Save dataset.
    ds.to_netcdf(paths.obs / 'GMIS_Kd490/GMIS_S_Kd490_{}{}.nc'.format(time_dim, '_interp'
                                                                      if interp else ''),
                 format='NETCDF4', compute=True,
                 encoding={'Kd490': {'shuffle': False, 'chunksizes': [12, 41, 331],
                                     'zlib': True, 'complevel': 5}})


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

    if msg is not None:
        ds = append_dataset_history(ds, msg)

    comp = dict(zlib=True, complevel=5, contiguous=False)

    if type(ds) is xr.Dataset:
        for var in ds:
            ds[var].encoding.update(comp)
    if type(ds) is xr.DataArray:
        ds.encoding.update(comp)
    ds = ds.chunk()
    ds.to_netcdf(filename, compute=True)
    return ds


def concat_exp_dimension(ds, add_diff=False):
    """Concatenate list of datasets along 'exp' dimension.

    Args:
        ds (list of xarray.Dataset): List of datasets.

    Returns:
        ds (xarray.Dataset): Concatenated dataset.

    """

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


def add_coord_attrs(ds):
    """Add time, lat, lon and depth coordinates like OFAM3."""
    ds.time.attrs = {'cartesian_axis': 'T', 'long_name': 'time', 'standard_name': 'time'}
    ds.lon.attrs = {'standard_name': 'longitude', 'long_name': 'longitude',
                    'units': 'degrees_east', 'cartesian_axis': 'X'}
    ds.lat.attrs = {'standard_name': 'latitude', 'long_name': 'latitude',
                    'units': 'degrees_north', 'cartesian_axis': 'Y'}
    ds.depth.attrs = {'standard_name': 'depth', 'long_name': 'depth',
                      'positive': 'down', 'units': 'm', 'cartesian_axis': 'Z'}
    return ds


def sub_missing_files_ofam3():
    """Create stand-in OFAM3 fields missing months.

    Fill missing month with ~50/50 (1-15/16-end) previous and next available day.

    Notes:
        - det_2008_01
        - zoo_2008_03
        - adic_2008_02 (not needed)
        - dic_2008_04 (not needed)
    """
    def get_dataset_zoo():
        """Get datasets with nearest month values and empty dataset for missing month.

        Returns:
            dv (xarray.DataSet): Dataset of variable for last/next available month.
            dt (xarray.DataSet): Empty Dataset with missing file's time coordinates.
            t_inds (list): Index of last (first) day of the month in prev (next) in dv.

        """
        dt = xr.open_dataset(paths.ofam / 'ocean_det_2008_03.nc', decode_cf=True,
                             drop_variables='det')
        del dt.attrs['history']  # Delete history attr.

        files = [paths.ofam / 'ocean_zoo_2008_02.nc', paths.ofam / 'ocean_zoo_2008_04.nc']
        dv = xr.open_mfdataset(files, decode_cf=True)
        t_inds = [28, 29]
        return dv, dt, t_inds

    def get_dataset_det():
        """Get datasets with nearest month values and empty dataset for missing month.

        Returns:
            dv (xarray.DataSet): Dataset of variable for last/next available month.
            dt (xarray.DataSet): Empty Dataset with missing file's time coordinates.
            t_inds (list): Index of last (first) day of the month in prev (next) in dv.

        """
        dt = xr.open_dataset(paths.ofam / 'ocean_zoo_2008_01.nc', decode_cf=True,
                             drop_variables='zoo')
        del dt.attrs['history']  # Delete history attr.

        files = [paths.ofam / 'ocean_det_2007_12.nc', paths.ofam / 'ocean_det_2008_02.nc']
        dv = xr.open_mfdataset(files, decode_cf=True)
        t_inds = [30, 31]
        return dv, dt, t_inds

    def create_sub_dataset(var, dv, dt, t_inds):
        """Generate dataset with nearest month values..

        Args:
            var (str): Variable name.
            dv (xarray.DataSet): Dataset of variable for last/next available month.
            dt (xarray.DataSet): Empty Dataset with missing file's time coordinates.
            t_inds (list): Index of last (first) day of the month in prev (next) in dv.

        Returns:
            ds (xarray.DataSet): Nearest neighbour filled dataset.

        """
        # Create new monthly file using coords from different variable.
        ds = dt.copy()

        # Initialise DataArray using NaN filled time subset of prev/next mfdataset.
        ds[var] = dv[var].isel(Time=slice(dt.Time.size)) * np.nan

        # Set the first (last) 15 days using last (next) available day.
        t1, t2 = [dv[var].isel(Time=t, drop=True) for t in t_inds]

        for t in range(0, 15):
            ds[var][dict(Time=t)] = t1

        for t in range(15, ds.Time.size):
            ds[var][dict(Time=t)] = t2

        # ds.encoding['zlib'] = True
        # ds.encoding['complevel'] = 5
        # encoding = {var: comp for var in ds.data_vars}
        return ds

    var = 'zoo'
    dv, dt, t_inds = get_dataset_zoo()
    ds = create_sub_dataset(var, dv, dt, t_inds)
    filename = paths.ofam / 'ocean_zoo_2008_03_sub.nc'
    msg = 'Original file corrupted. Generated substitute using repeat of 2008-02-29 for ' \
        'the first 15 days and 2008-03-01 for the remaining days.'
    ds = append_dataset_history(ds, msg)
    ds.to_netcdf(filename, compute=True)
    ds.close()
    dt.close()

    var = 'det'
    dv, dt, t_inds = get_dataset_det()
    ds = create_sub_dataset(var, dv, dt, t_inds)
    filename = paths.ofam / 'ocean_det_2008_01_sub.nc'
    msg = 'Original file corrupted. Generated substitute using repeat of 2007-12-31 for ' \
        'the first 15 days and 2008-02-01 for the remaining days.'
    ds = append_dataset_history(ds, msg)
    ds.to_netcdf(filename, compute=True)
    ds.close()
    dt.close()
    return


# def fix_corrupted_files_ofam3():
#     """Create stand-in file for corrupted OFAM3 ocean_temp_2000_04.nc (missing last 6 days)."""
#     files = [paths.ofam / 'ocean_temp_2000_{:02d}.nc'.format(m) for m in [4, 5]]
#     file_new = paths.ofam / '{}_sub.nc'.format(files[0].stem)

#     ds1, ds2 = [xr.open_dataset(file, decode_cf=True, decode_times=True) for file in files]
#     ds = ds1.copy()

#     # Replace missing time values.
#     time = pd.date_range('2000-04-01T12', periods=30, freq='D')
#     # time = np.array([ds.Time[0].item() + i for i in range(30)])  # decode_times=False.
#     # time = np.arange('2000-04-01T12', '2000-05-01T12', np.timedelta64(1, 'D'),
#     #                  dtype='datetime64[ns]')
#     # time = np.array([cftime.DatetimeGregorian(2000, 4, i+1, 12, has_year_zero=False)
#     #                  for i in range(30)])  # import cftime & use_cftime=True.

#     ds = ds.assign_coords(Time=time)
#     ds['Time'].attrs = ds1.Time.attrs
#     ds['Time'].encoding = ds1.Time.encoding

#     # Fill in first (last) 3 days with previous (next) available day of data.
#     for i in range(3):
#         ds.temp[dict(Time=24+i)] = ds1.temp.isel(Time=23, drop=True)
#         ds.temp[dict(Time=24+3+i)] = ds2.temp.isel(Time=0, drop=True)

#     msg = 'Original file corrupted - missing last six days of data. Generated substitute' \
#         'using repeat of the last available day.'
#     ds = append_dataset_history(ds, msg)
#     ds.to_netcdf(file_new, compute=True, format='NETCDF4', engine='netcdf4')
#     ds.close()
#     ds1.close()
#     ds2.close()

#     # Test file is able to be opened without error.
#     df = xr.open_mfdataset([file_new, files[1]], chunks=False, decode_cf=True, decode_times=1)
#     df.close()


def convert_plx_times(dt, obs, attrs, attrs_new):
    """Convert particle time calendar (use with np.apply_along_axis).

    Args:
        dt (numpy.array): Particle time array.
        obs (numpy.array): DESCRIPTION.
        attrs (dict): DESCRIPTION.
        attrs_new (dict): DESCRIPTION.

    Returns:
        time (numpy.array): Converted times for particle.

    Example:
        attrs = dict(units=ds.time.units, calendar=ds.time.calendar)
        attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='GREGORIAN')
        times = np.apply_along_axis(convert_plx_times, 1, arr=ds.time,
                                    obs=ds.obs.astype(dtype=int),
                                    attrs=attrs, attrs_new=attrs_new)
        ds['time'] = (('traj', 'obs'), times)
    """
    time = dt.copy().astype(dtype=object)
    mask = ~np.isnan(dt)
    time[mask] = num2date(dt[mask], units=attrs['units'], calendar=attrs['calendar'])
    time[mask] = date2num(time[mask], units=attrs_new['units'], calendar=attrs_new['calendar'])
    return time


def ofam_cell_depth():
    """Retur cell depth of OFAM3 vertical coordinate."""
    dz = get_ofam3_coords()
    st_ocean = dz['st_ocean']  # Copy st_ocean coords
    dz = dz.st_edges_ocean.diff(dim='st_edges_ocean')
    dz = dz.rename({'st_edges_ocean': 'st_ocean'})
    dz.coords['st_ocean'] = st_ocean
    dz = dz.rename({'st_ocean': 'lev'})
    return dz


def llwbc_transport_projections():
    """Get the Eulerian transport projections of the LLWBCs (for fe_high forcing).

    Notes:
        - Depth integrated to 775m
        - Requires transport_LLWBCs_hist.nc and transport_LLWBCs_rcp.nc (see plx repo)

    Results:
        Solomon Strait (ssx)  = 21.28% (Solomon Strait and St George's Channel)
        Vitiaz Strait (vs)    = 15.03%
        Mindanao CUrrent (mc) = -8.42%

    """
    ds = [xr.open_dataset(paths.data / 'transport_LLWBCs_{}.nc'.format(i)) for i in ['hist', 'rcp']]

    ds[0] = ds[0].sel(time=slice('2000-01-01', '2012-12-31'))
    ds[1] = ds[1].sel(time=slice('2089-01-01', '2101-12-31'))

    ds = [dx.sel(lev=slice(0, 776)).sum('lev') for dx in ds]
    ds = [dx.groupby('time.month').mean('time').mean('month') for dx in ds]
    ds = concat_exp_dimension(ds, add_diff=True)
    print((ds.isel(exp=2) / ds.isel(exp=0)) * 100)


    # Get lon-integrated velocity from transport (divide by depth width)
    dz = ofam_cell_depth()
    ds = [xr.open_dataset(paths.data / 'transport_LLWBCs_{}.nc'.format(i)) for i in ['hist', 'rcp']]

    ds[0] = ds[0].sel(time=slice('2000-01-01', '2012-12-31'))
    ds[1] = ds[1].sel(time=slice('2089-01-01', '2101-12-31'))
    ds = [ds[i] / dz for i in [0, 1]]
    ds = [dx.groupby('time.month').mean('time').mean('month') for dx in ds]
    ds = concat_exp_dimension(ds, add_diff=True)
    ds['mc'] *= -1

    for i, v in enumerate(['mc', 'vs', 'ssx']):
        for s in [0, 1]:
            plt.plot(ds[v].isel(exp=s), ds.lev, c=['k', 'r', 'b'][i], ls=['-', '--'][s], label=v)
    plt.legend()
    plt.axhline(50)
    plt.axhline(100)
    plt.axhline(400)
    plt.ylim(800, 0)

    ds = xr.open_mfdataset([paths.ofam / 'clim/ocean_{}_1981-2012_climo.nc'.format(v) for v in ['u', 'v']])
    ds = rename_ofam3_coords(ds)
    ds = ds.sel(depth=slice(0, 750), lon=slice(120, 280)).mean('time')

    kwargs = dict(yincrease=0, vmax=0.05, vmin=-0.05, cmap=plt.cm.seismic)

    ds.v.sel(lat=-6, method='nearest').plot(**kwargs)
    ds.v.sel(lat=8, method='nearest').plot(**kwargs)
    ds.u.sel(lon=120, method='nearest').plot(**kwargs)


def convert_to_transport(ds, lat=None, var='v', sum_dims=['lon', 'lev']):
    """Sum OFAM3 meridional velocity multiplied by cell depth and width.

    Args:
        ds (xr.DataArray or xr.DataSet): Contains variable v & dims renamed.
        lat (float): Latitude when transport is calculated (arrays not tested).

    Returns:
        ds (xr.DataArray or xr.DataSet): Meridional transport in Sverdrups.

    """
    dz = ofam_cell_depth()
    dz = dz.sel(lev=ds.lev.values)
    if var == 'v':
        dxdy = cfg.LON_DEG(lat) * 0.1 / 1e6

    elif var == 'u':
        dxdy = cfg.LAT_DEG * 0.1 / 1e6

    if isinstance(ds, xr.DataArray):
        ds *= dxdy * dz
        if sum_dims:
            ds = ds.sum(sum_dims, skipna=True, keep_attrs=True)
    else:
        ds[var] = ds[var] * dxdy * dz
        if sum_dims:
            ds[var] = ds[var].sum(sum_dims, skipna=True, keep_attrs=True)

    return ds


def get_ofam_clim_circ():
    """Sum OFAM3 meridional velocity multiplied by cell depth and width.

    Args:
        ds (xr.DataArray or xr.DataSet): Contains variable v & dims renamed.
        lat (float): Latitude when transport is calculated (arrays not tested).

    Returns:
        ds (xr.DataArray or xr.DataSet): Meridional transport in Sverdrups.

    """
    files = [*[paths.data / 'transport_{}_hist.nc'.format(v) for v in ['EUC', 'LLWBCs']],
             *[paths.data / 'transport_{}_rcp.nc'.format(v) for v in ['EUC', 'LLWBCs']]]
    df = [xr.open_mfdataset(f) for f in files]

    for i in [0, 1]:
        df[i] = df[i].sel(time=slice('2000', '2012'))
        df[i + 2] = df[i + 2].sel(time=slice('2089', '2101'))

    for i in range(4):
        df[i] = df[i].groupby('time.month').mean('time').rename({'month': 'time'})
    for i in [1, 3]:
        df[i].coords['lon'] = df[0].lon

    dg = [xr.merge(df[:2]).expand_dims(dict(exp=[0])),
          xr.merge(df[2:]).expand_dims(dict(exp=[1]))]

    ds = xr.concat(dg, 'exp').mean('time')


    dz = ofam_cell_depth()
    dz = dz.sel(lev=ds.lev.values)
    ds = ds / dz

    return ds
