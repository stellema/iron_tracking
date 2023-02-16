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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cfg import paths, zones, bgc_vars


def plx_particle_dataset(exp, **kwargs):
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
    ds = xr.open_dataset(str(exp.file_plx), **kwargs)
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
    for y in range(times[0].year, times[1].year + 1):
        for m in range(times[0].month, times[1].month + 1):
            files.append('ocean_{}_{}_{:02d}.nc'.format(var, y, m))

    # Check for known missing files and replace with sub files.
    for var, file in zip(['zoo', 'det'], ['ocean_det_2008_01.nc', 'ocean_zoo_2008_03.nc']):
        if file in files:
            i = np.argwhere(np.array(files) == file)[0][0]
            files[i] = file[:-3] + '_sub.nc'

    files = [str(paths.ofam / f) for f in files]
    return files


def rename_ofam3_coords(ds):
    # OFAM3 coords dataset.
    mesh = xr.open_dataset(paths.data / 'ofam_mesh_grid_part.nc')
    dims_name_map = {'lon': ['xt_ocean', 'xu_ocean'],
                     'lat': ['yt_ocean', 'yu_ocean'],
                     'depth': ['st_ocean', 'sw_ocean']}
    rename_map = {'Time': 'time'}
    for k, v in dims_name_map.items():
        c = [i for i in v if i in ds.dims][0]
        rename_map[c] = k
        ds.coords[c] = mesh[v[0]].values
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
                       decode_cf=None, decode_times=False)

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
    return ds


class BGCFields(object):
    """Biogeochemistry field datasets from OFAM3 and ."""

    def __init__(self, exp):
        """Initialise & format felx particle dataset."""
        # self.filter_particles_by_start_time()
        # Select vairables
        self.exp = exp
        self.vars_ofam = ['phy', 'zoo', 'det', 'temp', 'fe', 'no3']
        self.vars_clim = ['kd']

        if self.exp.test:
            self.vars_ofam = ['phy', 'zoo', 'det', 'u', 'v', 'w']

        self.variables = np.concatenate((self.vars_ofam, self.vars_clim))

        # Initialise ofam3 dataset.
        self.ofam = ofam3_datasets(exp, variables=self.vars_ofam[0])
        for var in self.vars_ofam[1:]:
            self.ofam[var] = ofam3_datasets(exp, variables=var)[var]

    def kd490_dataset(self):
        """Get Kd490 climatology field."""
        kd = xr.open_dataset(paths.obs / 'GMIS_Kd490/GMIS_S_Kd490_month.nc', chunks='auto')
        kd = kd.assign_coords(lon=kd.lon % 360)
        kd = kd.rename(dict(month='time'))
        kd = kd.drop_duplicates('lon')

        # Subset lat and lons to match ofam3 data.
        kd = kd.sel(lat=slice(self.ofam.lat.min(), self.ofam.lat.max()),
                    lon=slice(self.ofam.lon.min(), self.ofam.lon.max()))
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
    # Open kd490 dataset.
    files = [paths.obs / 'GMIS_Kd490/GMIS_S_K490_{:02d}.nc'.format(m) for m in range(1, 13)]
    df = xr.open_mfdataset(files, combine='nested', concat_dim='time', decode_cf=0)

    # Create New Dataset.
    # Format longitudes.
    df = df.assign_coords(lon=df.lon % 360)
    df = df.sortby(df.lon)

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

    ds = xr.Dataset({'Kd490': (['time', 'lat', 'lon'],
                               df['Kd490'].values, df['Kd490'].attrs)},
                    coords=coords, attrs=attrs)

    ds.time.encoding = {'units': 'seconds since 1981-01-01 00:00:00'}

    # Save dataset.
    ds.to_netcdf(paths.obs / 'GMIS_Kd490/GMIS_S_Kd490_{}.nc'.format(time_dim), format='NETCDF4',
                 encoding={'Kd490': {'shuffle': True, 'chunksizes': [1, 3503, 9577],
                                     'zlib': True, 'complevel': 5}})

    # # Interpolate lat and lon to OFAM3
    # # Open OFAM3 mesh coordinates.
    # mesh = xr.open_dataset(paths.data / 'ofam_mesh_grid_part.nc')

    # Plot
    # ds.Kd490.plot(col='time', col_wrap=4)
    # plt.tight_layout()
    # plt.savefig(paths.figs / 'particle_source_map.png', bbox_inches='tight',
    #             dpi=300)
    # files = paths.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc'
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

    if type(ds) is xr.Dataset:
        for var in ds:
            ds[var].encoding.update(comp)
    if type(ds) is xr.DataArray:
        ds.encoding.update(comp)

    ds.to_netcdf(filename, compute=True)


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
