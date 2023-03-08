# -*- coding: utf-8 -*-
"""Iron observational dataset and plot functions.

Opens and formats iron observational data from:
    - GEOTRACES IDP2021
    - Tagliabue et al. (2012) [Updated 2015]
    - Huang et al. (2022) data-driven climatology dataset (Machine Learning)

Notes:
    - Only works for pandas

Todo:
    - Replace/repeat dFe surface depths
    -

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Nov  7 17:05:56 2022

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import re
import xarray as xr

import cfg
from cfg import paths, mon
from tools import timeit, mlogger
from datasets import get_ofam_filenames, add_coord_attrs, save_dataset

logger = mlogger('misc')


def GEOTRACES_iron_dataset(var='var88'):
    """Open GEOTRACES idp2021 dataset.

    Todo:
        - Drop bad data np.unique(ds.var5)
        - Bin model depths?

    GEOTRACES Iron variables.
    metavar1	Cruise
    metavar2	Station
    metavar3	Type
    longitude	Longitude	degrees_east
    latitude	Latitude	degrees_north
    metavar4	Bot. Depth	m	Bottom depth
    metavar5	Operator's Cruise Name
    metavar6	Ship Name
    metavar7	Period
    metavar8	Chief Scientist
    metavar9	GEOTRACES Scientist
    metavar10	Cruise Aliases
    metavar11	Cruise Information Link
    metavar12	BODC Cruise Number
    date_time	Decimal Gregorian Days of the station days since 2006-01-01 UTC
    var1	CTDPRS_T_VALUE_SENSOR	dbar	Pressure from CTD sensor
    var2	DEPTH	m	Depth below sea surface calculated from pressure
    var3	Rosette Bottle Number		Bottle number on the rosette
    var4	GEOTRACES Sample ID		GEOTRACES sample id
    var5	Bottle Flag		Quality flag for the entire bottle
    var6	Cast Identifier		Cast identifier as string
    var7	Sampling Device		Sampling device descriptor
    var8	BODC Bottle Number		Unique BODC bottle number
    var9	BODC Event Number		Unique BODC event number
    var88	Fe_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved Fe
    var89	Fe_II_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved Fe(II)
    var90	Fe_S_CONC_BOTTLE	Conc. of operationally defined soluble Fe (no colloids)

    """
    f = 'GEOTRACES/idp2021/seawater/GEOTRACES_IDP2021_Seawater_Discrete_Sample_Data_v1.nc'
    ds = xr.open_dataset(paths.obs / f)

    ds = ds.rename({'date_time': 't', 'var2': 'z', 'metavar1': 'cruise',
                    'var5': 'bottle_flag', 'latitude': 'y', 'longitude': 'x'})
    ds['t'] = ds.t.astype(dtype='datetime64[D]')  # Remove H,M,S values from time.
    ds['cruise'] = ds.cruise.astype(dtype=str)

    # Drop extra data variables.
    keep_vars = ['t', 'z', 'y', 'x', 'cruise', var]
    ds = ds.drop([v for v in ds.data_vars if v not in keep_vars])

    ds = ds.where(~np.isnan(ds[var]), drop=True)

    # Round coords.
    for var in ['z', 'y', 'x']:
        ds[var] = np.around(ds[var], 1).astype(dtype=np.float32)

    ds = ds.stack({'index': ('N_STATIONS', 'N_SAMPLES')})
    ds = ds.sel(index=ds[var].dropna('index', 'all').index)
    ds = ds.drop_vars(['N_SAMPLES', 'N_STATIONS'])
    ds.coords['index'] = ds.index
    return ds


def GEOTRACES_iron_dataset_4D():
    """Open geotraces data, subset location and convert to multi-dim dataset."""
    file = paths.obs / 'GEOTRACES_IDP2021_fe_trop_pacific.nc'
    if file.exists():
        ds = xr.open_dataset(file)
        ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x'))
        return ds.dFe

    var = 'var88'
    ds = GEOTRACES_iron_dataset(var)
    ds = ds.drop('cruise')
    ds = ds.where(~np.isnan(ds[var]), drop=True)

    if 'var88' in ds.data_vars:
        ds = ds.rename(dict(var88='dFe'))

    lons = [110, 290]
    lats = [-30, 30]

    # Subset lat/lons.
    ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                  (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)

    # Convert to multi-dim array.
    coords = [ds.t.values, ds.z.values, ds.y.values, ds.x.values]
    index = pd.MultiIndex.from_arrays(coords, names=['t', 'z', 'y', 'x'],)
    ds = ds.drop_vars({'index', 'x', 'y', 't', 'z'})
    ds = ds.assign_coords({'index': index})
    ds = ds.unstack('index')

    # Add data var attrs.
    ds = ds.rename(dict(t='time', z='depth', y='lat', x='lon'))
    ds = add_coord_attrs(ds)
    save_dataset(ds, file, msg='Created multi-dimensional tropical Pacific subset.')
    return ds.dFe


def Tagliabue_iron_dataset():
    """Open Tagliabue 2015 iron [nM] dataset."""
    file = paths.obs / 'tagliabue_fe_database_jun2015_public_formatted_ref.csv'
    ds = pd.read_csv(file)
    ds = pd.DataFrame(ds)
    ds = ds.to_xarray()

    # Rename & format coords.
    ds = ds.rename({'lon': 'x', 'lat': 'y', 'depth': 'z'})
    # Convert longituge to degrees east (0-360).
    ds['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.where(ds.month > 0, drop=True)  # Remove single -99 month.

    ds = ds.where(~np.isnan(ds.dFe), drop=True)

    # Round coords.
    for var in ['z', 'y', 'x']:
        ds[var] = np.around(ds[var], 1).astype(dtype=np.float32)
    t = pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(ds.year[i].item(), ds.month[i].item())
                        for i in range(ds.index.size)])
    ds['t'] = ('index', t)

    # Convert full reference to in-text citation style.
    ds['ref'] = ('index', ds.Reference.values.copy())

    for i, r in enumerate(ds.Reference.values):
        ref = [s.replace(',', '').replace('.', '') for s in r.split(' ')]
        author = ref[0]
        year = re.findall('\d+', r)[0]  # NOQA

        if len(year) < 4:
            if author == 'Slemons':
                year = 2009
            if author == 'Obata':
                year = 2008

        if ref[3].isdigit():
            ds['ref'][i] = '{} ({})'.format(author, year)

        else:
            ds['ref'][i] = '{} et al. ({})'.format(author, year)
    return ds


def Tagliabue_iron_dataset_4D():
    """Open Tagliabue dataset, subset location and convert to multi-dim dataset."""
    file = paths.obs / 'tagliabue_fe_database_jun2015_trop_pacific.nc'
    if file.exists():
        ds = xr.open_dataset(file)
        ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x'))
        # ds = ds.sel(y=slice(*lats), x=slice(*lons))
        return ds.dFe

    # Subset to tropical ocean.
    ds = Tagliabue_iron_dataset()

    ds = ds.drop_vars(['month', 'year', 'ref', 'Reference'])
    lons = [110, 290]
    lats = [-30, 30]
    ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                  (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)

    # Convert to multi-dimensional array.
    coords = [ds.t.values, ds.z.values, ds.y.values, ds.x.values]
    index = pd.MultiIndex.from_arrays(coords, names=['t', 'z', 'y', 'x'], )
    ds = ds.drop_vars(['t', 'z', 'y', 'x'])
    ds = ds.assign_coords({'index': index})
    # ds = ds.dropna('index')
    ds = ds.unstack('index')

    ds = ds.rename(dict(t='time', z='depth', y='lat', x='lon'))
    ds = add_coord_attrs(ds)
    save_dataset(ds, file, msg='Created multi-dimensional tropical Pacific subset.')
    return ds.dFe


def iron_obs_dataset_information(name='GEOTRACES', lats=[-15, 15], lons=[110, 290]):
    """Open Tagliabue dataset and get obs information."""
    # Subset to tropical ocean.
    if name == 'GEOTRACES':
        ds = GEOTRACES_iron_dataset()
        ds = ds.rename({'cruise': 'ref'})
    elif name == 'Tagliabue':

        ds = Tagliabue_iron_dataset()
        ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                      (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)

        # Convert to multi-dimensional array.
        t = pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(ds.year[i].item(), ds.month[i].item())
                            for i in range(ds.index.size)])
        ds['t'] = ('index', t)
        # ds = ds.drop_vars(['month', 'year'])

    refs = np.unique(ds.ref)

    if name == 'Tagliabue':
        refs = refs[np.argsort([re.findall('\d+', r)[0] for r in refs])]  # NOQA

    for i, ref in enumerate(refs):
        dr = ds.where(ds.ref == ref, drop=True)

        x = '{:.1f}-{:.1f}°E (N={})'.format(dr.x.min(), dr.x.max(), np.unique(dr.x).size)
        y = '{:.1f}°-{:.1f}° (N={})'.format(dr.y.min(), dr.y.max(), np.unique(dr.y).size)
        z = '{:.0f}-{:.0f} m (N={})'.format(dr.z.min(), dr.z.max(), np.unique(dr.z).size)

        t = ', '.join(np.unique(dr.t.dt.strftime('%Y-%m')))
        print('{: >26}: N={: >3}, y={}, x={}, z={}, t={}'
              .format(ref, dr.index.size, y, x, z, t))

        # crd = [None, None, None]
        # for i, v, u, dp in zip(range(3), ['y', 'x', 'z'], ['°', '°E', ' m'], [1, 1, 0]):
        #     crd[i] = '{:.{dp}f}-{:.{dp}f}{} (N={})'.format(dr[v].min(), dr[v].max(), u,
        #                                                     np.unique(dr[v]).size, dp=dp)
        #     if np.unique(dr[v]).size <= 6:
        #         crd[i] = ', '.join(['{:.{dp}f}{}'.format(s, u, dp=dp) for s in np.unique(dr[v])])

        # t = ', '.join(np.unique(dr.t.dt.strftime('%Y-%m')))
        # print('{: >26}: N={: >3}, y={}, x={}, z={}, t={}'
        #       .format(ref, dr.index.size, *crd, t))

        # Print lat,lon pairs and corresponding depth ranges.
        crd = np.zeros((dr.index.size, 2))
        for j in range(dr.index.size):
            dj = dr.isel(index=j)
            crd[j] = [dj.y, dj.x]
        crd = np.unique(crd, axis=0)

        crds = crd[:, 0].astype(dtype=str)
        zz = crd[:, 0].astype(dtype=str)
        tt = crd[:, 0].astype(dtype=str)

        for j in range(crd.shape[0]):
            dj = dr.where((dr.y == crd[j, 0]) & (dr.x == crd[j, 1]), drop=True)
            d = 'S' if crd[j, 0] < 0 else 'N'
            crds[j] = ('{:.1f}°{}, {:.1f}°E'
                       .format(np.fabs(crd[j, 0]), d, crd[j, 1]))

            zz[j] = ('{:.0f}-{:.0f} m'.format(dj.z.min(), dj.z.max()))

            tt[j] = ', '.join(np.unique(dj.t.dt.strftime('%b/%Y')))

        if crds.size < 20:
            print('\n'.join(crds))
            print('\n'.join(zz))
            print('\n'.join(tt))
    return ds.dFe


def Huang_iron_dataset():
    """Huang et al. (2022).

    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    file = paths.obs / 'Huang_et_al_2022_monthly_dFe_V2.nc'
    ds = xr.open_dataset(file)
    ds = ds.rename({'Longitude': 'x', 'Latitude': 'y', 'Depth': 'z', 'Month': 't'})
    ds.coords['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.sortby(ds.x)
    ds.t.attrs['units'] = 'Month'

    for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'],
                       ['°', '°', 'm']):
        ds = ds.sortby(ds[v])
        ds[v].attrs['long_name'] = n
        ds[v].attrs['standard_name'] = n.lower()
        ds[v].attrs['units'] = u
    return ds.dFe_RF


def subset_obs_data(ds, y=None, x=None, z=None, drop=True):
    """Subset data array and drop NaNs along dimensions."""
    for dim, c in zip(['x', 'y', 'z'], [x, y, z]):
        if c is not None:
            if isinstance(c, (int, float)):
                kwargs = {dim: c, 'method': 'nearest'}
            else:
                kwargs = {dim: c}
            ds = ds.sel(**kwargs)

    if drop:
        for dim in [d for d in ds.dims if d != 't']:
            ds = ds.dropna(dim, 'all')
    ds = ds.squeeze()
    return ds


@timeit(my_logger=logger)
def interpolate_depths(ds, method='slinear'):
    """Interpolate observation depths to OFAM3 depths."""
    def valid_coord_indexes(dx, dim):
        """Dimension indexes with non NaN elements."""
        dim_valid = dx.dropna(dim, 'all')[dim]
        indexes = dx[dim].searchsorted(dim_valid)
        return indexes

    def interpolate_depth_array(dx, ds_interp, loc):
        """Interpolate along depths indexes with non NaN elements."""
        if dx.z.size >= 3:
            z_new_subset = ds_interp.z.copy().sel(z=slice(*[dx.z[a].item() for a in [0, -1]]))
            dx = dx.interp(z=z_new_subset, method=method)
        else:
            z_new_subset = ds_interp.z.copy().sel(z=dx.z.values, method='nearest')
            dx.coords['z'] = z_new_subset

        ds_interp[loc] = dx.broadcast_like(ds_interp[loc])
        return ds_interp

    # Drop any non-NaN dims (speeds up performance if data was subset).
    for v in ['t', 'z', 'y', 'x']:
        ds = ds.dropna(v, how='all')

    # Create all-NaN obs data array with OFAM3 depths.
    mesh = xr.open_dataset(str(paths.data / 'ofam_mesh_grid.nc'))
    ds_interp = (ds.copy() * np.nan).interp(z=mesh.st_edges_ocean.values)

    # loc = dict(t=6, y=37, x=197)  # test Tagliabue.

    # Interpolate depths at each point (only iterates through non-NaN dim elements).
    for t in range(ds.t.size):
        loc = dict(t=t)
        for y in valid_coord_indexes(ds.isel(loc), dim='y'):
            loc = dict(t=t, y=y)
            for x in valid_coord_indexes(ds.isel(loc), dim='x'):
                loc = dict(t=t, y=y, x=x)
                dx = ds.isel(loc).dropna('z')

                if dx.z.size > 0:
                    # Interpolate depths (unless all NaN).
                    ds_interp = interpolate_depth_array(dx, ds_interp, loc)
    return ds_interp


@timeit(my_logger=logger)
def combined_iron_obs_datasets(lats=[-15, 15], lons=[110, 290], add_Huang=False,
                               interpolate_z=True, pad_uniform_grid=False):
    """Create combined multi-dim iron observation datasets.

    Merged datasets:
        * GEOTRACES
        * Tagliabue et al. (2012)
        * [Optional] Huange et al. (2022)
        * [Optional] Evenly spaced (0.5 deg lat x 0.5 deg lon)

    Args:
        lats (list, optional): Latitude subset range. Defaults to [-15, 15].
        lons (list, optional): Longitude subset range. Defaults to [110, 290].
        pad_uniform_grid (bool, optional): Create evenly spaced NaN grid. Defaults to False.
        add_Huang (bool, optional): Merge data with the Huang et al. . Defaults to False.

    Returns:
        ds_merged (xarray.Dataset): Dataset with merged

    Notes:
        * Interpolates to OFAM3 depth grid.
        * Slice depth AFTER interpolating depths.
        * Interpolating depths takes around takes ~8 mins for Huang dataset
        * 00h:08m:38.18s: using pad_uniform_grid=False, add_Huang=True
        * Resulting grid won't be totally even (requires 0.1 deg spacing - too big).

    """
    def format_obs_dataset(ds, name):
        """Format (subset & interpolate) observation dataset."""
        # Subset to tropical ocean.
        ds = ds.sel(y=slice(*lats), x=slice(*lons))

        # Interpolate depths.
        if interpolate_z:
            ds = interpolate_depths(ds)

        # Drop/rename vars and convert to Datasets (must be after interpolate).
        ds = ds.to_dataset(name=name)

        # Drop any non-NaN dims (speeds up performance if data was subset).
        for v in ['t', 'z', 'y', 'x']:
            ds = ds.dropna(v, how='all')

        # Convert dtype to float32 (except for times).
        for var in list(ds.variables):
            if ds[var].dtype not in ['float32', 'datetime64[ns]']:
                ds[var] = ds[var].astype('float32')
        return ds

    name = 'fe'

    dss = []
    dss.append(GEOTRACES_iron_dataset_4D())
    dss.append(Tagliabue_iron_dataset_4D())

    if add_Huang:
        dss.append(Huang_iron_dataset())
        dss[-1] = dss[-1].isel(t=slice(12))
        dss[-1]['t'] = pd.to_datetime(['2022-{:02d}-01'.format(i + 1) for i in range(12)])

    dss = [format_obs_dataset(ds_, name) for ds_ in dss]

    ds_merged = xr.merge(dss)

    if pad_uniform_grid:
        # Evenly spaced lat and lon (single level time and depth arrays).
        y = np.arange(lats[0], lats[-1] + 0.5, 0.5, dtype=np.float32)
        x = np.arange(lons[0], lons[-1] + 0.5, 0.5, dtype=np.float32)
        z = np.array([dss[1].z.values[0]])
        t = np.array([dss[1].t.values[0]])

        coords = dict(t=('t', t), z=('z', z), y=('y', y), x=('x', x))
        temp = xr.DataArray(np.full((1, 1, y.size, x.size), np.nan),
                            dims=list(coords.keys()), coords=coords)
        temp = temp.to_dataset(name=name)

        ds_merged = xr.merge([ds_merged, temp])

    return ds_merged


def create_map_axis(fig, ax, extent=[120, 285, -10, 10]):
    """Create a figure and axis with cartopy."""
    proj = ccrs.PlateCarree()
    proj._threshold /= 20.

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent(extent, crs=proj)

    # Features.
    ax.add_feature(cfeature.LAND, color=cfeature.COLORS['land_alt1'])
    ax.add_feature(cfeature.COASTLINE)

    # Set ticks.
    xticks = np.arange(*extent[:2], 20)
    ax.set_xticks(xticks, crs=proj)
    ax.set_xticklabels(['{}°E'.format(i) for i in xticks])
    ax.set_yticks(ax.get_yticks(), crs=proj)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # # Minor ticks.
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.subplots_adjust(bottom=0.2, top=0.8)
    ax.set_aspect('auto')
    return fig, ax, proj


def plot_iron_obs_GEOTRACES():
    """Plot geotraces data."""
    ds = GEOTRACES_iron_dataset_4D()
    ds = ds.sel(y=slice(*[-10, 10]), x=slice(*[120, 285]))

    dx = ds.dFe  # .where(~np.isnan(ds.var88), drop=True)

    for v in ['x', 'y', 'z', 't']:
        dx = dx.dropna(v, 'all')

    dxx = dx.mean('t', skipna=True).mean('z', skipna=True)

    # dx = dx.sel(x=slice(147.7,  155), y=slice(-9.25, -3))
    # dx.mean('t').plot(col='z', col_wrap=5)
    # Plot: all x-y locations.
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[130, 290, -10, 10])
    cs = dxx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, vmin=0, transform=proj)
    ax.set_title('GEOTRACES mean concentration of dissolved Fe [nmol/kg]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/GEOTRACES_dFe_map.png')
    plt.show()

    # Plot West Pacific time/depth mean.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[135, 180, -10, 0])
    dxx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, vmin=0, transform=proj)
    plt.title('GEOTRACES iron observations (time/depth mean)')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/GEOTRACES_dFe_map_west_pacific.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.
    dx = ds.dFe.mean('t', skipna=True)
    da1 = subset_obs_data(dx, -5.9, 147.7)  # Vitiaz Strait
    da2 = subset_obs_data(dx, -5.139, 153.3)  # Solomon Strait
    da3 = subset_obs_data(dx, -6.166, 152.5)  # Solomon Sea
    da4 = subset_obs_data(dx, -8.337, 151.3)  # Solomon Sea
    da5 = subset_obs_data(dx, -3.998, 155.6)  # New Ireland

    color = ['r', 'darkviolet', 'b', 'dodgerblue', 'deeppink']
    name = ['Vitiaz Strait', 'Solomon Strait', 'Solomon Sea 6S-', 'Solomon Sea 8S',
            'New Ireland']

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i, dxx in enumerate([da1, da2, da3, da4, da5]):
        ax.plot(dxx, dxx.z, c=color[i], label=name[i], marker='o', ls='-',
                fillstyle='none')

    ax.set_title('GEOTRACES concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.axvline(x=0.6, c='k', lw=0.5)
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/GEOTRACES_dFe_straits.png')
    plt.show()


def plot_iron_obs_Tagliabue():
    """Plot Tagliabue data."""
    ds = Tagliabue_iron_dataset_4D()
    ds = ds.sel(y=slice(*[-10, 10]), x=slice(*[120, 285]))

    # Subset: Equatorial Pacific.
    dx = ds.sel(y=0, method='nearest').sel(z=slice(0, 550))
    for v in ['z', 't', 'x']:
        dx = dx.dropna(v, 'all')

    # Plot all equatorial timesteps.
    dx.plot(col='t', col_wrap=3, yincrease=False, vmax=1, cmap=plt.cm.rainbow)
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_equator.png')
    plt.show()

    # Plot equatorial time mean.
    dx.mean('t').plot(figsize=(12, 6), yincrease=False, vmax=1, cmap=plt.cm.rainbow)
    plt.title('Tagliabue equatorial iron observations (time mean)')
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_equator_mean.png')
    plt.show()

    # Subset: EUC.
    dx = ds.sel(y=slice(-2.6, 2.6)).sel(z=slice(0, 550))
    for v in ['z', 't', 'x', 'y']:
        dx = dx.dropna(v, 'all')

    # Plot EUC time mean.
    dx.mean('t').mean('y').plot(figsize=(12, 6), yincrease=False, vmax=1,
                                cmap=plt.cm.rainbow)
    plt.title('Tagliabue EUC(2.6S-2.6N) iron observations (time/lat mean)')
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_EUC_mean.png')
    plt.show()

    # Subset: Pacific.
    dx = ds.mean('z').mean('t')
    for v in ['y', 'x']:
        dx = dx.dropna(v, 'all')

    # Plot Pacific time/depth mean.
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -10, 10])
    dx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, transform=proj)
    plt.title('Tagliabue iron observations (time/depth mean)')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_map.png')
    plt.show()

    # Plot West Pacific time/depth mean.
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 158, -8, 10])
    dx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, transform=proj)
    plt.title('Tagliabue iron observations (time/depth mean)')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_map_west_pacific.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.
    dx = ds.mean('t')

    da1 = subset_obs_data(dx, -6, 147.6)  # Vitiaz Strait
    # PNG (mean)
    da2 = subset_obs_data(dx, slice(-4.5, -3), slice(140, 145)).mean('y').mean('x')
    da3 = subset_obs_data(dx, -5, 155)  # Solomon Strait
    da4 = subset_obs_data(dx, -3, 152)  # New Ireland
    da5 = subset_obs_data(dx, 7, 132)  # MC
    da6 = subset_obs_data(dx, 5.2, 125)  # MC

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    color = ['r', 'deeppink', 'darkviolet', 'b', 'lime', 'g']
    name = ['Vitiaz Strait', 'PNG', 'Solomon Strait', 'New Ireland', 'MC(7N)', 'MC(5N)']
    for i, dxx in enumerate([da1, da2, da3, da4, da5, da6]):
        ax.plot(dxx, dxx.z, c=color[i], label=name[i], marker='o', ls='-',
                fillstyle='none')

    ax.set_title('Tagliabue concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.axvline(x=0.6, c='k', lw=0.5)
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/Tagliabue_dFe_straits.png')
    plt.show()


def plot_iron_obs_Huang():
    """Huang et al. (2022).

    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    ds = Huang_iron_dataset()
    dx = ds.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000))

    # Plot: all x-y locations.
    dxx = dx.sel(z=slice(0, 400)).mean('z').isel(t=-1)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -30, 30])
    cs = dxx.plot(ax=ax, zorder=5, transform=proj, vmin=0, vmax=1, cmap=plt.cm.rainbow)
    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022) [mean 0-370m]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_map_Huang.png')
    plt.show()

    # Plot: equatorial depth profile.
    fig, ax = plt.subplots(figsize=(10, 7))
    dxx = dx.sel(y=0, x=slice(132, 279), z=slice(0, 400)).isel(t=-1)
    cs = dxx.plot(yincrease=False, vmin=0, vmax=1, cmap=plt.cm.rainbow)
    ax.set_title('Equatorial dissolved Fe ML compilation from Huang et al. (2022)')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/ML_obs_dFe_eq.png')
    plt.show()

    # Plot: equatorial depth profile (months).
    fig, ax = plt.subplots(figsize=(10, 7))
    dxx = dx.sel(y=0, x=slice(132, 279), z=slice(0, 400)).isel(t=slice(0, -1))
    dxx.coords['t'] = cfg.mon
    cs = dxx.plot(yincrease=False, vmin=0, vmax=1, cmap=plt.cm.rainbow, col='t',
                  col_wrap=4)
    ax.set_title('Equatorial dissolved Fe ML compilation from Huang et al. (2022)')
    plt.savefig(paths.figs / 'obs/ML_obs_dFe_eq_month.png')
    plt.show()

    # Plot: straits depth profile.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    dxx = subset_obs_data(dx, -7, 149)  # VS
    ax.plot(dxx, dxx.z, color='k', label='Vitiaz Strait')
    dxx = subset_obs_data(dx, -5, 155)  # SS
    ax.plot(dxx, dxx.z, color='r', label='Solomon Strait')

    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022)')
    ax.set_xlabel('dFe [nmol/kg]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(600, 0)
    ax.set_xlim(0.1, 0.9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/ML_obs_dFe_straits.png')
    plt.show()

    dx = ds.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000))
    # Plot: Vitiaz Strait & Solomon Strait.
    da1 = subset_obs_data(dx, -5.5, 147)  # Vitiaz Strait (not available)
    # PNG (mean)
    da2 = subset_obs_data(dx, slice(-4.5, -3), slice(140, 145)).mean('x')
    da3 = subset_obs_data(dx, -5, 155)  # Solomon Strait
    da4 = subset_obs_data(dx, -6.6, 152.5)  # Solomon Sea
    da5 = subset_obs_data(dx, -3, 156)  # New Ireland
    da6 = subset_obs_data(dx, 7, 132)  # MC
    da7 = subset_obs_data(dx, 5.2, 127)  # MC

    # All straits (annual mean).
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    color = ['r', 'deeppink', 'darkviolet', 'b', 'dodgerblue', 'lime', 'g']
    name = ['Vitiaz Strait', 'PNG', 'Solomon Strait', 'Solomon Sea',
            'New Ireland', 'MC(7N)', 'MC(5N)']
    for i, dxx in enumerate([da1, da2, da3, da4, da5, da6, da7]):
        ax.plot(dxx.isel(t=-1), dxx.z, c=color[i], label=name[i], marker='o', ls='-',
                fillstyle='none')

    ax.set_title('Huang concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.axvline(x=0.6, c='k', lw=0.5)
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/Huang_dFe_straits.png')
    plt.show()

    # All straits (month heatmap).
    name = ['MC(7N)', 'Vitiaz Strait', 'PNG', 'Solomon Sea', 'Solomon Strait',
            'New Ireland']

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    ax = ax.flatten()

    for i, dxx in enumerate([da6, da1, da2, da4, da3, da5]):
        cs = ax[i].pcolormesh(dxx.t, dxx.z, dxx.T, cmap=plt.cm.rainbow, vmin=0, vmax=1)

        ax[i].set_title('{} Huang dFe'.format(name[i]), loc='left')
        ax[i].set_ylabel('Depth [m]')

        ax[i].set_ylim(700, 0)
        ax[i].set_xlim(0.5, 12.5)
        ax[i].set_xticks(np.arange(1, 13))
        ax[i].set_xticklabels([t[:1] for t in cfg.mon])

    for i in [2, 5]:
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical', extend='max')
        cbar.set_label('dFe [nmol/kg]')

    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/Huang_dFe_straits_monthly.png')
    plt.show()


def plot_iron_obs_maps():
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    ds1 = GEOTRACES_iron_dataset()
    ds2 = Tagliabue_iron_dataset()
    names = ['a) GEOTRACES IDP2021', 'b) Tagliabue et al. (2012) Dataset [2015 Update]']

    # Global plot (seperate).
    fig = plt.figure(figsize=(12, 10))
    for i, ds in enumerate([ds1, ds2]):
        ax = fig.add_subplot(211 + i, projection=ccrs.PlateCarree(central_longitude=180))
        fig, ax, proj = create_map_axis(fig, ax, extent=[0, 360, -80, 80])
        ax.scatter(ds.x, ds.y, c='navy', s=9, marker='*', transform=proj)
        ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.
        ax.set_title('{} iron observations'.format(names[i]), loc='left')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_map_global.png')

    # Tropical Pacific.
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -30, 30])

    for i, ds, c in zip(range(2), [ds1, ds2], ['blue', 'red']):
        ax.scatter(ds.x, ds.y, c=c, s=15, marker='*', alpha=0.7, transform=proj)

    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.

    ax.set_title('Iron observations from (blue) GEOTRACES IDP2021 and \
                 (red) Tagliabue et al. (2012) dataset [2015 Update]')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_map_pacific.png')

    # Tropical Pacific (References/cruises).
    lats, lons = [-15, 15], [110, 290]
    colors = np.array(['navy', 'blue', 'orange', 'green', 'springgreen', 'red',
                       'purple', 'm', 'brown', 'pink', 'deeppink', 'cyan', 'y',
                       'darkviolet', 'k'])

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[110, 290, -15, 15])
    for i, ds, var, mrk in zip(range(2), [ds1, ds2], ['cruise', 'ref'], ['o', 'P']):

        ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                      (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)
        refs = np.unique(ds[var])
        for j, r in enumerate(refs):
            dx = ds.where(ds[var] == r, drop=True)
            ax.scatter(dx.x, dx.y, s=30, marker=mrk, label=r, c=colors[j], alpha=0.7,
                       transform=proj)

    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), markerscale=1.1, ncols=5)
    ax.set_title('Iron observations from GEOTRACES IDP2021 and ' +
                 'Tagliabue et al. (2012) dataset [2015 Update]')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_map_pacific_references.png', dpi=300)


def plot_iron_obs_straits():
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    ds1 = GEOTRACES_iron_dataset_4D()
    ds2 = Tagliabue_iron_dataset_4D()
    ds3 = Huang_iron_dataset()

    dx1, dx2, dx3 = xr.Dataset(), xr.Dataset(), xr.Dataset()
    drop = False
    # Vitiaz Strait
    dx1['vs'] = subset_obs_data(ds1, -5.9, 147.7, drop=drop)
    dx2['vs'] = subset_obs_data(ds2, -6, 147.6, drop=drop)
    dx3['vs'] = subset_obs_data(ds3, -5.5, 147, drop=drop)  # Alternatives available

    # PNG
    # dx1['png'] = ds1
    dx2['png'] = subset_obs_data(ds2, slice(-4.5, -3), slice(140, 145),
                                 drop=drop).mean('y').mean('x')
    dx3['png'] = subset_obs_data(ds3, slice(-4.5, -3), slice(140, 145),
                                 drop=drop).mean('y').mean('x')

    # Solomon Strait
    dx1['ss'] = subset_obs_data(ds1, -5.139, 153.3, drop=drop)
    dx2['ss'] = subset_obs_data(ds2, -5, 155, drop=drop)
    dx3['ss'] = subset_obs_data(ds3, -5, 155, drop=drop)

    # Solomon Sea
    dx1['ssea'] = subset_obs_data(ds1, -6.166, 152.5, drop=drop)
    # dx2['ssea'] = ds2
    dx3['ssea'] = subset_obs_data(ds3, -6.6, 152.5, drop=drop)

    # new Ireland
    dx1['ni'] = subset_obs_data(ds1, -3.998, 155.6, drop=drop)
    dx2['ni'] = subset_obs_data(ds2, -3, 152, drop=drop)
    dx3['ni'] = subset_obs_data(ds3, -3, 156, drop=drop)

    # MC 7N
    # dx1['mc7'] = ds1
    dx2['mc7'] = subset_obs_data(ds2, 7, 132, drop=drop)
    dx3['mc7'] = subset_obs_data(ds3, 7, 132, drop=drop)

    # MC 5N
    # dx1['mc'] = ds1
    dx2['mc5'] = subset_obs_data(ds2, 5.2, 125, drop=drop)
    dx3['mc5'] = subset_obs_data(ds3, 5.2, 127, drop=drop)

    dx1 = dx1.dropna('z', 'all').dropna('t', 'all')
    dx2 = dx2.dropna('z', 'all').dropna('t', 'all')
    dx3 = dx3.isel(t=-1)

    names = ['GEOTRACES', 'Tagliabue', 'Huang']
    name = ['MC(7N)', 'Vitiaz Strait', 'PNG', 'Solomon Sea', 'Solomon Strait', 'New Ireland']
    nvar = ['mc7', 'vs', 'png', 'ssea', 'ss',  'ni']
    # color = ['r', 'deeppink', 'darkviolet', 'b', 'dodgerblue', 'lime', 'g']

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    ax = ax.flatten()

    for i, v in enumerate(nvar):

        for m, ds, ls, c in zip(range(3), [dx1, dx2, dx3], ['d', '+', '*'], ['r', 'b', 'k']):
            label = names[m]
            if v in ds.data_vars:

                dv = ds[v].dropna('z', 'all')
                if 't' in dv.dims:
                    dv = dv.dropna('t', 'all')
                    label + ' ({})'.format(dv.t[0].dt.strftime("%Y-%m").item())
                    dv = dv.mean('t')

                ax[i].plot(dv, dv.z, c=c, label=label, marker='o', ls='-', fillstyle='none')

        ax[i].set_title(name[i], loc='left')

        ax[i].set_ylim(550, 0)
        # ax[i].set_xlim(0, 1.1)
        # ax[1].set_xlim(0, 2.7)
        ax[i].axvline(x=0.6, c='k', lw=0.5)  # Add reference line at 0.6nM.
        ax[i].legend()
        ax[i].set_xlabel('dFe [nM]')
        ax[i].set_ylabel('Depth [m]')

    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_straits.png')
    plt.show()


def plot_test_spline():
    """Plot interpolation method differences."""
    ds = Tagliabue_iron_dataset_4D()
    lats = [-10, 10]
    lons = [115, 220]
    ds = ds.sel(y=slice(*lats), x=slice(*lons))
    ds_sl = interpolate_depths(ds, method='slinear')
    ds_quad = interpolate_depths(ds, method='quadratic')
    ds_cubic = interpolate_depths(ds, method='cubic')

    x, y = 155, -5  # Solomon Strait
    f1 = ds.sel(y=y, x=x, method='nearest').dropna('t', 'all').isel(t=0)

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.scatter(f1, f1.z, c='k', s=100, zorder=10, label='obs')
    ax.set_title('Solomon Strait')
    for dx, n, c in zip([ds_cubic, ds_quad, ds_sl], ['cubic', 'quadratic', 'slinear'],
                        ['g', 'b', 'r']):
        f2 = dx.sel(y=y, x=x, method='nearest').dropna('t', 'all').isel(t=0)
        # ax.scatter(f2, f2.z, c=c, s=20, label=n)
        ax.plot(f2, f2.z, c=c, label=n)
        # ax.plot(f2, f2.z, c=c, marker='o', markerfacecolor='none', label=n)

    ax.invert_yaxis()
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('dFe [nmol/L]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/test_interp_solomon.png', dpi=250)


def get_iron_obs_locations(da, lat_slices, lon_slices):
    """Get iron observation lat & lon positions in observation data array.

    Args:
        da (xarray.DataArray): Iron obs dataset to search for observation locations.
        lat_slice (list of slice): Search region latitude slice(s).
        lon_slice (list of slice): Search region latitude slice(s).

    Returns:
        xx

    """
    dx = []
    for y, x in zip(lat_slices, lon_slices):
        dx.append(da.sel(y=slice(*y), x=slice(*x)))

    dx = xr.merge(dx).fe
    if 't' in dx.dims:
        dx = dx.mean('t')

    obs_x, obs_y = [], []
    for x in dx.x.values:
        for y in dx.y.values:
            dxx = dx.sel(x=x, y=y).dropna('z', 'all')
            if dxx.size > 0:
                obs_x.append(x)
                obs_y.append(y)

    coords = ['({:.1f}°E, {:.1f}°{})'.format(x, np.fabs(y), 'N' if y > 0 else 'S')
              for x, y in zip(obs_x, obs_y)]
    return dx, obs_x, obs_y, coords


def iron_obs_profiles():

    eps = 0.2
    loc = {}
    # loc[''] = dict(x=, y=)
    loc['mc'] = dict(y=[4, 9.2], x=[118, 135])
    loc['vs'] = dict(y=[-6, -3], x=[140, 150])
    loc['ss'] = dict(y=[-6, -3], x=[151, 156])
    loc['int_s'] = dict(y=[-10, -2.5], x=[170, 290])
    loc['int_n'] = dict(y=[2.5, 10], x=[138, 295])
    loc['mcx'] = dict(y=[7-eps, 7+eps], x=[130-eps, 130+eps])
    loc['vsx'] = dict(y=[-3.3-eps, -3.3+eps], x=[144-eps, 144+eps])
    loc['ssx'] = dict(y=[-5-eps, -5+eps], x=[155-eps, 155+eps])

    lats = [-15, 15]
    lons = [110, 290]

    ds = combined_iron_obs_datasets(lats, lons, add_Huang=False, interpolate_z=True)
    ds = ds.mean('t')

    # LLWBCs: X-Y location of obs.
    def sel_loc(da, loc, names):
        dx = []
        for n in names:
            dx.append(da.sel(y=slice(*loc[n]['y']), x=slice(*loc[n]['x'])))

        return xr.merge(dx).fe


    df = xr.Dataset()
    for n in loc.keys():
        df[n] = sel_loc(ds.fe, loc, [n]).mean(['x', 'y'])
    df['interior'] = sel_loc(ds.fe, loc, ['int_n', 'int_s']).mean(['x', 'y'])
    df['llwbcs'] = sel_loc(ds.fe, loc, ['mc', 'vs', 'ss']).mean(['x', 'y'])

def plot_combined_iron_obs_datasets_LLWBCs():
    """Plot combined iron observation datasets."""
    lats = [-15, 15]
    lons = [110, 290]

    ds = combined_iron_obs_datasets(lats, lons, add_Huang=False, interpolate_z=True)
    ds_all = combined_iron_obs_datasets(lats, lons, add_Huang=True, interpolate_z=True)
    ds_orig = combined_iron_obs_datasets(lats, lons, add_Huang=False, interpolate_z=False)
    ds = ds.mean('t')
    ds_all = ds_all.mean('t')
    ds_orig = ds_orig.mean('t')

    ##############################################################################################
    ##############################################################################################
    # LLWBCS
    colors = ['wheat', 'rosybrown', 'darkred', 'peru', 'red', 'darkorange', 'salmon', 'gold',
              'olive', 'lime', 'g', 'aqua', 'b', 'teal', 'dodgerblue', 'indigo', 'm', 'hotpink',
              'grey']

    # LLWBCs: X-Y location of obs.
    lat_slices, lon_slices = [[-6, -3], [4, 9.2]], [[140, 156], [118, 135]]
    dx, obs_x, obs_y, coords = get_iron_obs_locations(ds.fe, lat_slices, lon_slices)
    dx_all, obs_x, obs_y, coords = get_iron_obs_locations(ds_all.fe, lat_slices, lon_slices)

    ##############################################################################################
    # LLWBCs: orig vs interp depths (Tag & GEOTRACES)..
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[1].plot(dx.mean(['x', 'y']), dx.z, c='k', lw=4, label='mean', zorder=15)
    axes[1].plot(dx_all.mean(['x', 'y']), dx_all.z, c='k', lw=3, ls='--', label='mean (+Huang)',
                 zorder=15)

    for j, ax, ds_ in zip(range(2), axes, [ds_orig, ds]):

        for i, x, y in zip(range(len(obs_x[2:])), obs_x[2:], obs_y[2:]):
            dxx = ds_.fe.sel(x=x, y=y, method='nearest').dropna('z', 'all')
            label = coords[i] if j == 1 else None
            ax.plot(dxx, dxx.z, c=colors[i], label=label)
        ax.set_ylim(800, 0)

    axes[0].set_title('a) Southern LLWBCs dFe (Tagliabue & GEOTRACES)')
    axes[1].set_title('b) Southern LLWBCs dFe (interpolated)')
    fig.legend(ncols=6, loc='lower center', bbox_to_anchor=(0.5, -0.15))  # (x, y, width, height)
    for ax in axes:
        ax.set_xlabel('dFe [nmol/kg]')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dm"))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_combined_LLWBCs_interp.png', bbox_inches='tight')

    ##############################################################################################
    # LLWBCs: dFe and Map (Tag & GEOTRACES).
    fig = plt.figure(figsize=(12, 6))
    # Subplot 1: Map with obs location scatter.
    ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_title('a) LLWBC dFe observation locations', loc='left')
    ax.set_extent([117, 160, -7, 9], crs=ccrs.PlateCarree())
    ax.scatter(obs_x, obs_y, color=colors[:len(obs_x)], transform=ccrs.PlateCarree())
    ax.legend(ncols=2, loc='upper right', bbox_to_anchor=(1, 1))  # (x, y, width, height)
    ax.add_feature(cfeature.LAND, color=cfeature.COLORS['land_alt1'])
    ax.add_feature(cfeature.COASTLINE)
    xticks = np.arange(120, 165, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(['{}°E'.format(i) for i in xticks])
    ax.set_yticks(np.arange(-6, 10, 2), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_aspect('auto')
    # Subplot 2: dFe as a function of depth.
    ax = fig.add_subplot(122)
    ax.set_title('b) LLWBC dFe observations', loc='left')
    ax.plot(dx.mean(['x', 'y']), dx.z, c='k', lw=4, label='mean', zorder=15)
    ax.plot(dx_all.mean(['x', 'y']), dx_all.z, c='k', lw=3, ls='--', label='mean (+Huang)', zorder=15)

    for i, x, y in zip(range(len(obs_x)), obs_x, obs_y):
        dxx = dx.sel(x=x, y=y).dropna('z', 'all')
        ax.plot(dxx, dxx.z, c=colors[i], label=coords[i])
    ax.set_ylim(500, 0)
    fig.legend(ncols=6, loc='lower center', bbox_to_anchor=(0.5, -0.15))  # (x, y, width, height)
    ax.set_xlabel('dFe [nmol/kg]')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dm"))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_combined_LLWBCs.png', bbox_inches='tight')

    ##############################################################################################
    ##############################################################################################

    # Equator.
    dx = ds.fe.sel(y=slice(-2.5, 2.5)).mean('y')

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    dx.mean('t').plot(ax=ax, yincrease=False, vmin=0, vmax=1.5, cmap=plt.cm.rainbow)

    ax.set_title('Combined iron obs around the equator (time mean - no groupby)')
    ax.set_ylim(600, 0)
    ax.set_xlim(130, 280)
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_equator_combined.png')
    plt.show()

    # Equator.
    dx = ds.fe.groupby('t.month').mean('t').rename({'month': 't'})
    dx = dx.sel(y=slice(-2.5, 2.5)).mean('y')

    fig, ax = plt.subplots(4, 3, figsize=(14, 12))
    ax = ax.flatten()
    for i in range(12):
        cs = ax[i].pcolormesh(dx.x, dx.z, dx.isel(t=i), vmin=0, vmax=1.5, cmap=plt.cm.rainbow)
        ax[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%dm'))
        ax[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
        ax[i].set_xlim(130, 280)
        ax[i].set_title(cfg.mon[i])
        ax[i].invert_yaxis()

    fig.suptitle('Combined iron obs around the equator (time.month mean)', y=0.92)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.025])  # [left, bottom, width, height].
    cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
    fig.subplots_adjust(hspace=0.3, bottom=0.1)
    plt.savefig(paths.figs / 'obs/iron_obs_equator_combined_months.png', bbox_inches='tight', dpi=350)
    plt.show()

    ##############################################################################################
    ##############################################################################################
    # Interior: X-Y location of obs.

    lat_slices, lon_slices = [[-10, -2.5], [2.5, 10]], [[170, 290], [138, 295]]
    dx, obs_x, obs_y, coords = get_iron_obs_locations(ds.fe, lat_slices, lon_slices)
    dx_all, obs_x, obs_y, coords = get_iron_obs_locations(ds_all.fe, lat_slices, lon_slices)

    ##############################################################################################
    # Interior: dFe and Map (Tag & GEOTRACES).
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Map with obs location scatter.
    ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_title('a) Interior dFe observation locations', loc='left')
    ax.set_extent([135, 290, -11, 11], crs=ccrs.PlateCarree())
    ct = plt.rcParams['axes.prop_cycle'].by_key()['color']
    N = int(np.ceil(len(ct)*len(obs_x)/len(ct))/10) + 5
    cols = np.tile(ct, N)[:len(obs_x)]
    ax.scatter(obs_x, obs_y, color=cols, transform=ccrs.PlateCarree())
    # ax.legend(ncols=2, loc='upper right', bbox_to_anchor=(1, 1))  # (x, y, width, height)
    ax.add_feature(cfeature.LAND, color=cfeature.COLORS['land_alt1'])
    ax.add_feature(cfeature.COASTLINE)
    xticks = np.arange(130, 295, 20)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(['{}°E'.format(i) for i in xticks])
    ax.set_yticks(np.arange(-10, 12, 2), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_aspect('auto')

    # Subplot 2: dFe as a function of depth.
    ax = fig.add_subplot(122)
    ax.set_title('b) Interior dFe observations', loc='left')
    ax.plot(dx.mean(['x', 'y']), dx.z, c='k', lw=4, label='mean', zorder=15)
    ax.plot(dx_all.mean(['x', 'y']), dx_all.z, c='k', lw=3, ls='--',
            label='mean (+Huang)', zorder=15)

    lss = ['-', '--', ':', '-.', (0, (1, 10)), (0, (5, 10)), (0, (1, 1)), (0, (5, 1)),
           (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)),
           (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]

    for i, x, y in zip(range(len(obs_x)), obs_x, obs_y):
        dxx = dx.sel(x=x, y=y).dropna('z', 'all')
        ax.plot(dxx, dxx.z, color=cols[i], ls=lss[int(np.floor(i/10))], label=coords[i])

    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1.5)
    fig.legend(ncols=6, loc='upper center', bbox_to_anchor=(0.5, 0))  # (x, y, width, height)
    ax.set_xlabel('dFe [nmol/kg]')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dm"))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_combined_interior_no_euc.png', bbox_inches='tight')
