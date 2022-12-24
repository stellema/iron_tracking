# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:


GEOTRACES variables.
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
date_time	Decimal Gregorian Days of the station	days since 2006-01-01 00:00:00 UTC
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
var90	Fe_S_CONC_BOTTLE	Conc. of operationally defined soluble Fe (colloids excluded)

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Nov  7 17:05:56 2022

"""
import math
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg


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


def GEOTRACES_iron_dataset(var='var88'):
    """Open GEOTRACES dataset."""
    # @todo: drop bad data np.unique(ds.var5)
    # @todo: bin model depths?
    file = 'GEOTRACES/idp2021/seawater/GEOTRACES_IDP2021_Seawater_Discrete_Sample_Data_v1.nc'
    ds = xr.open_dataset(cfg.obs / file)

    ds = ds.rename({'date_time': 'time', 'var2': 'depth', 'metavar1': 'cruise',
                    'var5': 'bottle_flag'})
    ds['time'] = ds.time.astype(dtype='datetime64[D]')  # Remove H,M,S values from time.
    ds['cruise'] = ds.cruise.astype(dtype=str)

    # Drop extra data variables.
    keep_vars = ['time', 'depth', 'latitude', 'longitude', 'cruise', var]
    ds = ds.drop([v for v in ds.data_vars if v not in keep_vars])
    return ds


def GEOTRACES_iron_dataset_4D(var='var88', lats=[-15, 15], lons=[120, 290]):
    """Open geotraces data, subset location and convert to multi-dim dataset."""
    # @todo: drop bad data np.unique(ds.var5)
    # @todo: bin model depths?
    ds = GEOTRACES_iron_dataset(var)

    # Subset lat/lons.
    ds = ds.where((ds.latitude >= lats[0]) & (ds.latitude <= lats[1]) &
                  (ds.longitude >= lons[0]) & (ds.longitude <= lons[1]), drop=True)

    # Drop coords where var is all NaN.
    # ds = ds.dropna('N_SAMPLES', how='all')
    # ds = ds.sel(N_SAMPLES=ds[var].dropna('N_SAMPLES', 'all').N_SAMPLES)

    # Convert to multi-dim array.
    ds = ds.stack({'index': ('N_STATIONS', 'N_SAMPLES')})
    ds = ds.sel(index=ds[var].dropna('index', 'all').index)

    # ds = ds.drop_vars(['N_SAMPLES', 'N_STATIONS'])
    index = pd.MultiIndex.from_arrays([ds.time.values, ds.depth.values,
                                       ds.latitude.values, ds.longitude.values],
                                      names=['t', 'z', 'y', 'x'],)
    ds = ds.drop_vars({'index', 'N_STATIONS', 'N_SAMPLES', 'longitude', 'latitude',
                       'time', 'depth'})
    ds = ds.assign_coords({'index': index})

    # ds = ds.dropna('index', how='all')
    ds = ds.unstack('index')
    # ds = ds.drop(['time', 'depth', 'latitude', 'longitude'])

    # Add data var attrs.
    for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'],
                       ['°', '°', 'm']):
        # ds = ds.sortby(ds[v])  # slow - only needed if new coordinates are out of order.
        # ds = ds.dropna(v, 'all')
        ds[v].attrs['long_name'] = n
        ds[v].attrs['standard_name'] = n.lower()
        ds[v].attrs['units'] = u
    return ds


def Tagliabue_iron_dataset():
    """Open Tagliabue 2015 iron [nM] dataset."""
    file = cfg.obs / 'tagliabue_fe_database_jun2015_public_formatted_ref.csv'
    ds = pd.read_csv(file)
    ds = pd.DataFrame(ds)
    ds = ds.to_xarray()

    ds = ds.rename({'lon': 'x', 'lat': 'y', 'depth': 'z'})
    # Convert longituge to degrees east (0-360).
    ds['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.where(ds.month > 0, drop=True)  # Remove single -99 month.
    return ds


def Tagliabue_iron_dataset_4D(lats=[-15, 15], lons=[120, 290]):
    """Open Tagliabue iron dataset, subset location and convert to multi-dim dataset."""
    # Subset to tropical ocean.
    ds = Tagliabue_iron_dataset()
    ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                  (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)

    # Convert to multi-dimensional array.
    t = pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(ds.year[i].item(), ds.month[i].item())
                        for i in range(ds.index.size)])
    ds['t'] = ('index', t)
    ds = ds.drop_vars(['month', 'year'])
    index = pd.MultiIndex.from_arrays([ds.t.values, ds.z.values, ds.y.values, ds.x.values],
                                      names=['t', 'z', 'y', "x"], )
    ds = ds.assign_coords({"index": index})
    ds = ds.dropna('index')
    ds = ds.unstack("index")
    return ds.dFe


def Huang_iron_dataset():
    """Huang et al. (2022)
    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    file = cfg.obs / 'Huang_et_al_2022_monthly_dFe_V2.nc'
    ds = xr.open_dataset(file)
    ds = ds.rename({'Longitude': 'x', 'Latitude': 'y', 'Depth': 'z', 'Month': 't'})
    ds.coords['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.sortby(ds.x)
    ds.t.attrs['units'] = 'Month'

    for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'], ['°', '°', 'm']):
        ds = ds.sortby(ds[v])
        ds[v].attrs['long_name'] = n
        ds[v].attrs['standard_name'] = n.lower()
        ds[v].attrs['units'] = u
    return ds.dFe_RF


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
    ds = GEOTRACES_iron_dataset_4D(var='var88', lats=[-10, 10], lons=[120, 285])
    dx = ds.var88#.where(~np.isnan(ds.var88), drop=True)

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
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_map.png')
    plt.show()

    # Plot West Pacific time/depth mean.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[135, 180, -10, 0])
    dxx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, vmin=0, transform=proj)
    plt.title('GEOTRACES iron observations (time/depth mean)')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_map_west_pacific.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.
    dx = ds.var88.mean('t', skipna=True)
    da1 = subset_obs_data(dx, -5.9, 147.7)  # Vitiaz Strait
    da2 = subset_obs_data(dx, -5.139, 153.3)  # Solomon Strait
    da3 = subset_obs_data(dx, -6.166, 152.5)  # Solomon Sea
    da4 = subset_obs_data(dx, -8.337, 151.3)  # Solomon Sea
    da5 = subset_obs_data(dx, -3.998, 155.6)  # New Ireland

    color = ['r', 'darkviolet', 'b', 'dodgerblue', 'deeppink']
    name = ['Vitiaz Strait', 'Solomon Strait', 'Solomon Sea 6S', 'Solomon Sea 8S',
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
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_straits.png')
    plt.show()


def plot_iron_obs_Tagliabue():
    """Plot Tagliabue data."""
    ds = Tagliabue_iron_dataset_4D(lats=[-10, 10], lons=[120, 285])

    # Subset: Equatorial Pacific.
    dx = ds.sel(y=0, method='nearest').sel(z=slice(0, 550))
    for v in ['z', 't', 'x']:
        dx = dx.dropna(v, 'all')

    # Plot all equatorial timesteps.
    dx.plot(col='t', col_wrap=3, yincrease=False, vmax=1, cmap=plt.cm.rainbow)
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_equator.png')
    plt.show()

    # Plot equatorial time mean.
    dx.mean('t').plot(figsize=(12, 6), yincrease=False, vmax=1, cmap=plt.cm.rainbow)
    plt.title('Tagliabue equatorial iron observations (time mean)')
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_equator_mean.png')
    plt.show()

    # Subset: EUC.
    dx = ds.sel(y=slice(-2.6, 2.6)).sel(z=slice(0, 550))
    for v in ['z', 't', 'x', 'y']:
        dx = dx.dropna(v, 'all')

    # Plot EUC time mean.
    dx.mean('t').mean('y').plot(figsize=(12, 6), yincrease=False, vmax=1, cmap=plt.cm.rainbow)
    plt.title('Tagliabue EUC(2.6S-2.6N) iron observations (time/lat mean)')
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_EUC_mean.png')
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
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_map.png')
    plt.show()

    # Plot West Pacific time/depth mean.
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 158, -8, 10])
    dx.plot(ax=ax, zorder=5, cmap=plt.cm.rainbow, vmax=1, transform=proj)
    plt.title('Tagliabue iron observations (time/depth mean)')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_map_west_pacific.png')
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
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_straits.png')
    plt.show()


def plot_iron_obs_Huang():
    """Huang et al. (2022)
    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    ds = Huang_iron_dataset()
    dx = ds.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000)).isel(t=-1)

    # Plot: all x-y locations.
    dxx = dx.sel(z=slice(0, 400)).mean('z')
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -30, 30])
    cs = dxx.plot(ax=ax, zorder=5, transform=proj, vmin=0, vmax=1, cmap=plt.cm.rainbow)
    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022) [mean 0-370m]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/iron_obs_map_Huang.png')
    plt.show()

    # Plot: equatorial depth profile.
    fig, ax = plt.subplots(figsize=(10, 7))
    dxx = dx.sel(y=0, x=slice(132, 279), z=slice(0, 400))
    cs = dxx.plot(yincrease=False, vmin=0, vmax=1, cmap=plt.cm.rainbow)
    ax.set_title('Equatorial dissolved Fe ML compilation from Huang et al. (2022)')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/ML_obs_dFe_eq.png')
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
    plt.savefig(cfg.fig / 'obs/ML_obs_dFe_straits.png')
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
    name = ['Vitiaz Strait', 'PNG', 'Solomon Strait', 'Solomon Sea', 'New Ireland', 'MC(7N)', 'MC(5N)']
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
    plt.savefig(cfg.fig / 'obs/Huang_dFe_straits.png')
    plt.show()

    # All straits (month heatmap).
    name = ['MC(7N)', 'Vitiaz Strait', 'PNG', 'Solomon Sea', 'Solomon Strait', 'New Ireland']

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
    plt.savefig(cfg.fig / 'obs/Huang_dFe_straits_monthly.png')
    plt.show()


def plot_iron_obs_maps():
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    ds1 = GEOTRACES_iron_dataset().rename({'latitude': 'y', 'longitude': 'x'})
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
    plt.savefig(cfg.fig / 'obs/iron_obs_map_global.png')

    # Tropical Pacific.
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -30, 30])

    for i, ds, c in zip(range(2), [ds1, ds2], ['blue', 'red']):
        ax.scatter(ds.x, ds.y, c=c, s=15, marker='*', alpha=0.7, transform=proj)

    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.

    ax.set_title('Iron observations from (blue) GEOTRACES IDP2021 and (red) Tagliabue et al. (2012) dataset [2015 Update]')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/iron_obs_map_pacific.png')


def plot_iron_obs_straits():
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    ds1 = GEOTRACES_iron_dataset_4D().var88
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
    dx2['png'] = subset_obs_data(ds2, slice(-4.5, -3), slice(140, 145), drop=drop).mean('y').mean('x')
    dx3['png'] = subset_obs_data(ds3, slice(-4.5, -3), slice(140, 145), drop=drop).mean('y').mean('x')

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
    name = ['MC(7N)', 'Vitiaz Strait', 'PNG', 'Solomon Sea', 'Solomon Strait', 'New Ireland']  # 'MC(5N)'
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
    plt.savefig(cfg.fig / 'obs/iron_obs_straits.png')
    plt.show()
