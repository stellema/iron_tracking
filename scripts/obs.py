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

import cfg


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

    for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'], ['°', '°', 'm']):
        ds = ds.sortby(ds[v])
        ds[v].attrs['long_name'] = n
        ds[v].attrs['standard_name'] = n.lower()
        ds[v].attrs['units'] = u
    return ds.dFe_RF


def Tagliabue_iron_dataset():
    """Open Tagliabue 2015 iron [nM] dataset."""
    file = cfg.obs / 'tagliabue_fe_database_jun2015_public_formatted.csv'
    ds = pd.read_csv(file)
    ds = pd.DataFrame(ds)
    ds = ds.to_xarray()

    ds = ds.rename({'lon': 'x', 'lat': 'y', 'depth': 'z'})
    # Convert longituge to degrees east (0-360).
    ds['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.where(ds.month > 0, drop=True)  # Remove single -99 month.
    return ds


def Tagliabue_iron_dataset_4D(lats=[-30, 30], lons=[120, 290]):
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


def GEOTRACES_iron_dataset(var='var88'):
    """Open GEOTRACES dataset."""
    # @todo: drop bad data np.unique(ds.var5)
    # @todo: bin model depths?
    file = 'GEOTRACES/idp2021/seawater/GEOTRACES_IDP2021_Seawater_Discrete_Sample_Data_v1.nc'
    ds = xr.open_dataset(cfg.obs / file)

    ds = ds.rename({'date_time': 'time', 'var2': 'depth', 'metavar1': 'cruise',
                    'var5': 'bottle_flag'})
    ds['time'] = ds.time.astype(dtype='datetime64[D]')  # Remove H,M,S values from time.

    # Drop extra data variables.
    keep_vars = ['time', 'depth', 'latitude', 'longitude', var]
    ds = ds.drop([v for v in ds.data_vars if v not in keep_vars])
    return ds


def GEOTRACES_iron_dataset_4D(var='var88', lats=[-10, 10], lons=[130, 285]):
    """Open geotraces data, subset location and convert to multi-dim dataset."""
    # @todo: drop bad data np.unique(ds.var5)
    # @todo: bin model depths?
    ds = GEOTRACES_iron_dataset(var)

    # Subset lat/lons.
    ds = ds.where((ds.latitude >= lats[0]) & (ds.latitude <= lats[1]) &
                  (ds.longitude >= lons[0]) & (ds.longitude <= lons[1]), drop=True)

    # Drop coords where var is all NaN.
    ds = ds.dropna('N_SAMPLES', how='all')
    ds = ds.sel(N_SAMPLES=ds[var].dropna('N_SAMPLES', 'all').N_SAMPLES)

    # Convert to multi-dim array.
    ds = ds.stack({'index': ('N_STATIONS', 'N_SAMPLES')})
    ds = ds.drop_vars(['N_SAMPLES', 'N_STATIONS'])
    index = pd.MultiIndex.from_arrays([ds.time.values, ds.depth.values,
                                       ds.latitude.values, ds.longitude.values],
                                      names=['t', 'z', 'y', 'x'],)
    ds = ds.assign_coords({'index': index})
    ds = ds.dropna('index', how='all')
    ds = ds.unstack('index')
    ds = ds.drop(['time', 'depth', 'latitude', 'longitude'])

    # Add data var attrs.
    for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'],
                       ['°', '°', 'm']):
        # ds = ds.sortby(ds[v])  # slow - only needed if new coordinates are out of order.
        ds = ds.dropna(v, 'all')
        ds[v].attrs['long_name'] = n
        ds[v].attrs['standard_name'] = n.lower()
        ds[v].attrs['units'] = u
    return ds


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
    # Vitiaz Strait
    da1 = dx.sel(y=-5.9, x=147.7, method='nearest').dropna('z', 'all')
    # Solomon Strait
    da2 = dx.sel(y=-5.139, x=153.3, method='nearest').dropna('z', 'all')
    # Solomon Sea
    da3 = dx.sel(y=-3.998, x=155.6, method='nearest').dropna('z', 'all')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    color = ['r', 'darkviolet', 'b']
    name = ['Vitiaz Strait', 'Solomon Strait', 'New Ireland']
    for i, dxx in enumerate([da1, da2, da3]):
        ls = '--' if i in [3] else '-'
        ax.plot(dxx.squeeze(), dxx.z, c=color[i], label=name[i], ls=ls)

    ax.set_title('GEOTRACES concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_straits.png')
    plt.show()

def plot_iron_obs_Huang():
    """Huang et al. (2022)
    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    dx = Huang_iron_dataset()
    dx = dx.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000)).isel(t=-1)

    # Plot: all x-y locations.
    dxx = dx.sel(z=slice(0, 400)).mean('z')
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=[120, 290, -30, 30])
    cs = dxx.plot(ax=ax, zorder=5, transform=proj)
    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022) [mean 0-370m]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/iron_obs_map_Huang.png')
    plt.show()

    # Plot: equatorial depth profile.
    fig, ax = plt.subplots(figsize=(10, 7))
    dxx = dx.sel(y=0, x=slice(132, 279), z=slice(0, 400))
    cs = dxx.plot(yincrease=False)
    ax.set_title('Equatorial dissolved Fe ML compilation from Huang et al. (2022)')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/ML_obs_dFe_eq.png')
    plt.show()

    # Plot: straits depth profile.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    dxx = dx.sel(y=-7, x=149, method='nearest')  # VS
    ax.plot(dxx.squeeze(), dxx.z, color='k', label='Vitiaz Strait')
    dxx = dx.sel(y=-5, x=155, method='nearest')  # SS
    ax.plot(dxx.squeeze(), dxx.z, color='r', label='Solomon Strait')
    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022)')
    ax.set_xlabel('dFe [nmol/kg]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(600, 0)
    ax.set_xlim(0.1, 0.9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/ML_obs_dFe_straits.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.

    # Vitiaz Strait (not available)
    da1 = dx.sel(y=-6, x=147.6, method='nearest').dropna('z', 'all')
    # PNG (mean)
    da2 = dx.sel(y=slice(-4.5, -3), x=slice(140, 145)).mean('y').mean('x').dropna('z', 'all')
    # Solomon Strait
    da3 = dx.sel(y=-5, x=155, method='nearest').dropna('z', 'all')
    # New Ireland
    da4 = dx.sel(y=-3, x=156, method='nearest').dropna('z', 'all')
    # MC
    da5 = dx.sel(y=7, x=132, method='nearest').dropna('z', 'all')
    # MC
    da6 = dx.sel(y=5.2, x=127, method='nearest').dropna('z', 'all')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    color = ['r', 'deeppink', 'darkviolet', 'b', 'lime', 'g']
    name = ['Vitiaz Strait', 'PNG', 'Solomon Strait', 'New Ireland', 'MC(7N)', 'MC(5N)']
    for i, dxx in enumerate([da1, da2, da3, da4, da5, da6]):
        ls = '--' if i in [3] else '-'
        ax.plot(dxx.squeeze(), dxx.z, c=color[i], label=name[i], ls=ls)

    ax.set_title('Huang concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/Huang_dFe_straits.png')
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

    # Vitiaz Strait
    da1 = dx.sel(y=-6, x=147.6, method='nearest').dropna('z', 'all')
    # PNG (mean)
    da2 = dx.sel(y=slice(-4.5, -3), x=slice(140, 145)).mean('y').mean('x').dropna('z', 'all')
    # Solomon Strait
    da3 = dx.sel(y=-5, x=155, method='nearest').dropna('z', 'all')
    # New Ireland
    da4 = dx.sel(y=-3, x=152, method='nearest').dropna('z', 'all')
    # MC
    da5 = dx.sel(y=7, x=132, method='nearest').dropna('z', 'all')
    # MC
    da6 = dx.sel(y=5.2, x=125, method='nearest').dropna('z', 'all')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    color = ['r', 'deeppink', 'darkviolet', 'b', 'lime', 'g']
    name = ['Vitiaz Strait', 'PNG', 'Solomon Strait', 'New Ireland', 'MC(7N)', 'MC(5N)']
    for i, dxx in enumerate([da1, da2, da3, da4, da5, da6]):
        ls = '--' if i in [3] else '-'
        ax.plot(dxx.squeeze(), dxx.z, c=color[i], label=name[i], ls=ls)

    ax.set_title('Tagliabue concentration of dissolved Fe')
    ax.set_xlabel('dFe [nM]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(500, 0)
    ax.set_xlim(0, 4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_straits.png')
    plt.show()


def plot_iron_obs_maps():
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    ds1 = GEOTRACES_iron_dataset().rename({'latitude': 'y', 'longitude': 'x'})
    ds2 = Tagliabue_iron_dataset()
    names = ['a) GEOTRACES', 'b) Tagliabue']

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

    for i, ds, c in zip(range(2), [ds1, ds2], ['red', 'blue']):
        ax.scatter(ds.x, ds.y, c=c, s=15, marker='*', alpha=0.7, transform=proj)

    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.

    ax.set_title('GEOTRACES (red) and Tagliabue (blue) iron observations', loc='left')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/iron_obs_map_pacific.png')
