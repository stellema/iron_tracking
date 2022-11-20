# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:


metavar1	Cruise
metavar2	Station
metavar3	Type
longitude	Longitude	degrees_east
latitude	Latitude	degrees_north
date_time	Decimal Gregorian Days of the station	days since 2006-01-01 00:00:00 UTC	Relative Gregorian Days with decimal part

var1	CTDPRS_T_VALUE_SENSOR	dbar	Pressure from CTD sensor
var2	DEPTH	m	Depth below sea surface calculated from pressure
var3	Fe_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved Fe
var4	Fe_D_CONC_FISH	nmol/kg	Concentration of dissolved Fe
var5	Fe_D_CONC_BOAT_PUMP	nmol/kg	Concentration of dissolved Fe
var6	Fe_D_CONC_SUBICE_PUMP	nmol/kg	Concentration of dissolved Fe
var7	Fe_II_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved Fe(II)
var8	Fe_II_D_CONC_FISH	nmol/kg	Concentration of dissolved Fe(II)
var9	Fe_II_D_CONC_BOAT_PUMP	nmol/kg	Concentration of dissolved Fe(II)
var10	Fe_S_CONC_BOTTLE	nmol/kg	Concentration of operationally defined soluble Fe (colloids excluded)
var11	Fe_S_CONC_FISH	nmol/kg	Concentration of operationally defined soluble Fe (colloids excluded)
var1	CTDPRS_T_VALUE_SENSOR	dbar	Pressure from CTD sensor
var2	DEPTH	m	Depth below sea surface calculated from pressure

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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cfg


# # Seawater
"""
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
date_time	Decimal Gregorian Days of the station	days since 2006-01-01 00:00:00 UTC	Relative Gregorian Days with decimal part
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
var90	Fe_S_CONC_BOTTLE	nmol/kg	Concentration of operationally defined soluble Fe (colloids excluded)
var117	Fe_D_CONC_FISH	nmol/kg	Concentration of dissolved Fe
var118	Fe_II_D_CONC_FISH	nmol/kg	Concentration of dissolved Fe(II)
var119	Fe_S_CONC_FISH	nmol/kg	Concentration of operationally defined soluble Fe (colloids excluded)
var137	Fe_D_CONC_BOAT_PUMP	nmol/kg	Concentration of dissolved Fe
var138	Fe_II_D_CONC_BOAT_PUMP	nmol/kg	Concentration of dissolved Fe(II)
var150	Fe_D_CONC_SUBICE_PUMP	nmol/kg	Concentration of dissolved Fe
var316	LFe_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved iron binding ligand when only one ligand is determined
var317	LFe_D_LogK_BOTTLE		Log of the conditional stability constant for binding of Fe by the dissolved Fe-binding ligand when only one ligand is determined
var318	L1Fe_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved L1 Fe-binding ligand
var319	L1Fe_D_LogK_BOTTLE		Log of the conditional stability constant for binding of Fe by L1 Fe-binding ligand
var320	L2Fe_D_CONC_BOTTLE	nmol/kg	Concentration of dissolved L2 Fe-binding ligand
var321	L2Fe_D_LogK_BOTTLE		Log of the conditional stability constant for binding of Fe by L2 Fe-binding ligand
var325	L1Fe_D_CONC_FISH	nmol/kg	Concentration of dissolved L1 Fe-binding ligand
var326	L1Fe_D_LogK_FISH		Log of the conditional stability constant for binding of Fe by L1 Fe-binding ligand
var327	L2Fe_D_CONC_FISH	nmol/kg	Concentration of dissolved L2 Fe-binding ligand
var328	L2Fe_D_LogK_FISH		Log of the conditional stability constant for binding of Fe by L2 Fe-binding ligand
var343	Fe_TP_CONC_BOTTLE	nmol/kg	Concentration of total particulate iron determined by filtration from a water sampling bottle
var344	Fe_TPL_CONC_BOTTLE	nmol/kg	Concentration of labile particulate iron determined by filtration from a water sampling bottle
var345	Fe_TPR_CONC_BOTTLE	nmol/kg	Concentration of refractory particulate iron determined by filtration from a water sampling bottle
var385	Fe_TP_CONC_PUMP	nmol/kg	Concentration of total particulate iron determined by in situ filtration (pump) without size fractionation
var386	Fe_LPT_CONC_PUMP	nmol/kg	Concentration of total particulate iron determined by in situ filtration (pump) collected on a prefilter (large particles)
var387	Fe_SPT_CONC_PUMP	nmol/kg	Concentration of total particulate iron determined by in situ filtration (pump) collected on a main filter (small particles)
var388	Fe_SPL_CONC_PUMP	nmol/kg	Concentration of labile particulate iron determined by in situ filtration (pump) collected on a main filter (small particles)
var431	Fe_TP_CONC_FISH	nmol/kg	Concentration of total particulate iron determined by towed fish without size fractionation
var432	Fe_TPL_CONC_FISH	nmol/kg	Concentration of labile particulate iron determined by towed fish without size fractionation
var507	Fe_56_54_TP_DELTA_BOTTLE	per 10^3	Atom ratio of total particulate Fe isotopes expressed in conventional DELTA notation determined by filtration from a water sampling bottle referenced to {IRMM-14}
var575	Fe_CELL_CONC_BOTTLE	amol/cell	Fe measured in a specific individual cell with SXRF (attomoles (10^-18 moles) per cell
"""
def create_map_axis(figsize=(12, 5), extent=[115, 285, -10, 10],
                    xticks=np.arange(120, 290, 10),
                    yticks=np.arange(-10, 11, 2)):
    """Create a figure and axis with cartopy."""
    zorder = 2  # Placement of features (reduce to move to bottom.)
    projection = ccrs.PlateCarree(central_longitude=180)
    proj = ccrs.PlateCarree()
    proj._threshold /= 20.

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent(extent, crs=proj)

    # Features.
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', ec='k',
                                        fc=cfeature.COLORS['land'])

    ax.add_feature(land, fc=cfeature.COLORS['land_alt1'], lw=0.4, zorder=zorder)


    # ax.add_feature(cfeature.OCEAN, color='azure', zorder=zorder - 1)

    try:
        # Make edge frame on top.
        ax.outline_patch.set_zorder(zorder + 1)  # Depreciated.
    except AttributeError:
        ax.spines['geo'].set_zorder(zorder + 1)  # Updated for Cartopy

    if xticks is not None and yticks is not None:
        # Draw tick marks (without labels here - 180 centre issue).
        ax.set_xticks(xticks, crs=proj)
        ax.set_yticks(yticks, crs=proj)
        # ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.set_xticklabels(xticks)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # Minor ticks.
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        fig.subplots_adjust(bottom=0.2, top=0.8)

    ax.set_aspect('auto')
    return fig, ax, proj



def Fe_obs_GEOTRACES():
    file = 'GEOTRACES/idp2021/seawater/GEOTRACES_IDP2021_Seawater_Discrete_Sample_Data_v1.nc'
    ds = xr.open_dataset(cfg.obs / file)
    # @todo: drop bad data np.unique(ds.var5)
    # @todo: bin model depths
    ds = ds.rename({'var2': 'depths'})
    var = ['var{}'.format(v) for v in np.concatenate([np.arange(3), [2, 88]])]
    drop_vars = [v for v in ds.data_vars if v not in var]
    drop_vars = [v for v in drop_vars if v not in ['date_time', 'depths', 'latitude', 'longitude']]
    ds = ds.drop(drop_vars)

    ds = ds.where((ds.latitude < 10) & (ds.latitude > -10), drop=True)
    ds = ds.where((ds.longitude > 130) & (ds.longitude < 285), drop=True)

    # convert to multi-dim array.
    ds = ds.stack({'index': ('N_STATIONS', 'N_SAMPLES')})
    ds = ds.drop_vars(['N_SAMPLES', 'N_STATIONS'])
    index = pd.MultiIndex.from_arrays([ds.date_time.values, ds.depths.values,
                                       ds.latitude.values, ds.longitude.values],
                                      names=['t', "z", "y", "x"], )
    ds = ds.assign_coords({"index": index})
    ds = ds.dropna('index')
    ds = ds.unstack("index")

    # # Bin Depths.
    # dz = xr.open_dataset(cfg.data / 'ofam_mesh_grid.nc')
    # dx = ds.groupby_bins(ds.z, bins=dz.st_edges_ocean.values, labels=dz.st_ocean.values)
    # dx = dx.mean().rename({'z_bins': 'z'})

    for v, d, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'], ['°', '°', 'm']):
        ds = ds.sortby(ds[v])
        ds[v].attrs['long_name'] = d
        ds[v].attrs['standard_name'] = d.lower()
        ds[v].attrs['units'] = u

    dx = ds.var88.where(~np.isnan(ds.var88), drop=True)

    for v in ['x', 'y', 'z', 't']:
        dx = dx.dropna(v, 'all')

    # dx = dx.sel(x=slice(147.7,  155), y=slice(-9.25, -3))
    # dx.mean('t').plot(col='z', col_wrap=5)

    # Plot: all x-y locations.
    fig, ax, proj = create_map_axis()
    cs = dx.mean(['t', 'z'], skipna=True).plot(ax=ax, zorder=5, transform=proj)
    ax.set_title('GEOTRACES mean concentration of dissolved Fe [nmol/kg]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_map.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # Plot Vitiaz Strait
    dxx = dx.sel(y=-5.9, x=147.7, method='nearest')
    for v in ['z', 't']:
        dxx = dxx.dropna(v, 'all')
    ax.plot(dxx.squeeze(), dxx.z, color='k', label='Vitiaz Strait')

    # Plot Solomon Strait
    dxx = dx.sel(y=-5.139, x=153.3, method='nearest')
    for v in ['z', 't']:
        dxx = dxx.dropna(v, 'all')
    ax.plot(dxx.squeeze(), dxx.z, color='r', label='Solomon Strait')

    ax.set_title('GEOTRACES concentration of dissolved Fe')
    ax.set_xlabel('dFe [nmol/kg]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(600, 0)
    ax.set_xlim(0.2, 0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/GEOTRACES_dFe_straits.png')
    plt.show()


def Fe_obs_ML():
    """Huang et al. (2022)
    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    file = 'Huang_et_al_2022_monthly_dFe.nc'
    ds = xr.open_dataset(cfg.obs / file)
    ds = ds.rename({'Longitude': 'x', 'Latitude': 'y', 'Depth': 'z', 'Month': 't'})
    ds.coords['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
    ds = ds.sortby(ds.x)
    for v, d, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'], ['°', '°', 'm']):
        ds = ds.sortby(ds[v])
        ds[v].attrs['long_name'] = d
        ds[v].attrs['standard_name'] = d.lower()
        ds[v].attrs['units'] = u

    dx = ds.dFe_RF
    dx = dx.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000)).isel(t=-1)

    # Plot: all x-y locations.
    dxx = dx.sel(z=slice(0, 400)).mean('z')
    fig, ax, proj = create_map_axis()
    cs = dxx.plot(ax=ax, zorder=5, transform=proj)
    ax.set_title('Dissolved Fe ML compilation from Huang et al. (2022) [mean 0-370m]')
    cbar = cs.colorbar
    cbar.set_label('dFe [nmol/kg]')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/ML_obs_dFe_map.png')
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

def Fe_obs_tagliabue():
    ds = pd.read_csv(cfg.obs / 'tagliabue_fe_database_jun2015_publicp.csv')
    ds = pd.DataFrame(ds)#.set_index(['depth', "x", "lat"])
    ds = ds.to_xarray()

    ds = ds.rename({'lon': 'x', 'lat': 'y', 'depth': 'z'})
    # Convert longituge to degrees east (0-360).
    ds['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)

    # Subset to tropical pacific ocean.
    ds = ds.where((ds.y < 10) & (ds.y > -10), drop=True)
    ds = ds.where((ds.x > 120) & (ds.x < 290), drop=True)
    ds = ds.where(ds.month > 0, drop=True)  # Remove single -99 month.

    # Convert to multi-dimensional array.
    t = pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(ds.year[i].item(), ds.month[i].item())
                          for i in range(ds.index.size)])
    ds['t'] = ('index', t)
    ds = ds.drop_vars(['month', 'year'])

    index = pd.MultiIndex.from_arrays([ds.t.values, ds.z.values, ds.y.values, ds.x.values],
                                      names=['t', "z", "y", "x"], )
    ds = ds.assign_coords({"index": index})
    ds = ds.dropna('index')
    ds = ds.unstack("index")

    # Plot equatorial Pacific.
    dx = ds.dFe.sel(y=0, method='nearest').sel(z=slice(0, 400))
    for v in ['z', 't', 'x']:
        dx = dx.dropna(v, 'all')

    dx.plot(col='t', col_wrap=3, yincrease=False)


    # Plot Pacific.
    dx = ds.dFe.sel(z=slice(0, 400)).mean('z').mean('t')
    for v in ['y', 'x']:
        dx = dx.dropna(v, 'all')

    fig, ax, proj = create_map_axis()
    dx.plot(yincrease=False, ax=ax, zorder=5, transform=proj)
    ax.set_title('tagliabue_fe_database_jun2015')
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_map.png')
    plt.show()

    # Plot: Vitiaz Strait & Solomon Strait.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # Plot Vitiaz Strait
    dxx = ds.sel(y=-5.9, x=147.7, method='nearest')
    for v in ['z', 't']:
        dxx = dxx.dropna(v, 'all')
    ax.plot(dxx.squeeze(), dxx.z, color='k', label='Vitiaz Strait')

    # Plot Solomon Strait
    dxx = ds.sel(y=-5.139, x=153.3, method='nearest')
    for v in ['z', 't']:
        dxx = dxx.dropna(v, 'all')
    ax.plot(dxx.squeeze(), dxx.z, color='r', label='Solomon Strait')

    ax.set_title('Tagliabue concentration of dissolved Fe')
    ax.set_xlabel('dFe [nmol/kg]')
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(600, 0)
    ax.set_xlim(0.2, 0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cfg.fig / 'obs/Tagliabue_dFe_straits.png')
    plt.show()
