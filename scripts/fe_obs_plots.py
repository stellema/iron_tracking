# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Fri Mar 10 17:07:35 2023

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter
from dataclasses import dataclass, field
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
from datasets import get_ofam_filenames, add_coord_attrs, save_dataset
from fe_obs_dataset import FeObsDataset, FeObsDatasets
from tools import timeit, mlogger


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


def plot_iron_obs_Huang(ds):
    """Huang et al. (2022).

    dFe climatology based on compilation of dFe observations + environmental predictors
    from satellite observations and reanalysis products using machine-learning approaches
    (random forest).

    """
    dx = ds.fe.sel(y=slice(-30, 30), x=slice(120, 290), z=slice(0, 2000))
    wbcs = {}
    wbcs['vs'] = dict(name='Vitiaz Strait', c='r', ds=dx.sel(y=-7, x=149))  # (not available)
    wbcs['png'] = dict(name='PNG', c='deeppink', ds=dx.sel(y=slice(-4.5, -3), x=slice(140, 145)).mean(['y', 'x']))
    wbcs['ss'] = dict(name='Solomon Strait', c='darkviolet', ds=dx.sel(y=-5, x=155))
    wbcs['ssea'] = dict(name='Solomon Sea', c='b', ds=dx.sel(y=-7, x=153))
    wbcs['ni'] = dict(name='New Ireland', c='dodgerblue', ds=dx.sel(y=-3, x=156))
    wbcs['mc'] = dict(name='MC(7N)', c='lime', ds=dx.sel(y=7, x=132))
    wbcs['mc5'] = dict(name='MC(5N)', c='g', ds=dx.sel(y=5, x=127))

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

    # Plot: All straits (annual mean).
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for n in wbcs.keys():
        ax.plot(wbcs[n]['ds'].isel(t=-1), wbcs[n]['ds'].z, c=wbcs[n]['c'],
                label=wbcs[n]['name'], marker='o', ls='-', fillstyle='none')
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

    # Plot: All straits (month heatmap).
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    ax = ax.flatten()

    for i, n in enumerate(['mc', 'vs', 'png', 'ssea', 'ss', 'ni']):
        cs = ax[i].pcolormesh(wbcs[n]['ds'].t, wbcs[n]['ds'].z, wbcs[n]['ds'].T,
                              cmap=plt.cm.rainbow, vmin=0, vmax=1)

        ax[i].set_title('{} Huang dFe'.format(wbcs[n]['name']), loc='left')
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


def plot_iron_obs_maps(ds_geo, ds_tag):
    """Plot Tagliabue & GEOTRACES data positions."""
    # Plot positions of observations in database.
    # ds1 = GEOTRACES_iron_dataset()
    # ds2 = Tagliabue_iron_dataset()
    names = ['a) GEOTRACES IDP2021', 'b) Tagliabue et al. (2012) Dataset [2015 Update]']

    # Global plot (seperate).
    fig = plt.figure(figsize=(12, 10))
    for i, ds in enumerate([ds_geo, ds_tag]):
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

    for i, ds, c in zip(range(2), [ds_geo, ds_tag], ['blue', 'red']):
        ax.scatter(ds.x, ds.y, c=c, s=15, marker='*', alpha=0.7, transform=proj)
    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.
    ax.set_title('Iron observations from (blue) GEOTRACES IDP2021 and' +
                 ' (red) Tagliabue et al. (2012) dataset [2015 Update]')
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
    for i, ds, mrk in zip(range(2), [ds_geo, ds_tag], ['o', 'P']):

        ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                      (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)
        refs = np.unique(ds['ref'])
        for j, r in enumerate(refs):
            dx = ds.where(ds['ref'] == r, drop=True)
            ax.scatter(dx.x, dx.y, s=30, marker=mrk, label=r, c=colors[j], alpha=0.7,
                       transform=proj)
    ax.axhline(y=0, c='k', lw=0.5)  # Add reference line at the equator.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), markerscale=1.1, ncols=5)
    ax.set_title('Iron observations from GEOTRACES IDP2021 and ' +
                 'Tagliabue et al. (2012) dataset [2015 Update]')
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_map_pacific_references.png', dpi=300)


def plot_iron_obs_straits(dfe_geo, dfe_tag, dfe_ml):
    """Plot Tagliabue & GEOTRACES data positions."""
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    ax = ax.flatten()

    for i, v in enumerate(['mcx', 'vs', 'png', 'ssea', 'ss',  'ni']):
        for m, ds in enumerate([dfe_geo.ds_avg, dfe_tag.ds_avg, dfe_ml.ds_avg.isel(t=-1)]):
            label = ['GEOTRACES', 'Tagliabue', 'Huang'][m]
            c = ['r', 'b', 'k'][m]
            if v in ds.data_vars:
                dv = ds[v].dropna('z', 'all')
                if 't' in dv.dims:
                    dv = dv.dropna('t', 'all')
                    # label + ' ({})'.format(dv.t[0].dt.strftime("%Y-%m").item())
                    dv = dv.mean('t')

                ax[i].plot(dv, dv.z, c=c, label=label, marker='o', ls='-', fillstyle='none')

        nm = dfe_geo.obs_site[v]['name'] if v in dfe_geo.obs_site.keys() else v
        ax[i].set_title(nm, loc='left')

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


def plot_test_spline(dfs):
    """Plot interpolation method differences."""
    if not hasattr(dfs, 'dfe_tag'):
        setattr(dfs, 'dfe_tag', FeObsDataset(dfs.Tagliabue_iron_dataset_4D()))

    lats, lons = [-10, 10], [115, 220]
    ds = dfs.dfe_tag.ds.fe.sel(y=slice(*lats), x=slice(*lons))
    ds_sl = dfs.interpolate_depths(ds, method='slinear')
    ds_quad = dfs.interpolate_depths(ds, method='quadratic')
    ds_cubic = dfs.interpolate_depths(ds, method='cubic')

    names = ['obs', 'cubic', 'quadratic', 'slinear']
    colors = ['k', 'g', 'b', 'r']

    x, y = 155, -5  # Solomon Strait

    fig, ax = plt.subplots(figsize=(7, 10))
    # Plot observations as circles.
    dx = ds.sel(y=y, x=x, method='nearest').dropna('t', 'all').isel(t=0)
    ax.scatter(dx, dx.z, c=colors[0], s=100, label=names[0], zorder=10)

    for i, dx in enumerate([ds_cubic, ds_quad, ds_sl]):
        dx = dx.sel(y=y, x=x, method='nearest').dropna('t', 'all').isel(t=0)
        ax.plot(dx, dx.z, c=colors[i+1], label=names[i+1])

    ax.set_title('Solomon Strait')
    ax.invert_yaxis()
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('dFe [nmol/L]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/test_interp_solomon.png', dpi=250)


def plot_combined_iron_obs_datasets_LLWBCs(dfs):
    """Plot combined iron observation datasets."""
    if not hasattr(dfs, 'dfe'):
        setattr(dfs, 'dfe', FeObsDataset(dfs.combined_iron_obs_datasets(add_Huang=False, interp_z=True)))
    if not hasattr(dfs, 'dfe_all'):
        setattr(dfs, 'dfe_all', FeObsDataset(dfs.combined_iron_obs_datasets(add_Huang=True, interp_z=True)))
    # if not hasattr(dfs, 'dfe_orig'):
    #     setattr(dfs, 'dfe_orig', FeObsDataset(dfs.combined_iron_obs_datasets(add_Huang=True, interp_z=False)))

    ##############################################################################################
    # LLWBCS
    # LLWBCs: X-Y location of obs.
    name = 'llwbcs'
    dx, obs, coords = dfs.dfe.get_obs_locations(name)
    dx_all, obs, coords = dfs.dfe_all.get_obs_locations(name)

    ##############################################################################################
    # LLWBCs: orig vs interp depths (Tag & GEOTRACES)..
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[1].plot(dfs.dfe.ds_avg[name], dfs.dfe.ds_avg[name].z, c='k', lw=4, label='mean', zorder=15)
    axes[1].plot(dfs.dfe_all.ds_avg[name], dfs.dfe_all.ds_avg[name].z, c='k', lw=3, ls='--',
                 label='mean (+Huang)', zorder=15)

    for j, ax, ds_ in zip(range(2), axes, [dfs.dfe_orig, dfs.dfe]):

        for i, x, y in zip(range(len(obs[2:])), obs[2:]):
            dxx = ds_.fe.sel(x=x, y=y, method='nearest').dropna('z', 'all')
            label = coords[i] if j == 1 else None
            ax.plot(dxx, dxx.z, c=dfs.colors[i], label=label)
        ax.set_ylim(800, 0)

    axes[0].set_title('a) Southern LLWBCs dFe (Tagliabue & GEOTRACES)')
    axes[1].set_title('b) Southern LLWBCs dFe (interpolat ed)')
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
    ax.scatter(obs[0], obs[1], color=dfs.dfe.colors[:len(obs[0])], transform=ccrs.PlateCarree())
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

    j = 0
    for i, xy in enumerate(obs[:22]):
        dxx = dx.sel(x=xy[1], y=xy[0]).dropna('z', 'all')
        ax.plot(dxx, dxx.z, c=dfs.dfe.colors[i], label=coords[i])

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
    dx = dfs.dfe.ds.fe.sel(y=slice(-2.5, 2.5)).mean('y')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    dx.mean('t').plot(ax=ax, yincrease=False, vmin=0, vmax=1.5, cmap=plt.cm.rainbow)

    ax.set_title('Combined iron obs around the equator (time mean - no groupby)')
    ax.set_ylim(600, 0)
    ax.set_xlim(130, 280)
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_equator_combined.png')
    plt.show()

    # Equator.
    dx = dfs.dfe.ds.fe.groupby('t.month').mean('t').rename({'month': 't'})
    dx = dx.sel(y=slice(-2.5, 2.5)).mean('y')

    fig, ax = plt.subplots(4, 3, figsize=(14, 12))
    ax = ax.flatten()
    for i in range(dx.t.size):
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

    name = 'interior'
    dx, obs, coords = dfs.dfe.get_obs_locations(name)
    dx_all, obs, coords = dfs.dfe_all.get_obs_locations(name)

    ##############################################################################################
    # Interior: dFe and Map (Tag & GEOTRACES).
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Map with obs location scatter.
    ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_title('a) Interior dFe observation locations', loc='left')
    ax.set_extent([135, 290, -11, 11], crs=ccrs.PlateCarree())
    ct = plt.rcParams['axes.prop_cycle'].by_key()['color']
    N = int(np.ceil(len(ct)*len(obs[0])/len(ct))/10) + 5
    cols = np.tile(ct, N)[:len(obs[0])]
    ax.scatter(obs[0], obs[1], color=cols, transform=ccrs.PlateCarree())
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
    ax.plot(dx_all.mean(['x', 'y']), dx_all.z, c='k', lw=3, ls='--', label='mean (+Huang)', zorder=15)

    for i, x, y in enumerate(obs):
        dxx = dx.sel(x=x, y=y).dropna('z', 'all')
        ax.plot(dxx, dxx.z, color=cols[i], ls=dfs.lss[int(np.floor(i/10))], label=coords[i])

    ax.set_ylim(500, 0)
    ax.set_xlim(0, 1.5)
    fig.legend(ncols=6, loc='upper center', bbox_to_anchor=(0.5, 0))  # (x, y, width, height)
    ax.set_xlabel('dFe [nmol/kg]')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%dm"))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.savefig(paths.figs / 'obs/iron_obs_combined_interior_no_euc.png', bbox_inches='tight')


logger = mlogger('iron_observations')
dfs = FeObsDatasets()
setattr(dfs, 'ds', dfs.combined_iron_obs_datasets(add_Huang=False, interp_z=True))
setattr(dfs, 'dfe', FeObsDataset(dfs.ds))

setattr(dfs, 'ds_geo', dfs.GEOTRACES_iron_dataset())
setattr(dfs, 'ds_tag', dfs.Tagliabue_iron_dataset())

setattr(dfs, 'dfe_ml', FeObsDataset(dfs.Huang_iron_dataset()))
setattr(dfs, 'dfe_geo', FeObsDataset(dfs.GEOTRACES_iron_dataset_4D()))
setattr(dfs, 'dfe_tag', FeObsDataset(dfs.Tagliabue_iron_dataset_4D()))


plot_iron_obs_Huang(dfs.dfe_ml.ds)
plot_iron_obs_maps(dfs.ds_geo, dfs.ds_tag)
plot_iron_obs_straits(dfs.dfe_geo, dfs.dfe_tag, dfs.dfe_ml)
plot_combined_iron_obs_datasets_LLWBCs(dfs)
