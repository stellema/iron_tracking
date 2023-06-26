# -*- coding: utf-8 -*-
"""Plot iron model output.

Notes:

Example:

Todo:

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr  # NOQA

from cfg import paths, ExpData
from tools import unique_name, mlogger, timeit
from fe_exp import FelxDataSet

scenario, lon, version, index = 0, 165, 0, 7
exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index)
pds = FelxDataSet(exp)

ds = xr.open_dataset(pds.exp.file_felx)

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun  7 18:19:06 2023

"""
from cartopy.mpl.gridliner import LATITUDE_FORMATTER  # LONGITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# import seaborn as sns
import xarray as xr  # NOQA

from cfg import paths, ExpData, ltr, release_lons
from tools import unique_name, mlogger, timeit
from fe_exp import FelxDataSet
# from stats import format_pvalue_str, get_min_weighted_bins, weighted_bins_fd

fsize = 10
plt.rcParams.update({'font.size': fsize})
plt.rc('font', size=fsize)
plt.rc('axes', titlesize=fsize)
plt.rcParams['figure.figsize'] = [10, 7]
# plt.rcParams['figure.dpi'] = 300

def test_plot_iron_paths(pds, ds, ntraj=5):
    """Plot particle map, depth, dFe, remin, scav and phyto."""
    fig, ax = plt.subplots(3, 2, figsize=(15, 13), squeeze=True)
    ax = ax.flatten()

    for j, p in enumerate(range(ntraj)):
        c = ['k', 'b', 'r', 'g', 'm', 'y', 'darkorange', 'hotpink', 'lime', 'navy'][j % 10]
        ls = ['-', '--', '-.', ':'][int(np.floor((j / 10)) % 3)]
        lw = 1
        dx = ds.isel(traj=p).dropna('obs', 'all')
        ax[0].plot(dx.lon, dx.lat, c=c, ls=ls, lw=lw, label=p)  # map
        ax[1].plot(dx.obs, dx.z, c=c, ls=ls, lw=lw, label=p)  # Depth
        ax[2].plot(dx.obs, dx.fe, c=c, ls=ls, lw=lw, label=p)  # lagrangian iron
        if 'Fe' in dx.data_vars:
            ax[2].plot(dx.obs, dx.Fe, c=c, ls=ls, lw=1.5, label=p + '(OFAM3)')  # ofam iron

        for i, var in zip([3, 4, 5], ['fe_reg', 'fe_scav', 'fe_phy']):
            ax[i].set_title(var)
            ax[i].plot(dx.obs, dx[var], c=c, ls=ls, lw=lw, label=p)

    ax[0].set_title('Particle lat & lon')
    ax[1].set_title('Particle depth')
    ax[2].set_title('Lagrangian iron')

    for i in range(1, 6):
        ax[i].set_ylabel('Fe [nM Fe]')
        ax[i].set_xlabel('obs')
        ax[i].set_xmargin(0)
        ax[i].set_ymargin(0)

    ax[1].set_ylabel('Depth')
    ax[1].invert_yaxis()
    ax[0].set_xlabel('Lon')
    ax[0].set_ylabel('Lat')
    lgd = ax[-2].legend(ncol=16, loc='upper left', bbox_to_anchor=(0, -0.1))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(paths.figs / 'felx/dFe_paths_{}_{}.png'.format(pds.exp.id, ntraj),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def test_plot_EUC_iron_depth_profile(pds, ds, dfs):
    """Plot particle depth vs dFe (at source and EUC) and scatter start vs final dFe."""
    # Particle data
    ds_i = ds.isel(obs=0)  # at source
    ds_f = ds.ffill('obs').isel(obs=-1, drop=True)  # at EUC
    ds_f_z_mean = (ds_f.fe * ds_f.u).groupby(ds_f.z).sum() / (ds_f.u).groupby(ds_f.z).sum()
    ds_f_mean = ds_f.fe.weighted(ds_f.u).mean().load().item()

    # Observations
    # df_h = dfs.Huang_iron_dataset().fe.sel(x=pds.exp.lon, method='nearest').sel(
    #     y=slice(-2.6, 2.6)).mean('y')
    df_avg = dfs.dfe.ds_avg.euc_avg.sel(lon=pds.exp.lon, method='nearest')
    df_avg_mean = df_avg.mean().item()

    # Plot particle depth vs dFe (at initial/source and final/EUC).
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), squeeze=True)
    ax[0].set_title('a) {} dFe at source'.format(pds.exp.scenario_name), loc='left')
    ax[0].scatter(ds_i.fe, ds_i.z, c='k', s=7)  # dFe at source.
    ax[0].invert_yaxis()
    ax[0].set_ylabel('Depth [m]')

    # Iron at the EUC (inc. obs & weighted mean depth profile)
    ax[1].set_title('b) {} EUC dFe at {}'.format(pds.exp.scenario_name, pds.exp.lon_str), loc='left')
    # ax[1].plot(df_h.isel(t=-1), df_h.z, c='r', label='Huang et al. (2022)')
    ax[1].plot(df_avg, df_avg.z, c='m', label='Obs. mean ({:.2f} nM)'.format(df_avg_mean))
    ax[1].plot(ds_f_z_mean, ds_f_z_mean.z, c='k', label='Model mean ({:.2f} nM)'.format(ds_f_mean))
    ax[1].scatter(ds_f.fe, ds_f.z, c='k', s=7, label='Model')  # dFe at EUC.

    ax[1].axvline(0.6, c='darkgrey', lw=1)
    ax[1].set_ylim(355, 20)
    ax[1].set_xlim(min(0.1, ds_f.fe.min() - 0.1), max(1.2, ds_f.fe.max() + 0.1))
    ax[1].set_xmargin(0.05)
    lgd = ax[1].legend(ncol=1, loc='upper left', bbox_to_anchor=(1, 1))
    for i in range(2):
        ax[i].set_xlabel('dFe [nM]')

    textstr = ', '.join(['{}={}'.format(i, pds.params[i])
                         for i in ['c_scav', 'k_inorg', 'k_org', 'mu_D', 'mu_D_180']])
    # ax[0].text(0, -0.1, 'params: ' + textstr, transform=ax[0].transAxes)
    title = plt.suptitle('params: ' + textstr, x=0.42, y=0.005)

    # plt.subplots_adjust(bottom=-0.1)
    plt.tight_layout()
    file = unique_name(paths.figs / 'felx/iron_z_profile_{}.png'.format(pds.exp.id))
    plt.savefig(file, bbox_extra_artists=(lgd, title,), bbox_inches='tight')
    plt.show()

    # # Plot start vs final dFe.
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=True)
    # ax.scatter(ds_i.fe, ds_f.fe, c='k', s=12)  # dFe at source.
    # ax.set_title('{} EUC dFe at source vs EUC at {}'.format(pds.exp.scenario_name, pds.exp.lon_str), loc='left')
    # ax.set_ylabel('EUC dFe [nM]')
    # ax.set_xlabel('Source dFe [nM]')
    # plt.tight_layout()
    # plt.savefig(cfg.paths.figs / 'felx/dFe_scatter_source_EUC_{}.png'.format(pds.exp.id))
    # plt.show()
    return


def transport_source_bar_graph(pds, var='u_sum_src'):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        pds ():
        var (str, optional):

    """
    z_ids = [1, 2, 6, 7, 8, 3, 4, 5, 0]

    # Open data.
    dss = []

    for i, lon in enumerate(pds.release_lons):
        pds = FelxDataSet(ExpData(scenario=0, lon=lon, version=pds.exp.version))
        ds = pds.fe_model_sources(add_diff=False)
        ds = ds.isel(zone=z_ids)  # Must select source order (for late use of ds)
        dss.append(ds[var].mean('rtime'))

    xlim_max = np.max([d.max('exp') for d in dss])
    xlim_max += xlim_max * 0.2
    ylabels, c = ds.names.values, ds.colors.values

    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = pds.release_lons[i]
        dx = dss[i]
        ticks = np.arange(dx.zone.size)  # Source y-axis ticks.

        ax.set_title('{} {} for EUC at {}°E'.format(ltr[i], dx.attrs['long_name'], lon), loc='left')

        # Historical horizontal bar graph.
        ax.barh(ticks, dx.isel(exp=0), width, color=c, **kwargs)

        # RCP8.5: Hatched overlay.
        # ax.barh(ticks, dx.isel(exp=1), width, fill=False, hatch='//', label='RCP8.5', **kwargs)
        ax.barh(ticks, dx.isel(exp=1), width / 3, color='k', alpha=0.6, label='RCP8.5')

        g = ax.barh(ticks, dx.max('exp'), width, fill=False, alpha=0)
        diff = ((dx.isel(exp=1) - dx.isel(exp=0)) / dx.isel(exp=0)) * 100
        diff = ['{:+0.1f}%'.format(i) for i in diff.values]
        ax.bar_label(g, labels=diff, label_type='edge', padding=5)

        # Remove bar white space at ends of axis (-0.5 & +0.5).
        ax.set_ylim([ticks[x] + c * 0.5 for x, c in zip([0, -1], [-1, 1])])
        ax.set_yticks(ticks)
        ax.set_xlim(right=xlim_max)

        # Add x-axis label on bottom row.
        if i >= 2:
            ax.set_xlabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))

        # Add y-axis ticklabels on first column.
        if i in [0, 2]:
            ax.set_yticklabels(ylabels)

        # Add RCP legend at ends of rows.
        ax.legend(loc='lower right')
        ax.invert_yaxis()
        # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_tick_params(labelbottom=True)

    plt.tight_layout()
    plt.savefig(paths.figs / 'source_bar_{}.png'.format(var), dpi=300)
    plt.show()
    return


if __name__ == '__main__':
    scenario, lon, version, index = 0, 220, 0, 0
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index)
    pds = FelxDataSet(exp)

    # Weighted mean. @todo
    var = 'fe_avg'
    dss = []
    for i, lon in enumerate(pds.release_lons):
        pds = FelxDataSet(ExpData(scenario=0, lon=lon, version=pds.exp.version))
        ds = pds.fe_model_sources(add_diff=False)
        dss.append(ds[var].mean('rtime'))

    # Weighted mean per source
    transport_source_bar_graph(pds, var='fe_avg_src')
    transport_source_bar_graph(pds, var='fe_src_avg_src')
    transport_source_bar_graph(pds, var='u_sum_src')

    # Weighted mean depth profile @todo
