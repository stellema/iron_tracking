# -*- coding: utf-8 -*-
"""Plot iron model output.

Notes:

Example:

Todo:


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun  7 18:19:06 2023

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr  # NOQA

from cfg import paths, ExpData, ltr, release_lons, scenario_name, scenario_abbr, zones
from fe_exp import FelxDataSet
from fe_obs_dataset import iron_source_profiles
from stats import get_min_weighted_bins
from tools import unique_name

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


def test_plot_EUC_iron_depth_profile(pds, ds, ds_fe):
    """Plot particle depth vs dFe (at source and EUC) and scatter start vs final dFe."""
    # Particle data
    ds_i = ds.isel(obs=0)  # at source
    ds_f = ds.ffill('obs').isel(obs=-1, drop=True)  # at EUC
    ds_f_z_mean = (ds_f.fe * ds_f.u).groupby(ds_f.z).sum() / (ds_f.u).groupby(ds_f.z).sum()
    ds_f_mean = ds_f.fe.weighted(ds_f.u).mean().load().item()

    # Observations
    df_avg = ds_fe.euc_avg.sel(lon=pds.exp.lon, method='nearest')
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


def source_bar_graph(pds, ds, var='u_sum_src'):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        pds ():
        var (str, optional):

    """
    z_ids = [1, 2, 6, 7, 8, 3, 4, 5, 0]

    ds = ds.isel(zone=z_ids)  # Must select source order (for late use of ds)

    xlim_max = ds[var].mean('rtime').max()
    xlim_max += xlim_max * 0.2
    ylabels, c = ds.names.values, ds.colors.values

    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = pds.release_lons[i]
        dx = ds[var].sel(x=lon).mean('rtime')
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
    plt.savefig(paths.figs / 'fe_model/source_bar_{}_v{}.png'.format(var, pds.exp.version), dpi=300)
    plt.show()
    return


def plot_var_scatter(pds, ds, var):
    """Scatter plot of variable mean for each release longitude."""
    ds_fe = iron_source_profiles()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Add observations.
    if var in ['fe_avg']:
        ds_fe = ds_fe.sel(z=slice(25, 350), x=slice(160, 260)).dropna('x', 'all')
        ax.plot(ds_fe.x, ds_fe.EUC.mean('z'), 'x', c='g', label='Obs.')
        ax.plot(ds_fe.lon, ds_fe.euc_avg.mean('z'), 'o', c='g', label='Obs. Mean')

        for j, v in enumerate(['fe_avg', 'fe_src_avg']):
            dx = ds[v].mean('rtime')
            for source_iron in ['LLWBC-obs']:
                for s in [0, 1]:
                    ax.plot(dx.x, dx.isel(exp=s), ['-o', '--*'][s], c=['k', 'darkviolet'][j],
                            label='{} {}'.format(['EUC', 'Source'][j], scenario_name[s],
                                                 source_iron))
    else:
        dx = ds[var].mean('rtime')
        for source_iron in ['LLWBC-obs']:
            for s in [0, 1]:
                ax.plot(dx.x, dx.isel(exp=s), ['-o', '--*'][s], c='k',
                        label='{} ({})'.format(scenario_name[s], source_iron))

    ax.set_title('{}'.format(ds[var].attrs['long_name']), loc='left')
    ax.set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_xticks(dx.x)
    ax.legend()
    # fig.legend(loc='upper right', bbox_to_anchor=(0.97, 0.93))  # (x, y, width, height)
    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/lon_scatter_{}_v{}.png'.format(var, pds.exp.version), dpi=300)


def plot_var_depth_profile(pds, ds):
    """Plot var as a function of depth."""
    ds = ds.mean('rtime')
    var = 'fe_avg_depth'
    var_src = 'fe_avg_src_depth'
    ds_fe = iron_source_profiles().sel(z=slice(25, 350))

    fig, axes = plt.subplots(2, 4, figsize=(12, 12), sharey='row', sharex='all')
    for i in range(4):

        dx = ds.isel(x=i)
        axes[0, i].plot(ds_fe.euc_avg.isel(lon=i), ds_fe.z, c='g', label='Obs. Mean')
        for j in range(2):
            ax = axes[0, i]
            ax.set_title('{} EUC iron at {}°E'.format(ltr[i], dx.x.item()), loc='left')
            ax.plot(dx[var].isel(exp=j), dx.z, c='k', ls=['-', '--'][j], label=scenario_name[j])  # Mean

            ax = axes[1, i]
            ax.set_title('{} EUC source iron at {}°E'.format(ltr[4 + i], dx.x.item()), loc='left')
            for z in [1, 2, 3]:
                dx_src = dx.isel(exp=j).sel(zone=z)
                label = dx_src.names.item() if j == 0 else None
                ax.plot(dx_src[var_src], dx.z, c=dx_src.colors.item(), ls=['-', '--'][j],
                        label=label)

            axes[j, i].set_xlabel('{} [{}]'.format(dx[var].attrs['standard_name'],
                                                   dx[var].attrs['units']))
            axes[j, i].set_ylim(*[dx[dx[var].dims[-1]][a] for a in [-1, 0]])
            axes[j, i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%dm'))

    # x-axis (dFe) and y-axis (depth)
    axes[0, 3].legend(ncol=1, loc='upper left', bbox_to_anchor=(1, 1))
    axes[1, 3].legend(ncol=1, loc='upper left', bbox_to_anchor=(1, 1))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/depth_profile_{}_v{}.png'.format(var, pds.exp.version), dpi=300)
    plt.show()


def plot_iron_at_source_scatter(pds, ds):
    """Plot scatter of iron at source for each longitude."""
    ds = ds.sel(zone=[1, 2, 6, 7, 8, 3, 4, 5, 0])
    ds = ds.mean('rtime')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')
    for i, ax in enumerate(axes.flatten()):
        ax.set_title('{} Iron at source (EUC at {}°E)'.format(ltr[i], release_lons[i]), loc='left')
        for s, marker in enumerate(['o', '*']):
            ax.scatter(range(9), ds.fe_src_avg_src.isel(x=i, exp=s), marker=marker,
                       c=ds.colors.values, label=scenario_name[s])
        ax.set_ylabel('{} [{}]'.format(ds['fe_avg'].attrs['standard_name'],
                                       ds['fe_avg'].attrs['units']))
        ax.set_xticks(range(9), ds.names.values, rotation='vertical')
        ax.legend()

    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/iron_src_avg_v{}.png'.format(pds.exp.version), dpi=300)
    plt.show()


def plot_var_scatter_and_depth(pds, ds, ds2, ds3):
    """Scatter plot of variable mean for each release longitude."""
    ds_all = [ds, ds2, ds3]
    ds_fe = iron_source_profiles()
    ds_fe = ds_fe.sel(z=slice(25, 350))
    ds_fe = ds_fe.sel(x=slice(160, 260)).dropna('x', 'all')

    version_names = ['Depth-mean', 'Depth-profile', 'Proj-adjusted']
    version_colors = ['k', 'k', 'b']
    ls = [['-', ':'], ['--', '--'], ['-', '-.']]

    fig, axes = plt.subplot_mosaic("AABB;CDEF", figsize=(14, 12), height_ratios=[2.75, 4],
                                   gridspec_kw={"wspace": 0.4})
    var = 'fe_avg'
    ax = axes['A']
    # Add observations.
    ax.plot(ds_fe.x, ds_fe.EUC.mean('z'), 'x', c='g', label='Obs.')
    ax.plot(ds_fe.lon, ds_fe.euc_avg.mean('z'), 'o', c='g', label='Obs. Mean')

    for i, dx in enumerate(ds_all):
        dx = dx[var].mean('rtime')
        for s in [0, 1]:
            label = '{}'.format(version_names[i]) if s == 0 else None
            ax.plot(dx.x, dx.isel(exp=s), ls[i][s], c=version_colors[i], label=label, zorder=4 - i)

    ax.set_title('{} EUC Mean dFe'.format(ltr[0]), loc='left')
    ax.set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_xticks(dx.x)
    ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(1, 1.1))

    var = 'fe_flux_sum'
    ax = axes['B']
    for i, dx in enumerate(ds_all):
        dx = dx[var].mean('rtime')
        for s in [0, 1]:
            ax.plot(dx.x, dx.isel(exp=s), ls[i][s], c=version_colors[i], label=None, zorder=4 - i)

    ax.set_title('{} EUC dFe Flux and Transport'.format(ltr[1]), loc='left')
    ax.set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_xticks(dx.x)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    var = 'u_sum'
    color = 'tab:red'
    ax2.set_ylabel('Transport [Sv]', color=color)  # we already handled the x-label with ax1
    dx = ds[var].mean('rtime')
    for s in [0, 1]:
        ax2.plot(dx.x, dx.isel(exp=s), ls[0][s], c=color, label='Transport', zorder=4 - i)
    ax2.tick_params(axis='y', labelcolor=color)

    var = 'fe_avg_depth'
    for i, a in enumerate(['C', 'D', 'E', 'F']):
        ax = axes[a]

        for j, dx in enumerate(ds_all):
            dx = dx.isel(x=i).mean('rtime')
            lon = dx.x.item()
            for s in range(2):
                ax.plot(dx[var].isel(exp=s), dx.z, c=version_colors[j], ls=ls[j][s], zorder=4 - j)
        ax.set_title('{} EUC Mean dFe at {}°E'.format(ltr[i + 2], dx.x.item()), loc='left')

        # Plot observations.
        ax.plot(ds_fe.euc_avg.isel(lon=i), ds_fe.z, c='g', label='Obs. Mean')
        obs = ds_fe.EUC.sel(x=slice(lon - 5, lon + 5))
        if obs.x.size > 0:
            for j in obs.x:
                ax.scatter(obs.sel(x=lon), ds_fe.z, marker='x', c='g', label='Obs.')

        # x-axis (dFe) and y-axis (depth)
        ax.set_xlabel('{} [{}]'.format(dx[var].attrs['standard_name'], dx[var].attrs['units']))
        ax.set_ylim(*[dx[dx[var].dims[-1]][a] for a in [-1, 0]])
        ax.set_xlim(0, 1.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%dm'))
        if a != 'C':
            ax.set_yticklabels([])
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    # plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/scatter_depth_profile_{}.png'.format(var),
                dpi=300, bbox_inches='tight')
    plt.show()


# def plot_fe_multvar_scatter(pds, ds, scenario=2):
#     """Plot all fe variables (time mean).

#     Args:
#         ds (xarray.dataset): pds.fe_model_sources_all(add_diff=True)

#     """
#     dvars = ['fe_avg', 'fe_src_avg', 'fe_reg_avg', 'fe_scav_avg', 'fe_phy_avg']
#     vlabels = [ds[var].attrs['long_name'] for var in dvars]
#     colors = ['k', 'darkviolet', 'g', 'r', 'blue']  # for each fe variable
#     titles = ['Historical and RCP8.5', 'Projected Change']  # Depending on scenario input

#     names = ['', *['from the {}'.format(s) for s in ds.names.sel(zone=z_ids).values]]  # For titles

#     # Multi-Source.
#     fig, ax = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
#     ax = ax.flatten()

#     for i, z in enumerate([None, *z_ids]):

#         ax[i].set_title('{} EUC Dissolved Iron {}'.format(ltr[i], names[i]), loc='left')

#         for j, var in enumerate(dvars):
#             v = var + '_src' if i > 0 else var

#             for k, dx in zip([0, 1], [ds, ds2]):
#                 dx = dx[v].mean('rtime')
#                 if z is not None:
#                     dx = dx.sel(zone=z)
#                 dx = dx * -1 if var in ['fe_scav_avg', 'fe_phy_avg'] else dx
#                 if scenario == 0:
#                     ax[i].plot(dx.x, dx.sel(exp=scenario), ['o-', 'dashdot'][k], c=colors[j],
#                                label=vlabels[j] if k == 0 else None)
#                     ax[i].plot(dx.x, dx.isel(exp=1), ['--', 'dashdot'][k], c=colors[j])
#                 else:
#                     ax[i].plot(dx.x, dx.sel(exp=1) - dx.sel(exp=0), ['.--', '.-.'][k],
#                                c=colors[j],
#                                label=vlabels[j] if k == 0 else None)

#         ax[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
#         ax[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
#         ax[i].set_xticks(dx.x)
#         ax[i].axhline(0, c='grey', lw=1)
#         if i % 3 == 0:
#             ax[0].set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))

#     handles, labels = ax[0].get_legend_handles_labels()
#     fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.02))
#     plt.tight_layout()
#     plt.savefig(paths.figs / 'fe_model/fe_multi_var_scatter_sources_{}.png'
#                 .format(scenario_abbr[scenario]), dpi=350, bbox_inches='tight')


def plot_fe_multvar_scatter_sources(pds, ds, ds2, ds3=None, scenario=2):
    """Plot all fe variables (time mean).

    Args:
        ds (xarray.dataset): pds.fe_model_sources_all(add_diff=True)

    """
    dvars = ['fe_avg', 'fe_src_avg', 'fe_reg_avg', 'fe_scav_avg', 'fe_phy_avg']
    vlabels = [ds[var].attrs['long_name'] for var in dvars]
    colors = ['k', 'darkviolet', 'blue', 'r', 'g']  # for each fe variable
    # titles = ['Historical and RCP8.5', 'Projected Change']  # Depending on scenario input
    ls = ['-', '--', '-', ':']  # Historical vs RCP markers
    ds_list = [ds, ds2, ds3] if scenario == 2 else [ds, ds2]

    z_ids = [1, 2, 3, 7, 8]
    names = ['', *['from the {}'.format(s) for s in ds.names.sel(zone=z_ids).values]]  # For titles

    # Multi-Source.
    fig, ax = plt.subplots(2, 3, figsize=(12, 12), sharex=True, sharey=0)
    ax = ax.flatten()

    for i, z in enumerate([None, *z_ids]):

        ax[i].set_title('{} EUC dFe {}'.format(ltr[i], names[i]), loc='left')

        for j, var in enumerate(dvars):
            v = var + '_src' if i > 0 else var

            for k, dx in enumerate(ds_list):
                dx = dx[v].mean('rtime')
                if z is not None:
                    dx = dx.sel(zone=z)
                dx = dx * -1 if var in ['fe_scav_avg', 'fe_phy_avg'] else dx

                if k == 0:
                    label = vlabels[j]
                    lw = 1.2
                else:
                    label = None
                    lw = 1.2

                if scenario == 0:
                    ax[i].plot(dx.x, dx.sel(exp=scenario), ['o-', '--.', '.-.'][k], c=colors[j],
                               label=label)
                    ax[i].plot(dx.x, dx.isel(exp=1), ['.:', '.--', '.-.'][k], c=colors[j])
                else:
                    ax[i].plot(dx.x, dx.sel(exp=1) - dx.sel(exp=0), ['o-', '*--', 'd-.'][k],
                               c=colors[j], label=label, lw=lw)

        ax[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
        ax[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax[i].set_xticks(dx.x)
        ax[i].axhline(0, c='grey', lw=1)
        if i % 3 == 0:
            ax[0].set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/fe_multi_var_scatter_sources_{}.png'
                .format(scenario_abbr[scenario]), dpi=350, bbox_inches='tight')


def plot_particle_histogram(pds, ds, var='age'):
    """Plot histogram (draft)."""
    ds = ds.sel(zone=[1, 2, 6, 7, 8, 3, 4, 5, 0])
    xmax = 1000 if var == 'age' else 600

    fig, ax = plt.subplots(7, 4, figsize=(12, 15), sharey='row')
    for i, x in enumerate(release_lons):
        for j, z in enumerate(ds.zone[:-2].values):
            for s, scenario in enumerate([0, 1]):
                # Subset dataset.
                dx = ds.sel(exp=scenario, x=x, zone=z).dropna('traj', 'all')

                # Plot histogram (weighted).
                ax[j, i].hist(dx[var], color=[zones.colors[z], 'k'][s], alpha=[0.5, 0.3][s],
                              bins='fd', weights=ds.u)

            ax[j, 0].set_title('{}{}'.format(ltr[j], zones.names[z]), loc='left')
            ax[j, i].set_xlim(0, xmax)

        # Add the longitude above each column.
        ax[0, i].text(0.25, 1.25, 'EUC at {}°E'.format(x), weight='bold', transform=ax[0, i].transAxes)

        # Add units at the bottom of each column.
        ax[-1, i].set_xlabel('{} [{}]'.format(dx[var].attrs['long_name'], dx[var].attrs['units']))

    fig.subplots_adjust(wspace=0.2, hspace=0.5, top=0.9)
    plt.savefig(paths.figs / 'fe_model/histogram_{}_v{}.png'.format(var, pds.exp.version),
                bbox_inches='tight', dpi=350)


def plot_KDE_source(ax, ds, var, z, color=None, add_IQR=False, axis='x',
                    kde_kws=dict(bw_adjust=0.5)):
    """Plot KDE of source var for historical (solid) and RCP(dashed).

    Args:
        ax (matplotlib.axes): Plot axis.
        ds (xarray.Dataset): Particle source dataset.
        var (str): Variable to plot.
        z (int): Source ID.
        color (str, optional): Color of variable to plot. Defaults to None.
        add_IQR (bool, optional): Add verticle median and IQR lines. Defaults to False.
        axis (str, optional): Axis to plot variable (use 'y' for depth). Defaults to 'x'.

    Returns:
        ax (matplotlib.axes): Plot axis.

    """
    ds = [ds.sel(zone=z).isel(exp=x).dropna('traj', 'all') for x in [0, 1]]
    bins = get_min_weighted_bins([d[var] for d in ds], [d.u / d.rtime.size for d in ds])

    for s in range(2):
        dx = ds[s]
        c = dx.colors.item() if color is None else color
        ls = ['-', ':'][s]  # Linestyle
        n = dx.names.item() if s == 0 else None  # Legend name.

        ax = sns.histplot(**{axis: dx[var]}, weights=dx.u / dx.rtime.size, ax=ax,
                          bins=bins, color=c, element='step', alpha=0,
                          fill=False, kde=True, kde_kws=kde_kws,
                          line_kws=dict(color=c, linestyle=ls, label=n))

        # xmax & xmin cutoff.
        if axis == 'x':
            hist = ax.get_lines()[-1]
            x, y = hist.get_xdata(), hist.get_ydata()
            xlim = x[np.cumsum(y) < (sum(y) * 0.85)]
            ax.set_xlim(max([0, xlim[0]]), xlim[-1])

        # Median & IQR.
        if add_IQR:
            for q, lw, h in zip([0.5, 0.25, 0.75], [1.7, 1.5, 1.5], [0.11, 0.07, 0.07]):
                ax.axvline(x[sum(np.cumsum(y) < (sum(y) * q))], ymax=h, c=color, ls=ls, lw=lw)
    return ax


def plot_KDE_multi_var(ds, lon, var, add_IQR=False):
    """Plot KDE of source var for historical (solid) and RCP(dashed)."""
    z_inds = [[1, 2, 6], [3, 4], [7, 8, 5]]  # Source subsets.
    colors = ['m', 'b', 'g', 'k']
    axis = 'y' if var in ['depth', 'depth_at_src'] else 'x'
    var = [var] if isinstance(var, str) else var
    nc = len(var)

    fig, ax = plt.subplots(3, nc, figsize=(5 * nc, 10), squeeze=0)
    for j, v in enumerate(var):  # Iterate through variables.
        for i in range(ax.size // nc):  # Iterate through source subsets.
            for iz, z in enumerate(z_inds[i]):
                c = colors[iz]
                ax[i, j] = plot_KDE_source(ax[i, j], ds, v, z, c, add_IQR, axis)

            # Plot extras.
            ax[i, j].set_title('{}'.format(ltr[j + i * nc]), loc='left')
            ax[i, j].legend()
            ax[i, j].set_ylabel('Transport [Sv] / day')
            ax[i, j].set_xlabel('{} [{}]'.format(*[ds[v].attrs[s] for s in ['long_name', 'units']]))

            # Subplot ymax (excluding histogram)
            ymax = [max(y.get_ydata()) for y in list(ax[i, j].get_lines())]
            ymax = max(ymax[1::5]) if add_IQR else max(ymax[1::2])
            ymax += (ymax * 0.05)
            if j > 0:
                ymax = max([ymax, ax[i, j - 1].get_ybound()[-1]])
            ax[i, j].set_ylim(0, ymax)
            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/KDE_{}_{}.png'.format('_'.join(var), lon), dpi=350)
    plt.show()
    return


def source_KDE_multi_lon(ds, var='age', add_IQR=False):
    """Plot variable KDE at each longitude."""
    z_inds = [[1, 2, 6], [3, 4], [7, 8]]  # Source subsets.
    colors = ['m', 'b', 'g', 'k']
    axis = 'y' if var in ['depth', 'depth_at_src'] else 'x'

    nr, nc = 3, 4
    fig, ax = plt.subplots(nr, nc, figsize=(14, 9), sharey='row')

    for i, lon in enumerate(release_lons):
        dx = ds.sel(x=lon)
        name, units = [dx[var].attrs[a] for a in ['long_name', 'units']]

        # Suptitles.
        ax[0, i].text(0.25, 1.1, 'EUC at {}°E'.format(lon), weight='bold',
                      transform=ax[0, i].transAxes)

        for j in range(ax.size // nc):  # iterate through zones.
            for iz, z in enumerate(z_inds[j]):
                ax[j, i] = plot_KDE_source(ax[j, i], dx, var, z, colors[iz], add_IQR, axis)

            # Plot extras.
            ax[j, i].set_title('{}'.format(ltr[i + j * nc]), loc='left')
            ax[j, i].set_xlabel('{} [{}]'.format(name, units))
            ax[j, 0].set_ylabel('Transport [Sv] / day')

            # Subplot ymax (excluding histogram)
            ymax = [max(y.get_ydata()) for y in list(ax[j, i].get_lines())]
            ymax = max(ymax[1::5]) if add_IQR else max(ymax[1::2])
            ymax += (ymax * 0.05)
            if i > 0:
                ymax = max([ymax, ax[j, i - 1].get_ybound()[-1]])
            ax[j, i].set_ylim(0, ymax)
            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            if i == nc - 1:
                ax[j, nc - 1].legend()  # Last col.

    fig.subplots_adjust(wspace=0.1, hspace=0.3, top=1)
    plt.savefig(paths.figs / 'fe_model/KDE_{}.png'.format(var), dpi=350, bbox_inches='tight')
    plt.show()
    return


def source_KDE_depth(ds, add_IQR=False):
    """Plot variable KDE at each longitude."""
    z_inds = [1, 2, 3, 7, 8]  # Source subsets.
    colors = ['k', 'darkviolet']

    nr, nc = len(z_inds), 4
    fig, ax = plt.subplots(nr, nc, figsize=(13, 18), sharey=1)

    for j, z in enumerate(z_inds):  # Rows: iterate through zones.
        for i, lon in enumerate(release_lons):  # Cols: iterate through lons.
            dx = ds.sel(x=lon)

            for k, var in enumerate(['depth', 'depth_at_src']):
                ax[j, i] = plot_KDE_source(ax[j, i], dx, var, z, colors[k], add_IQR, axis='y',
                                           kde_kws=dict(bw_adjust=[2, 1.5][k]))

            # Plot extras.
            ax[j, 0].set_title('{} {}'.format(ltr[j], zones.names[z]), loc='left')

            # Axis
            ax[j, i].set_xlabel('Transport [Sv] / day' if j == nr - 1 else None)
            ax[j, i].set_ylabel('Depth [m]' if i == 0 else None)
            ax[j, i].xaxis.set_major_formatter('{x:.2f}')  # Limit decimal places.
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

            # Subplot ymax (excluding histogram)
            ymax = [max(y.get_xdata()) for y in list(ax[j, i].get_lines())]
            ymax = max(ymax[1::5]) if add_IQR else max(ymax[1::2])
            ymax += (ymax * 0.02)
            if i > 0:
                ymax = max([ymax, ax[j, i - 1].get_xbound()[-1]])
            ax[j, i].set_ylim(400, 0)
            ax[j, i].set_xlim(1e-6, ymax)

            # Add ticks inside subplot.
            ax[j, i].tick_params(axis='y', direction='inout')

            # Add EUC lon inside subplot.
            ax[j, i].text(0.89, 0.1, '{}°E'.format(lon), ha='center',
                          va='center', transform=ax[j, i].transAxes)

    # Legend.
    nm = [mpl.lines.Line2D([], [], color=c, label=n, lw=5)
          for n, c in zip(['EUC', 'Source'], colors)][::-1]
    lgd = ax[0, 1].legend(handles=nm, bbox_to_anchor=(1.05, 1.1), loc=8, ncol=2)

    fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.6)
    plt.savefig(paths.figs / 'fe_model/KDE_depth.png', dpi=350, bbox_inches='tight')
    plt.show()
    return


if __name__ == '__main__':
    exp = ExpData(scenario=1, lon=250, version=0, file_index=7)
    pds = FelxDataSet(exp)
    ds = pds.fe_model_sources_all(add_diff=True)
    ds2 = FelxDataSet(ExpData(version=2)).fe_model_sources_all(add_diff=True)
    ds3 = FelxDataSet(ExpData(version=3)).fe_model_sources_all(add_diff=True)

    # # df = xr.open_dataset(exp.file_felx)
    # # dx = pds.fe_model_sources(add_diff=False)

    # # Weighted mean.
    # for var in ['fe_avg', 'fe_src_avg', 'fe_flux_sum', 'fe_scav_avg', 'fe_phy_avg', 'fe_reg_avg']:
    #     plot_var_scatter(pds, ds, var=var)
    #     source_bar_graph(pds, ds, var=var + '_src')

    # plot_var_scatter_and_depth(pds, ds, ds2, var='fe_avg')
    # plot_fe_multvar_scatter(pds, ds, ds3, scenario=0)
    # plot_fe_multvar_scatter(pds, ds, ds2, scenario=2)
    # plot_iron_at_source_scatter(pds, ds)

    # # Weighted mean depth profile @todo
    # plot_var_depth_profile(pds, ds)
    # plot_iron_at_source_scatter(pds, ds)

    # # KDE/Historgrams
    # # source_KDE_multi_lon(ds, var='age', add_IQR=False)
    # plot_particle_histogram(pds, ds, var='age')


    # # Plot start vs final dFe.
    # var2 = 'fe_scav'
    # var1 = 'age'
    # lon = 250
    # s = 2
    # dx = ds.sel(x=lon, exp=s)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=True)
    # ax.scatter(dx[var1], dx[var2], marker='.', c='k', s=20, alpha=0.2)  # dFe at source.
    # ax.set_title('{} EUC at {}'.format(scenario_name[s], lon), loc='left')
    # ax.set_xlabel('{} [{}]'.format(*[dx[var1].attrs[s] for s in ['long_name', 'units']]))
    # ax.set_ylabel('{} [{}]'.format(*[dx[var2].attrs[s] for s in ['long_name', 'units']]))
    # if s == 2:
    #     ax.axhline(0, c='k')
    #     ax.axvline(0, c='k')
    # plt.tight_layout()
    # plt.savefig(cfg.paths.figs / 'felx/dFe_scatter_source_EUC_{}.png'.format(pds.exp.id))
    # plt.show()
