# -*- coding: utf-8 -*-
"""Plot iron model output.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun  7 18:19:06 2023

"""
from cartopy.mpl.gridliner import LatitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import xarray as xr  # NOQA

from cfg import paths, ExpData, ltr, release_lons, scenario_name, scenario_abbr, zones
from datasets import ofam_clim
from fe_exp import FelxDataSet
from fe_obs_dataset import iron_source_profiles
from stats import get_min_weighted_bins, test_signifiance
from tools import unique_name

warnings.filterwarnings("ignore")

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
    plt.savefig(paths.figs / 'model_test/dFe_paths_{}_{}.png'.format(pds.exp.id, ntraj),
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
    file = unique_name(paths.figs / 'model_test/iron_z_profile_{}.png'.format(pds.exp.id))
    plt.savefig(file, bbox_extra_artists=(lgd, title,), bbox_inches='tight')
    plt.show()

    # # Plot start vs final dFe.
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=True)
    # ax.scatter(ds_i.fe, ds_f.fe, c='k', s=12)  # dFe at source.
    # ax.set_title('{} EUC dFe at source vs EUC at {}'.format(pds.exp.scenario_name, pds.exp.lon_str), loc='left')
    # ax.set_ylabel('EUC dFe [nM]')
    # ax.set_xlabel('Source dFe [nM]')
    # plt.tight_layout()
    # plt.savefig(cfg.paths.figs / 'model_test/dFe_scatter_source_EUC_{}.png'.format(pds.exp.id))
    # plt.show()
    return


def scatter_var(pds, ds, var):
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
                            label='{} {}'.format(['EUC', 'Source'][j], scenario_name[s]))
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


def bar_graph_sources(pds, ds, var='u_sum_src'):
    """Bar graph of source transport for each release longitude.

    Horizontal bar graph (sources on y-axis) with or without RCP8.5.
    4 (2x2) subplots for each release longitude.

    Args:
        pds ():
        var (str, optional):

    """
    z_ids = [1, 2, 6, 7, 8, 3, 4, 5, 0]

    ds = ds.sel(zone=z_ids)  # Must select source order (for late use of ds)

    xlim_max = ds[var].mean('rtime').max()
    xlim_max += xlim_max * 0.16
    ylabels, c = ds.names.values, ds.colors.values

    # P-values
    pv = np.zeros((4, len(z_ids)))
    pct = np.full_like(pv, '', dtype=object)
    for i in range(4):
        for z in range(len(z_ids)):
            dx = ds[var].sel(x=pds.release_lons[i], zone=z_ids[z])
            pv[i, z] = test_signifiance(dx[0], dx[1], format_string=False)
            # pv = np.where(pv <= 0.05, 1, 0)  # 1 where significant.
            dx = dx.mean('rtime')
            diff = ((dx.isel(exp=1) - dx.isel(exp=0)) / dx.isel(exp=0)).item() * 100
            if pv[i, z] <= 0.05:
                if diff < 100:
                    pct[i, z] = '{0:+1.2g}%'.format(diff)
                else:
                    pct[i, z] = '{:+1.0f}%'.format(diff)

    width = 0.9  # Bar widths.
    kwargs = dict(alpha=0.7)  # Historical bar transparency.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row', sharex='all')

    for i, ax in enumerate(axes.flatten()):
        lon = pds.release_lons[i]
        dx = ds[var].sel(x=lon).mean('rtime')
        ticks = np.arange(dx.zone.size)  # Source y-axis ticks.

        ax.set_title('{} EUC {} at {}°E'.format(ltr[i], dx.attrs['long_name'], lon), loc='left')

        # Historical horizontal bar graph.
        ax.barh(ticks, dx.isel(exp=0), width, color=c, **kwargs)

        # RCP8.5: Hatched overlay.
        # ax.barh(ticks, dx.isel(exp=1), width, fill=False, hatch='//', label='RCP8.5', **kwargs)
        ax.barh(ticks, dx.isel(exp=1), width / 3, color='k', alpha=0.6, label='RCP8.5')

        g = ax.barh(ticks, dx.max('exp'), width, fill=False, alpha=0)

        ax.bar_label(g, labels=pct[i], label_type='edge', padding=5)

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

        ax.invert_yaxis()
        # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_tick_params(labelbottom=True)
    axes.flatten()[1].legend(loc='lower right', bbox_to_anchor=(1, 1.06))
    plt.tight_layout()
    var_ = var[:-8] if var in ['u_sum_src'] else '{}_v{}'.format(var[:-8], pds.exp.version)
    plt.savefig(paths.figs / 'fe_model/bar_{}.png'.format(var_), bbox_inches='tight', dpi=300)
    plt.show()
    return


def var_depth_profile(pds, ds):
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
            ax.plot(dx[var].isel(exp=j), dx.z, c='k', ls=['-', '--'][j], label=scenario_name[j])

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
    plt.savefig(paths.figs / 'fe_model/depth_profile_{}_v{}.png'.format(var, pds.exp.version),
                dpi=300)
    plt.show()


def scatter_iron_at_source(pds, ds):
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


def var_significance(ds, var, fill=[1, 0], stack_exp=True):
    """Get array that is 1 where change is significant.

    Args:
        ds (xarray.dataset): Source dataset.
        dvar (str): Variables [exp, rtime, *dims] (not traj).
        fill (list, optional): Fill values where significant/not. Defaults to [1, 0].


    Returns:
        sig (numpy.array): significance(*dims)

    """
    # P-values
    dims = [dim for dim in ds[var].dims if dim not in ['exp', 'rtime']]
    pval = np.zeros(tuple([ds[dim].size for dim in dims]))

    # Iterate through first dimension (e.g., longitude).
    for i in range(ds[dims[0]].size):
        dx = ds[var].isel({dims[0]: i})
        if len(dims) == 1:
            pval[i] = test_signifiance(dx[0], dx[1], format_string=False)
        else:
            # Iterate through second dimension (e.g., zone).
            for j in range(ds[dims[1]].size):
                dxx = dx.isel({dims[1]: j})
                pval[i, j] = test_signifiance(dxx[0], dxx[1], format_string=False)

    sig = pval.copy()
    if fill is not None:
        # pval = np.where(pval <= 0.05, fill[0], fill[1])  # 1 where significant.
        sig[pval <= 0.05] = fill[0]  # 1 where significant.
        sig[pval > 0.05] = fill[1]

    if stack_exp:
        t = np.ones((3, *list(sig.shape)))
        t[1:] = sig
        sig = t
    return sig


def dfe_mean_flux_depth(pds, ds, ds1, ds2):
    """Plot dFe mean, dFe flux & transport, dFe depth profiles at each release longitude.

    Args:
        pds (FelxDataSet instance): Iron model info.
        ds (xarray.dataset): Source dataset (multi-lon) for obs-profile dFe.
        ds1 (xarray.dataset): Source dataset (multi-lon) for proj-adjusted dFe.
        ds2 (xarray.dataset): Source dataset (multi-lon) for depth-mean dFe.
    """
    ds_all = [ds, ds1, ds2]
    ds_fe = iron_source_profiles()
    ds_fe = ds_fe.sel(z=slice(0, 350))
    ds_fe = ds_fe.sel(x=slice(160, 300)).dropna('x', 'all')
    version_colors = ['k', 'k', 'b', 'tab:red']
    ls = [['-', ':'], ['-', '--'], ['-', ':']]  # [version, scenario]
    lw = 1.5  # linewidth for model outout

    # ---------------------------------------------------------------------------------------
    fig, axes = plt.subplot_mosaic("AABB;CDEF", figsize=(12, 11), height_ratios=[2.75, 4],
                                   gridspec_kw={"wspace": 0.22, "hspace": 0.14})
    var = 'fe_avg'
    ax = axes['A']

    # Legend Historical/RCP linestyles.
    for i, name in enumerate(['Historical', 'RCP8.5', 'RCP8.5 (Proj-Adjusted)']):
        ax.plot([0], [0], c='k', ls=['-', ':', '--'][i], lw=2, label=name)

    # Add observations.
    ax.plot(ds_fe.x, ds_fe.EUC.mean('z'), 'x', ms=7, c='g', lw=1, label='Obs.')
    ax.plot(ds_fe.lon, ds_fe.euc_avg.mean('z'), 'o', ms=6, c='g', lw=1, label='Obs. Mean')

    for i, dx in enumerate(ds_all):
        dx = dx[var].mean('rtime')
        for s in [0, 1]:
            ax.plot(dx.x, dx.isel(exp=s), ls=ls[i][s], c=version_colors[i], lw=lw, zorder=4 - i)

    ax.set_title('{} EUC Mean dFe'.format(ltr[0]), loc='left')
    ax.set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.set_xticks(dx.x)
    ax.set_xlim(161, 270)
    ax.set_ylim(0.01)

    # Legend experiment version colors.
    for i, name in enumerate(['dFe Obs. Depth-Profile', 'dFe Obs. Depth-Mean', 'Transport']):
        ax.plot([0], [0], c=version_colors[i + 1], lw=5, label=name)
    handles, labels = ax.get_legend_handles_labels()

    # Variable legend.
    fig.legend(handles[:5], labels[:5], ncols=5, loc='upper center', bbox_to_anchor=(0.5, 0.93))
    # Experiment legend.
    fig.legend(handles[5:], labels[5:], ncols=3, markerscale=1.4, handlelength=3.7,
               loc='upper center', bbox_to_anchor=(0.5, 0.957), numpoints=2)

    # ---------------------------------------------------------------------------------------
    var = 'fe_flux_sum'
    ax = axes['B']
    for i, dx in enumerate(ds_all):
        dx = dx[var].mean('rtime')
        for s in [0, 1]:
            ax.plot(dx.x, dx.isel(exp=s), ls=ls[i][s], c=version_colors[i], label=None,
                    lw=1.2, zorder=4 - i)

    ax.set_title('{} EUC dFe Flux and Transport'.format(ltr[1]), loc='left')
    ax.set_ylabel('dFe Flux [{}]'.format(dx.attrs['units']), labelpad=0)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_tick_params(pad=0)
    ax.set_xticks(dx.x)
    ax.set_xlim(162, 253)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    var = 'u_sum'
    color = 'tab:red'
    ax2.set_ylabel('Transport [Sv]', color=color)  # we already handled the x-label with ax1
    dx = ds[var].mean('rtime')
    for s in [0, 1]:
        ax2.plot(dx.x, dx.isel(exp=s), ls=ls[0][s], c=color, lw=lw, label='Transport',
                 zorder=4 - i)
    ax2.tick_params(axis='y', labelcolor=color)

    # ---------------------------------------------------------------------------------------
    var = 'fe_avg_depth'
    for i, a in enumerate(['C', 'D', 'E', 'F']):
        ax = axes[a]
        ax.set_title('{} EUC Mean dFe at {}°E'.format(ltr[i + 2], release_lons[i]), loc='left')

        for j, dx in enumerate(ds_all):
            dx = dx.isel(x=i).mean('rtime')
            lon = dx.x.item()
            for s in range(2):
                ax.plot(dx[var].isel(exp=s), dx.z, c=version_colors[j], ls=ls[j][s],
                        lw=lw, zorder=5 - j)

        # Plot observations.
        ax.plot(ds_fe.euc_avg.isel(lon=i), ds_fe.z, 'o', ms=3, c='g', lw=0.8, label='Obs. Mean',
                zorder=1)
        # obs = ds_fe.EUC.sel(x=slice(lon - 5, lon + 5))
        # if obs.x.size > 0:
        #     for j in obs.x:
        #         ax.plot(obs.sel(x=lon), ds_fe.z, 'x', ms=5, c='g', lw=1, mew=0.8,
        #                 label='Obs.', zorder=1)

        # x-axis (dFe) and y-axis (depth)
        ax.set_xlabel('{} [{}]'.format(dx[var].attrs['standard_name'], dx[var].attrs['units']))
        ax.set_ylim(*[dx[dx[var].dims[-1]][a] for a in [-1, 0]])
        ax.set_xlim(0, 1.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%dm'))
        if a != 'C':
            ax.set_yticklabels([])
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.tick_params(which='major', length=5)

    plt.savefig(paths.figs / 'fe_model/dfe_mean_flux_depth.png', dpi=300, bbox_inches='tight')
    plt.show()


def fe_multivar(pds, ds):
    """Plot all fe variables (time mean) for historical & RCP (left) and projected change(right).

    Args:
        pds (FelxDataSet instance): Iron model info.
        ds (xarray.dataset): Source dataset (multi-lon) fe_model_sources_all(add_diff=True)

    """
    dvars = ['fe_avg', 'fe_src_avg', 'fe_reg_avg', 'fe_scav_avg', 'fe_phy_avg']
    titles = ['Historical and RCP-8.5 EUC dFe ', 'EUC dFe Projected Change']
    vlabels = [ds[var].attrs['long_name'] for var in dvars]
    colors = ['k', 'darkviolet', 'blue', 'r', 'g']  # for each fe variable
    lw = 2  # linewidth
    ls = [['o-', 'o-', 'd-.'],  # Historical & RCP markers [scenario, version]
          ['.:', '.-', 'd-.'],
          ['o-', '.--', 'd-.']]

    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=False)
    ax = ax.flatten()

    for i, scenario in enumerate([0, 2]):
        ax[i].set_title('{} {}'.format(ltr[i], titles[i]), loc='left')
        for j, var in enumerate(dvars):
            for k, dx in enumerate([ds]):
                pv = var_significance(dx, var, fill=[np.nan, 1], stack_exp=True)
                dx = dx[var].mean('rtime')
                dx = dx * -1 if var in ['fe_scav_avg', 'fe_phy_avg'] else dx

                label = vlabels[j] if k == 0 else None
                kwargs = dict(c=colors[j], lw=lw)

                if scenario == 0:
                    ax[i].plot(dx.x, dx.sel(exp=0), ls[0][k], label=label, zorder=5 - j, **kwargs)
                    ax[i].plot(dx.x, dx.isel(exp=1), ls[1][k], markersize=10, **kwargs)
                    ax[i].plot(dx.x, dx.isel(exp=1) * pv[1], ls[1][k][0], markersize=10,
                               markerfacecolor='w', zorder=10, **kwargs)
                else:
                    ax[i].plot(dx.x, dx.sel(exp=1) - dx.sel(exp=0), ls[2][k], label=label,
                               zorder=5 - j, **kwargs)
                    ax[i].plot(dx.x, (dx.sel(exp=1) - dx.sel(exp=0)) * pv[2], ls[2][k][0],
                               markerfacecolor='w', zorder=10, **kwargs)

            ax[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
            ax[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[i].set_xticks(dx.x)
            ax[i].set_xlim(160, 255)  # Needed because of extra legend markers at (0, 0).
            ax[i].axhline(0, c='grey', lw=1)
            if i % 3 == 0:
                ax[0].set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']))

    # Legend Historical/RCP linestyles.
    for i, name in enumerate(['Historical', 'RCP-8.5']):
        ax[0].plot([0], [0], ls[i][0], c='k', lw=2, label=name)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles[:5], labels[:5], ncols=5, loc='upper center', bbox_to_anchor=(0.53, 1.05))
    ax[0].legend(handles[5:], labels[5:], ncols=1, fontsize='small', loc='center right',
                 bbox_to_anchor=(1, 1.05))
    plt.tight_layout()
    plt.savefig(paths.figs / 'fe_model/fe_multi_var_v{}.png'.format(pds.exp.version),
                dpi=300, bbox_inches='tight')


def fe_multivar_sources(pds, ds, ds1, ds2, scenario=2, version_inds=[0, 1, 2], all_sources=False):
    """Plot all fe model variables as a function of longitude (EUC & sources).

    Args:
        pds (FelxDataSet instance): Iron model info.
        ds (xarray.dataset): Source dataset (multi-lon) for obs-profile dFe.
        ds1 (xarray.dataset): Source dataset (multi-lon) for proj-adjusted dFe.
        ds2 (xarray.dataset): Source dataset (multi-lon) for depth-mean dFe.
        scenario (int, optional): Scenario index. Defaults to 2.

    Notes:
        - Scenario = 0: Plot historical only (obs & depth-mean)
        - Scenario = 1: Plot historical & RCP (obs, proj-adjusted & (?)depth-mean)
        - Scenario = 2: Plot projected change (obs, proj-adjusted & depth-mean)
    """
    # Source regions.
    z_ids = [1, 2, 6, 3, 4, 7, 8] if all_sources else [1, 2, 3, 7, 8]
    # Titles
    # if all_sources:
    names = ['EUC', *['{}'.format(s) for s in ds.names.sel(zone=z_ids).values]]

    # names = ['EUC dFe', *['EUC dFe from the {}'.format(s)
    #                       for s in ds.names.sel(zone=z_ids).values]]
    dvars = ['fe_avg', 'fe_src_avg', 'fe_reg_avg', 'fe_scav_avg', 'fe_phy_avg']  # Variables
    vlabels = [ds[var].attrs['long_name'] for var in dvars]  # Variable names
    colors = ['k', 'darkviolet', 'blue', 'r', 'g']  # Variable colours
    version_labels = ['Obs. Depth-Profile', 'Projection-Adjusted Obs. Depth-Profile',
                      'Obs. Depth-Mean']

    # version_inds = [0, 1, 2] if scenario in [1, 2] else [0, 2]
    ds_list = [ds, ds1, ds2]

    # Linestyles ls[exp][fe-version: obs-profile, proj-adjusted, depth-mean]
    ls = [['o-', '*--', 'd-.'],
          ['.:', '*--', 'd-.'],
          ['o-', '*--', 'd-.']]

    # Plot figure.
    nr = 2 if all_sources else 1
    nc = int((len(z_ids) + 1) / nr)# + 1
    figsize = (11, 12) if all_sources else (13, 7)

    fig, ax = plt.subplots(nr, nc, figsize=figsize, sharex=False, sharey=1,
                           gridspec_kw=dict(wspace=0.12, hspace=0.13))
    # fig, ax = plt.subplots(nr, nc, figsize=(13, 8), sharex=False, sharey=0,
    #                        gridspec_kw=dict(wspace=0.12, width_ratios=[1, 0.12, 1, 1, 1, 1, 1]))
    ax = ax.flatten()
    # ax[1].set_visible(False)

    for i, z in enumerate([None, *z_ids]):
        ax[i].set_title('{} {}'.format(ltr[i], names[i]), loc='left')
        # if z != None:
            # i += 1
            # ax[i].set_title('{} {}'.format(ltr[i-1], names[i-1]), loc='left', x=-0.05)
        # elif z != 1:
            # ax[i].set_title('{} {}'.format(ltr[i], names[i]), loc='left')

        for j, var in enumerate(dvars):
            zorder = 5 - j  # Last to plot on bottom.
            v = var + '_src' if i > 0 else var

            for k, vid in enumerate(version_inds):
                dx = ds_list[vid]
                if z is not None:
                    dx = dx.sel(zone=z)

                pv = var_significance(dx, v, fill=[np.nan, 1], stack_exp=True)
                dx = dx[v].mean('rtime')

                dx = dx * -1 if var in ['fe_scav_avg', 'fe_phy_avg'] else dx  # flip sign
                label = vlabels[j] if k == 0 else None  # Only plot v0 labels
                lw = 1.5 if var == 'fe_avg' else 1.2  # Thicker dFe
                kwargs = dict(c=colors[j], lw=lw)

                # Historical
                if scenario in [0, 1]:
                    # (except for v1 interior change).
                    if vid != 1:
                        ax[i].plot(dx.x, dx.sel(exp=scenario), ls[0][vid], zorder=zorder,
                                   label=label, **kwargs)

                    # RCP85 (except for v1 interior change).
                    if scenario == 1 and not (vid == 1 and z in [7, 8]):
                        ax[i].plot(dx.x, dx.isel(exp=1), ls[1][vid], zorder=zorder, **kwargs)
                        ax[i].plot(dx.x, dx.isel(exp=1) * pv[1], ls[1][vid], markerfacecolor='w',
                                   zorder=zorder, **kwargs)

                # Projected Change (except for v1 interior change or v2 source changes).
                elif (not (vid == 2 and z is not None and var == 'fe_src_avg')
                      and not (vid == 1 and z in [7, 8])):
                    ax[i].plot(dx.x, dx.sel(exp=1) - dx.sel(exp=0), ls[2][vid], label=label,
                               zorder=zorder, **kwargs)
                    ax[i].plot(dx.x, (dx.sel(exp=1) - dx.sel(exp=0)) * pv[2], ls[2][vid],
                               markerfacecolor='w', zorder=zorder, **kwargs)

        ax[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d°E'))
        ax[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax[i].set_xticks(dx.x)
        ax[i].set_xlim(160, 257)  # Needed because of extra legend markers at (0, 0).
        ax[i].axhline(0, c='grey', lw=1)
        # if all_sources:
        ax[i].tick_params(axis='both', which='major', labelsize='small', length=4)
        # if nr == 1:
        #     ax[i].tick_params(axis='x', which='major', labelsize='small', length=5)
        #     ax[i].tick_params(axis='y', which='major', labelsize='small', length=4)

        if i % nc == 0:
            ax[i].set_ylabel('{} [{}]'.format(dx.attrs['standard_name'], dx.attrs['units']),
                             labelpad=-4)
        ax[i].yaxis.set_tick_params(pad=0)
    # if nr == 1:
    #     for i, z in enumerate(z_ids):
    #         ylim = [[-0.85, 1.26], [-0.85, 1.4], [-0.2, 0.21]]
    #         ax[i+1].set_ylim(*ylim[scenario])
    #         if i > 0:
    #             ax[i+1].set_yticklabels([])

    # Legend.
    # Legend variable colors (no markers).
    for j, var in enumerate(dvars):
        ax[0].plot([0], [0], c=colors[j], lw=5, label=vlabels[j])

    # Legend experiment version linestyles.
    for i in version_inds:
        ax[0].plot([0], [0], ls[0][i], c='k', lw=2, label=version_labels[i])
    handles, labels = ax[0].get_legend_handles_labels()

    # Variable legend.
    c = -0.04 if all_sources else 0
    fig.legend(handles[5:10], labels[5:10], ncols=5, loc='upper center', bbox_to_anchor=(0.5, 1.03+c*1.5))
    # Experiment legend.
    fig.legend(handles[10:], labels[10:], ncols=3, markerscale=1.4, handlelength=3.7,
               loc='upper center', bbox_to_anchor=(0.5, 0.98+c), numpoints=2)

    plt.tight_layout()
    if all_sources:
        fig.subplots_adjust(wspace=0.22)
    # if nr == 1:
        # fig.subplots_adjust(wspace=0.2)
    plt.savefig(paths.figs / 'fe_model/fe_multi_var_{}_v{}_sources_{}.png'
                .format('-'.join([str(x) for x in version_inds]), 'all_' if all_sources else '',
                        scenario_abbr[scenario]),
                dpi=300, bbox_inches='tight')



def get_histogram(ax, dx, var, color, bins='fd', cutoff=0.85, weighted=True,
                  outline=True, median=False, **plot_kwargs):
    """Plot histogram with historical (solid) & projection (dashed).

    Histogram bins weighted by transport / sum of all transport.
    This gives the probability (not density)

    Args:
        ax (plt.AxesSubplot): Axes Subplot.
        dx (xarray.Dataset): Dataset (hist & proj; var and 'u').
        var (str): Data variable.
        color (list of str): Bar colour.
        xlim_percent (float, optional): Shown % of bins. Defaults to 0.75.
        weighted (bool, optional): Weight by transport. Defaults to True.

    Returns:
        ax (plt.AxesSubplot): Axes Subplot.

    """
    kwargs = dict(histtype='stepfilled', density=0, range=None, stacked=False,
                  alpha=0.6, cumulative=False, color=color[0], lw=0.8,
                  hatch=None, edgecolor=color[0], orientation='vertical')

    # Update hist kwargs based on input.
    for k, p in plot_kwargs.items():
        kwargs[k] = p

    dx = [dx.isel(exp=i).dropna('traj', 'all') for i in [0, 1]]

    weights = None
    if weighted:
        weights = [dx[i].u / dx[i].rtime.size for i in [0, 1]]

    if weighted and isinstance(bins, str):
        bins = get_min_weighted_bins([d[var] for d in dx], weights)

    # Historical.
    y1, x1, _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)

    # RCP8.5.
    bins = bins if weighted else y1
    kwargs.update(dict(color=color[1], alpha=0.3, edgecolor=color[1]))
    if color[0] == color[-1]:
        kwargs.update(dict(linestyle=(0, (1, 1))), alpha=1, fill=0, edgecolor=color[-1])
    y2, x2, _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

    if outline:
        # Black outline (Hist & RCP).
        kwargs.update(dict(histtype='step', color='k', alpha=1))
        _ = ax.hist(dx[0][var], bins, weights=weights[0], **kwargs)
        _ = ax.hist(dx[1][var], bins, weights=weights[1], **kwargs)

    # Median & IQR.
    if median:
        axline = ax.axvline if kwargs['orientation'] == 'vertical' else ax.axhline
        for q, ls in zip([0.25, 0.5, 0.75], ['--', '-', '--']):
            # Historical
            axline(x1[sum(np.cumsum(y1) < (sum(y1) * q))], c=color[0], ls=ls)
            # RCP8.5
            axline(x2[sum(np.cumsum(y2) < (sum(y2) * q))], c='k', ls=ls, alpha=0.8)

    # Cut off last 5% of xaxis (index where <95% of total counts).
    if cutoff is not None:
        xmax = x1[sum(np.cumsum(y1) < sum(y1) * cutoff)]
        xmin = x1[max([sum(np.cumsum(y1) < sum(y1) * 0.01) - 1, 0])]
        ax.set_xlim(xmin=xmin, xmax=xmax)
    return ax


def particle_histogram(pds, ds, var='age'):
    """Histograms of variables for each source (rows) and longitude (columns).

    Args:
        pds (FelxDataSet instance): Iron model info.
        ds (xarray.dataset): Source dataset (multi-lon; any version; add_diff=False).
        var (str, optional): Variable to plot. Defaults to 'age'.

    """
    kwargs = dict(bins='fd', cutoff=0.7, median=True)
    # xmax = 1000 if var == 'age' else 600

    fig, ax = plt.subplots(7, 4, figsize=(14, 15), sharey='row')

    # Iterate through longitudes for each column.
    for i, x in enumerate(release_lons):

        # Iterate through source regions for each row.
        for j, z in enumerate([1, 2, 6, 3, 4, 7, 8, 5, 0][:ax.shape[0]]):

            # Subset dataset.
            dx = ds.sel(x=x, zone=z).dropna('traj', 'all')
            c = [zones.colors[z], 'k']

            ax[j, i] = get_histogram(ax[j, i], dx, var, c, **kwargs)

            ax[j, 0].set_title('{} {}'.format(ltr[j], zones.names[z]), loc='left')
            ax[j, 0].set_ylabel('Transport [Sv] / day')  # First column.

            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

            # Add ticks inside subplot.
            ax[j, i].tick_params(which='major', direction='inout', right=True, length=7)
            ax[j, i].tick_params(which='minor', direction='inout', right=True, length=4)

        # Add the longitude above each column.
        ax[0, i].text(0.25, 1.25, 'EUC at {}°E'.format(x), weight='bold',
                      transform=ax[0, i].transAxes)

        # Add units at the bottom of each column.
        ax[-1, i].set_xlabel('{} [{}]'.format(dx[var].attrs['long_name'], dx[var].attrs['units']))

    fig.subplots_adjust(wspace=0.1, hspace=0.4)
    var_ = var if var in ['age', 'distance'] else '{}_v{}.png'.format(var, pds.exp.version)
    plt.savefig(paths.figs / 'fe_model/histogram_{}.png'.format(var_), bbox_inches='tight',
                dpi=300)
    plt.show()


def particle_histogram_depth(pds, ds):
    """Histograms of single source variables."""
    kwargs = dict(bins=np.arange(0, 500, 25), cutoff=None, fill=True, lw=1.3, edgecolor='none',
                  orientation='horizontal', histtype='step', alpha=0.25, outline=False)
    colors = ['darkviolet', 'k']

    # # OFAM3 climatology (for EUC core depth).
    # clim = ofam_clim(['u', 'v']).mean('time')
    # euc_clim = clim.u.sel(lat=slice(-2.6, 2.6), lev=slice(2.5, 400))
    # euc_z_max = euc_clim.max('lat').idxmax('lev')
    # clim['u'] = clim.u.sel(lat=slice(-2.6, 2.6), lev=slice(2.5, 400))
    # z_max = clim.max('lat').idxmax('lev')

    nr, nc = 5, 4
    fig, ax = plt.subplots(nr, nc, figsize=(11, 10), sharey='row')

    for i, x in enumerate(release_lons):
        for j, z in enumerate([1, 2, 3, 7, 8, 5, 0][:nr]):
            # Subset dataset.
            dx = ds.sel(x=x, zone=z).dropna('traj', 'all')

            c = [colors[1]] * 2
            ax[j, i] = get_histogram(ax[j, i], dx, 'depth_at_src', c, **kwargs)

            c = [colors[0]] * 2
            ax[j, i] = get_histogram(ax[j, i], dx, 'depth', c, **kwargs)

            # # Add EUC max depth line.
            # for s, ls in zip(range(2), ['-', ':']):
            #     ax[j, i].axhline(z_max.sel(lon=x).isel(time=s).item(), c='k', ls=ls, lw=1)

            # Axes.
            if i == 0:  # First column.
                ax[j, 0].set_title('{} {}'.format(ltr[j], zones.names[z]), loc='left')
            ax[nr - 1, i].set_xlabel('Transport [Sv] / day')  # Last row.
            ax[j, i].set_ymargin(0)
            ax[j, i].set_ylim(400, 0)  # Flip yaxis 0m at top.

            ax[j, i].set_xmargin(0.10)
            ax[j, i].set_xlim(0)
            xticks = ax[j, i].get_xticks()
            if i > 0:
                ax[j, i].set_xticks(xticks[1:])  # limit to avoid overlap.

            ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax[j, i].yaxis.set_major_formatter('{x:.0f}m')

            # Add ticks inside subplot.
            ax[j, i].tick_params(which='major', direction='inout', right=True, length=7)
            ax[j, i].tick_params(which='minor', direction='inout', right=True, length=4)

            # Add EUC lon inside subplot.
            ax[j, i].text(0.89, 0.1, '{}°E'.format(x), ha='center', va='center',
                          transform=ax[j, i].transAxes)

    # Legend.
    ln = [mpl.lines.Line2D([], [], c=colors[i], label=['EUC', 'Source'][i], lw=5) for i in [0, 1]]
    lgd = fig.legend(handles=ln, ncols=2, bbox_to_anchor=(0.5, 0.965), loc=8, fontsize='large')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.35)
    plt.savefig(paths.figs / 'fe_model/histogram_depth.png', bbox_inches='tight', dpi=300,
                bbox_extra_artists=(lgd,))
    plt.show()


def get_KDE_source(ax, ds, var, z, color=None, add_IQR=False, axis='x',
                   kde_kws=dict(bw_adjust=0.5), stat='count'):
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
        ls = ['-', (0, (1, 1))][s]  # Linestyle
        lw = [1, 1.3][s]
        n = dx.names.item() if s == 0 else None  # Legend name.

        ax = sns.histplot(**{axis: dx[var]}, weights=dx.u / dx.rtime.size, ax=ax,
                          bins=bins, color=c, stat=stat, element='step', alpha=0,
                          fill=False, kde=True, kde_kws=kde_kws,
                          line_kws=dict(color=c, linestyle=ls, label=n, linewidth=lw))
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


def KDE_multi_var(ds, lon, var, add_IQR=False):
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
                ax[i, j] = get_KDE_source(ax[i, j], ds, v, z, c, add_IQR, axis)

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
    plt.savefig(paths.figs / 'fe_model/KDE_{}_{}.png'.format('_'.join(var), lon), dpi=300)
    plt.show()
    return


def KDE_source_multi_lon(ds, var='age', stat='count', add_IQR=False):
    """Plot variable KDE at each longitude."""
    z_inds = [[1, 2, 6], [3, 4], [7, 8]]  # Source subsets.
    if stat != 'count':
        z_inds[0] = z_inds[0][:2]
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
                ax[j, i] = get_KDE_source(ax[j, i], dx, var, z, colors[iz], add_IQR, axis, stat=stat)

            # Plot extras.
            ax[j, i].set_title('{}'.format(ltr[i + j * nc]), loc='left')
            ax[j, i].set_xlabel('{} [{}]'.format(name, units))
            ax[j, 0].set_ylabel('Transport [Sv] / day' if stat == 'count' else stat.capitalize())

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
    plt.savefig(paths.figs / 'fe_model/KDE_{}_{}.png'.format(var, stat), dpi=300,
                bbox_inches='tight')
    plt.show()
    return


def KDE_source_depth(ds, stat='count', add_IQR=False):
    """Plot variable KDE at each longitude."""
    z_inds = [1, 2, 3, 7, 8]  # Source subsets.
    colors = ['k', 'darkviolet']

    nr, nc = len(z_inds), 4
    fig, ax = plt.subplots(nr, nc, figsize=(12, 19), sharey=1)

    for j, z in enumerate(z_inds):  # Rows: iterate through zones.
        for i, lon in enumerate(release_lons):  # Cols: iterate through lons.
            dx = ds.sel(x=lon)

            for k, var in enumerate(['depth', 'depth_at_src']):
                ax[j, i] = get_KDE_source(ax[j, i], dx, var, z, colors[k], add_IQR, axis='y',
                                          stat=stat, kde_kws=dict(bw_adjust=[2, 1.5][k]))

            # Plot extras.
            ax[j, 0].set_title('{} {}'.format(ltr[j], zones.names[z]), loc='left')

            # Axis
            if j == nr - 1:
                ax[j, i].set_xlabel('Transport [Sv] / day' if stat == 'count' else stat.capitalize())
            else:
                ax[j, i].set_xlabel(None)
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
    ax[0, 1].legend(handles=nm, bbox_to_anchor=(1.05, 1.1), loc=8, ncol=2)

    fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.6)
    plt.savefig(paths.figs / 'fe_model/KDE_depth_{}.png'.format(stat), dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_yz_profile_EUC_sources(ds, lon, scenario, var='u'):
    """Plot 2D lat-depth histogram profile at a longitude and for a scenario.

    Args:
        ds (xarray.Dataset): Particle source dataset (any version).
        lon (int): Release Longitude.
        scenario (int): Scenario index.
        var (str, optional): Variable to plot (traj, obs). Defaults to 'u'.
    """
    ds = ds.sel(zone=[0, 1, 2, 6, 3, 4, 7, 8, 5])
    dss = ds.drop([v for v in ds.data_vars if v not in ['u', var, 'lat', 'depth']])

    names = ds.names.values
    names[0] = 'EUC'  # Replace zone=0 with overall EUC

    # Stack transport by lat and depth then sum over trajectories.
    data = [[], []]
    # Iterate over scenarios.
    for s in [0, 1]:
        dxx = dss.sel(exp=s, x=lon).dropna('traj', 'all')
        # Overall EUC (replace zone=0 with overall EUC)
        dx = dxx.stack(t=['zone', 'traj'])
        dx = dx.dropna('t', 'all')
        dx = dx.set_index(xy=['t', 'lat', 'depth'])
        dx = dx.unstack('xy')
        data[s].append(dx[var].sum('t').expand_dims(dict(zone=[0])))

        # Iterate over source regions (exclude zone=0 if replacing with overall).
        for z in dss.zone.values[1:]:
            dx = dxx.sel(zone=z).dropna('traj', 'all')
            dx = dx.set_index(xy=['traj', 'lat', 'depth'])
            dx = dx.unstack('xy')
            data[s].append(dx[var].sum('traj'))

    # Concatenate sources, calculate projected change and concatenate scenarios.
    tmp = [xr.concat(data[s], 'zone') for s in [0, 1]]
    tmp.append((tmp[1].drop('exp') - tmp[0].drop('exp')).expand_dims(dict(exp=[2])))
    df = xr.Dataset()
    df[var] = xr.concat(tmp, 'exp') / ds.rtime.size

    # Contour climatology/overall EUC for reference.
    # clim = df.u.isel(exp=scenario if scenario != 2 else 0).isel(zone=0)
    cmap = plt.cm.plasma if scenario != 2 else plt.cm.seismic

    # Plot profile (each source).
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharey=1, sharex='row')
    for i, ax in enumerate(axes.flatten()):
        dx = df.isel(exp=scenario).isel(zone=i)
        if scenario == 2:
            vlim = [max(np.fabs([dx[var].min(), dx[var].max()])) * c for c in [-1, 1]]
        else:
            vlim = [0, max(np.fabs([df.isel(zone=i)[var].min(), df.isel(zone=i)[var].max()]))]
        cs = ax.pcolormesh(dx.lat, dx.depth, dx[var].T, cmap=cmap, vmax=vlim[1], vmin=vlim[0])
        cbar = fig.colorbar(cs, ax=ax)
        cbar.ax.tick_params(labelsize='small')

        # EUC contour (Contour where velocity is 50% of maximum EUC velocity.)
        for s in [0, 1]:
            clim = df[var].isel(exp=s).isel(zone=0)
            lvls = [clim.max().item() * c for c in [0.70]]
            ax.contour(clim.lat, clim.depth, clim.T, lvls, colors='k', linewidths=1.5,
                       linestyles=['-', '--'][s])

        # Axes.
        ax.set_title('{} {}'.format(ltr[i], names[i]), loc='left')
        ax.set_ylim(362.5, 12.5)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(LatitudeFormatter())
        ax.yaxis.set_major_formatter('{x:.0f}m')
        if (i + 1) % 3 == 0:  # Last column.
            cbar.set_label('[{}]'.format(ds[var].attrs['units']))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    plt.savefig(paths.figs / 'EUC_sources_yz_profile_{}_{}_{}.png'
                .format(var, lon, scenario_abbr[scenario]), bbox_inches='tight', dpi=300)
    plt.show()


# if __name__ == '__main__':
#     exp = ExpData(scenario=0, lon=250, version=4, file_index=0)
#     pds = FelxDataSet(exp)
#     ds = pds.fe_model_sources_all(add_diff=True)
#     ds1 = FelxDataSet(ExpData(version=5)).fe_model_sources_all(add_diff=True)
#     ds2 = FelxDataSet(ExpData(version=6)).fe_model_sources_all(add_diff=True)

#     # # df = xr.open_dataset(exp.file_felx)
#     # # dx = pds.fe_model_sources(add_diff=False)

#     # dfe_mean_flux_depth(pds, ds, ds1, ds2)

#     # # KDE/Historgrams
#     # KDE_source_multi_lon(ds, var='age', stat='density', add_IQR=False)
#     # KDE_source_multi_lon(ds, var='age', stat='count', add_IQR=False)
#     # KDE_source_depth(ds, stat='density', add_IQR=False)
#     # KDE_source_depth(ds, stat='count', add_IQR=False)
#     # particle_histogram(pds, ds, var='age')
#     # particle_histogram_depth(pds, ds)

#     # for version, ds_ in zip([4, 5, 6], [ds, ds1, ds2]):
#     #     pds_ = FelxDataSet(ExpData(version=version))
#     #     fe_multivar(pds_, ds_)

#     # EUC & source water dFe variables
#     for scenario in range(3):
#         fe_multivar_sources(pds, ds, ds1, ds2, version_inds=[0, 1, 2], scenario=scenario)
#         fe_multivar_sources(pds, ds, ds1, ds2, version_inds=[0, 1], scenario=scenario)
#         fe_multivar_sources(pds, ds, ds1, ds2, version_inds=[0, 2], scenario=scenario)
#         fe_multivar_sources(pds, ds, ds1, ds2, version_inds=[0, 1, 2], scenario=scenario, all_sources=True)

#     #     plot_yz_profile_EUC_sources(ds, 165, scenario, var='u')

#     # # Weighted mean.
#     # for var in ['fe_avg', 'fe_src_avg', 'fe_flux_sum', 'fe_scav_avg', 'fe_phy_avg', 'fe_reg_avg']:
#     #     # scatter_var(pds, ds, var=var)
#     #     bar_graph_sources(pds, ds, var=var + '_src')

#     # # Weighted mean depth profile
#     # var_depth_profile(pds, ds)
#     # scatter_iron_at_source(pds, ds)
#     # scatter_iron_at_source(pds, ds)
