# -*- coding: utf-8 -*-
"""Plot iron model output.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun  7 18:19:06 2023

"""
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import xarray as xr  # NOQA

from cfg import paths
from tools import unique_name


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
    plt.savefig(paths.figs / 'felx/dFe_paths_{}_{}.png'.format(pds.exp.file_base, ntraj),
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
    df_h = dfs.Huang_iron_dataset().fe.sel(x=pds.exp.lon, method='nearest').sel(
        y=slice(-2.6, 2.6)).mean('y')
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
    ax[1].plot(df_h.isel(t=-1), df_h.z, c='r', label='Huang et al. (2022)')
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
    file = unique_name(paths.figs / 'felx/iron_z_profile_{}.png'.format(pds.exp.file_base))
    plt.savefig(file, bbox_extra_artists=(lgd, title,), bbox_inches='tight')
    plt.show()

    # # Plot start vs final dFe.
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=True)
    # ax.scatter(ds_i.fe, ds_f.fe, c='k', s=12)  # dFe at source.
    # ax.set_title('{} EUC dFe at source vs EUC at {}'.format(pds.exp.scenario_name, pds.exp.lon_str), loc='left')
    # ax.set_ylabel('EUC dFe [nM]')
    # ax.set_xlabel('Source dFe [nM]')
    # plt.tight_layout()
    # plt.savefig(cfg.paths.figs / 'felx/dFe_scatter_source_EUC_{}.png'.format(pds.exp.file_base))
    # plt.show()
    return
