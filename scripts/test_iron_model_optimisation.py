# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:
    # OFAM3.
    if cfg.test:
        fieldset = BGCFields(exp)
        ds_fe_ofam = fieldset.ofam_dataset(variables=['det', 'phy', 'zoo', 'fe'],
                                            decode_times=True, chunks='auto')
        ds_fe_ofam = ds_fe_ofam.rename({'depth': 'z', 'fe': 'Fe'})  # For SourceIron compatability.
        ds_fe = [ds_fe_obs, ds_fe_ofam][1]  # Pick source iron dataset.

        # Sample OFAM3 iron along trajectories (for testing).
        dim_map = fieldset.dim_map_ofam.copy()
        dim_map['z'] = dim_map.pop('depth')
        ds['Fe'] = pds.empty_DataArray(ds)
        fe = update_field_AAA(ds, ds_fe_ofam, 'Fe', dim_map=dim_map)
        ds['Fe'] = (['traj', 'obs'], fe)


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Apr 29 20:45:13 2023

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr  # NOQA

import cfg
from cfg import paths
from datasets import ofam_clim, BGCFields
from felx_dataset import FelxDataSet
from tools import mlogger, unique_name
from particle_BGC_fields import update_field_AAA

logger = mlogger('fe_model_test')


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
    plt.savefig(cfg.paths.figs / 'felx/dFe_paths_{}_{}.png'.format(pds.exp.file_base, ntraj),
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
    file = unique_name(cfg.paths.figs / 'felx/iron_z_profile_{}.png'.format(pds.exp.file_base))
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


def plot_OFAM3_fields():
    # OFAM3.
    ds = ofam_clim().mean('time')

    # KD490
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ds['kd'].plot(vmax=0.05, vmin=-1.7, cmap=plt.cm.rainbow)
    ax.set_title('Diffuse Attenuation Coefficient Climatology at 490 nm [m^-1]')


def plot_ofam_clim():
    var_list = ['phy', 'zoo', 'det', 'fe', 'no3', 'temp']
    ds = ofam_clim().mean('time')

    # Equator
    dx = ds.sel(lat=0, method='nearest').sel(depth=slice(0, 700))
    # dx.det.plot(yincrease=False, cmap=plt.cm.rainbow)

    # Define the figure and axis objects
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharey=True)
    axs = axs.flatten()
    for i, var in enumerate(var_list):
        ax = axs[i]
        im = ax.pcolormesh(dx.lon, dx.depth, dx[var], cmap=plt.cm.rainbow)

        # Add a color bar to the subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(dx[var].attrs['units'], rotation=270)
        ax.set_xlabel('Longitude [°E]')
        ax.set_xlabel('Depth [m]')
        ax.set_title(var)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/OFAM_hist_clim.png', bbox_inches='tight')
    plt.show()


def plot_ofam_clim_limits():
    exp = cfg.ExpData(scenario=0)
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    ds = ofam_clim().mean('time')

    # Equator
    dx = ds.sel(lat=0, method='nearest').sel(depth=slice(0, 400))
    N = dx['no3'] / (dx['no3'] + pds.k_N)
    Fe = dx['fe'] / (dx['fe'] + (pds.k_fe * 0.02))

    J_max = pds.a * pds.b**(pds.c * dx.temp)
    light = pds.PAR * pds.I_0 * np.exp(-dx.depth * dx.kd)
    J_lim = J_max * (1 - np.exp((-pds.alpha * light) / J_max))
    J = J_max * np.min([J_lim / J_max, N, Fe], axis=0)
    phy_up = 0.02 * J * dx.phy

    fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(13, 18), sharey=True, sharex=True)
    ax = ax.flatten()
    i = 0
    ax[i].set_title('NO3/(NO3+k_N)')
    im = ax[i].pcolormesh(dx.lon, dx.depth, N, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[no units]', rotation=270)

    i = 1
    ax[i].set_title('Fe/(Fe+k_Fe)')
    im = ax[i].pcolormesh(dx.lon, dx.depth, Fe, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[no units]', rotation=270)

    i = 2
    ax[i].set_title('J_max')
    im = ax[i].pcolormesh(dx.lon, dx.depth, J_max, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[no units]', rotation=270)

    i = 3
    ax[i].set_title('J/J_max')
    im = ax[i].pcolormesh(dx.lon, dx.depth, J_lim / J_max, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[no units]', rotation=270)

    i = 4
    ax[i].set_title('J=min(J/J_max, NO3/(NO3+k_N), Fe/(Fe+k_Fe))')
    im = ax[i].pcolormesh(dx.lon, dx.depth, J / J_max, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[no units]', rotation=270)

    i = 5
    ax[i].set_title('J=J_max*min(J/J_max, NO3/(NO3+k_N), Fe/(Fe+k_Fe))')
    im = ax[i].pcolormesh(dx.lon, dx.depth, J, cmap=plt.cm.rainbow, vmin=0)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[day^-1]', rotation=270)

    i = 6
    ax[i].set_title('Phyto uptake (J*phy*0.02)')
    im = ax[i].pcolormesh(dx.lon, dx.depth, phy_up, cmap=plt.cm.rainbow)
    cbar = fig.colorbar(im, ax=ax[i])
    cbar.ax.set_ylabel('[umol Fe m^-3 day^-1]', rotation=270)

    ax[i].set_ylim(375, 0)
    ax[i].set_xlim(130, 275)
    ax[-1].set_xlabel('Longitude [°E')
    ax[0].set_xlabel('Depth [m]')

    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/phyto_limiting.png', bbox_inches='tight')
    plt.show()


def plot_phyto_param_constants():
    def phyto_up(dx, a, b, c, PAR, I_0, alpha, k_N, k_fe):
        N = dx['no3'] / (dx['no3'] + k_N)
        Fe = dx['fe'] / (dx['fe'] + (k_fe * 0.02))
        J_max = a * b**(c * dx.temp)
        light = PAR * I_0 * np.exp(-dx.depth * dx.kd)
        J_lim = J_max * (1 - np.exp((-alpha * light) / J_max))
        J = J_max * np.min([J_lim / J_max, N, Fe], axis=0)
        phy_up = 0.02 * J * dx.phy
        return phy_up

    exp = cfg.ExpData(scenario=0)
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    ds = ofam_clim().mean('time')
    dx = ds.sel(lat=0, lon=exp.lon, method='nearest').sel(depth=slice(0, 180))

    constant_names = ['a', 'b', 'c', 'k_N', 'k_fe', 'PAR', 'I_0', 'alpha']
    a, b, c, k_N, k_fe, PAR, I_0, alpha = [pds.params[i] for i in constant_names]

    colors = ['b', 'k', 'r']
    bnds = [[250, 300, 350],  # I_0
            [0.015, 0.025, 0.035],  # alpha
            [0.23, 0.43, 0.63],  # PAR
            [0.3, 0.6, 0.9],  # a
            [1, 1.066, 1.1],  # b
            [0.8, 1, 1.2],  # c
            [0.8, 1, 1.2],  # k_N
            [0.8, 1, 1.2]]  # k_fe

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), sharey=True)
    ax = ax.flatten()
    params = pds.params.copy()

    for i, var in enumerate(['I_0', 'alpha', 'PAR', 'a', 'b', 'c', 'k_N', 'k_fe']):
        params = pds.params.copy()
        for j, p in enumerate(bnds[i]):
            params[var] = p
            a, b, c, k_N, k_fe, PAR, I_0, alpha = [params[i] for i in constant_names]
            ax[i].set_title('Phytoplankton uptake ({})'.format(var))
            phy = phyto_up(dx, a, b, c, k_N, k_fe, PAR, I_0, alpha)
            ax[i].plot(phy, phy.depth, c=colors[j], label='{}={}'.format(var, p))
            ax[i].legend(loc='lower right')
    ax[0].set_ymargin(0)
    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/params_phyto.png', bbox_inches='tight')
    plt.show()


def plot_remin_param_constants():
    def remin(dx, b, c, mu_P, gamma_2, mu_D, mu_D_180):
        bcT = b**(c * dx.temp)
        mu_P = mu_P * bcT
        gamma_2 = gamma_2 * bcT
        mu_D = xr.where(dx.depth < 180, mu_D * bcT, mu_D_180 * bcT)
        fe_reg = 0.02 * (mu_D * dx.det + gamma_2 * dx.zoo + mu_P * dx.phy)
        return fe_reg

    exp = cfg.ExpData(scenario=0)
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    ds = ofam_clim().mean('time')
    dx = ds.sel(lat=0, lon=exp.lon, method='nearest').sel(depth=slice(0, 350))

    constant_names = ['b', 'c', 'mu_P', 'gamma_2', 'mu_D', 'mu_D_180']
    b, c, mu_P, gamma_2, mu_D, mu_D_180 = [pds.params[i] for i in constant_names]

    colors = ['b', 'k', 'r']
    bnds = [[1, 1.066, 1.1],  # b
            [0.8, 1, 1.2],  # c
            [0.005, 0.01, 0.05],  # mup
            [0.005, 0.01, 0.05],  # ga
            [0.005, 0.02, 0.05],  # mud
            [0.005, 0.01, 0.05]]  # mud180

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 16), sharey=True)
    ax = ax.flatten()
    params = pds.params.copy()

    for i, var in enumerate(constant_names):
        params = pds.params.copy()
        for j, p in enumerate(bnds[i]):
            params[var] = p
            b, c, mu_P, gamma_2, mu_D, mu_D_180 = [params[i] for i in constant_names]
            ax[i].set_title('Fe remin ({})'.format(var))
            dr = remin(dx, b, c, mu_P, gamma_2, mu_D, mu_D_180)
            ax[i].plot(dr, dr.depth, c=colors[j], label='{}={}'.format(var, p))
            ax[i].legend(loc='lower right')
            ax[i].axhline(80, c='darkgrey', lw=1)
            ax[i].axhline(180, c='darkgrey', lw=1)
    ax[0].set_ymargin(0)
    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/params_remin.png', bbox_inches='tight')
    plt.show()


def plot_scav_param_constants():
    """Test iron model stuff (unfinished)."""
    def fe_scav(fe, det, k_org, k_inorg, c_scav):
        org = fe * k_org * (0.02 * det)**0.58
        inorg = k_inorg * (fe**c_scav)
        scav = org + inorg
        return scav, org, inorg

    exp = cfg.ExpData(scenario=0)
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    # ds = ofam_clim().mean('time')
    # dx = ds.sel(lat=0, lon=exp.lon, method='nearest').sel(depth=slice(0, 350))

    constant_names = ['c_scav', 'k_inorg', 'k_org']

    # Plot scavenging
    fe = np.arange(0, 1.2, 0.1)
    det = [0.01372703, 0.07713306][1]
    colors = ['b', 'k', 'r']
    bnds = [[1.5, 2, 2.5],  # c
            [1e-6, 1e-5, 1e-4],  # k_inorg
            [1e-6, 1e-5, 1e-4]]  # k_org

    k_org, k_inorg, c_scav = [pds.params[i] for i in constant_names]
    # scav, org, inorg = fe_scav(fe, det, k_org, k_inorg, c_scav)

    fig, ax = plt.subplots(3, 3, figsize=(11, 11), sharex=True)
    ax = ax.flatten()
    i = 0

    for k_inorg in bnds[1]:
        for k_org in bnds[2]:
            for j, c_scav in enumerate(bnds[0]):
                scav, org, inorg = fe_scav(fe, det, k_org, k_inorg, c_scav)
                ax[i].set_title('inorg={:.1e}, org={:.1e}'.format(k_inorg, k_org))
                ax[i].plot(fe, scav, c=colors[j], label='c={}'.format(c_scav))
                ax[i].legend(loc='lower right')
                ax[i].axvline(0.8, c='darkgrey', lw=1)
                ax[i].set_xmargin(0)
            i += 1
    ax[-1].set_xlabel('Fe [nM Fe]')
    ax[0].set_ylabel('Scavenged Fe [nM Fe]')

    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/params_scav.png', bbox_inches='tight')
    plt.show()
    # ax[0].plot(f, org, 'b', label='org={:.1e}'.format(inorg))
    # ax[0].plot(f, inorg, 'hotpink', label='inorg={:.1e}'.format(inorg))
    # ax[0].plot(f, scav, 'k', label='c-{}, inorg={:.1e}, org={:.1e}'.format(c_scav, k_inorg, k_org))
    return


def add_particles_dD_dz(ds, exp):
    """Add OFAM det at depth above."""
    df = ofam_clim().rename({'depth': 'z', 'det': 'dD_dz'}).mean('time')

    # Calculate dD/dz (detritus flux)
    zi_1 = (np.abs(df.z - ds.z.fillna(1))).argmin('z')  # Index of depth.
    # zi_0 = zi_1 + 1 if zi_1 == 0 else zi_1 - 1  # Index of depth above/below
    zi_0 = xr.where(zi_1 == 0, zi_1 + 1, zi_1 - 1)  # Index of depth above/below
    D_z0 = df.dD_dz.sel(lat=0, lon=exp.lon, method='nearest').isel(z=zi_0, drop=True).drop(['lat', 'lon', 'z'])
    # D_z0 = D_z0.sel(lat=ds.lat, lon=ds.lon, method='nearest', drop=True).drop(['lat', 'lon', 'z'])
    dz = df.z.isel(z=zi_1, drop=True) - df.z.isel(z=zi_0, drop=True)  # Depth difference.
    ds['dD_dz'] = np.fabs((ds.det - D_z0) / dz).load()
    return ds


def add_particles_ofam_iron(ds, pds):
    """Add OFAM3 iron at particle locations."""
    fieldset = BGCFields(pds.exp)
    ds_fe_ofam = fieldset.ofam_dataset(variables=['det', 'phy', 'zoo', 'fe'], decode_times=True, chunks='auto')
    ds_fe_ofam = ds_fe_ofam.rename({'depth': 'z', 'fe': 'Fe'})  # For SourceIron compatability.
    # Sample OFAM3 iron along trajectories (for testing).
    dim_map = fieldset.dim_map_ofam.copy()
    dim_map['z'] = dim_map.pop('depth')
    ds['Fe'] = pds.empty_DataArray(ds)
    fe = update_field_AAA(ds, ds_fe_ofam, 'Fe', dim_map=dim_map)
    ds['Fe'] = (['traj', 'obs'], fe)
    return ds
