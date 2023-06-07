# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:


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
from fe_exp import FelxDataSet
from tools import mlogger, unique_name

logger = mlogger('fe_model_test')


def plot_seawifs_chlorophyll():
    """Plot seaWIFIS chlorophyll climatology map."""
    ds = xr.open_mfdataset([paths.obs / 'GMIS_SEAWIFS/GMIS_A_CHLA_{:02d}.nc'.format(i)
                            for i in range(1, 13)], combine='nested', concat_dim='time',
                           decode_cf=0)
    ds = ds.Chl_a
    ds = ds.assign_coords(lon=ds.lon % 360)
    ds = ds.sortby(ds.lon)
    ds = ds.drop_duplicates('lon')
    dx = 0.1
    ds = ds.interp(lon=np.arange(0, 360 + dx, dx), lat=np.arange(90, -90 - dx, -dx))
    ds = ds.where((ds != -9999).compute(), drop=1)
    df = ds.mean('time')
    df = 10**df

    cmap = plt.cm.jet
    cmap.set_bad('k')
    norm = mpl.colors.LogNorm(vmin=0.01, vmax=40)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    pcm = ax.pcolormesh(df.lon, df.lat, df.where(df > 0), cmap=cmap, norm=norm)

    # Colourbar
    cb = fig.colorbar(pcm, ax=ax, extend='max', orientation='horizontal', pad=0.05,
                      fraction=0.08, shrink=0.4, format='%.2f')
    cb.set_label('Chlorophyll [mg m^-3]')

    # Ticks
    ax.set_ylim(-80, 85)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d°E"))
    yticks = np.arange(-80, 85, 20)
    ytick_labels = ['{:.0f}°{}'.format(np.fabs(i), 'N' if i >= 0 else 'S') for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    plt.tight_layout()
    plt.savefig(paths.figs / 'clim/chl-a_clim_seawifs.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_euc_obs_xz_profile():
    # Johnson EUC
    ds = xr.open_dataset(cfg.paths.obs / 'pac_mean_johnson_2002.cdf')
    ds = ds.rename(dict(ZDEP1_50='z', YLAT11_101='y', XLON='x', UM='u'))
    # ds.u.sel(y=0).plot(figsize=(14, 5), yincrease=0, cmap=plt.cm.seismic)
    ds = ds.sel(x=slice(156, 270))
    xe = ds.XLONedges[1:]

    cmap = plt.cm.seismic
    cmap.set_bad('grey')
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # levels = np.arange(-1.1, 1.12, 0.02)
    # im = ax.contourf(ds.x, ds.z, ds.u.sel(y=0), cmap=plt.cm.seismic, levels=levels)
    # ax.contour(ds.x, ds.z, ds.SIGMAM.sel(y=0), 10, colors='k')
    im = ax.pcolormesh(xe, ds.z, ds.u.sel(y=0, z=slice(5, 485)), cmap=cmap, vmax=1.1, vmin=-1)

    # Colour bar
    cb = fig.colorbar(im, ax=ax, extend='both', pad=0.01)
    cb.set_label('zonal velocity [m^-1]')

    # Ticks
    ax.set_ylim(400, 5)
    ax.set_xlim(156, 270)
    ax.set_xticks(ds.x)
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))

    plt.tight_layout()

    plt.savefig(paths.figs / 'clim/euc_obs_xz_profile.png', dpi=300, bbox_inches='tight')
    plt.show()



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
