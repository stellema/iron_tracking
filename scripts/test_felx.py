# -*- coding: utf-8 -*-
"""Tests functions for felx experiment (non-parcels version).

Notes:
    - Uses Python=3.9
    - Primarily uses parcels version 2.4.0

Example:

Todo:


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 10 11:52:04 2023

"""
import math
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cfg
from cfg import ExpData
from datasets import BGCFields
from felx_dataset import FelxDataSet
from fe_obs_dataset import FeObsDatasets, FeObsDataset
from particle_BGC_fields import update_field_AAA


def test_plot_iron_output(ds, ntraj=5):
    # Plot output (for testing).
    fig, ax = plt.subplots(3, 2, figsize=(16, 16), squeeze=True)
    ax = ax.flatten()

    for j, p in enumerate(range(ntraj)):
        c = ['k', 'b', 'r', 'g', 'm', 'y', 'darkorange', 'hotpink', 'lime', 'navy'][j]
        dx = ds.isel(traj=p).dropna('obs', 'all')
        ax[0].plot(dx.lon, dx.lat, c=c, label=p)  # map
        ax[1].plot(dx.obs, dx.z, c=c, label=p)  # Depth
        ax[2].plot(dx.obs, dx.fe, c=c, label=p)  # lagrangian iron
        ax[2].plot(dx.obs, dx.Fe, c=c, ls='--', label=p)  # ofam iron

        for i, var in zip([3, 4, 5], ['fe_reg', 'fe_scav', 'fe_phy']):
            ax[i].set_title(var)
            ax[i].plot(dx.obs, dx[var], c=c, label=p)

    ax[0].set_title('Particle lat & lon')
    ax[1].set_title('Particle depth')
    ax[2].set_title('lagrangian iron (solid) & OFAM3 iron (dashed)')

    for i in range(1, 5):
        ax[i].set_ylabel('[mmol Fe m^-3]')
        ax[i].set_xlabel('obs')
    ax[1].set_ylabel('Depth')
    ax[1].invert_yaxis()
    ax[0].set_xlabel('lon')
    ax[0].set_ylabel('lat')
    ax[0].set_ylabel('Fe [nM Fe]')
    ax[1].legend()


def UpdateIron(pds, ds, p, t, df, k_org, k_inorg, c, dt=2, scav_method='notofam'):
    """Update iron at a particle location.

    Args:
        pds (FelxDataSet): FelxDataSet instance (contains constants and dataset functions).
        ds (xarray.Dataset): Particle dataset (e.g., lat(traj, obs), zoo(traj, obs), fe_scav(traj, obs), etc).
        p (int): Particle index.
        t (int): Time index.

    Returns:
        ds (xarray.Dataset): Particle dataset.

    Notes:
        * p and time must be position index (not value).
        * Redfield ratio:
            * 1P: 16N: 106C: 16*2e-5 Fe (Oke et al., 2012, Christian et al., 2002)
            * ---> 1N:(2e-5 *1e3) 0.02 Fe (multiply Fe by 1e3 because mmol N m^-3 & umol Fe m^-3)
             -> [mmol N m^-3 * 1e-3]=[1N: 10,000 Fe]  (1 uM N = 1 nM Fe)
        * Variables and units:
            * Temperature (T) [degrees C]
            * Detritus (D) [mmol N m^-3]
            * Zooplankton (Z) [mmol N m^-3]
            * Phytoplankton (P) [mmol N m^-3]
            * Nitrate/NO3 (N) [mmol N m^-3]
            * Diffuse Attenuation Coefficient at 490 nm (Kd) [m^-1]
            * Iron (fe) [umol Fe m^-3]
            * Remineralised iron (fge_reg) [umol Fe m^-3 day^-1] (add to fe)
            * Iron uptake for phytoplankton growth (fe_scav) [umol Fe m^-3 day^-1] (minus from fe)
            * Scavenged iron [fe_scav] [umol Fe m^-3 day^-1] (minus from fe)
    Todo:
        * check mu_D values
        * How to calculate dDet/dt (remineralisation)?

    """
    dt = 2  # Timestep [days]
    loc = dict(traj=p, obs=t)  # Particle location indexes for subseting ds
    loc_prev = dict(traj=p, obs=max(0, t - 1))

    # Variables from OFAM3 subset at previous particle location.
    T, D, Z, P, N, Kd = [ds[v].isel(loc_prev, drop=True)
                         for v in ['temp', 'det', 'zoo', 'phy', 'no3', 'kd']]



    # Iron at previous particle location.
    Fe = ds['fe'].isel(loc_prev, drop=True)  # Lagrangian Iron.
    z = ds['z'].isel(loc_prev, drop=True)  # Depth.
    # --------------------------------------------------------------------------------
    # Iron Scavenging.
    if scav_method == 'ofam':
        # Scavenging Option 1: Oke et al., 2012 (Qin thesis pg. 144)
        ds['fe_scav'][loc] = pds.tau_scav * max(0, Fe - 0.6)  # [umol Fe m^-3]

    else:
        # Scavenging Option 2: Galbraith et al., 2010
        # w_sink = 16 + max(0, (z - 80)) * 0.05  # [m day^-1] linearly increases by 0.5 below 80m
        # D_flux = 0.02 * D * pds.w_D  # ??? [mmol N m^-2 day^-1] (0.02 * mmol N m^-3 * m day^-1)

        # Calculate dD/dz (detritus flux)
        zi_1 = (np.abs(df.z - z)).argmin()  # Index of depth.
        zi_0 = zi_1 + 1 if zi_1 == 0 else zi_1 - 1  # Index of depth above/below.
        dx = ds.isel(loc_prev, drop=True)  # Dataset of of particle positions.
        # Detritus field at particle position (at depth above/below).
        D_z0 = df.det.isel(z=zi_0, drop=True).sel(time=dx.time, lat=dx.lat, lon=dx.lon,
                                                  method='nearest', drop=True)
        dz = df.z.isel(z=zi_1, drop=True) - df.z.isel(z=zi_0, drop=True)  # Depth difference.
        dD_dz = 0.02 * np.fabs((D - D_z0) / dz)

        # Convert detritus (D) units: 0.02 * [mmol N m^-3] x ??? -> 0.02 * [mmol N m^-2 day^-1] = [umol Fe m^-3]
        # ([umol Fe m^-3] * [(nM Fe m)^-0.58 day^-1] * [((umol Fe m^-3)/(m day^-1))^0.58]) + ([(nM Fe m)^-0.5 day^-1] * [(umol Fe m^-3)^1.5])
        # = ([umol Fe m^-3] * [(umol Fe m^-2)^-0.58 day^-1] * [(umol Fe m^-2)^0.58]) + ([(umol Fe m^-3)^-0.5 day^-1] * [(umol Fe m^-3)^1.5])
        # = [umol Fe m^-3 day^-1] + [umol Fe m^-3 day^-1] - (Note that (nM Fe)^-0.5 * (nM Fe)^1.5 = (nM Fe)^1)
        ds['fe_scav'][loc] = (Fe * k_org * (dD_dz)**0.58) + (k_inorg * Fe**c)

    # --------------------------------------------------------------------------------
    # Iron Phytoplankton Uptake.
    # Maximum phytoplankton growth [J_max] (assumes no light or nutrient limitations).
    J_max = (pds.a * pds.b)**(pds.c * T)  # [day^-1] ((day^-1 * const)^const)

    # Nitrate limit on phytoplankton growth rate.
    J_limit_N = N / (N + pds.k_N)  # [no units] (mmol N m^-3 / mmol N m^-3)

    # Iron limit on phytoplankton growth rate.
    k_fe = (0.02 * pds.k_fe)  # !!! Convert k_fe using 1N: 0.02Fe [mmol N m^-3 -> umol Fe m^-3]
    J_limit_Fe = Fe / (Fe + k_fe)  # [no units] (umol Fe m^-3 / (umol Fe m^-3 + umol Fe m^-3)

    # Light limit on phytoplankton growth rate.
    Frac_z = math.exp(-z * Kd)  # [no units] (m * m^-1)
    light = pds.PAR * pds.I_0 * Frac_z  # [W/m^2] (const * W m^-2 * const)
    # day^-1 x (exp^(day^-1 * Wm^2x Wm^-2) / day^-1) -> [day^-1 * exp(const)]
    J_I = J_max * (1 - math.exp(-(pds.alpha * light) / J_max))  # [day^-1]
    J_limit_I = J_I / J_max  # [no units] (day^-1 / day^-1)

    # Phytoplankton growth rate (iron consumption associated with growth in amount of phytoplankton).
    J = J_max * min(J_limit_I, J_limit_N, J_limit_Fe)  # [day^-1]
    ds['fe_phy'][loc] = 0.02 * J * P  # [umol Fe m^-3 day^-1] (0.02 * day^-1 * mmol N m^-3)
    # --------------------------------------------------------------------------------
    # Iron Remineralisation.
    bcT = pds.b**(pds.c * T)  # [no units]
    gamma_2 = 0.01 * bcT  # [day^-1]
    mu_P = 0.01 * bcT  # [day^-1]
    if z < 180:
        mu_D = 0.02 * bcT  # [day^-1]
    else:
        mu_D = 0.01 * bcT  # [day^-1]
    mu_P = 0.01 * bcT  # [day^-1]
    gamma_2 = 0.01 * bcT  # [day^-1]

    ds['fe_reg'][loc] = 0.02 * (mu_D * D + gamma_2 * Z + mu_P * P)  # [umol Fe m^-3 day^-1] (0.02 * mmol N m^-3 * day^-1)
    # G = ((pds.g * pds.epsilon * P**2) / (pds.g + (pds.epsilon * P**2))) * Z
    # dD_dz = D * pds.w_D # ??? [mmol N m^-3 * m day^-1 = mmol N m^-2 day^-1]
    # # dD_dt = ((1 - pds.gamma_1) * G) + (pds.mu_Z * Z**2) - (mu_D * D) - (pds.w_D * dD_dz)
    # dD_dt = ((1 - pds.gamma_1) * G) + (pds.mu_Z * Z**2) - (mu_D * D) - (pds.w_D * D)
    # dP_dt = (J * P) - G - (mu_P * P)
    # dZ_dt = (pds.gamma_1 * G) - (gamma_2 * Z) - (pds.mu_Z * Z**2)
    # ds['fe_reg'][loc] = 0.02 * (mu_D * dD_dt + gamma_2 *dZ_dt + mu_P * dP_dt)  # [umol Fe m^-3 day^-1] (0.02 * mmol N m^-3 * day^-1)

    # --------------------------------------------------------------------------------
    # Iron (multiply by 2 because iron is updated every 2nd day).
    # n days * [umol Fe m^-3 day^-1]
    ds['fe'][loc] = ds.fe[loc_prev] + dt * (ds.fe_reg[loc] - ds.fe_phy[loc] - ds.fe_scav[loc])
    return ds


def SourceIron(pds, ds, p, ds_fe):
    """Assign Source Iron.

    Args:
        pds (FelxDataSet): FelxDataSet instance.
        ds (xarray.Dataset): Particle dataset.
        p (int): Particle index.
        ds_fe (xarray.Dataset): Dataset of iron depth profiles.

    Returns:
        ds (xarray.Dataset): Particle dataset.

    Notes:
        * Source iron fields:
            * Option 1: Dataset with regions as variables var(depth) and select based on zone.
            * Option 2: Dataset with stacked variable var(zone, depth)
            * Option 3: OFAM3 iron var(time, depth, lat, lon)

        * Expect iron values of around:
            * fe_reg: 0.03 nM
            * fe_scav & fe_phy: 5-7 nM

    """
    t = 0
    dx = ds.isel(traj=p, obs=t, drop=True)
    zone = int(dx.zone.item())

    # Option 1: Select region data variable based on source ID.
    if 'llwbcs' in ds_fe:
        if zone in [1, 2, 3, 4, 6]:
            var = 'llwbcs'
        elif zone in [0, 5] or zone >= 7:
            var = 'interior'
        da_fe = ds_fe[var]

    elif 'Fe' in ds_fe:
        # Use OFAM3 iron.
        var = 'Fe'
        da_fe = ds_fe[var].sel(time=dx.time, drop=True)

    # Particle location map (e.g., lat, lon, depth)
    particle_loc = dict([[c, dx[c].item()] for c in da_fe.dims])

    fe = da_fe.sel(particle_loc, method='nearest', drop=True).load().item()

    # Assign Source Iron.
    ds['fe'][dict(traj=p, obs=t)] = fe
    ds['fe_src'][dict(traj=p)] = fe  # For reference.
    return ds


def update_particles_iron(pds, ds, ds_fe, df, k_org, k_inorg, c):
    """ Run iron model for each particle."""
    num_particles = ds.traj.size
    for p in range(num_particles):
        ds = SourceIron(pds, ds, p, ds_fe)  # Assign initial iron.

        num_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for t in range(num_obs):
            ds = UpdateIron(pds, ds, p, t, df, k_org, k_inorg, c)  # Update Iron.
    return ds

def run_iron_model():
    exp = ExpData(scenario=0, lon=250, test=True, test_month_bnds=12)

    # Source Iron fields.
    # Observations.
    dfs = FeObsDatasets()
    setattr(dfs, 'ds', dfs.combined_iron_obs_datasets(add_Huang=False, interp_z=True))
    setattr(dfs, 'dfe', FeObsDataset(dfs.ds))
    ds_fe_obs = dfs.dfe.ds_avg.mean('t')
    # 1nM = 1e-3 mmol/m3  (1nM = 1e-9M = 1e-6 mol -- 1M = 1e3mol)

    # OFAM3.
    fieldset = BGCFields(exp)
    ds_fe_ofam = fieldset.ofam_dataset(variables=['det', 'phy', 'zoo', 'fe'], decode_times=True, chunks='auto')
    ds_fe_ofam = ds_fe_ofam.rename({'depth': 'z', 'fe': 'Fe'})  # For SourceIron compatability.

    ds_fe = [ds_fe_obs, ds_fe_ofam][1]  # Pick source iron dataset.

    # particle dataset
    pds = FelxDataSet(exp)
    pds.add_iron_model_constants()

    ds = pds.init_felx_bgc_dataset()

    # Cut off particles in early years to load less ofam fields.
    trajs = ds.traj.where(ds.isel(obs=0, drop=True).time.dt.year > 2011, drop=True)
    ds = ds.sel(traj=trajs)
    ntraj = 10  # num particles to run.
    ds = ds.isel(traj=slice(ntraj)).dropna('obs', 'all')

    # Sample OFAM3 iron along trajectories (for testing).
    dim_map = fieldset.dim_map_ofam.copy()
    dim_map['z'] = dim_map.pop('depth')
    ds['Fe'] = pds.empty_DataArray(ds)
    fe = update_field_AAA(ds, ds_fe_ofam, 'Fe', dim_map=dim_map)
    ds['Fe'] = (['traj', 'obs'], fe)

    df = ds_fe_ofam

    # paramters to optmise

    cc = np.arange(1.5, 2, 0.1)[::-1]
    rmse = np.zeros(cc.size)

    for i, c in enumerate(cc):
        for k_org in np.arange(0.01e-4, 8e-4, 0.0001):
            for k_inorg in np.arange(0.01e-4, 8e-4, 0.0001):

                # k_org = 1e-4  # (Qin: 1.0521e-4, Galbraith: 4e-4)
                # k_inorg = 6e-4  #  (Qin: 6.10e-4, Galbraith: 6e-4)
                # c = 1.7

                ds = update_particles_iron(pds, ds, ds_fe, df, k_org, k_inorg, c)
                rms = np.mean(np.sqrt(((ds.Fe-ds.fe)**2).mean('obs')))
                print('k_org={}, k_inorg={}, c={}, rms={}'.format(k_org, k_inorg, c, rms.item()))
                # test_plot_iron_output(ds, ntraj=5)
                rmse[i] = rms
