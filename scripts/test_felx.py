# -*- coding: utf-8 -*-
"""Tests functions for felx experiment (non-parcels version).

Notes:
    - Uses Python=3.9
    - Primarily uses parcels version 2.4.0

Example:

Todo:
    - Add Kd490 Fields
    - Calculate background Iron fields
    - Add background Iron fields
    - Calculate & save BGC fields at particle positions
        - Calculate OOM
        - Test fieldset vs xarray times
        - Research best method for:
            - indexing 4D data
            - chunking/opening data
            - parallelisation
    - Filter & save particle datasets
        - Spinup:
            - Calculate num of affected particles
            - Test effect of removing particles
        - Available BGC fields:
            - Figure out availble times
            - Calculate num of affected particles
            - Test effect of removing particles
    - Write UpdateIron function
        - Calculate OOM & memory
    - Run simulation+
    C
        - Add parallelisation
        - Optimise parallelisation

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


def UpdateIron(pds, ds, p, t):
    """Update Lagrangian Iron.

    Args:
        pds (FelxDataSet): FelxDataSet instance (contains constants).
        ds (xarray.Dataset): Particle dataset.
        p (int): Particle index.
        t (int): Time index.

    Returns:
        ds (xarray.Dataset): Particle dataset.

    Notes:
        * p and time must be position index
        *Redfield ratio:
            * 1P: 16N: 106C: 16*2e-5 Fe (Oke et al., 2012, Christian et al., 2002)
            * ---> 1N:(2e-5 *1e3) 0.02 Fe (multiply Fe by 1e3 because mmol N m^-3 & umol Fe m^-3)

    """
    loc = dict(traj=p, obs=t)
    loc_prev = dict(traj=p, obs=max(0, t-1))
    # Variables from OFAM3 subset at particle location.
    # ??? subset at previous or current particle location? (Qin thesis pg. 133)
    T, D, Z, P, N, Kd, z = [ds[v].isel(loc_prev, drop=True)
                            for v in ['temp', 'det', 'zoo', 'phy', 'no3', 'kd', 'z']]

    # Iron at previous particle location.
    Fe = ds['fe'].isel(loc_prev, drop=True)

    # Iron Remineralisation.
    bcT = math.pow(pds.b, pds.c * T)  # [no units]
    y_2 = 0.01 * bcT  # [no units]
    u_P = 0.01 * bcT  # [no units]
    if ds[loc].z < 180:
        u_D = 0.02 * bcT  # [no units]
    else:
        u_D = 0.01 * bcT  # [no units]

    ds['fe_reg'][loc] = 0.02 * (u_D * D + y_2 * Z + u_P * P)  # [umol Fe m^-3] (0.02 * [mmol N m^-3])

    # Iron Scavenging.
    # # Scavenging Option 1: Oke et al., 2012 (Qin thesis pg. 144)
    # ds['fe_scav'][loc] = 1.24e-2 * min(0, Fe - 0.6)  # [umol Fe m^-3]
    # # ds['fe_scav'][loc] = 0.02 * dN/dt - (1 * max(0, Fe - 0.6))  # [umol Fe m^-3]

    # Scavenging Option 2: Galbraith et al., 2010
    # ??? Detritus sinking velocity (w_sink) - which one to use?
    w_sink = 16 + max(0, (z - 80)) * 0.05  # [m day^-1] linearly increases by 0.5 below 80m
    w_sink = pds.w_det  # 10 m day^-1 ( from Oke et al., 2012)

    # Detritus (D) [mmol N m^-1 day^-1]
    # ??? Convert to [nmol N m^2 day^-1] (units given in Qin et al.) - How to get d/dz(Det)?
    # [(nM Fe m)^-0.58 day^-1] * [[umol Fe m^-1 day^-1]/[m day^-1]]^0.58 +
    # [nM Fe m]^-0.5 day^-1 * [umol Fe m^-3]^1.5
    ds['fe_scav'][loc] = ((pds.k_org * math.pow(((D * 0.02) / w_sink), 0.58) * Fe)
                          + (pds.k_inorg * math.pow(Fe, 1.5)))

    # Iron Phyto Uptake.
    # Maximum phytoplankton growth [J_max] (assumes no light or nutrient limitations).
    J_max = math.pow(pds.a * pds.b, pds.c * T)  # [(day^-1)] (([day^-1] * const) **const)

    # Nitrate limit on phytoplankton growth rate.
    J_limit_N = N / (N + pds.k_N)  # [no units][ (mmol N m^-3] / [mmol N m^-3])

    # Iron limit on phytoplankton growth rate.
    k_fe = (pds.k_fe * 0.02)  # !!! Convert k_fe using 1N: 0.02Fe [mmol N m^-3 -> umol Fe m^-3]
    J_limit_Fe = Fe / (Fe + k_fe)  # [no units] ([umol Fe m^-3] / ([umol Fe m^-3]))

    # Light limit on phytoplankton growth rate.
    Frac_z = math.exp(-z * Kd)  # [no units] (m * m^-1)
    light = pds.PAR * pds.I_0 * Frac_z  # [W/m^2] (const * [W/m^2] * const)
    # [(day^-1)] x exp^(([day^-1 * Wm^2]x[Wm^-2]) / [day^-1]) -> [day^-1 * exp(const)]
    J_I = J_max * (1 - math.exp(-(pds.alpha * light) / J_max))  # [day^-1]
    J_limit_I = J_I / J_max  # [no units] ([day^-1] / [day^-1])

    # Phytoplankton growth rate (iron consumption associated with growth in amount of phytoplankton).
    J = J_max * min(J_limit_I, J_limit_N, J_limit_Fe)  # [day^-1]
    ds['fe_phy'][loc] = 0.02 * J * P  # [umol Fe m^-3 day^-1] ([0.02 * day^-1 * mmol N m^-3])

    # ds['fe_phy'][loc] = ds['fe_phy'][loc] * 1e3  # [umol Fe m^-3 day^-1]
    # ds['fe_scav'][loc] = ds['fe_scav'][loc] * 1e3
    # ds['fe_reg'][loc] = ds['fe_reg'][loc] * 1e3

    # Iron (multiply by 2 because iron is updated every 2nd day.)
    ds['fe'][loc] = Fe + 2 * (ds.fe_reg[loc] - ds.fe_phy[loc] - ds.fe_scav[loc])
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


def felx_alt_test_simple():
    """Test simple config of fe experiment."""
    exp = ExpData(scenario=0, lon=165, test=True, test_month_bnds=15)

    # Source Iron fields.
    # Observations.
    dfs = FeObsDatasets()
    setattr(dfs, 'ds', dfs.combined_iron_obs_datasets(add_Huang=False, interp_z=True))
    setattr(dfs, 'dfe', FeObsDataset(dfs.ds))
    ds_fe_obs = dfs.dfe.ds_avg.mean('t')
    # 1nM = 1e-3 mmol/m3  (1nM = 1e-9M = 1e-6 mol -- 1M = 1e3mol)

    # OFAM3.
    fieldset = BGCFields(exp)
    ds_fe_ofam = fieldset.ofam_dataset(variables='fe', decode_times=True, chunks='auto')
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

    # --------------------------------------------------------------------------------
    num_particles = ds.traj.size
    for p in range(num_particles):
        ds = SourceIron(pds, ds, p, ds_fe)  # Assign initial iron.

        num_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for t in range(num_obs):
            ds = UpdateIron(pds, ds, p, t)

    # -------------------------------------------------------------------------------

    fig, ax = plt.subplots(2, 2, figsize=(16, 12), squeeze=True)
    ax = ax.flatten()

    for j, p in enumerate(range(10)):
        c = ['k', 'b', 'r', 'g', 'm', 'y', 'darkorange', 'hotpink', 'lime', 'navy'][j]
        dx = ds.isel(traj=p).dropna('obs', 'all')
        ax[0].set_title('Lagrangian iron (solid) and OFAM3 iron (dashed)')
        ax[0].plot(dx.obs, dx.fe, c=c, ls='-', label=p)
        ax[0].plot(dx.obs, dx.Fe, c=c, ls='--', label=p)

        for i, var in zip([1, 2, 3], ['fe_scav', 'fe_reg', 'fe_phy']):

            ax[i].set_title(var)
            ax[i].plot(dx.obs, dx[var], c=c, label=p)

    for i in range(4):
        ax[i].set_ylabel('Fe [M]')
        ax[i].set_xlabel('obs')
    ax[1].legend()

    return ds
