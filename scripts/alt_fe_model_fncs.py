# -*- coding: utf-8 -*-
"""FelxDataSet class contains constants and functions the lagrangian iron model.

Includes functions for sampling and saving BGC fields at particle positions.

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
@created: Sun Feb 19 14:24:37 2023

"""
import math
import numba
import numpy as np
import xarray as xr  # NOQA

from cfg import ExpData, test
from datasets import ofam_clim, BGCFields
from fe_model import SourceIron
from particle_BGC_fields import update_field_AAA
from tools import mlogger, timeit

logger = mlogger('alt_fe_model')

@numba.njit
def update_iron_NPZD_jit(p, t, fe, T, D, Z, P, N, Kd, z, J_max, J_I, dD_dz, k_org, k_inorg, c_scav,
                         mu_D, mu_D_180, gamma_2, mu_P, a, b, c, k_N, k_fe, tau, g, epsilon,
                         gamma_1, mu_Z, w_D):
    dt = 2  # Timestep [days]
    bcT = b**(c * T)  # [no units]
    mu_P = mu_P * bcT  # [day^-1]
    gamma_2 = gamma_2 * bcT  # [day^-1]
    if z < 180:
        mu_D = mu_D * bcT  # [day^-1]
    else:
        mu_D = mu_D_180 * bcT  # [day^-1]

    # Nitrate limit on phytoplankton growth rate
    J_limit_N = N / (N + k_N)  # [no units] (mmol N m^-3 / mmol N m^-3)

    # Iron limit on phytoplankton growth rate
    J_limit_Fe = fe / (fe + (0.02 * k_fe))  # [no units] (umol Fe m^-3 / (umol Fe m^-3 + umol Fe m^-3))

    # Light limit on phytoplankton growth rate
    J_limit_I = J_I / J_max  # [no units] (day^-1 / day^-1)

    # Phytoplankton growth rate [day^-1] ((day^-1 * const)^const)
    J = J_max * min(J_limit_I, J_limit_N, J_limit_Fe)  # [day^-1]

    # Update NPZD.
    G = ((g * epsilon * P**2) / (g + (epsilon * P**2))) * Z
    dP_dt = (J * P) - G - (mu_P * P)
    dZ_dt = (gamma_1 * G) - (gamma_2 * Z) - (mu_Z * Z**2)
    dD_dt = ((1 - gamma_1) * G) + (mu_Z * Z**2) - (mu_D * D) - (w_D * dD_dz)
    dN_dt = (mu_D * D) - (gamma_2 * Z) + (mu_P * P) - (J * P)

    P = P + dt * dP_dt
    Z = Z + dt * dZ_dt
    D = D + dt * dD_dt
    N = N + dt * dN_dt

    # Iron Phytoplankton Uptake
    fe_phy = 0.02 * J * P  # [umol Fe m^-3 day^-1] (0.02 * day^-1 * mmol N m^-3)

    # Iron Remineralisation
    fe_reg = 0.02 * (mu_D * D + gamma_2 * Z + mu_P * P)  # [umol Fe m^-3 day^-1]

    # Iron Scavenging
    fe_scav = (fe * k_org * (0.02 * D)**0.58) + (k_inorg * fe**c_scav)  # [umol Fe m^-3 day^-1]
    # fe_scav = tau * max(0, fe - 0.6)  # [umol Fe m^-3  day^-1]

    # Iron
    fe = fe + dt * (fe_reg - fe_phy - fe_scav)
    return fe, fe_scav, fe_reg, fe_phy


def update_iron(pds, ds, p, t):
    """Update iron at a particle location.

    Args:
        pds (FelxDataSet): FelxDataSet instance (contains constants and dataset functions).
        ds (xarray.Dataset): Particle dataset (e.g., lat(traj, obs), zoo(traj, obs), fe_scav(traj, obs), etc).
        p (int): Particle index.
        t (int): Time index.

    Returns:
        ds (xarray.Dataset): Particle dataset.

    Notes:
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
    """
    dt = 2  # Timestep [days]
    loc_new = dict(traj=p, obs=t)  # Particle location indexes for subseting ds
    loc = dict(traj=p, obs=t - 1)

    # Variables from OFAM3 subset at previous particle location.
    Fe, T, D, Z, P, N, Kd, z = [ds[v].isel(loc, drop=True)
                                for v in ['fe', 'temp', 'det', 'zoo', 'phy', 'no3', 'kd', 'z']]

    # --------------------------------------------------------------------------------
    # Iron Phytoplankton Uptake.
    # Maximum phytoplankton growth [J_max] (assumes no light or nutrient limitations).
    J_max = pds.a * pds.b**(pds.c * T)  # [day^-1] ((day^-1 * const)^const)

    # Nitrate limit on phytoplankton growth rate.
    J_limit_N = N / (N + pds.k_N)  # [no units] (mmol N m^-3 / mmol N m^-3)

    # Iron limit on phytoplankton growth rate.
    # Convert k_fe using 1N: 0.02Fe [mmol N m^-3 -> umol Fe m^-3]
    J_limit_Fe = Fe / (Fe + (0.02 * pds.k_fe))  # [no units] (umol Fe m^-3 / (umol Fe m^-3 + umol Fe m^-3)

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
    mu_P = pds.mu_P * bcT  # [day^-1]
    gamma_2 = pds.gamma_2 * bcT  # [day^-1]
    if z < 180:
        mu_D = pds.mu_D * bcT  # [day^-1]
    else:
        mu_D = pds.mu_D_180 * bcT  # [day^-1]

    # [umol Fe m^-3 day^-1] (0.02 * mmol N m^-3 * day^-1)
    ds['fe_reg'][loc] = 0.02 * (mu_D * D + gamma_2 * Z + mu_P * P)

    # --------------------------------------------------------------------------------
    # Iron Scavenging.
    if pds.exp.scav_eq == 'OFAM':
        # Scavenging Option 1: Oke et al., 2012 (Qin thesis pg. 144)
        ds['fe_scav'][loc] = pds.tau_scav * max(0, Fe - 0.6)  # [umol Fe m^-3]

    elif pds.exp.scav_eq == 'Galibraith':
        # Scavenging Option 2: Galbraith et al., 2010
        # Convert detritus (D) units: -> 0.02 * [mmol N m^-2 day^-1] = [umol Fe m^-3]
        # =([umol Fe m^-3] * [(umol Fe m^-3)^-0.58 day^-1] * [((umol Fe m^-3))^0.58])
        # + ([(nM Fe m)^-0.5 day^-1] * [(umol Fe m^-3)^1.5])
        # = [umol Fe m^-3 day^-1] + [umol Fe m^-3 day^-1] - (Note that (nM Fe)^-0.5 * (nM Fe)^1.5 = (nM Fe)^1)
        ds['fe_scav'][loc] = (Fe * pds.k_org * (0.02 * D)**0.58) + (pds.k_inorg * Fe**pds.c_scav)

    # --------------------------------------------------------------------------------
    # Iron (multiply by 2 because iron is updated every 2nd day). n days * [umol Fe m^-3 day^-1]
    ds['fe'][loc_new] = ds.fe[loc] + dt * (ds.fe_reg[loc] - ds.fe_phy[loc] - ds.fe_scav[loc])
    return ds



@timeit(my_logger=logger)
def update_particles(pds, ds, ds_fe, param_names=None, params=None):
    """Run iron model for each particle.

    Args:
        pds (FelxDataSet): FelxDataSet instance (contains constants and dataset functions).
        ds (xarray.Dataset): Particle dataset (e.g., lat(traj, obs), zoo(traj, obs), fe_scav(traj, obs), etc).
        ds_fe (xarray.Dataset): Source dFe depth profiles dataset.
        params (list, optional): List of non-default params to use (see optmisise).

    Returns:
        ds (xarray.Dataset): Particle dataset with updated iron, fe_scav, etc.
    """
    if params is not None:
        pds.update_params(dict([(k, v) for k, v in zip(param_names, params)]))
    particles = range(ds.traj.size)
    for p in particles:
        ds = SourceIron(pds, ds, p, ds_fe)  # Assign initial iron.

        timesteps = range(1, ds.isel(traj=p).dropna('obs', 'all').obs.size)
        for t in timesteps:
            ds = update_iron(pds, ds, p, t)  # Update Iron.
    return ds


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
