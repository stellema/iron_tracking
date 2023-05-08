# -*- coding: utf-8 -*-
"""Functions for iron model experiment & tests.

Notes:
    - Uses Python=3.9
    - Don't use @numba.njit(parallel=True) for update_iron - need prange to parallelise.
    - 1 year run time (ndays * particles/day * runtime/particles):
        - ((365 / 6) * (193 * ((2427.86*4)/386)) / (60*60) = 82 hours / 48 cpu = 1.7 hours

Example:

Todo:
# Calculate dD/dz (detritus flux)
zi_1 = (np.abs(df.z - ds.z)).argmin()  # Index of depth.
zi_0 = zi_1 + 1 if zi_1 == 0 else zi_1 - 1  # Index of depth above/below
D_z0 = df.det.isel(z=zi_0, drop=True).sel(time=ds.time, lat=ds.lat, lon=ds.lon, method='nearest', drop=True)
dz = df.z.isel(z=zi_1, drop=True) - df.z.isel(z=zi_0, drop=True)  # Depth difference.
dD_dz = 0.02 * np.fabs((D - D_z0) / dz)


G = ((pds.g * pds.epsilon * P**2) / (pds.g + (pds.epsilon * P**2))) * Z
D_z0 = df.det.isel(z=zi_0, drop=True).sel(time=dx.time, lat=dx.lat, lon=dx.lon, method='nearest', drop=True)
dz = df.z.isel(z=zi_1, drop=True) - df.z.isel(z=zi_0, drop=True)  # Depth difference.
dD_dz = 0.02 * np.fabs((D - D_z0) / dz)
dD_dt = ((1 - pds.gamma_1) * G) + (pds.mu_Z * Z**2) - (mu_D * D) - (pds.w_D * dD_dz)
dP_dt = (J * P) - G - (mu_P * P)
dZ_dt = (pds.gamma_1 * G) - (gamma_2 * Z) - (pds.mu_Z * Z**2)
dN_dt = (mu_D * D) - (gamma_2 * Z) + (mu_P * P) - (J * P)

Z = Z + dt * dZ_dt
D = D + dt * dD_dt
P = P + dt * dP_dt
N = N + dt * dN_dt

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 10 11:52:04 2023

"""
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import os
from scipy.optimize import minimize
import xarray as xr  # NOQA

import cfg
from cfg import ExpData
from tools import mlogger, timeit
from datasets import BGCFields, ofam_clim, save_dataset
from felx_dataset import FelxDataSet
from fe_obs_dataset import get_merged_FeObsDataset
from particle_BGC_fields import update_field_AAA
from test_iron_model_optimisation import (test_plot_iron_paths, test_plot_EUC_iron_depth_profile,
                                          add_particles_dD_dz)

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None
    rank = 0

logger = mlogger('fe_model_test')


def estimate_runtime(ntraj, ndays, runtime):
    """1 year run time (ndays * particles/day * runtime/particles).
    estimate_runtime(386, 2, 2427.86)
    """
    ncpus = 48
    hours = ((ntraj / ndays) * ((runtime * 4) / ntraj)) / (60 * 60)
    hours_per_year = (365 / 6) * hours / ncpus
    logger.info('Estimated yearly runtime using {} cpus: particles={}, ndays={}, hours/year={:.2f}'
                .format(ncpus, ntraj, ndays, hours_per_year))


def SourceIron(pds, ds, p, ds_fe, method='seperate'):
    """Assign Source Iron.

    Args:
        pds (FelxDataSet): FelxDataSet instance.
        ds (xarray.Dataset): Particle dataset.
        p (int): Particle index.
        ds_fe (xarray.Dataset): Dataset of iron depth profiles.
        method (str, optional): ['combined', 'seperate', 'ML', 'ofam']

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
    if method == 'combined':
        if zone in [1, 2, 3, 4, 6]:
            var = 'llwbcs'
        elif zone in [0, 5] or zone >= 7:
            var = 'interior'
        da_fe = ds_fe[var]

    elif method == 'seperate':
        var = ['interior', 'ngcu', 'nicu', 'mc', 'mc', 'interior', 'nicu', 'interior', 'interior'][zone]
        da_fe = ds_fe[var]

    elif method == 'ofam':
        # Use OFAM3 iron.
        var = 'Fe'
        da_fe = ds_fe[var].sel(time=dx.time, drop=True)

    # Particle location map (e.g., lat, lon, depth)
    particle_loc = dict([[c, dx[c].item()] for c in da_fe.dims])

    fe = da_fe.sel(particle_loc, method='nearest', drop=True).load().item()

    # Assign Source Iron.
    ds['fe'][dict(traj=p, obs=t)] = fe
    # ds['fe_src'][dict(traj=p)] = fe  # For reference.
    return ds


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


@numba.njit
def update_iron_jit(p, t, fe, T, D, Z, P, N, Kd, z, J_max, J_I, k_org, k_inorg, c_scav, mu_D,
                    mu_D_180, gamma_2, mu_P, a, b, c, k_N, k_fe, tau):
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


@timeit(my_logger=logger)
def update_particles_MPI(pds, dss, ds_fe, param_names=None, params=None, NPZD=False):
    """Run iron model for each particle.

    Args:
        pds (FelxDataSet): class instance (contains constants and dataset functions).
        ds (xarray.Dataset): Particle dataset (e.g., lat(traj, obs), zoo(traj, obs), fe_scav(traj, obs), etc).
        ds_fe (xarray.Dataset): Source dFe depth profiles dataset.

    Returns:
        ds (xarray.Dataset): Particle dataset with updated iron, fe_scav, etc.
    """
    ds = dss.copy()
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Distribute particles among processes
        particle_subset = pds.particle_subsets(ds, size)[rank]
        ds = ds.isel(traj=slice(*particle_subset))
        # particles = range(*particle_subset)  # !!! Check range
    else:
        rank = 0
    particles = range(ds.traj.size)
    logger.info('{}: rank={}: particles:{}'.format(pds.exp.file_base, rank, ds.traj.size))

    # Constants.
    if params is not None:
        pds.update_params(dict([(k, v) for k, v in zip(param_names, params)]))

    field_names = ['fe', 'temp', 'det', 'zoo', 'phy', 'no3', 'kd', 'z', 'J_max', 'J_I']
    constant_names = ['k_org', 'k_inorg', 'c_scav', 'mu_D', 'mu_D_180', 'gamma_2', 'mu_P',
                      'a', 'b', 'c', 'k_N', 'k_fe', 'tau']  # 'PAR', 'I_0', 'alpha'
    if NPZD:
        field_names = ['fe', 'temp', 'det', 'zoo', 'phy', 'no3', 'kd', 'z',
                       'J_max', 'J_I', 'dD_dz']
        constant_names = ['k_org', 'k_inorg', 'c_scav', 'mu_D', 'mu_D_180', 'gamma_2', 'mu_P',
                          'a', 'b', 'c', 'k_N', 'k_fe', 'tau', 'g', 'epsilon', 'gamma_1',
                          'mu_Z', 'w_D']  # 'PAR', 'I_0', 'alpha'
        ds = add_particles_dD_dz(ds, pds.exp)

    constants = [pds.params[i] for i in constant_names]

    # Pre calculate data_variables that require function calls.
    ds['J_max'] = pds.a * (pds.b**(pds.c * ds.temp))
    light = pds.PAR * pds.I_0 * np.exp(-ds.z * ds.kd)
    ds['J_I'] = ds.J_max * (1 - np.exp((-pds.alpha * light) / ds.J_max))

    # Run simulation for each particle.
    for p in particles:
        # Assign initial iron.
        ds = SourceIron(pds, ds, p, ds_fe)

        # Number of non-NaN observations.
        timesteps = range(1, ds.z.isel(traj=p).dropna('obs', 'all').obs.size)

        for t in timesteps:
            # Data variables at particle location.
            fields = [ds[i].isel(traj=p, obs=t - 1, drop=True).item() for i in field_names]
            fe, fe_scav, fe_reg, fe_phy = update_iron_jit(p, t, *tuple([*fields, *constants]))
            # fe, fe_scav, fe_reg, fe_phy, P, Z = update_iron_NPZD_jit(p, t, *args)

            ds['fe'][dict(traj=p, obs=t)] = fe
            ds['fe_scav'][dict(traj=p, obs=t - 1)] = fe_scav
            ds['fe_reg'][dict(traj=p, obs=t - 1)] = fe_reg
            ds['fe_phy'][dict(traj=p, obs=t - 1)] = fe_phy

    # Synchronize all processes
    if MPI is not None:

        tmp_files = pds.felx_tmp_filenames(size)
        save_dataset(ds, tmp_files[rank])
        comm.Barrier()

        ds = xr.open_mfdataset(tmp_files)
        if rank == 0:
            for path in tmp_files:
                os.remove(path)

    # if rank == 0 and not cfg.test:

    #     # Write output to netCDF file
    #     encoding = {var: {'zlib': True} for var in ds.data_vars}
    #     with MPI.File.Open(comm, pds.exp.file_felx, amode=MPI.MODE_CREATE | MPI.MODE_WRONLY) as fh:
    #         ds.to_netcdf(cfg.paths.data / (pds.exp.file_base + '_test.nc'), mode='w',
    #                      format='NETCDF4', engine='h5netcdf', encoding=encoding, group=fh)

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


def run_iron_model(NPZD=False):
    """Set up and run Lagrangian iron model.

    Args:
        exp (ExpData instance): Experiment information.

    Returns:
        ds (xarray.Dataset): Particle dataset with updated iron, fe_scav, etc.
    """
    ndays = 6
    scav_eq = ['Galibraith', 'OFAM'][0]
    exp = ExpData(scenario=0, lon=220, test=True, scav_eq=scav_eq)

    # Source Iron fields (observations or OFAM3 Fe).
    dfs = get_merged_FeObsDataset()
    ds_fe_obs = dfs.dfe.ds_avg
    ds_fe = ds_fe_obs

    # Particle dataset
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()
    ds = pds.init_felx_dataset()

    if cfg.test:
        ds = ds.isel(traj=slice(750 * ndays))
        target = np.datetime64('2012-12-31T12') - np.timedelta64((ndays)*6 - 1, 'D')
        traj = ds.traj.where(ds.time.ffill('obs').isel(obs=-1, drop=True) >= target, drop=True)
        ds = ds.sel(traj=traj)

        # # Cut off particles in early years to load less ofam fields.
        # trajs = ds.traj.where(ds.isel(obs=0, drop=True).time.dt.year > 2011, drop=True)
        # ds = ds.sel(traj=trajs)
        # ds = ds.isel(traj=slice(200)).dropna('obs', 'all')

    param_dict = ['{}={}'.format(k, pds.params[k])
                  for k in ['c_scav', 'k_org', 'k_inorg', 'mu_D', 'mu_D_180', 'I_0', 'a', 'b']]
    logger.info('{}({} particles): {}'.format(exp.file_base, ds.traj.size, param_dict))
    pds, ds = update_particles_MPI(pds, ds, ds_fe, NPZD=False)

    # test_plot_iron_paths(pds, ds, ntraj=min(ds.traj.size, 35))
    test_plot_EUC_iron_depth_profile(pds, ds, dfs)
    return ds


@timeit(my_logger=logger)
def optimise_iron_model_params():
    """Optimise Lagrangian iron model paramaters.

    Args:
        exp (ExpData instance): Experiment information.
        ntraj (int): Number of particles.

    Returns:
        params_optimized (list): Optmised paramters.

    Tests:
    hist_250_v0_00(31 particles):
        Optmimal {'c_scav': 2.0, 'k_org': 1e-08, 'k_inorg': 8.2e-06, 'mu_D': 0.01, 'mu_D_180': 0.02})
        optimise_iron_model_params: 01h:01m:26.78s (total=3686.78 seconds).
    hist_250_v0_00(31 particles):
        [k_org mistake]
        Optmimal {'c_scav': 2.0, 'k_org': 1e-08, 'k_inorg': 8.2e-06, 'mu_D': 0.01, 'mu_D_180': 0.02})
        optimise_iron_model_params: 00h:36m:03.90s (total=2163.90 seconds).
    hist_250_v0_00(61 particles):
        Optmimal {'k_org': 5e-07, 'k_inorg': 8.2e-06, 'mu_D_180': 0.02})
        optimise_iron_model_params: 00h:45m:07.72s (total=2707.72 seconds).
    """
    def cost_function(F_obs, pds, ds, dfs, param_names, params):
        pds.update_params(dict([(k, v) for k, v in zip(param_names, params)]))
        F_pred = update_particles_MPI(pds, ds, dfs.dfe.ds_avg, param_names, params)  # Particle dataset (all vars & obs).
        F_pred_f = F_pred.fe.ffill('obs').isel(obs=-1, drop=True)  # Final particle Fe.

        # Error for each particle.
        # cost = np.sqrt(np.sum((F_obs - F_pred)**2) / F_obs.size)  # RSMD
        cost = np.fabs((F_obs - F_pred_f)).mean().load().item()  # Least absolute deviations

        if rank == 0:
            # Log & plot.
            logger.info('{}({} particles): cost={} {}'
                        .format(exp.file_base, ds.traj.size, cost,
                                ['{}={}'.format(p, v) for p, v in zip(param_names, params)]))
            test_plot_EUC_iron_depth_profile(pds, F_pred, dfs)
        return cost

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    # Particle dataset.
    exp = ExpData(scenario=0, lon=220, scav_eq='Galibraith')
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    if rank == 0:
        ds = pds.init_felx_optimise_dataset()

        # # Subset number of particles.
        # ndays = 24
        # ds = ds.isel(traj=slice(742 * ndays))
        # target = np.datetime64('{}-12-31T12'.format(exp.year_bnds[-1])) - np.timedelta64((ndays) * 6 - 1, 'D')
        # traj = ds.traj.where(ds.time.ffill('obs').isel(obs=-1, drop=True) >= target, drop=True)
        # ds = ds.sel(traj=traj)
    else:
        ds = None

    if MPI is not None:
        ds = comm.bcast(ds, root=0)

    # Source iron profile.
    dfs = get_merged_FeObsDataset()

    # Fe observations at particle depths.
    z = ds.z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
    F_obs = dfs.dfe.ds_avg.euc_avg.sel(lon=pds.exp.lon, z=z, method='nearest', drop=True)

    # Paramaters to optimise (e.g., a, b, c = params).
    # Copy inside update_iron: ', '.join([i for i in params])
    param_names = ['c_scav', 'k_inorg', 'k_org']
    params_init = [pds.params[i] for i in param_names]  # params = params_init

    param_bnds = dict(k_org=[1e-8, 1e-3], k_inorg=[1e-8, 1e-3], c_scav=[1.5, 2.5], tau=None,
                      mu_D=[0.005, 0.04], mu_D_180=[0.005, 0.03], mu_P=[0.005, 0.02],
                      gamma_2=[0.005, 0.02], I_0=[280, 350], alpha=None, PAR=None,
                      a=None, b=[0.8, 1.2], c=[0.5, 1.5], k_fe=[0.5, 1.5], k_N=[0.5, 1.5],
                      gamma_1=None, g=None, epsilon=None, mu_Z=None)
    bounds = [param_bnds[i] for i in param_names]

    res = minimize(lambda params: cost_function(F_obs, pds, ds, dfs, param_names, params),
                   params_init, method='powell', bounds=bounds, options={'disp': True})
    params_optimized = res.x

    # Update pds params dict with optimised values.
    if rank == 0:
        param_dict = dict([(k, v) for k, v in zip(param_names, params_optimized)])
        pds.update_params(param_dict)
        logger.info('{}({} particles): Init {}'.format(exp.file_base, ds.traj.size, params_init))
        logger.info('{}({} particles): Optimimal {}'.format(exp.file_base, ds.traj.size, param_dict))
        # # Calculate the predicted iron concentration using the optimized parameters.
        # ds_opt = update_particles_MPI(pds, ds, dfs, param_names, params_optimized)

        # # Plot the observed and predicted iron concentrations.
        # test_plot_EUC_iron_depth_profile(pds, ds_opt, dfs)
    return



if __name__ == '__main__' and not cfg.test:

    if MPI is not None:
        optimise_iron_model_params()
