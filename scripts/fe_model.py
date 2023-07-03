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
from argparse import ArgumentParser
import gc
import numba
import numpy as np
import xarray as xr  # NOQA

from cfg import ExpData, test
from datasets import save_dataset
from tools import mlogger, timeit
from fe_exp import FelxDataSet
from fe_obs_dataset import iron_source_profiles
#from plot_fe_model import test_plot_EUC_iron_depth_profile, test_plot_iron_paths

try:
    from mpi4py import MPI

except ModuleNotFoundError:
    MPI = None


logger = mlogger('fe_model', level=10)


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
    zone = int(dx.zone.load().item())

    # Option 1: Select region data variable based on source ID.
    if method == 'LLWBC-obs':
        var = ['interior', 'png', 'nicu', 'mc', 'mc', 'interior', 'nicu', 'interior', 'interior'][zone]
        da_fe = ds_fe[var]

    elif method == 'LLWBC-proj':
        var = ['interior', 'png_proj', 'nicu_proj', 'mc_proj', 'mc_proj', 'interior', 'nicu_proj',
               'interior', 'interior'][zone]
        da_fe = ds_fe[var]

    elif method == 'LLWBC-low':
        da_fe = ds_fe['interior']

    elif method == 'depth-mean':
        var = ['interior', 'png', 'nicu', 'mc', 'mc', 'interior', 'nicu', 'interior', 'interior'][zone]
        da_fe = ds_fe[var + '_mean']

    elif method == 'LLWBC-mean':
        var = 'llwbcs' if zone in [1, 2, 3, 4, 6] else 'interior'
        da_fe = ds_fe[var]

    elif method == 'ofam':
        # Use OFAM3 iron.
        da_fe = ds_fe['Fe'].sel(time=dx.time, drop=True)

    # Particle location map (e.g., lat, lon, depth)
    particle_loc = dict([[c, dx[c].load().item()] for c in da_fe.dims])

    fe = da_fe.sel(particle_loc, method='nearest', drop=True).load().item()

    # Assign Source Iron.
    ds['fe'][dict(traj=p, obs=t)] = fe

    return ds


@numba.njit
def update_iron_jit(p, t, fe, T, D, Z, P, N, z, J_max, J_I, k_org, k_inorg, c_scav, mu_D,
                    mu_D_180, gamma_2, mu_P, b, c, k_N, k_fe):
    bcT = b**(c * T)  # [no units]
    if z < 180:
        mu_D = mu_D * bcT  # [day^-1]
    else:
        mu_D = mu_D_180 * bcT  # [day^-1]

    # # Nitrate limit on phytoplankton growth rate
    # J_limit_N = N / (N + k_N)  # [no units] (mmol N m^-3 / mmol N m^-3)

    # # Iron limit on phytoplankton growth rate
    # J_limit_Fe = fe / (fe + (0.02 * k_fe))  # [no units] (umol Fe m^-3 / (umol Fe m^-3 + umol Fe m^-3))

    # # Light limit on phytoplankton growth rate
    # J_limit_I = J_I / J_max  # [no units] (day^-1 / day^-1)

    # Phytoplankton growth rate [day^-1] ((day^-1 * const)^const)
    J = J_max * min(J_I / J_max, N / (N + k_N), fe / (fe + (0.02 * k_fe)), 0)  # [day^-1]

    # Iron Phytoplankton Uptake
    fe_phy = 0.02 * J * P  # [umol Fe m^-3 day^-1] (0.02 * day^-1 * mmol N m^-3)

    # Iron Remineralisation
    fe_reg = 0.02 * (mu_D * D + gamma_2 * bcT * Z + mu_P * bcT * P)  # [umol Fe m^-3 day^-1]

    # Iron Scavenging
    fe_scav = (fe * k_org * (0.02 * D)**0.58) + (k_inorg * fe**c_scav)  # [umol Fe m^-3 day^-1]
    # fe_scav = tau * max(0, fe - 0.6)  # [umol Fe m^-3  day^-1]

    fe_phy = max([fe_phy, 0])
    fe_scav = max([fe_scav, 0])
    fe_reg = max([fe_reg, 0])

    # Iron
    fe = fe + 2 * (fe_reg - fe_phy - fe_scav)
    fe = max([fe, 0])
    return fe, fe_scav, fe_reg, fe_phy


@timeit(my_logger=logger)  # Turn off for optimisation
def update_particles_MPI(pds, ds, ds_fe, param_names=None, params=None):
    """Run iron model for each particle.

    Args:
        pds (FelxDataSet): class instance (contains constants and dataset functions).
        ds (xarray.Dataset): Particle dataset (e.g., lat(traj, obs), zoo(traj, obs), fe_scav(traj, obs), etc).
        ds_fe (xarray.Dataset): Source dFe depth profiles dataset.

    Returns:
        ds (xarray.Dataset): Particle dataset with updated iron, fe_scav, etc.
    """
    rank = 0
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        logger.debug('{}: p={}: Subsetting dataset.'.format(pds.exp.file_felx.stem, ds.traj.size))

        # Distribute particles among processes
        if rank == 0:
            particle_subsets = pds.particle_subsets(ds, size)
        else:
            particle_subsets = None

        particle_subsets = comm.bcast(particle_subsets if rank == 0 else None, root=0)
        particle_subset = particle_subsets[rank]
        ds = ds.isel(traj=slice(*particle_subset))

    particles = range(ds.traj.size)

    # Constants.
    if params is not None:
        pds.update_params(dict([(k, v) for k, v in zip(param_names, params)]))

    field_names = ['fe', 'temp', 'det', 'zoo', 'phy', 'no3', 'z', 'J_max', 'J_I']
    constant_names = ['k_org', 'k_inorg', 'c_scav', 'mu_D', 'mu_D_180', 'gamma_2', 'mu_P',
                      'b', 'c', 'k_N', 'k_fe']  # 'PAR', 'I_0', 'alpha''a', kd
    constants = [pds.params[i] for i in constant_names]

    # Pre calculate data_variables that require function calls.
    logger.debug('{}: Calculating J_max and J_I.'.format(pds.exp.file_felx.stem))

    ds['J_max'] = pds.a * np.power(pds.b, (pds.c * ds.temp))
    light = pds.PAR * pds.I_0 * np.exp(-ds.z * np.power(10, ds.kd))
    ds['J_I'] = ds.J_max * (1 - np.exp((-pds.alpha * light) / ds.J_max))

    # ds['J_max'] = pds.a * xr.apply_ufunc(np.power, pds.b, (pds.c * ds.temp))
    # # N.B. Use 10**Kd instead of Kd because it was scaled by log10.
    # light = pds.PAR * pds.I_0 * xr.apply_ufunc(np.exp, -ds.z * xr.apply_ufunc(np.power, 10, ds.kd))
    # ds['J_I'] = ds.J_max * (1 - xr.apply_ufunc(np.exp, (-pds.alpha * light) / ds.J_max))

    # Free memory
    del light
    ds = ds.drop('kd')
    ds = ds.drop('u')
    gc.collect()

    # Run simulation for each particle.
    logger.info('{}: particles={}: Updating iron.'.format(pds.exp.id, ds.traj.size))

    for p in particles:
        # Assign initial iron.
        ds = SourceIron(pds, ds, p, ds_fe, method=pds.exp.source_iron)

        # Number of non-NaN observations.
        timesteps = range(1, ds.z.isel(traj=p).dropna('obs', 'all').obs.size)

        for t in timesteps:
            # Data variables at particle location.
            fields = [ds[i].isel(traj=p, obs=t - 1, drop=True).load().item() for i in field_names]
            fe, fe_scav, fe_reg, fe_phy = update_iron_jit(p, t, *tuple([*fields, *constants]))

            ds['fe'][dict(traj=p, obs=t)] = fe
            ds['fe_scav'][dict(traj=p, obs=t - 1)] = fe_scav
            ds['fe_reg'][dict(traj=p, obs=t - 1)] = fe_reg
            ds['fe_phy'][dict(traj=p, obs=t - 1)] = fe_phy
            del fields
            gc.collect()

    if MPI is not None:
        logger.info('{}: Saving dataset...'.format(pds.exp.file_felx.stem))

        # Save proc dataset as tmp file.
        tmp_files = pds.felx_tmp_filenames(size)
        ds = ds.chunk()

        # Drop existing variables.
        dvars = [v for v in ds.data_vars if v not in ['trajectory', *pds.variables]]
        ds = ds.drop(dvars)
        ds = ds.where(~np.isnan(ds.trajectory))  # Fill empty with NaNs

        # Set variable data type (otherwise dtype='object')
        for var in pds.variables:
            ds[var] = ds[var].astype(dtype=np.float64)

        # Save output as temp dataset.
        ds.attrs['history'] += str(np.datetime64('now', 's')).replace('T', ' ')
        ds.attrs['history'] += ' Iron model ({})'.format(pds.exp.source_iron)
        ds.to_netcdf(tmp_files[rank], compute=True)
        logger.info('{}: Saved dataset.'.format(pds.exp.file_felx.stem))
        ds.close()

    return ds


@timeit(my_logger=logger)
def run_iron_model(pds):
    """Set up and run Lagrangian iron model.

    Args:
        exp (ExpData instance): Experiment information.

    Returns:
        ds (xarray.Dataset): Particle dataset with updated iron, fe_scav, etc.
    """
    logger.debug('{}: Running update_particles_MPI ({})'.format(pds.exp.file_felx.stem, pds.exp.source_iron))

    # Source Iron fields (observations).
    ds_fe = iron_source_profiles()

    # Particle dataset
    pds.add_iron_model_params()

    logger.debug('{}: Initializing dataset...'.format(pds.exp.file_felx.stem))
    ds = pds.init_felx_dataset()

    logger.debug('{}: Dataset initializion complete.'.format(pds.exp.file_felx.stem))

    if pds.exp.source_iron not in ['LLWBC-obs', 'depth-mean']:
        logger.debug('{}: Dropping background sources p={}.'.format(pds.exp.file_felx.stem, ds.traj.size))
        pid_source_map = np.load(pds.exp.file_source_map, allow_pickle=True).item()
        pids = np.concatenate([pid_source_map[i] for i in [1, 2, 3, 4, 6]])
        ds = ds.sel(traj=pids)
        logger.debug('{}: Dropped background sources p={}.'.format(pds.exp.file_felx.stem, ds.traj.size))

    # Run iron model for particles.
    ds = update_particles_MPI(pds, ds, ds_fe)
    return ds


if __name__ == '__main__':
    p = ArgumentParser(description="""Optimise iron model paramaters.""")
    p.add_argument('-x', '--lon', default=250, type=int,
                   help='Release longitude [165, 190, 220, 250].')
    p.add_argument('-s', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-v', '--version', default=0, type=int, help='Version index.')
    p.add_argument('-r', '--index', default=7, type=int, help='File repeat index [0-7].')
    p.add_argument('-f', '--func', default='run', type=str, help='run, fix or save.')
    args = p.parse_args()
    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    source_iron = ['LLWBC-obs', 'LLWBC-low', 'LLWBC-proj', 'depth-mean', 'LLWBC-mean'][version]

    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, source_iron=source_iron)
    pds = FelxDataSet(exp)

    if args.func == 'run':
        ds = run_iron_model(pds)

    # elif args.func == 'fix':
    #     fix_particles_v0_err(pds)

    elif args.func == 'save':
        if pds.exp.version in [0, 3]:
            felx_file_tmp = pds.exp.out_subdir / 'tmp_err/{}'.format(pds.exp.file_felx.name)
            if felx_file_tmp.exists():
                ds = pds.save_felx_dataset_particle_subset(v0_error_fix=True)
            else:
                ds = pds.save_felx_dataset()
        else:
            ds = pds.save_felx_dataset_particle_subset()

        ds_fe = iron_source_profiles()

        # Log Cost
        z = ds.z.ffill('obs').isel(obs=-1)
        fe_obs = ds_fe.euc_avg.sel(lon=pds.exp.lon, z=z, method='nearest', drop=True)
        fe_pred = ds.fe.ffill('obs').isel(obs=-1, drop=True)
        cost = np.fabs((fe_obs - fe_pred)).weighted(ds.u).mean().load().item()
        logger.info('{}: p={}: cost={}:'.format(pds.exp.id, ds.traj.size, cost))

        # Plot output
        # test_plot_EUC_iron_depth_profile(pds, ds, ds_fe)
        # test_plot_iron_paths(pds, ds, ntraj=min(ds.traj.size, 35))
