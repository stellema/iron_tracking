# -*- coding: utf-8 -*-
"""Functions to optmimise iron model parameters.

Notes:

Example:

Todo:


hist_165_v0_00: p=2135: cost=0.3138301968574524 {'c_scav': 2.5, 'k_inorg': 1.000000000000001e-08, 'k_org': 0.00018144043748232996, 'mu_D': 0.03999999999999999, 'mu_D_180': 0.03}
hist_165_v0_00: p=2135, method=Nelder-Mead, nit=37, nfev=59, success=True, message=Optimization terminated successfully.
hist_165_v0_00: Optimimal {'c_scav': 2.5, 'k_inorg': 1.000000000000001e-08, 'k_org': 0.00018144043748232996, 'mu_D': 0.03999999999999999, 'mu_D_180': 0.03}

hist_190_v0_00: p=2782: cost=0.2835429012775421 {'c_scav': 2.0, 'k_inorg': 0.0006257817077636716, 'k_org': 0.00010003890991210938, 'mu_D': 0.02000781250464934, 'mu_D_180': 0.010062507627066224}

hist_220_v0_00: p=3211: cost=0.21084801852703094 {'c_scav': 2.5, 'k_inorg': 0.001, 'k_org': 2.6156331054350042e-05, 'mu_D': 0.03999999999999993, 'mu_D_180': 0.03}
hist_220_v0_00: p=3211, method=Nelder-Mead, nit=55, nfev=82, success=True, message=Optimization terminated successfully.
hist_220_v0_00: Optimimal {'c_scav': 2.5, 'k_inorg': 0.001, 'k_org': 2.6156331054350042e-05, 'mu_D': 0.03999999999999993, 'mu_D_180': 0.03}

hist_250_v0_00: p=3294: cost=0.30697357654571533 {'c_scav': 1.5, 'k_inorg': 0.001, 'k_org': 0.001, 'mu_D': 0.005, 'mu_D_180': 0.03}
hist_250_v0_00: p=3294, method=Nelder-Mead, nit=59, nfev=90, success=True, message=Optimization terminated successfully.
hist_250_v0_00: Optimimal {'c_scav': 1.5, 'k_inorg': 0.001, 'k_org': 0.001, 'mu_D': 0.005, 'mu_D_180': 0.03}


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed May 10 17:27:31 2023

"""
from argparse import ArgumentParser
import numpy as np
from scipy.optimize import minimize
import xarray as xr  # NOQA

import cfg  # NOQA
from cfg import ExpData
from datasets import save_dataset
from felx_dataset import FelxDataSet
from felx import update_particles_MPI
from fe_obs_dataset import iron_source_profiles
from test_iron_model_optimisation import (test_plot_iron_paths, test_plot_EUC_iron_depth_profile)
from tools import mlogger, timeit

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None
    rank = 0

logger = mlogger('optimise_iron_model_params')


def cost_function(pds, ds, fe_obs, dfs, params, rank=0):
    """Cost function for iron model optmisation (uses weighted least absolute deviations)."""
    params_dict = dict([(k, v) for k, v in zip(pds.param_names, params)])
    pds.update_params(params_dict)

    # Particle dataset (all vars & obs).
    fe_pred = update_particles_MPI(pds, ds, dfs.dfe.ds_avg, pds.param_names, params)

    # Final particle Fe.
    fe_pred = fe_pred.fe.ffill('obs').isel(obs=-1, drop=True)

    # Error for each particle.
    cost = np.fabs((fe_obs - fe_pred)).weighted(ds.u).mean()  # Least absolute deviations
    # cost = np.sqrt(np.sum((F_obs - F_pred)**2) / F_obs.size)  # RSMD
    cost = cost.load().item()

    if rank == 0:
        logger.info('{}: p={}: cost={} {}'.format(pds.exp.file_base, ds.traj.size, cost, params_dict))
    fe_pred.close()
    return cost


def optimise_iron_model_params(lon, method):
    """Optimise Lagrangian iron model paramaters.

    Args:
        exp (ExpData instance): Experiment information.
        ntraj (int): Number of particles.

    Returns:
        params_optimized (list): Optmised paramters.

    """
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    # Particle dataset.
    exp = ExpData(scenario=0, lon=lon)
    pds = FelxDataSet(exp)
    pds.add_iron_model_params()

    if rank == 0:
        logger.info('felx_hist: Optimisation method={}'.format(method))
        ds = pds.init_felx_optimise_dataset()

        # Subset number of particles.
        if cfg.test:
            ndays = 1
            ds = ds.isel(traj=slice(742 * ndays))
            target = np.datetime64('{}-12-31T12'.format(exp.year_bnds[-1])) - np.timedelta64((ndays) * 6 - 1, 'D')
            traj = ds.traj.where(ds.time.ffill('obs').isel(obs=-1, drop=True) >= target, drop=True)
            ds = ds.sel(traj=traj)
    else:
        ds = None

    if MPI is not None:
        ds = comm.bcast(ds, root=0)

    # Source iron profile.
    dfs = iron_source_profiles()

    # Fe observations at particle depths.
    z = ds.z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
    fe_obs = dfs.dfe.ds_avg.euc_avg.sel(lon=pds.exp.lon, z=z, method='nearest', drop=True)

    # Paramaters to optimise (e.g., a, b, c = params).
    # Copy inside update_iron: ', '.join([i for i in params])
    pds.param_names = ['c_scav', 'k_inorg', 'k_org', 'mu_D', 'mu_D_180']

    params_init = [pds.params[i] for i in pds.param_names]  # params = params_init

    param_bnds = dict(k_org=[1e-8, 1e-3], k_inorg=[1e-8, 1e-3], c_scav=[1.5, 2.5], tau=None,
                      mu_D=[0.005, 0.04], mu_D_180=[0.005, 0.03], mu_P=[0.005, 0.02],
                      gamma_2=[0.005, 0.02], I_0=[280, 350], alpha=None, PAR=None,
                      a=None, b=[0.8, 1.2], c=[0.5, 1.5], k_fe=[0.5, 1.5], k_N=[0.5, 1.5],
                      gamma_1=None, g=None, epsilon=None, mu_Z=None)
    bounds = [tuple(param_bnds[i]) for i in pds.param_names]

    res = minimize(lambda params: cost_function(pds, ds, fe_obs, dfs, params, rank),
                   params_init, method=method, bounds=bounds, options={'disp': True})
    params_optimized = res.x

    # Update pds params dict with optimised values.
    if rank == 0:
        param_dict = dict([(k, v) for k, v in zip(pds.param_names, params_optimized)])
        pds.update_params(param_dict)

        logger.info('{}: p={}, method={}, nit={}, nfev={}, success={}, message={}'
                    .format(exp.file_base, ds.traj.size, method, res.nit, res.nfev, res.success, res.message))
        logger.info('{}: Init {}'.format(exp.file_base, params_init))
        logger.info('{}: Optimimal {}'.format(exp.file_base, param_dict))

        # Calculate the predicted iron concentration using the optimized parameters.
        ds_opt = update_particles_MPI(pds, ds, dfs.dfe.ds_avg, pds.param_names, params_optimized)

        # Plot the observed and predicted iron concentrations.
        test_plot_EUC_iron_depth_profile(pds, ds_opt, dfs)
    return res


# @timeit(my_logger=logger)
def optimise_multi_lon_dataset():
    """Save dataset for multi-lon paramater optimisation."""
    # Source iron profile.
    dfs = iron_source_profiles()
    file_ds = cfg.paths.data / 'felx/felx_optimise_hist_multi.nc'
    file_obs = cfg.paths.data / 'felx/fe_obs_optimise_hist_multi.nc'

    if file_ds.exists() and file_obs.exists():
        ds = xr.open_dataset(file_ds)
        fe_obs = xr.open_dataset(file_obs)
        return dfs, ds, fe_obs

    lons = [165, 190, 220, 250]
    n = len(lons)
    ds_list = [None] * n
    fe_obs_list = [None] * n

    # Particle dataset.
    for i in range(n):
        logger.info('Init dataset: {}'.format(i))
        # Experiment class.
        pds = FelxDataSet(ExpData(scenario=0, lon=lons[i]))

        if cfg.test:
            ds_list[i] = ds_list[i].isel(traj=slice(10))

        # Particle dataset.
        ds_list[i] = pds.init_felx_optimise_dataset()

        # Fe observations at particle depths.
        z = ds_list[i].z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
        fe_obs_list[i] = dfs.dfe.ds_avg.euc_avg.sel(lon=lons[i], z=z, method='nearest', drop=True)

    # Merge datasets.
    logger.info('Merging datasets')
    for i in range(1, n):
        id_next = int(ds_list[i - 1].traj[-1].load().item()) + 1
        id_new = list(range(id_next, id_next + ds_list[i].traj.size))
        ds_list[i]['traj'] = id_new
        fe_obs_list[i]['traj'] = id_new

    logger.info('Concat datasets')
    ds = xr.concat(ds_list, 'traj')
    fe_obs = xr.concat(fe_obs_list, 'traj')

    logger.info('Save datasets')
    save_dataset(ds, file_ds)
    save_dataset(fe_obs, file_obs)
    logger.info('Datasets saved')
    return dfs, ds, fe_obs


@timeit(my_logger=logger)
def optimise_iron_model_params_multi_lon(lon, method):
    """Optimise Lagrangian iron model paramaters.

    Args:
        exp (ExpData instance): Experiment information.
        ntraj (int): Number of particles.

    Returns:
        params_optimized (list): Optmised paramters.

    Notes:
        params not worth optmising:
            - alpha, k_N, k_fe
        params worth optmising:
            - I_0, PAR, a, b, c, gamma_2, mu_P

    """
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    dfs, ds, fe_obs = optimise_multi_lon_dataset()
    fe_obs = fe_obs['euc_avg']

    if cfg.test:
        ds = ds.isel(traj=slice(10))

    # Source iron profile.
    pds = FelxDataSet(ExpData(scenario=0, lon=lon))

    # Paramaters to optimise (e.g., a, b, c = params).
    #pds.param_names = ['c_scav', 'k_inorg', 'k_org', 'I_0', 'PAR', 'mu_D', 'mu_D_180', 'gamma_2', 'mu_P', 'a', 'b', 'c']
    pds.param_names = ['c_scav', 'k_inorg', 'k_org', 'mu_D', 'mu_D_180']
    params_init = [pds.params[i] for i in pds.param_names]  # params = params_init

    param_bnds = dict(k_org=[1e-8, 1e-3], k_inorg=[1e-8, 1e-3], c_scav=[1.5, 2.5],
                      mu_D=[0.005, 0.03], mu_D_180=[0.005, 0.03], mu_P=[0.005, 0.02],
                      gamma_2=[0.005, 0.02], I_0=[280, 350], PAR=[0.323, 0.5375],
                      a=[0.45, 0.75], b=[0.8, 1.33], c=[0.75, 1.25],
                      alpha=[0.02, 0.03], k_fe=[0.75, 1.25], k_N=[0.75, 1.25],
                      tau=None, gamma_1=None, g=None, epsilon=None, mu_Z=None)
    bounds = [tuple(param_bnds[i]) for i in pds.param_names]

    if rank == 0:
        logger.info('felx_hist: Optimisation {} method - init {}'.format(method, params_init))
    res = minimize(lambda params: cost_function(pds, ds, fe_obs, dfs, params, rank),
                   params_init, method=method, bounds=bounds, options={'disp': True})
    params_optimized = res.x

    # Update pds params dict with optimised values.
    if rank == 0:
        param_dict = dict([(k, v) for k, v in zip(pds.param_names, params_optimized)])
        pds.update_params(param_dict)

        logger.info('felx_hist: p={}, method={}, nit={}, nfev={}, success={}, message={}'
                    .format(ds.traj.size, method, res.nit, res.nfev, res.success, res.message))
        logger.info('felx_hist: Init {}'.format(params_init))
        logger.info('felx_hist: Optimimal {}'.format(param_dict))
    return res


if __name__ == '__main__':
    p = ArgumentParser(description="""Optimise iron model paramaters.""")
    p.add_argument('-x', '--lon', default=220, type=int,
                   help='Release longitude [165, 190, 220, 250].')
    args = p.parse_args()
    lon = args.lon
    method = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC'][0]
    logger = mlogger('optimise_iron_model_params_{}'.format(lon))
    # optimise_multi_lon_dataset()
    if lon in [165, 190, 220, 250]:
        res = optimise_iron_model_params(lon, method=method)
    else:
        res = optimise_iron_model_params_multi_lon(lon, method=method)
