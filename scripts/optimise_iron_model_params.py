# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

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
from tools import mlogger, timeit
from felx_dataset import FelxDataSet
from felx import update_particles_MPI
from fe_obs_dataset import get_merged_FeObsDataset
from test_iron_model_optimisation import (test_plot_iron_paths, test_plot_EUC_iron_depth_profile)

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None
    rank = 0

logger = mlogger('optimise_iron_model_params')


def cost_function(pds, ds, fe_obs, dfs, params):
    """Cost function for iron model optmisation (uses weighted least absolute deviations)."""
    pds.update_params(dict([(k, v) for k, v in zip(pds.param_names, params)]))

    # Particle dataset (all vars & obs).
    fe_pred = update_particles_MPI(pds, ds, dfs.dfe.ds_avg, pds.param_names, params)

    # Final particle Fe.
    fe_pred_f = fe_pred.fe.ffill('obs').isel(obs=-1, drop=True)

    # Error for each particle.
    cost = np.fabs((fe_obs - fe_pred_f)).weighted(ds.u).mean()  # Least absolute deviations
    # cost = np.sqrt(np.sum((F_obs - F_pred)**2) / F_obs.size)  # RSMD
    cost = cost.load().item()

    if rank == 0:
        logger.info('{}: p={}: cost={} {}'.format(pds.exp.file_base, ds.traj.size, cost,
                                                  ['{}={}'.format(p, v) for p, v in zip(pds.param_names, params)]))
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
    dfs = get_merged_FeObsDataset()

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

    logger.info('felx_hist: Optimisation {} method - init {}'.format(method, params_init))

    res = minimize(lambda params: cost_function(pds, ds, fe_obs, dfs, params),
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
        ds_opt = update_particles_MPI(pds, ds, dfs, pds.param_names, params_optimized)

        # Plot the observed and predicted iron concentrations.
        test_plot_EUC_iron_depth_profile(pds, ds_opt, dfs)
    return res


@timeit(my_logger=logger)
def optimise_iron_model_params_multi_lon(method):
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

    # Source iron profile.
    dfs = get_merged_FeObsDataset()
    pds = FelxDataSet(ExpData(scenario=0, lon=0))

    lons = [165, 190, 220, 250]
    n = len(lons)
    pds_list = [None] * n
    ds_list = [None] * n
    fe_obs_list = [None] * n

    # Particle dataset.
    if rank == 0:
        for i in range(n):
            # Experiment class.
            pds_list[i] = FelxDataSet(ExpData(scenario=0, lon=lons[i]))

            # Particle dataset.
            ds_list[i] = pds_list[i].init_felx_optimise_dataset()
            if cfg.test:
                ds_list[i] = ds_list[i].isel(traj=slice(10))

            # Fe observations at particle depths.
            z = ds_list[i].z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
            fe_obs_list[i] = dfs.dfe.ds_avg.euc_avg.sel(lon=lons[i], z=z, method='nearest', drop=True)

        # Merge datasets.
        for i in range(1, n):
            id_next = int(ds_list[i - 1].traj[-1].load().item()) + 1
            id_new = list(range(id_next, id_next + ds_list[i].traj.size))
            ds_list[i]['traj'] = id_new
            fe_obs_list[i]['traj'] = id_new

        ds = xr.concat(ds_list, 'traj')
        fe_obs = xr.concat(fe_obs_list, 'traj')

    else:
        fe_obs = None
        ds = None

    if MPI is not None:
        ds = comm.bcast(ds, root=0)
        fe_obs = comm.bcast(fe_obs, root=0)

    # Paramaters to optimise (e.g., a, b, c = params).
    pds.param_names = ['c_scav', 'k_inorg', 'k_org', 'mu_D', 'mu_D_180']

    params_init = [pds.params[i] for i in pds.param_names]  # params = params_init

    param_bnds = dict(k_org=[1e-8, 1e-3], k_inorg=[1e-8, 1e-3], c_scav=[1.5, 2.5], tau=None,
                      mu_D=[0.005, 0.04], mu_D_180=[0.005, 0.03], mu_P=[0.005, 0.02],
                      gamma_2=[0.005, 0.02], I_0=[280, 350], alpha=None, PAR=None,
                      a=None, b=[0.8, 1.2], c=[0.5, 1.5], k_fe=[0.5, 1.5], k_N=[0.5, 1.5],
                      gamma_1=None, g=None, epsilon=None, mu_Z=None)
    bounds = [tuple(param_bnds[i]) for i in pds.param_names]

    logger.info('felx_hist: Optimisation {} method - init {}'.format(method, params_init))
    res = minimize(lambda params: cost_function(pds, ds, fe_obs, dfs, params),
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
    if lon is not None:
        res = optimise_iron_model_params(lon, method=method)
    else:
        res = optimise_iron_model_params_multi_lon(method=method)
