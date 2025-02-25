# -*- coding: utf-8 -*-
"""Functions to optmimise iron model parameters.

Notes:
    Multi-lon:
    cost=0.29325631260871887: c_scav=2.5 k_inorg=0.001 k_org=0.001 mu_D=0.005 mu_D_180=0.03 (diff PAR & I_0)
    cost=0.2939159870147705: c_scav=2.5 k_inorg=0.001 k_org=0.0009568104638292959 mu_D=0.00, mu_D_180=0.03
    cost=0.29445257782936096: c_scav=2.5 k_inorg=0.001 k_org=1.4780934529146143e-05 mu_D=0.005 mu_D_180=0.03
    cost=0.29581040143966675: c_scav=2.499546896146215 k_inorg=0.001 k_org=0.00099 mu_D=0.01 mu_D_180=0.01
    cost=0.2875936031341553: c_scav=2.5 k_inorg=0.001 k_org=0.0001171 I_0=280 PAR=0.3391 mu_D=0.0175
                             mu_D_180=0.0121317 gamma_2=0.0059699 mu_P=0.014364 a=0.48645 b=0.8 c=1.1398
    Single lon:
    hist_190: p=2782: cost=0.2835429012775421: c_scav=2.0 k_inorg=0.0006257817077636716
                        k_org=0.00010003890991210938 mu_D=0.02000781250464934 mu_D_180=0.0100625
    hist_220: p=3211: cost=0.21084801852703094: c_scav=2.5 k_inorg=0.001 k_org=2.6156331054350042e-05
                        mu_D=0.03999999999999993 mu_D_180=0.03
    hist_250: p=3294: cost=0.30697357654571: c_scav=1.5 k_inorg=0.001 k_org=0.001 mu_D=0.005 mu_D_180=0.03


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed May 10 17:27:31 2023

"""
from argparse import ArgumentParser
import numpy as np
import os
from scipy.optimize import minimize
import xarray as xr

import cfg
from cfg import ExpData
from datasets import save_dataset
from fe_exp import FelxDataSet
from fe_model import update_particles_MPI
from fe_obs_dataset import iron_source_profiles
from tools import mlogger, timeit
from plot_fe_model import test_plot_EUC_iron_depth_profile

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None
    rank = 0

logger = mlogger('optimise_iron_model_params')


def cost_function(params, pds, ds, fe_obs, ds_fe, comm):
    """Cost function for iron model optmisation (uses weighted least absolute deviations)."""
    params_dict = dict([(k, v) for k, v in zip(pds.param_names, params)])
    pds.update_params(params_dict)

    # Particle dataset (only fe vars).
    fe_pred = update_particles_MPI(pds, ds, ds_fe, pds.param_names, params, comm=comm)

    if MPI is not None:
        # Synchronize all processes.
        rank = comm.Get_rank()
        comm.Barrier()
        tmp_files = pds.felx_tmp_filenames(comm.Get_size())
        fe_pred = xr.open_mfdataset(tmp_files)
    else:
        rank = 0

    # Final particle Fe.
    fe_pred = fe_pred.fe.ffill('obs').isel(obs=-1, drop=True)

    # Error for each particle.
    # cost = np.fabs((fe_obs - fe_pred)).weighted(ds.u).mean()  # Least absolute deviations
    cost = np.fabs((fe_obs - fe_pred)).mean()  # Least absolute deviations
    # cost = np.sqrt(np.sum((F_obs - F_pred)**2) / F_obs.size)  # RSMD
    cost = cost.load().item() * 10

    if rank == 0:
        logger.info('{}: p={}: cost={} {}'.format(pds.exp.id, ds.traj.size, cost, params_dict))
    fe_pred.close()

    if MPI is not None:
        if tmp_files[rank].exists():  # Delete tmp_file
            os.remove(tmp_files[rank])
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
        comm = None
        rank = 0

    # Particle dataset.
    exp = ExpData(scenario=0, lon=lon, version=9)
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

    if cfg.test:
        ds = ds.isel(traj=slice(30))

    # Source iron profile.
    ds_fe = iron_source_profiles()

    # Fe observations at particle depths.
    z = ds.z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
    fe_obs = ds_fe.euc_avg.sel(lon=pds.exp.lon, z=z, method='nearest', drop=True)

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

    res = minimize(cost_function, params_init, args=(pds, ds, fe_obs, ds_fe, comm),
                   method=method, bounds=bounds, options={'disp': True})
    params_optimized = res.x

    # Update pds params dict with optimised values.
    if rank == 0:
        param_dict = dict([(k, v) for k, v in zip(pds.param_names, params_optimized)])
        pds.update_params(param_dict)

        logger.info('{}: p={}, method={}, nit={}, nfev={}, success={}, message={}'
                    .format(exp.id, ds.traj.size, method, res.nit, res.nfev, res.success, res.message))
        logger.info('{}: Init {}'.format(exp.id, params_init))
        logger.info('{}: Optimimal {}'.format(exp.id, param_dict))

        # Calculate the predicted iron concentration using the optimized parameters.
        ds_opt = update_particles_MPI(pds, ds, ds_fe, pds.param_names, params_optimized, comm=comm)

        # Plot the observed and predicted iron concentrations.
        test_plot_EUC_iron_depth_profile(pds, ds_opt, ds_fe)
    return res


# @timeit(my_logger=logger)
def optimise_multi_lon_dataset():
    """Save dataset for multi-lon paramater optimisation."""
    # Source iron profile.
    ds_fe = iron_source_profiles()
    file_ds = cfg.paths.data / 'fe_model/felx_optimise_hist_multi.nc'
    file_obs = cfg.paths.data / 'fe_model/fe_obs_optimise_hist_multi.nc'

    if file_ds.exists() and file_obs.exists():
        ds = xr.open_dataset(file_ds)
        fe_obs = xr.open_dataset(file_obs)
        return ds_fe, ds, fe_obs

    lons = [165, 190, 220, 250]
    n = len(lons)
    ds_list = [None] * n
    fe_obs_list = [None] * n

    # Particle dataset.
    for i in range(n):
        logger.info('Init dataset: {}'.format(i))
        # Experiment class.
        pds = FelxDataSet(ExpData(scenario=0, lon=lons[i], version=9))

        if cfg.test:
            ds_list[i] = ds_list[i].isel(traj=slice(10))

        # Particle dataset.
        ds_list[i] = pds.init_felx_optimise_dataset()

        # Fe observations at particle depths.
        z = ds_list[i].z.ffill('obs').isel(obs=-1)  # Particle depth in EUC.
        fe_obs_list[i] = ds_fe.euc_avg.sel(lon=lons[i], z=z, method='nearest', drop=True)

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
    return ds_fe, ds, fe_obs


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
        comm = None
        rank = 0

    ds_fe, ds, fe_obs = optimise_multi_lon_dataset()
    fe_obs = fe_obs['euc_avg']

    # Source iron profile.
    pds = FelxDataSet(ExpData(scenario=0, lon=lon, version=9))
    pds.add_iron_model_params()

    if cfg.test:
        n = 20
        ds = ds.isel(traj=slice(n))
        fe_obs = fe_obs.isel(traj=slice(n))

    # Paramaters to optimise (e.g., a, b, c = params).
    # pds.param_names = ['k_inorg', 'k_org', 'c_scav', 'I_0', 'PAR', 'mu_D', 'mu_D_180', 'gamma_2', 'mu_P', 'a', 'b', 'c']
    # pds.param_names = ['k_inorg', 'k_org', 'c_scav', 'mu_D', 'mu_D_180']
    pds.param_names = ['k_inorg', 'k_org', 'mu_D', 'mu_D_180', 'c_scav']
    params_init = np.array([pds.params[i] for i in pds.param_names])  # params = params_init
    params_init[0] = 1.05e-3
    params_init[1] = 6e-4
    params_init[2] = 0.001
    params_init[3] = 0.05
    params_init[4] = 2.5
    param_bnds = dict(k_org=[1e-6, 1e-2], k_inorg=[1e-6, 1e-2], c_scav=[1.5, 2.5],
                      mu_D=[0.001, 0.05], mu_D_180=[0.001, 0.05], mu_P=[0.005, 0.02],
                      gamma_2=[0.005, 0.03], I_0=[280, 350], PAR=[0.323, 0.5375],
                      a=[0.45, 0.75], b=[0.8, 1.33], c=[0.75, 1.25],
                      alpha=[0.02, 0.03], k_fe=[0.75, 1.25], k_N=[0.75, 1.25],
                      tau=None, gamma_1=None, g=None, epsilon=None, mu_Z=None)
    bounds = [tuple(param_bnds[i]) for i in pds.param_names]

    if rank == 0:
        logger.info('felx_hist: Optimisation {} method - init {} (fixed scavenging detritus conversion (k_org*=0.02; unweighted)'.format(method, params_init))

    res = minimize(cost_function, params_init, args=(pds, ds, fe_obs, ds_fe, comm),
                   method=method, bounds=bounds, options={'disp': True})
    params_optimized = res.x

    # Update pds params dict with optimised values.
    param_dict = dict([(k, v) for k, v in zip(pds.param_names, params_optimized)])
    pds.update_params(param_dict)

    if rank == 0:
        logger.info('felx_hist: p={}, method={}, nit={}, nfev={}, success={}, message={}'
                    .format(ds.traj.size, method, res.nit, res.nfev, res.success, res.message))
        logger.info('felx_hist: Init {}'.format(params_init))
        logger.info('felx_hist: Optimimal {}'.format(param_dict))
    return res


def test_plot_optimise_iron_model_params_multi_lon(lon):
    """Plot and save dataset with optmised values."""
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0

    pds = FelxDataSet(ExpData(scenario=0, lon=lon, version=9))
    pds.add_iron_model_params()
    file = pds.exp.file_felx

    ds_fe, ds, fe_obs = optimise_multi_lon_dataset()
    fe_obs = fe_obs['euc_avg']
    param_dict = {'c_scav': 2.5, 'k_inorg': 0.0011708984374999998, 'k_org': 0.0006181640625000001, 'mu_D': 0.001, 'mu_D_180': 0.05}
    param_dict = {'c_scav': 2, 'k_inorg': 0.0009053696000000005, 'k_org': 0.00040948480000000005, 'mu_D': 0.001, 'mu_D_180': 0.05}
    # param_dict = {'k_inorg': 0.0011076, 'k_org': 0.00057648, 'mu_D': 0.0010112, 'mu_D_180': 0.05, 'c_scav': 2.5}
    pds.param_names = ['c_scav', 'k_inorg', 'k_org', 'mu_D', 'mu_D_180']
    params = np.array([param_dict[i] for i in pds.param_names])  # params = params_init

    # Calculate the predicted iron concentration using the optimized parameters.
    ds_opt = update_particles_MPI(pds, ds, ds_fe, pds.param_names, params, comm=comm)

    if rank == 0:
        tmp_files = pds.felx_tmp_filenames(comm.Get_size())
        ds_opt = xr.open_mfdataset(tmp_files)
        for var in pds.variables:
            ds[var] = ds_opt[var]
        ds = ds.unify_chunks()

        if 'obs' in ds.zone.dims:
            ds['zone'] = ds.zone.isel(obs=0)  # Not run in v0

        # Add metadata
        ds = pds.add_variable_attrs(ds)

        # Save dataset.
        comp = dict(zlib=True, complevel=5, contiguous=False)
        for var in ds:
            ds[var].encoding.update(comp)

        # Tests
        # all iron non-negative
        # phyto uptake okay
        # remineralisation okay
        # Scavenging okay

        # Plot
        p = [0, 2135, 2782, 3211, 3294]  # Particle ID of lontitude subsets

        for i in range(4):
            dx = ds.isel(traj=slice(p[i], p[i+1]))
            pds = FelxDataSet(ExpData(scenario=0, lon=cfg.release_lons[i], version=9))
            test_plot_EUC_iron_depth_profile(pds, dx, ds_fe)

        ds = ds.chunk()
        ds.to_netcdf(file)

    #if MPI is not None:
        #comm.Barrier()
        #if tmp_files[rank].exists():  # Delete tmp_file
            #os.remove(tmp_files[rank])


if __name__ == '__main__':
    p = ArgumentParser(description="""Optimise iron model paramaters.""")
    p.add_argument('-x', '--lon', default=0, type=int,
                   help='Release longitude [165, 190, 220, 250].')
    p.add_argument('-f', '--function', default='run', type=str, help='Run or plot.')
    args = p.parse_args()
    lon = args.lon
    method = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC'][0]
    logger = mlogger('optimise_iron_model_params_{}'.format(lon))
    # cfg.test = True

    if args.function == 'run':
        if lon in [165, 190, 220, 250]:
            res = optimise_iron_model_params(lon, method=method)
        else:
            res = optimise_iron_model_params_multi_lon(lon, method=method)
    else:
        test_plot_optimise_iron_model_params_multi_lon(lon)
