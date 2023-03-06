# -*- coding: utf-8 -*-
"""Calculate BGC fields at particle positions.

Fields: T, Det, Z, P, N, Kd490, Fe (optional)

Notes:
    - Could sample background iron (OFAM or Huang et al.)
    - Saved reversed felx dataset.
    - Historical: 2000-2012 (12 years = ~4383 days)
    - RCP: 2087-2101 (12 years=2089-2101)
    - Initial year is a La Nina year (for spinup)
    - 85% of particles finish
    - Missing
Example:

Questions:

TODO:
    * check monotonic index error

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jan 14 15:03:35 2023

"""
from argparse import ArgumentParser
from datetime import datetime, timedelta  # NOQA
import matplotlib.pyplot as plt  # NOQA
from memory_profiler import profile
import numpy as np
import pandas as pd  # NOQA
import time
import xarray as xr  # NOQA

import cfg
from cfg import paths, ExpData
from datasets import save_dataset, BGCFields
from felx_dataset import FelxDataSet
from tools import timeit, mlogger
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

logger = mlogger('files_felx')


@profile
@timeit(my_logger=logger)
def update_field_AAA(ds, field_dataset, var, dim_map):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """
    def update_field_particle(traj, ds, field, dim_map):
        """Subset using entire array for a particle."""
        dx = ds.sel(traj=traj[~np.isnan(traj)][0].item())
        loc = {}
        for k, v in dim_map.items():
            loc[k] = dx[v]
        try:
            x = field.sel(loc, method='nearest')

        except ValueError as err:
            logger.debug('{}'.format(err))
            logger.debug('traj={}, obs={}, non-NaN obs={}'
                         .format(dx.traj.item(), dx.obs.size, dx.lat.dropna('obs', 'all').obs.size))
            for k, v in dim_map.items():
                logger.debug('{}: {}'.format(k, dx[v].head().values))
            raise err
        return x

    kwargs = dict(ds=ds, field=field_dataset[var], dim_map=dim_map)
    dx = np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)
    return dx


@profile
@timeit(my_logger=logger)
def save_felx_BGC_field_subset(exp, var, n):
    """Sample OFAM3 BGC fields at particle positions.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.

    Returns:
        ds (xarray.Dataset): Particle trajectory dataset with BGC fields.

    Notes:
        - First save/create plx_inverse file.

    Todo:
        - Unzip OFAM3 fields [in progress]
        - Upload formatted Kd490 fields [in progress]a
        - Save plx reversed datasets
        - parallelise
        - add datset encoding [done]
        - Filter spinup?
        - Add kd490 fields
    """
    def save_subset(file, ds, p, field_dataset, var, dim_map):
        """Save temp particle BGC fields subset."""
        logger.info('{}: Calculating'.format(str(file)))
        dx = ds.isel(traj=slice(*p))
        dx = update_field_AAA(dx, field_dataset, var, dim_map)

        np.save(file, dx, allow_pickle=True)
        logger.info('{}: Saved.'.format(str(file)))
        return

    # Create fieldset for variable.
    fieldset = BGCFields(exp)

    if var in fieldset.vars_ofam:
        field_dataset = fieldset.ofam_dataset(variables=var)
        dim_map = fieldset.dim_map_ofam
    else:
        field_dataset = fieldset.kd490_dataset()
        dim_map = fieldset.dim_map_kd

    # Initialise particle dataset.
    pds = FelxDataSet(exp)
    # Create temp directory and filenames.
    tmp_files = pds.bgc_var_tmp_filenames(var)
    tmp_file = tmp_files[n]

    if pds.check_file_complete(tmp_file):
        return

    pds.check_bgc_prereq_files()
    ds = xr.open_dataset(exp.file_felx_bgc)

    # Save temp file subset for variable.
    logger.info('{}: Getting field.'.format(str(tmp_file)))
    traj_slices = pds.bgc_tmp_traj_subsets(ds)
    traj_slice = traj_slices[n]

    # Calculate & save particle subset.
    save_subset(tmp_file, ds, traj_slice, field_dataset, var, dim_map)
    logger.info('{}: Saved tmp subset field.'.format(str(tmp_file)))


@profile
def save_felx_BGC_fields(exp):
    """Run and save OFAM3 BGC fields at particle positions as n temp files."""
    pds = FelxDataSet(exp)
    pds.check_bgc_prereq_files()

    ds = xr.open_dataset(exp.file_felx_bgc)

    # Put all temp files together.
    logger.info('{}: Getting OFAM3 BGC fields.'.format(exp.file_felx_bgc.stem))
    file = exp.file_felx_bgc
    if cfg.test:
        file = paths.data / ('test/' + file.stem + '.nc')

    logger.info('{}: Setting dataset variables'.format(file.stem))
    ds = pds.set_bgc_dataset_vars_from_tmps(ds)

    ds = ds.where(ds.valid_mask)
    ds['time'] = ds.time_orig
    ds = ds.drop('valid_mask')
    ds = ds.drop('time_orig')

    # Save file.
    logger.info('{}: Saving...'.format(file.stem))
    save_dataset(pds.ds, file, msg='Added BGC fields at paticle positions.')
    logger.info('{}: Felx BGC fields file saved.'.format(file.stem))
    return


@profile
def parallelise_prereq_files(scenario):
    """Calculate plx subset and inverse_plx files.

    Notes:
        * Assumes 8 processors
        * 10 file x 16 min each = ~2.6 hours.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    lon = [165, 190, 220, 250][rank]

    for r in range(8):
        name = 'felx_bgc'
        exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=r, name=name)
        pds = FelxDataSet(exp)
        pds.check_bgc_prereq_files()
    return


@profile
def parallelise_BGC_fields(exp):
    """Step 2. Parallelise saving particle BGC fields as temp files.

    Notes:
        * Assumes 35 processors
        * Arounf 7-8 hours each file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    pds = FelxDataSet(exp)
    if rank == 0:
        pds.rename_bgc_tmp_files()  # TODO: can be deleted after run once.
    else:
        time.sleep(5)

    # Input ([[var0, 0], [var0, 1], ..., [varn, n]).
    var_n_all = [[v, i] for v in pds.bgc_variables for i in range(pds.num_subsets)]

    # Remove any finished saved subsets from list.
    for vn in var_n_all.copy():
        files = pds.bgc_var_tmp_filenames(vn[0])
        if pds.check_file_complete(files[vn[1]]):
            var_n_all.remove(vn)
    if rank == 0:
        logger.info('{}: Number of files to run={}'.format(exp.file_felx_bgc.stem, len(var_n_all)))

    var, n = var_n_all[rank]
    logger.info('{}: Rank={}, var={}, n={}/4'.format(exp.file_felx_bgc.stem, rank, var, n))
    save_felx_BGC_field_subset(exp, var, n)
    return


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-r', '--index', default=0, type=int, help='File repeat.')
    p.add_argument('-v', '--version', default=0, type=int, help='FeLX experiment version.')
    p.add_argument('-func', '--function', default='bgc_fields', type=str,
                   help='[bgc_fields, prereq_files, save_files]')
    # p.add_argument('-var', '--variable', default=0, type=int, help='FeLX experiment version.')
    args = p.parse_args()

    func = args.function
    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    # var = args.variable

    if cfg.test:
        func = ['bgc_fields', 'prereq_files', 'save_files'][0]
        scenario, lon, version, index, var = 0, 250, 0, 0, 'phy'

    name = 'felx_bgc'
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name=name,
                  out_subdir=name, test=cfg.test)
    # if cfg.test:
    #     n = 0
    #     var = 'phy'
    #     save_felx_BGC_field_subset(exp, var, n)

    # Step 1.
    if func == 'prereq_files':
        parallelise_prereq_files(scenario)

    # Step 2.
    if func == 'bgc_fields':
        parallelise_BGC_fields(exp)

    # Step 3.
    if func == 'save_files':
        pds = FelxDataSet(exp)
        if pds.check_bgc_tmp_files_complete():
            save_felx_BGC_fields(exp)
