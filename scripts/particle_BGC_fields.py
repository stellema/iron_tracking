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
    * Parallelise openmfdataset
    * Parallelise file index
    * Estimate computational resources

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jan 14 15:03:35 2023

"""
from argparse import ArgumentParser
from datetime import datetime, timedelta  # NOQA
import dask
from dask.distributed import Client, progress
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # NOQA
import xarray as xr  # NOQA

import cfg
from cfg import paths, ExpData
from datasets import save_dataset, BGCFields
from felx_dataset import FelxDataSet
from tools import timeit, mlogger, random_string
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

logger = mlogger('files_felx')


def save_subset(file, ds, p, field_dataset, var, dim_map):
    logger.info('{}: Calculating'.format(file.stem))
    dx = ds.isel(traj=slice(*p))
    dx = update_field_AAA(dx, field_dataset, var, dim_map)

    logger.info('{}: Saveing...'.format(file.stem))
    np.save(file, dx, allow_pickle=True)
    logger.info('{}: Saved.'.format(file.stem))
    return


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
        x = field.sel(loc, method='nearest')
        return x

    kwargs = dict(ds=ds, field=field_dataset[var], dim_map=dim_map)
    dx = np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)
    return dx


def test_runtime_BGC_fields():
    """Calculate BGC fields at particle positions.

    Runtime:
        - 100 particles:
            -update_ofam3_fields_AAA: 00h:04m:38.58s (total=278.58 seconds).
            update_kd490_field_AAA: 00h:15m:06.79s (total=906.79 seconds).
            get_felx_BGC_fields: 00h:19m:52.34s (total=1192.34 seconds).
        - 500 particles:
            - update_ofam3_fields_AAA: 00h:23m:33.38s (total=1413.38 seconds).
            - update_kd490_field_AAA: 01h:11m:27.42s (total=4287.42 seconds).
            - get_felx_BGC_fields: 01h:35m:10.64s (total=5710.64 seconds).
        - 550 particles:
            - update_ofam3_fields_AAA: 00h:25m:03.39s (total=1503.39 seconds).
            - update_kd490_field_AAA: 01h:23m:40.10s (total=5020.10 seconds).
            - get_felx_BGC_fields: 01h:48m:54.75s (total=6534.75 seconds).
        - 900 particles:
            -
        - 1000 particles:
            - # update_ofam3_fields_AAA: 00h:54m:15.27s (total=3255.27 seconds).
            - update_ofam3_fields_AAA: 00h:46m:01.28s (total=2761.28 seconds).
            - update_kd490_field_AAA: 02h:30m:17.30s (total=9017.30 seconds).
            - get_felx_BGC_fields: 03h:16m:34.74s (total=11794.74 seconds)

    """
    xx = np.array([100, 500, 550, 1000])
    xfit = np.array([100, 500, 550, 1000])
    t_ofam_update = np.array([278.58, 1413.38, 1503.39, 2761.28])
    t_kd_update = np.array([906.79, 4287.42, 5020.10, 9017.30])
    t_total = np.array([1192.34, 5710.64, 6534.75, 11794.74])

    # Plot.
    yy = [t_ofam_update, t_kd_update, t_total]
    cc = ['b', 'r', 'k']
    nn = ['OFAM3 Update', 'Kd Update', 'Total Time']
    for y, c, n in zip(yy, cc, nn):
        x = xx if y.size == xx.size else xx[:-1]
        y = y / 60
        a, b = np.polyfit(x, y, 1)
        plt.scatter(x, y, c=c)
        plt.plot(xfit, a*xfit+b, c=c, label=n)
    plt.legend()


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
    # Create fieldset for variable.
    fieldset = BGCFields(exp)

    if var in fieldset.vars_ofam:
        fieldset.ofam_dataset(variables=var)
        field_dataset = fieldset.ofam
        dim_map = fieldset.dim_map_ofam
    else:
        fieldset.kd490_dataset()
        field_dataset = fieldset.kd
        dim_map = fieldset.dim_map_kd

    # Initialise particle dataset.

    pds = FelxDataSet(exp)
    # Create temp directory and filenames.
    tmp_files = pds.var_tmp_files(var)
    tmp_file = tmp_files[n]

    if tmp_file.exists():
        return

    if exp.file_felx_bgc.exists():
        ds = xr.open_dataset(exp.file_felx_bgc)
    else:
        ds = pds.get_formatted_felx_file(variables=fieldset.variables)

    # Save temp file subset for variable.
    logger.info('{}: Getting field.'.format(tmp_file.stem))
    traj_slices = pds.traj_subsets(ds)
    traj_slice = traj_slices[n]

    # Calculate & save particle subset.
    save_subset(tmp_file, ds, traj_slice, field_dataset, var, dim_map)
    logger.info('{}: Saved tmp subset field.'.format(tmp_file.stem))


def save_felx_BGC_fields(exp):
    variables = ['phy', 'zoo', 'det', 'temp', 'fe', 'no3', 'kd']
    pds = FelxDataSet(exp)

    # for var in variables:
    #     for n in range(pds.num_subsets):
    #         save_felx_BGC_field_subset(exp, var, n)

    if exp.file_felx_bgc.exists():
        ds = xr.open_dataset(exp.file_felx_bgc)
    else:
        ds = pds.get_formatted_felx_file(variables=variables)

    # Put all temp files together.
    logger.info('{}: Getting OFAM3 BGC fields.'.format(exp.file_felx_bgc.stem))
    file = exp.file_felx_bgc
    if cfg.test:
        file = paths.data / ('test/' + file.stem + '.nc')

    logger.info('{}: Setting dataset variables'.format(file.stem))
    ds = pds.set_dataset_vars_from_tmps(exp, ds, variables)

    ds = ds.where(ds.valid_mask)
    ds['time'] = ds.time_orig
    ds = ds.drop('valid_mask')
    ds = ds.drop('time_orig')

    # Save file.
    logger.info('{}: Saving...'.format(file.stem))
    save_dataset(pds.ds, file, msg='Added BGC fields at paticle positions.')
    logger.info('{}: Felx BGC fields file saved.'.format(file.stem))
    return


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-r', '--index', default=0, type=int, help='File repeat.')
    p.add_argument('-v', '--version', default=0, type=int, help='FeLX experiment version.')
    # p.add_argument('-var', '--variable', default=0, type=int, help='FeLX experiment version.')
    args = p.parse_args()

    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    # var = args.variable

    if cfg.test:
        scenario, lon, version, index, var = 0, 190, 0, 0, 'phy'

    name = 'felx_bgc'
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name=name,
                  out_subdir=name, test=cfg.test)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    variables = ['phy', 'zoo', 'det', 'temp', 'fe', 'no3', 'kd']
    pds = FelxDataSet(exp)

    # for var in variables:
    #     for n in range(pds.num_subsets):
    #         save_felx_BGC_field_subset(exp, var, n)

    if rank == 0:
        # Input ([[var0, 0], [var0, 1], ..., [varn, n]).
        data = [[v, i] for v in variables for i in range(pds.num_subsets)]
        comm.send()
    else:
        data = None
    data = comm.scatter(data, root=0)
    var, n = data
    logger.info('{}: Rank={}, var={}, n={}/4'.format(exp.file_felx_bgc.stem, rank, var, n))
    save_felx_BGC_field_subset(exp, var, n)

    if rank == 0 and pds.check_tmp_files_complete(variables):
        save_felx_BGC_fields(exp)
