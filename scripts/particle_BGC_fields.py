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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # NOQA
import xarray as xr  # NOQA

import cfg
from cfg import paths, ExpData
from datasets import save_dataset, BGCFields
from felx_dataset import FelxDataSet
from tools import timeit, mlogger

logger = mlogger('files_felx')


@dask.delayed
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
def save_felx_BGC_fields(exp):
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
    def set_dataset_vars(ds, variables, arrays):
        for var, arr in zip(variables, arrays):
            ds[var] = (['traj', 'obs'], arr)
        return ds

    file = exp.file_felx_bgc
    if cfg.test:
        file = paths.data / ('test/' + file.stem + '.nc')

    logger.info('{}: Getting OFAM3 BGC fields.'.format(file.stem))

    # Select vairables
    fieldset = BGCFields(exp)
    fieldset.kd = fieldset.kd490_dataset()
    variables = fieldset.variables
    vars_ofam = fieldset.vars_ofam

    # Initialise particle dataset.
    pds = FelxDataSet(exp)

    if file.exists():
        ds = xr.open_dataset(file)
    else:
        ds = pds.get_formatted_felx_file(variables=variables)

    # Option 2
    dim_map_ofam = dict(time='time', depth='z', lat='lat', lon='lon')
    dim_map_kd = dict(time='month', lat='lat', lon='lon')

    ds = dask.delayed(ds)
    fieldset = dask.delayed(fieldset)

    results = []
    for var in vars_ofam:
        logger.info('{}: OFAM3 {} field.'.format(file.stem, var))
        result = update_field_AAA(ds, fieldset.ofam, var, dim_map=dim_map_ofam)
        results.append(result)

    var = 'kd'
    logger.info('{}: OFAM3 {} field.'.format(file.stem, var))
    result = update_field_AAA(ds, fieldset.kd, var, dim_map=dim_map_kd)
    results.append(result)

    logger.info('{}: Computing...'.format(file.stem))
    total = dask.compute(results)
    ds = dask.compute(ds)
    ds = list(ds)[0]

    logger.info('{}: Setting dataset variables'.format(file.stem))
    ds = pds.set_dataset_vars(ds, variables, total)

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
    args = p.parse_args()

    if not cfg.test:
        scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    else:
        scenario, lon, version, index = 0, 190, 0, 0

    name = 'felx_bgc'
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name=name,
                  out_subdir=name, test=cfg.test)
    save_felx_BGC_fields(exp)
