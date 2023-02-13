# -*- coding: utf-8 -*-
"""Inverse observations in plx particle trajectory files (save/get dataset).

Notes:
    - Might need more than 6GB (uses ~90% mem)
    - multiprocessing.Pool
        - pool.map: can take only one iterable as an argument (faster?).
        - pool.apply: takes agrs as an argument.
        - pool.starmap: map with each element in that iterable is also a iterable.
        - map_async(): next process can start as soon as previous one gets over
            - results may be out of order

Example:
    - inverse_plx_dataset(exp)

Todo:
    - mkdir /g/data/e14/as3189/stellema/felx/data/felx
    - Test output files are correct
    - Create jobs script

Runtimes:
    - plx_hist_250_v1r00_inverse: 0:7:26.23 total: 446.23 seconds.
    - ?plx_hist_165_v1r00_inverse:
        inverse_particle_obs: 0:2:15.70 total: 135.70 seconds.
        inverse_plx_dataset: 0:2:35.68 total: 155.68 seconds.
    - plx_hist_165_v1r00_inverse:
        inverse_particle_obs: 0:2:23.39 total: 143.39 seconds.
    - plx_hist_190_v1r00_inverse:
        inverse_particle_obs: 00h:02m:14.12s (total=134.12 seconds).
        inverse_plx_dataset: 00h:09m:20.65s (total=560.65 seconds).

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sun Jan 15 14:03:01 2023

"""
import numpy as np
import xarray as xr
import pandas as pd
from argparse import ArgumentParser

from cfg import paths, ExpData
from tools import timeit, mlogger
from datasets import plx_particle_dataset, save_dataset

logger = mlogger('files_felx')


@timeit(my_logger=logger)
def inverse_particle_obs(ds):
    """Reverse obs order for plx particles & remove leading NaNs.

    Args:
        ds (xarray.Dataset): Dataset particle trajectories.

    Returns:
        ds (xarray.Dataset): dataset particle trajectories with reversed coord 'obs'.

    Notes:
        - Dataset requires data variable 'trajectory'.

    Alternate:
        for p in range(ds.traj.size):
            # Index of first non-NaN value.
            N = np.isnan(ds.isel(traj=p).trajectory).sum().item()
            ds[dict(traj=p)] = ds.isel(traj=p).roll(obs=-N)

    """
    def align_ragged_array(ds):
        """Roll non-NaN 1D data array to start."""
        def roll_nan_data(ds):
            return np.roll(ds, -np.sum(pd.isnull(ds)), axis=0)
        return np.apply_along_axis(roll_nan_data, -1, ds)

    # Reverse order of obs indexes (forward).
    ds = ds.isel(obs=slice(None, None, -1))

    for var in ds.data_vars:
        ds[var] = (['traj', 'obs'], align_ragged_array(ds[var]))

    # Reorder coord 'obs'.
    ds.coords['obs'] = np.arange(ds.obs.size, dtype=ds.obs.dtype)
    return ds


@timeit(my_logger=logger)
def inverse_plx_dataset(exp, **open_kwargs):
    """Open, reverse and save plx dataset.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.
        open_kwargs (dict): xarray.open_dataset keywords.

    Returns:
        ds (xarray.Dataset): Dataset particle trajectories.

    Notes:
        - Saves file as './data/felx/plx_hist_165_v1r00.nc'
        - Drops 'unbeached' variable.
        - Runs in 5-10 mins per file.

    """
    file = exp.file_plx_inv

    if file.exists():
        ds = xr.open_dataset(file, **open_kwargs)

    else:
        logger.info('{}: Calculating inverse plx file'.format(file.stem))

        # Particle trajectories.
        ds = plx_particle_dataset(exp)
        ds = ds.drop({'unbeached'})

        # # Subset number of particl1es.
        # if __name__ != '__main__' and cfg.test:
        #     logger.debug('Calculating test subset: {}'.format(file.stem))
        #     # ds = ds.drop({'age', 'zone', 'distance'})
        #     ds = ds.isel(traj=slice(10)).dropna('obs', 'all')
        #     # Subset number of times.
        #     ds = ds.where(ds.time >= np.datetime64('2012-01-01'))

        # Copy & drop 1D data vars.
        u = ds.u.copy()
        zone = ds.u.copy()
        ds = ds.drop({'u', 'zone'})

        # Reverse particleset.
        ds = inverse_particle_obs(ds)

        # Replace 1D data vars.
        ds['u'] = u
        ds['zone'] = zone

        # Save dataset.
        logger.info('{}: Inverse plx file saving..'.format(file.stem))
        save_dataset(ds, file, msg='Inverse particle obs dimension.')
        logger.info('{}: Inverse plx file saved.'.format(file.stem))
    return ds


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Start longitude(s).')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index.')
    # p.add_argument('-ind', '--index', default=0, type=int, help='File repeat.')
    p.add_argument('-v', '--version', default=1, type=int, help='PLX experiment version.')
    args = p.parse_args()

    # for r in range(10):
    #     exp = ExpData(scenario=args.scenario, lon=args.lon, version=args.v,
    #                       file_index=r)
    #     inverse_plx_dataset(exp)

    scenario, lon, r = 0, 190, 0
    exp = ExpData(scenario=scenario, lon=lon, version=0, file_index=r, out_subdir='felx')
    inverse_plx_dataset(exp)
