# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Jan 16 22:07:05 2023

"""

def spinup_analysis():
    import numpy as np
    import xarray as xr

    import cfg
    from datasets import plx_particle_dataset

    exp, lon, r = 0, 165, 0
    ds = plx_particle_dataset(exp, lon, r)
    file = cfg.data / 'plx/plx_{}_{}_v1r{:02d}.nc'.format(['hist', 'rcp'][exp], lon, r)
    ds = xr.open_dataset(str(file), decode_cf=True)
    ds = ds.drop({'zone', 'distance', 'unbeached', 'u'})
    inds = ds.age.argmax('obs')
    # t_min = ds.time.sel(  # returns NaT
    t_min = ds.time.ffill('obs').min('obs')


def working_parallel_plx_reverse():
    import math
    import bottleneck
    import dask
    import numpy as np
    import xarray as xr
    import multiprocessing as mp
    from multiprocessing import Pool
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    from dask.distributed import Client
    import dask.array as da

    import cfg
    from datasets import (plx_particle_dataset)

    """Example 1."""
    # def cube(x):
    #     return math.sqrt(x)

    # if __name__ == "__main__":
    #     with Pool() as pool:
    #         print(pool)
    #         # pool.map will return list of range(10) cube results.
    #         result = pool.map(cube, range(10))
    #         print(result)
    #     print("Program finished!")

    # """Example 2."""
    # # Parallel processing with Pool.apply_async() without callback function
    # np.random.RandomState(100)
    # arr = np.random.randint(0, 10, size=[10, 5])
    # data = arr  #.tolist()

    # def howmany_within_range2(i, row):
    #     """Return how many numbers lie within `maximum` and `minimum` in a given row."""
    #     count = 0
    #     for n in row:
    #         if 2 <= n <= 8:
    #             count = count + 1
    #     return (i, count)

    # pool = mp.Pool(mp.cpu_count())

    # results = []

    # # call apply_async() without callback
    # result_objects = [pool.apply_async(howmany_within_range2, args=(i, row, 4, 8))
    #                   for i, row in enumerate(data)]

    # # result_objects is a list of pool.ApplyResult objects
    # results = [r.get()[1] for r in result_objects]

    # pool.close()
    # pool.join()
    # print(results[:10])
    # #  > [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]

    """Example 3."""
    # client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
    # client
    # shape = (1000, 4000)
    # chunk_shape = (1000, 1000)
    # ones = da.ones(shape, chunks=chunk_shape)

    """Example 4 - reverse plx"""

    def test_reverse(ds, p):
        # Index of first non-NaN value.
        N = np.isnan(ds.isel(traj=p).trajectory).sum().item()
        ds[dict(traj=p)] = ds.isel(traj=p).roll(obs=-N)
        return ds

    def test_reverse1(p):
        # Index of first non-NaN value.
        N = np.isnan(ds.isel(traj=p).trajectory).sum().item()
        return ds.isel(traj=p).roll(obs=-N)

    def test_reverse2(i, p, ds):
        # Index of first non-NaN value.
        N = np.isnan(ds.isel(traj=p).trajectory).sum().item()
        ds[dict(traj=p)] = ds.isel(traj=p).roll(obs=-N)
        return (i, ds[dict(traj=p)])

    def test_reverse3(ds):
        # Index of first non-NaN value.
        N = np.isnan(ds.trajectory).sum().item()
        return ds.roll(obs=-N)

    ds = plx_particle_dataset(0, 165, 0)
    ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})

    # Subset number of particles.
    if cfg.test:
        ds = ds.isel(traj=slice(10)).dropna('obs', 'all')
        # Subset number of times.
        ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    ds = ds.isel(obs=slice(None, None, -1))

    """Alternate 1 - pool.map [slow]."""
    # pool = mp.Pool(mp.cpu_count())
    # results = []
    # # result = pool.apply(test_reverse, np.arange(ds.traj.size))
    # # call apply_async() without callback
    # result_objects = [pool.apply_async(test_reverse2, args=(i, p, ds))
    #                   for i, p in enumerate(np.arange(ds.traj.size))]
    # results = [r.get()[1] for r in result_objects]

    """Alternate 2 - [slow]."""
    # pool = mp.Pool(mp.cpu_count())
    # results = pool.map(test_reverse1, np.arange(ds.traj.size, dtype=np.int32))

    """Alternate 3."""
    # input_core_dims=[[], ['time']]

    """Alternate 4."""
    # da = ds.to_dask_dataframe()
    # dx = dask.dataframe.apply_along_axis(test_reverse3, 0, ds, p=ds.traj)

    """Alternate 5."""
    # def test_reverse4(ds):
    #     # Index of first non-NaN value.
    #     return np.roll(ds, -np.sum(np.isnan(ds)), axis=0)

    # def func(dx):
    #     return np.apply_along_axis(test_reverse4, -1, dx)

    # dx = ds.z
    # np.apply_along_axis(test_reverse4, -1, dx)  # works
    # dx = xr.apply_ufunc(func, ds.z, dask='allowed', vectorize=True)  # doesnt work

    """Alternate 6 - works."""
    def test_reverse5(ds):
        def func(ds):
            return np.roll(ds, -np.sum(np.isnan(ds)), axis=0)
        return np.apply_along_axis(func, -1, ds)

    dx = ds.copy()
    for var in ds.data_vars:
        dx[var] = (['traj', 'obs'], test_reverse5(ds[var]))

    """Alternate 7 - does not work."""
    # def test_reverse6(ds):
    #     def func(ds):
    #         return np.roll(ds, -np.sum(np.isnan(ds)), axis=0)
    #     return np.apply_along_axis(func, 0, ds)

    # def func(ds):
    #     return np.roll(ds, -np.sum(np.isnan(ds)), axis=0)

    # da = ds.to_dask_dataframe()
    # # dask.dataframe.utils.make_meta
    # meta = ds.to_dataframe()
    # daz = da.z.apply(func, meta=meta.z)
    # # daz.visualize(str(cfg.fig/'test2.png'), optimize_graph=True)
    # # daz.dask.visualize()
    # daz = daz.compute()
    return

def working_particle_BGC_fields(ds):
    import numpy as np
    import xarray as xr
    from argparse import ArgumentParser

    import cfg
    from tools import timeit, mlogger
    from datasets import (ofam3_datasets, plx_particle_dataset, get_felx_filename,
                          save_dataset)
    from inverse_plx import inverse_plx_dataset
    def sample_fields(da, field, var):
        def sample_fields_particle(da, field):
            """Subset using entire array for a particle."""

            loc = dict(time=da.time, depth=da.z, lat=da.lat, lon=da.lon)
            return field.sel(loc, method='nearest')

        da[var] = np.apply_along_axis(sample_fields_particle, -1, da, field=field)
        return da[var]

    def sample_fields2(ds, field, var):
        def sample_fields_particle(depth, time, lat, lon, field):
            """Subset using entire array for a particle."""
            loc = dict(time=time, depth=depth, lat=lat, lon=lon)
            return field.sel(loc, method='nearest', drop=True)

        dx = np.apply_along_axis(sample_fields_particle, 1, ds.z, time=ds.time, lat=ds.lat, lon=ds.lon, field=field)
        return dx

    def sample_fields3(ds, field, var):  # works with right shape
        def sample_fields_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[0])
            loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
            return field.sel(loc, method='nearest', drop=True)

        dx = np.apply_along_axis(sample_fields_particle, 1, ds.trajectory, ds=ds, field=field)
        return dx


    exp = 0
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det'}
    fieldset = ofam3_datasets(exp, variables=variables)

    var = list(variables.items())[-1][1]
    field = fieldset[var]
    dt3 = sample_fields(ds, field, var)
    return
