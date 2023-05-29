# -*- coding: utf-8 -*-
"""Test calculating BGC fields at particle positions.

Fields: T, Det, Z, P, N, Kd490, Fe (optional)

Notes:
    - Parcels requires plx conda env
    - Could sample background iron (OFAM or Huang et al.)
    - Saved reversed felx dataset.
    - Historical: 2000-2012 (12 years = ~4383 days)
    - RCP: 2087-2101 (12 years=2089-2101)
    - Initial year is a La Nina year (for spinup)
    - 85% of particles finish
    - the t-cell land point (0.85°S, 185.35°E) corresponds to
    u-cell land points (0.9°S, 185.3°E), (0.8°S, 185.3°E), (0.9°S, 185.4°E), (-0.8°S, 185.4°E)

Example:

TODO:

Execution times:
    165
    - ds = ds.isel(traj=slice(10), obs=slice(1000)) [i think]
        - LOOP:
            inverse_plx_dataset: 0:2:35.43 total: 155.43 seconds.
        - AAA:
            inverse_plx_dataset: 0:0:01.38 total: 1.38 seconds.

    - 250r0.isel(traj=slice(10), obs=slice(1000))
    - Base:
        update_PZD_fields_AAA: 0:0:34.94 total: 34.94 seconds.
    - 2x particles:
        update_PZD_fields_AAA: 0:0:35.74 total: 35.74 seconds.

    - Base:
        update_PZD_fields_AAA: 00h:00m:15.73s (total=15.73 seconds).
    - 2x obs: [=1.66x time]
        update_PZD_fields_AAA: 00h:00m:25.76s (total=25.76 seconds).
    - 2x particles:  [=2x time]
        update_PZD_fields_AAA: 00h:00m:30.41s (total=30.41 seconds).
    - 2x particles + 2x obs:
        update_PZD_fields_AAA: 00h:00m:51.75s (total=51.75 seconds).

    ds = ds.isel(traj=slice(1000), obs=slice(1000))  # Base.
    update_PZD_fields_AAA: 00h:26m:12.66s (total=1572.66 seconds).

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jan 14 15:03:35 2023

"""
from datetime import datetime, timedelta
from memory_profiler import profile
import numpy as np
import pandas as pd
import xarray as xr

from cfg import paths, ExpData
from tools import timeit, mlogger
from datasets import ofam3_datasets, save_dataset, BGCFields, convert_plx_times
from fe_exp import FelxDataSet
from particle_BGC_fields import update_field_AAA, update_field_AAA_old
from fieldsets import (get_fe_pclass, add_fset_constants,
                       ofam3_fieldset_parcels221, add_Kd490_field_parcels221,
                       add_ofam3_bgc_fields_parcels221,
                       ofam3_fieldset_parcels240, add_Kd490_field_parcels240,
                       add_ofam3_bgc_fields_parcels240)

logger = mlogger('test_files_felx')
test = True


def repeat(a, n):
    """Make n copies of particle array to set as DataArray."""
    return (['traj', 'obs'], np.repeat(a[None, :], n, 0))


def sample_particle_dataset(df, ntraj, nobs):
    """Create test particle dataset with particle positions along OFAM3 coords."""
    if ntraj > df.time.size:
        print('Pick smaller number of obs (<{}).'.format(df.time.size))
    coords = dict(traj=np.arange(ntraj), obs=np.arange(nobs))
    ds = xr.Dataset(coords=coords)

    dvars = ['time', 'z', 'lat', 'lon', 'phy', 'det']
    ds['trajectory'] = (['traj', 'obs'], np.repeat(ds.traj.values[None, :], nobs, 0).T)
    for v in dvars:
        ds[v] = (['traj', 'obs'], np.zeros((ds.traj.size, ds.obs.size)))

    # Create particle positions based on OFAM3 coords.
    for v in ['time', 'lat', 'lon']:
        ds[v] = repeat(df[v].values[:nobs], n=ntraj)

    ds['z'] = repeat(np.tile(df.depth, int(np.ceil(204 / df.depth.size)))[:nobs], n=ntraj)
    return ds


@timeit(my_logger=logger)
def update_ofam3_fields_AAA(ds, fieldset, variables):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """
    def update_ofam3_field(ds, field):
        def update_field_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[~np.isnan(traj)][0].item())
            loc = dict(depth=dx.z, lat=dx.lat, lon=dx.lon)
            x = field.sel(loc, method='nearest').sel(time=dx.time, method='nearest')
            return x

        kwargs = dict(ds=ds, field=field)
        return np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)

    dims = ['traj', 'obs']
    for var in variables:
        # da = xr.DataArray(update_ofam3_field(ds, field=fieldset[var]), dims=dims)
        # ds[var] = da.pad(obs=(0, ds[var].obs.size - da.obs.size))
        ds[var] = (dims, update_ofam3_field(ds, field=fieldset[var]))
    return ds


# @profile
@timeit(my_logger=logger)
def update_ofam3_fields_loop(ds, fieldset, variables):
    """Calculate fields at plx particle positions (using for loop and xarray)."""
    def update_PZD_fields_xarray(dx, fieldset, variables):
        """Test function: sample a variable."""
        loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
        for v in variables:
            dx[v] = fieldset[v].sel(loc, method='nearest').load().item()
        return dx

    def loop_particle_obs(p, ds, fieldset, update_func, variables):
        n_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for time in range(n_obs):
            particle_loc = dict(traj=p, obs=time)
            ds[particle_loc] = update_func(ds[particle_loc], fieldset, variables)
        return ds

    update_func = update_PZD_fields_xarray
    # Sample fields at particle positions.
    n_particles = ds.traj.size
    for p in range(n_particles):
        ds = loop_particle_obs(p, ds, fieldset, update_func, variables)
    return ds

# @timeit(my_logger=logger)
# def update_ofam3_fields_parcels(dx, fieldset, time):
#     """Test function: sample a variable."""
#     fieldset.computeTimeChunk(time, 1)
#     dx['phy'] = fieldset.P[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
#     dx['det'] = fieldset.Det[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
#     return dx


def update_ofam3_fields_parcels(ds, fieldset):
    """Test sampling OFAM3 fields using parcels fieldset.

    Notes:
        - Interpolation test no point - parcels doesn't interpoalte b_grid tracer fields.
        - Assumes variables are 'phy' and 'det'.

    Todo:
        - Fix sampled out-of-bound at the surface, at (0.000000, 0.000000, 0.000000)

    Args:
        ds (xarray.Dataset): Particle dataset.
        fieldset (parcels.fieldset): OFAM3 fieldset.

    Returns:
        ds (xarray.Dataset): Particle dataset.

    """
    def update_ofam3_field(ds, field, fieldset):
        def update_field_particle(obs, dx, field, fieldset):
            """Subset using entire array for a particle."""
            dz = dx.sel(obs=obs)
            fieldset.computeTimeChunk(dz.time.item(), 1)  # correctly define dt
            x = field.eval(dz.time.item(), dz.z.item(), dz.lat.item(), dz.lon.item())
            return x

        def update_field_particles(traj, ds, field, fieldset):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[~np.isnan(traj)][0].item())
            kwargs = dict(dx=dx, field=field, fieldset=fieldset)
            dx = np.apply_along_axis(update_field_particle, -1, arr=dx.obs, **kwargs)
            return dx

        kwargs = dict(ds=ds, field=field, fieldset=fieldset)
        dx = np.apply_along_axis(update_field_particles, 1, arr=ds.trajectory, **kwargs)
        return dx

    dims = ['traj', 'obs']
    ds['phy'] = (dims, update_ofam3_field(ds, field=fieldset.phy, fieldset=fieldset))
    ds['det'] = (dims, update_ofam3_field(ds, field=fieldset.det, fieldset=fieldset))

    return ds


def test_field_sampling_exact_xarray(decode_times=True):
    """Test BGC field sampling using xarray 'nearest' and numpy.apply_along_axis.
    not decoded:
        p=5, obs=51:
            update_field_AAA: 00h:00m:11.70s (total=11.70 seconds). # @profile
            update_ofam3_fields_loop: 00h:00m:03.08s (total=3.08 seconds).
        p=100, obs=100:
            update_field_AAA: 00h:05m:01.49s (total=301.49 seconds). # @profile
            update_field_AAA: 00h:05m:45.02s (total=345.02 seconds).
            update_ofam3_fields_loop: 00h:02m:07.37s (total=127.37 seconds).
            update_ofam3_fields_loop: 00h:02m:18.76s (total=138.76 seconds).


    decoded:
        p=5, obs=51:
            update_field_AAA: 00h:00m:08.08s (total=8.08 seconds). # @profile
            update_ofam3_fields_loop: 00h:00m:03.47s (total=3.47 seconds).
        p=100, obs=100:
            update_field_AAA: 00h:05m:16.55s (total=316.55 seconds). # @profile
            update_ofam3_fields_loop: 00h:02m:28.41s (total=148.41 seconds).
        p=5, nobs=100:
            update_ofam3_fields_loop: 00h:00m:08.07s (total=8.07 seconds).
            update_field_AAA: 00h:00m:20.39s (total=20.39 seconds).

            update_ofam3_fields_loop: 00h:00m:07.91s (total=7.91 seconds).
            update_field_AAA: 00h:00m:17.91s (total=17.91 seconds).
            update_field_AAA_alt: 00h:00m:03.91s (total=3.91 seconds).
        p=20, obs=180:
            update_ofam3_fields_loop: 00h:00m:57.92s (total=57.92 seconds).
            update_field_AAA: 00h:02m:23.97s (total=143.97 seconds).
            update_field_AAA_alt: 00h:00m:27.37s (total=27.37 seconds).
        p=100, obs=180:
            update_ofam3_fields_loop: 00h:04m:46.75s (total=286.75 seconds).
            update_field_AAA: 00h:10m:58.78s (total=658.78 seconds).
            update_field_AAA_alt: 00h:02m:17.57s (total=137.57 seconds).
            update_field_AAA_alt: 00h:02m:20.83s (total=140.83 seconds). # mask
            update_field_AAA_alt: 00h:01m:56.79s (total=116.79 seconds) # mask


    """
    exp = ExpData(scenario=0, lon=250, test=True, file_index=0, test_month_bnds=12)
    ntraj = 10  # Number of particle trajectory (will be duplicates anyway).
    nobs = 180  # N.B. max based on number of ofam3 depths
    var = 'det'

    fieldset = BGCFields(exp)

    if var in fieldset.vars_ofam:
        field_dataset = fieldset.ofam_dataset(variables=var, decode_times=decode_times)
        dim_map = fieldset.dim_map_ofam
    else:
        dim_map = fieldset.dim_map_kd
        field_dataset = fieldset.kd490_dataset()
    field = field_dataset[var]

    pds = FelxDataSet(exp)

    # Test using time values from particle dataset.  # !!!
    dt = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=False)
    dtd = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=True)
    t = dt.isel(traj=0, drop=1).where(dtd.isel(traj=0, drop=1).time.dt.year >= 2012, drop=1)
    dt = dtd.sel(traj=0, obs=t.obs[:nobs]) if decode_times else dt.sel(traj=0, obs=t.obs[:nobs])
    times = dt.time

    if not decode_times:
        # Change time encoding (errors - can't overwrite file & doesen't work when not decoded).
        attrs = dict(units=dt.time.units, calendar=dt.time.calendar)
        attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='gregorian')  # OFAM3
        times = convert_plx_times(times, obs=times.obs.astype(dtype=int), attrs=attrs, attrs_new=attrs_new)
        dt['time'] = times
        dt['time'].attrs['units'] = attrs_new['units']
        dt['time'].attrs['calendar'] = attrs_new['calendar']

    # Test 1: Check OFAM3 and sampled fields are the same (using known x,y,z positions).
    ds = sample_particle_dataset(field, ntraj, nobs)
    ds['lon'] = repeat(field.lon.sel(lon=slice(165, 220)).values[:nobs], n=ntraj)
    ds['lat'] = repeat(field.lat.sel(lat=slice(-10, 10.5)).values[:nobs], n=ntraj)
    ds['z'] = repeat(np.repeat(field.depth, 4).values[:nobs], n=ntraj)
    # ds['time'] = repeat(field.time.values[:nobs], n=ntraj)
    ds['time'] = repeat(times.values[:nobs], n=ntraj)  # !!!

    # Copy sample dataset (before modifying it).
    ds1 = ds.copy()
    ds2 = ds.copy()
    ds3 = ds.copy()

    # Sample OFAM3.
    for p in range(ds.traj.size):
        for i in range(ds.obs.size):
            dx = ds.isel(traj=p, obs=i).drop(['obs', 'traj'])
            loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
            ds[var][dict(traj=p, obs=i)] = field.sel(loc).load().item()

    ds3[var] = (['traj', 'obs'], update_field_AAA(ds3, field_dataset, var, dim_map))
    ds2 = update_ofam3_fields_loop(ds2, field_dataset, [var])
    ds1[var] = (['traj', 'obs'], update_field_AAA_old(ds1, field_dataset, var, dim_map))

    try:
        # Note that this test fasils if any NaN present
        check = ds[var] == ds1[var]
        if not all(check):
            print('Error (AAA method):', var)
        check = ds[var] == ds2[var]
        if not all(check):
            print('Error (loop method):', var)
        check = ds[var] == ds3[var]
        if not all(check):
            print('Error (AAA alt method):', var)
    except:
        print('Not tested')
        pass
    return ds, ds1, ds2


def test_field_sampling_time_OOB_xarray():
    """Test BGC field sampling using numpy.apply_along_axis using OOB times.

    Notes:
       - picks last avail time.

    """
    exp = ExpData(scenario=0, lon=190, test=True)
    ntraj = 2  # Number of particle trajectory (will be duplicates anyway).
    nobs = 50
    variables = ['phy', 'det']

    # OFAM3 fields.
    df = ofam3_datasets(exp, variables=['phy', 'det'], decode_times=True, chunks=False)
    ds = sample_particle_dataset(df, ntraj, nobs)
    ds['time'] = repeat(np.arange('2012-12-01T12:00:00', '2013-01-20T12:00:00',
                                  np.timedelta64(1, 'D'), dtype='datetime64[ns]'), n=ntraj)
    ds['z'] = (ds['z'] * 0) + 2.5

    # Sample OFAM3.
    for p in range(ds.traj.size):
        for i in range(ds.obs.size):
            dx = ds.isel(traj=p, obs=i).drop(['obs', 'traj'])
            loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
            for v in variables:
                ds[v][dict(traj=p, obs=i)] = df[v].sel(loc, method='nearest').load().item()

    v = 'det'
    p, i = 0, -1
    dx = ds.isel(traj=p, obs=i).drop(['obs', 'traj'])
    loc = dict(depth=dx.z, lat=dx.lat, lon=dx.lon)
    print('nearest', ds[v][p, i].item(),
          'last', df[v].sel(loc, method='nearest').isel(time=-1).load().item())


def test_field_sampling_interpolation_xarray():
    """Test BGC field est interpolation of fields at particle positions.

    Notes:
       - Unfinished

    """

    exp = ExpData(scenario=0, lon=190, test=True, test_month_bnds=5)
    ntraj = 3  # Number of particle trajectory (will be duplicates anyway).
    nobs = 102  # N.B. 102 to double depths.
    variables = ['phy', 'det']

    # OFAM3 fields.
    df = ofam3_datasets(exp, variables=variables, decode_times=True, chunks=False)
    ds = sample_particle_dataset(df, ntraj, nobs)

    ds['lat'] = repeat(np.repeat([1], nobs), n=ntraj)
    ds['lon'] = repeat(np.repeat([165], nobs), n=ntraj)
    ds['z'] = repeat(np.repeat([2.5], nobs), n=ntraj)
    ds['time'] = repeat(np.repeat(df.time.values[0], nobs), n=ntraj)

    # Particle 0.
    # Sample OFAM3 depths at and between each OFAM3 depth level.
    p = 0
    depths = np.zeros(51*2)
    depths[::2] = df.st_edges_ocean.values[:-1]
    depths[1::2] = df.depth.values
    ds['z'][dict(traj=p)] = depths

    p = 1
    ds['lon'][dict(traj=p)] = np.arange(165, 275, 0.05)[:nobs]

    p = 2
    ds['time'][dict(traj=p)] = np.arange(df.time.values[0], df.time.values[-1],
                                         np.timedelta64(12, 'h'), dtype='datetime64[ns]')[:nobs]

    # Filter out depths at 0m (for OOB error).
    ds['z'] = xr.where(ds.z != 0, ds.z, 2.5)
    ds1 = ds.copy()
    ds1 = update_ofam3_fields_AAA(ds1, df, variables)
    return


def test_field_sampling_interpolation_parcels():
    """Test BGC field est interpolation of fields at particle positions.

    Notes:
       - Unfinished

    """

    exp = ExpData(scenario=0, lon=190, test=True, test_month_bnds=5)
    ntraj = 3  # Number of particle trajectory (will be duplicates anyway).
    nobs = 91  # N.B. 102 to double depths.
    variables = ['phy', 'det']

    # OFAM3 fields.

    cs = 300
    fieldset = ofam3_fieldset_parcels240(exp=exp, cs=cs)
    varz = {'phy': 'phy', 'det': 'det'}
    fieldset = add_ofam3_bgc_fields_parcels240(fieldset, varz, exp)

    # update_func = update_PZD_fields_parcels
    df = ofam3_datasets(exp, variables=variables, decode_times=True, chunks=False)
    ds = sample_particle_dataset(df, ntraj, nobs)

    ds['lat'] = repeat(np.repeat([1], nobs), n=ntraj)
    ds['lon'] = repeat(np.repeat([165], nobs), n=ntraj)
    ds['z'] = repeat(np.repeat([2.5], nobs), n=ntraj)
    ds['time'] = repeat(np.repeat(df.time.values[0], nobs), n=ntraj)

    # Particle 0.
    # Sample OFAM3 depths at and between each OFAM3 depth level.
    p = 0
    depths = np.zeros(51*2)
    depths[::2] = df.st_edges_ocean.values[:-1]
    depths[1::2] = df.depth.values
    ds['z'][dict(traj=p)] = depths[:nobs]

    p = 1
    ds['lon'][dict(traj=p)] = np.arange(165, 275, 0.05)[:nobs]

    p = 2
    ds['time'][dict(traj=p)] = np.arange(df.time.values[0], df.time.values[-1],
                                         np.timedelta64(12, 'h'), dtype='datetime64[ns]')[:nobs]

    # Convert times to time since fieldset.U.grid.time_origin
    ds['time'] = ((ds.time - np.datetime64('2012-01-01T12:00:00')) / (1e9))
    ds1 = ds.copy()

    ds1 = update_ofam3_fields_parcels(ds1, fieldset)

    return


@timeit(my_logger=logger)
def test_field_sampling(nparticles=10, nfields=3, method='xarray'):
    """Test sampling fields at all particle positions.

    Args:
        nparticles (int, optional): Number of particles. Defaults to 10.
        nfields (int, optional): Number of fields 1-6. Defaults to 3.
        method (str, optional): 'xarray' or 'parcels'. Defaults to 'xarray'.

    Returns:
        ds: (xarray.Dataset). Particle dataset.

    Notes:
        - if method == 'parcels', nfields in updatefunc needs to be manually changed.
        - All P & Z values are 0.00006104 (6.104e-05)
            - This is because phy and zoo are basically zero below 175m
        - Only selecting nearest neighbour (check if issue?)

        - Xarray
            - 10p 10y 3f:  (13-18s/particle).
            - 5p 10y 6f: 0:2:33.35 total: 153.35 seconds (25-32s/particle).
            - Double nfields -> double runtime.

        - Parcels
            - 10p 10y 3f: 0:33:43.63 total: 2023.63 seconds (200s/particle).
            - 5p 10y 6f: 0:41:40.17 total: 2500.17 seconds (443-600s/particle).
            - Double nfields -> >double runtime.

    Todo:
        - 4D sampling without loop
        - Loop through time & search matching particles instead
        - check time search indexing correct
        - test mem usage
        - test output file size

    """
    def update_PZD_fields_xarray(dx, fieldset, time, variables):
        """Test function: sample a variable."""

        loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
        for k, v in variables.items():
            dx[k] = fieldset[v].sel(loc, method='nearest').load().item()
        return dx, fieldset

    def update_PZD_fields_parcels(dx, fieldset, time, variables):
        """Test function: sample a variable."""
        fieldset.computeTimeChunk(time, 1)
        dx['P'] = fieldset.P[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        dx['Z'] = fieldset.Z[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        dx['Det'] = fieldset.Det[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        dx['U'] = fieldset.U[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        dx['V'] = fieldset.V[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        dx['W'] = fieldset.W[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        return dx, fieldset

    @timeit(my_logger=logger)
    def loop_particle_obs(p, ds, fieldset, update_func, variables):
        n_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for time in range(n_obs):
            particle_loc = dict(traj=p, obs=time)
            ds[particle_loc], fieldset = update_func(ds[particle_loc], fieldset, time, variables)
        return ds, fieldset

    # logger.debug('Test field sampling ({}): {}p, 1y, {} fields.'
    #              .format(method, nparticles, nfields))
    exp = ExpData(0, lon=165, file_index=0)
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det', 'U': 'u', 'V': 'v', 'W': 'w'}
    variables = dict((k, variables[k]) for k in list(variables.keys())[:nfields])

    # Particle trajectories.
    df = ofam3_datasets(exp, variables=['phy', 'det'], decode_times=True, chunks=False)
    ds = sample_particle_dataset(df, 3, 50)
    ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})

    # Subset number of particles.
    ds = ds.isel(traj=slice(nparticles)).dropna('obs', 'all')
    # Subset number of times.
    ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    # Reverse particleset.
    ds = inverse_plx_dataset(ds)

    # Initialise fields in particle dataset.
    particle_vars = list(variables.keys())
    initial_vals = [np.nan] * len(particle_vars)  # Fill with NaN.
    for var, val in zip(particle_vars, initial_vals):
        ds[var] = xr.DataArray(np.full(ds.z.shape, np.nan, dtype=np.float32),
                               dims=ds.z.dims, coords=ds.z.coords)

    if method == 'xarray':
        update_func = update_PZD_fields_xarray
        fieldset = ofam3_datasets(exp, variables=variables)
    else:
        update_func = update_PZD_fields_parcels
        cs = 300
        fieldset = ofam3_fieldset_parcels240(exp=exp, cs=cs)
        varz = {'P': 'phy', 'Z': 'zoo', 'Det': 'det'}
        fieldset = add_ofam3_bgc_fields_parcels240(fieldset, varz, exp)
        # fieldset = add_Kd490_field_parcels240(fieldset)

    # Sample fields at particle positions.
    n_particles = ds.traj.size
    for p in range(n_particles):
        ds, fieldset = loop_particle_obs(p, ds, fieldset, update_func, variables)
    ds.to_netcdf(paths.data / 'test/test_field_sampling_{}_{}p_{}f.nc'.format(method, nparticles, nfields))
    return ds


def check_test_field_sampling_interpolation(nparticles, nfields):
    """Check xarray and parcels particle trajectory field sampling equality."""
    logger.debug('Check xarray and parcels fields: 1y {}p {}f.'.format(nparticles, nfields))
    ds1 = xr.open_dataset(paths.data / 'test/test_field_sampling_xarray_{}p_{}f.nc'.format(nparticles, nfields))
    ds2 = xr.open_dataset(paths.data / 'test/test_field_sampling_parcels_{}p_{}f.nc'.format(nparticles, nfields))

    for var in ds1.data_vars:
        if not all(~np.isnan(ds1[var]) == ~np.isnan(ds2[var])):
            logger.debug('Failed check for {} field: 1y {}p {}f.'.format(var, nparticles, nfields))
    return


# fieldset, ds = felx_alt_test_simple()
# nparticles = 5
# nfields = 6
# ds = test_field_sampling(nparticles, nfields, method='parcels')
# ds = test_field_sampling(nparticles, nfields, method='xarray')
# check_test_field_sampling_interpolation(nparticles, nfields)
