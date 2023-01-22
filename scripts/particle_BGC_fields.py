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

Example:

Questions:
    - How to ionterpolate kd490 lon fields.

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
import numpy as np
import xarray as xr
import pandas as pd
from argparse import ArgumentParser

import cfg
from tools import timeit, mlogger
from datasets import (ofam3_datasets, get_felx_filename, save_dataset)
from inverse_plx import inverse_plx_dataset

logger = mlogger('felx_files')
test = True


@timeit(my_logger=logger)
def update_ofam3_fields_loop(ds, fieldset, variables):
    """Calculate fields at plx particle positions (using for loop and xarray)."""
    def update_PZD_fields_xarray(dx, fieldset, variables):
        """Test function: sample a variable."""
        loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
        for k, v in variables.items():
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


@timeit(my_logger=logger)
def update_ofam3_fields_AAA(ds, fieldset, variables):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """
    def update_ofam3_field(ds, field):
        def update_field_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[~np.isnan(traj)][0])
            loc = dict(depth=dx.z, lat=dx.lat, lon=dx.lon)
            return field.sel(time=dx.time).sel(loc, method='nearest', drop=True)

        kwargs = dict(ds=ds, field=field)
        return np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)

    for var in variables:
        ds[var] = (['traj', 'obs'], update_ofam3_field(ds, field=fieldset[var]))
    return ds


def update_kd490_field_AAA(ds, fieldset, variables):
    """Calculate fields at plx particle positions dims=(month, lat, lon)."""
    def update_kd490_field(ds, field):
        def sample_field_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[0])
            loc = dict(time=dx.time.dt.month, lat=dx.lat, lon=dx.lon)
            return field.sel(loc, method='nearest', drop=True)
        return np.apply_along_axis(sample_field_particle, 1, ds.trajectory, ds=ds,
                                   field=field)

    for var in variables:
        field = fieldset[var]
        field = field.isel(lon=np.arange(0, field.lon.size, 4))  # fix duplicate index er.
        ds[var] = (['traj', 'obs'], update_kd490_field(ds, field=field))
    return ds


def filter_particles_by_start_time(ds, exp=0):
    """Drop particles that are out the time domain."""
    year = [2000, 2089][exp]
    # ni = ds.traj.size
    dx = ds.isel(obs=0)

    # # alt 1
    # trajs = dx.where(dx.time >= np.datetime64('{}-01-01'.format(year)), drop=True).traj
    # ds = ds.sel(traj=trajs.drop('obs'))

    # alt2
    p = ds.traj[dx.time >= np.datetime64(['2000-01-01', '2089-01-01'][exp])]
    ds = ds.sel(traj=p.drop('obs'))

    # nf = ds.traj.size
    # logger.info('n_traj initial={}, remaining={} %={:.1%}'.format(ni, nf, (nf-ni)/ni))
    return ds


@timeit(my_logger=logger)
def get_felx_BGC_fields(exp, lon, r, v=0):
    """Sample OFAM3 BGC fields at particle positions.

    Args:
        exp (int): Scenario {0, 1}.
        lon (int): Particle release longitude {165, 190, 220, 250}.
        r (int): Particle file repeat index {0-9}.
        v (int, optional): FELX version. Defaults to 0.

    Returns:
        ds (xarray.Dataset): Particle trajectory dataset with BGC fields.

    Notes:
        - First save/create plx_inverse file.

    Todo:
        - Unzip OFAM3 fields [in progress]
        - Upload formatted Kd490 fields [in progress]
        - Save plx reversed datasets
        - parallelise
        - add datset encoding [done]
        - Filter spinup?
        - Add kd490 fields
    """
    file = get_felx_filename(exp, lon, r, v)
    file = cfg.data / ('test/' + file.stem + '_BGC.nc')

    logger.info('{}: Getting OFAM3 BGC fields.'.format(file.stem))

    # Select vairables
    vars_ofam = ['phy', 'zoo', 'det', 'temp', 'fe', 'no3']
    vars_clim = ['Kd490']

    if cfg.test:
        nfields = 3
        vars_ofam = ['phy', 'zoo', 'det', 'u', 'v', 'w'][:nfields]

    variables = np.concatenate((vars_ofam, vars_clim))

    # Initialise ofam3 dataset.
    fieldset = ofam3_datasets(exp, variables=vars_ofam)

    # Kd490 fields.
    kd = xr.open_dataset(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490_month.nc', chunks='auto')
    kd = kd.assign_coords(lon=kd.lon % 360)
    kd = kd.rename(dict(month='time'))

    # Subset lat and lons to match ofam3 data.
    kd = kd.sel(lat=slice(fieldset.lat.min(), fieldset.lat.max()),
                lon=slice(fieldset.lon.min(), fieldset.lon.max()))

    # Initialise particle dataset.
    # Open inversed plx particle data.
    ds = inverse_plx_dataset(exp, lon, r, v=1)

    # Filter times.
    ds = filter_particles_by_start_time(ds)
    if test:
        ds = ds.isel(traj=slice(10))
        ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
        ds = ds.dropna('obs', 'all')
    # convert to days since 1979 (same as ofam3)
    # @ TODO
    ds['time'] = ((ds.time - np.datetime64('1979-01-01T00:00:00')) / (1e9*60*60*24)) - 0.5
    # (ds.time - np.datetime64('1979-01-01T00:00:00')) / (1e9*60*60*24)
    # ds['time'] = (ds.time - datetime(1979,1,1,0)) / (1e9*60*60*24)

    # Initialise fields in particle dataset.
    initial_vals = [np.nan] * len(variables)  # Fill with NaN.
    for var, val in zip(variables, initial_vals):
        ds[var] = xr.DataArray(np.full(ds.z.shape, np.nan, dtype=np.float32),
                               dims=ds.z.dims, coords=ds.z.coords)



    ds = update_ofam3_fields_AAA(ds, fieldset, vars_ofam)
    ds = update_kd490_field_AAA(ds, kd, vars_clim)  # Duplicate index error.

    # ds = update_kd490_field(ds, kd[variables[-1]])  # Duplicate index error.
    # ds = update_ofam3_fields_loop(ds, fieldset, variables)  # Alternate.

    # Save file.
    if not test:
        save_dataset(ds, file, msg='Added BGC fields at paticle positions.')
        logger.info('felx BGC fields file saved: {}'.format(file.stem))

    return ds


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--exp', default=0, type=int, help='Scenario index.')
    p.add_argument('-r', '--repeat', default=0, type=int, help='File repeat.')
    p.add_argument('-v', '--version', default=1, type=int, help='PLX experiment version.')
    args = p.parse_args()

    # get_felx_BGC_fields(lon=args.lon, exp=args.exp, r=args.repeat, v=args.version)

    exp, lon, r, v = 0, 250, 0, 0
    ds = get_felx_BGC_fields(exp, lon, r, v=0)
