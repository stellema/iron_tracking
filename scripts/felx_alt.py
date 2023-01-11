# -*- coding: utf-8 -*-
"""
python 3.9

Created on Tue Jan 10 11:52:04 2023

@author: a-ste
"""
import math
import math
import random
import parcels
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

import cfg
from tools import timeit, mlogger, get_datetime_bounds, get_ofam_filenames
from fncs import (from_ofam3, ofam3_fieldset, add_ofam3_bgc_fields, get_fe_pclass,
                  add_fset_constants, add_Kd490_field, ofam3_datasets, ParticleIron,
                  iron_model_constants)
from fncs_parcels_new import (from_ofam3, ofam3_fieldset, add_ofam3_bgc_fields,
                              add_Kd490_field)
from kernels import (AdvectionRK4_3D, recovery_kernels, IronScavenging,
                     IronRemineralisation, IronSourceInput, IronPhytoUptake, Iron)

logger = mlogger('felx_alt_test')


def plx_particle_dataset(exp, lon=165, r=0):
    """Get particle file.

    Args:
        exp (int): Scenario.
        lon (int, optional): Particle release longitude. Defaults to 165.
        r (in, optional): Particle file repeat index. Defaults to 0.

    Returns:
        ds (xarray.Dataset): Dataset particle trajectories.

    Todo:
        - chunk dataset
        - Filter spinup particles
        - Filter by available field times

    """
    file = cfg.data / 'plx/plx_{}_{}_v1r{:02d}.nc'.format(['hist', 'rcp'][exp], lon, r)
    ds = xr.open_dataset(str(file), decode_cf=True)
    ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})
    return ds


@timeit
def reverse_plx_particle_dataset(ds):
    """Get particle file.

    Args:
        ds (xarray.Dataset): Dataset particle trajectories.

    Returns:
        ds (xarray.Dataset): Reversed dataset particle trajectories.

    Todo:

    """
    # Reverse order of obs indexes (forward).
    ds = ds.isel(obs=slice(None, None, -1))

    # Remove leading NaNs
    for p in range(ds.traj.size):
        # Index of first non-NaN value
        N = np.isnan(ds.isel(traj=p).z).sum().item()
        ds[dict(traj=p)] = ds.isel(traj=p).roll(obs=-N)

    ds.coords['obs'] = np.arange(ds.obs.size, dtype=np.int32)
    ds.coords['traj'] = ds.traj.astype(dtype=np.int32)
    return ds


@timeit
def felx_alt_test_simple():
    """Test simple config of manual felx experiment.

    - Opens OFAM3 U & V fields
    - Opens particle data & reverses observation coordinate
    - Adds new particle data variables to particle dataset (filled with NaN)
    - Calculates test function for each particle and timestep.

    Todo:
        - Initialize particle data properly so time counter starts at 1 not zero

    """
    def UpdateU(dx, fieldset, time, constants):
        """Test function: add current zonal velocity to previous."""

        loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
        # loc = dict(time=particle.time, xt_ocean=particle.lon, yt_ocean=particle.lat, st_ocean=particle.z)

        U = fieldset.u.sel(loc, method='nearest').load().item()
        dx['u'] = dx['prev_u'].item() + U
        dx['u'] = dx['prev_u'].item() * U
        dx['u'] = dx['prev_u'].item() * U
        dx['u'] = dx['prev_u'].item() + U
        return dx

    #------------------------------------------------------------------------------------
    # Non-Parallelisable.
    exp = 0
    constants = iron_model_constants()
    # field_vars = cfg.bgc_name_map
    field_vars = {'U': 'u', 'V': 'v'}
    fieldset = ofam3_datasets(exp, variables=field_vars)

    ds = plx_particle_dataset(exp, lon=165, r=0)
    if cfg.test:
        ds = ds.isel(traj=slice(1)).dropna('obs', 'all')
        ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    ds = reverse_plx_particle_dataset(ds)

    # Add BGC particle arrays to dataset.
    particle_vars = ['u', 'prev_u']
    initial_vals = [np.nan, np.nan]  # Fill with NaN or zero?

    for var, val in zip(particle_vars, initial_vals):
        ds[var] = xr.DataArray(np.full(ds.z.shape, np.nan, dtype=np.float32),
                               dims=ds.z.dims, coords=ds.z.coords)
        ds[var][dict(obs=0)] = 0

    #------------------------------------------------------------------------------------
    # Parallelisable.
    # Todo: Split functions.
    n_particles = ds.traj.size
    for p in range(n_particles):
        n_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for time in range(n_obs):
            particle_loc = dict(traj=p, obs=time)

            dx = ds[particle_loc]

            ds[particle_loc] = UpdateU(dx, fieldset, time, constants)
            ds['prev_u'][dict(traj=p, obs=time + 1)] = ds['u'][dict(traj=p, obs=time)]

    #------------------------------------------------------------------------------------
    # Todo: Save dataset.

    return fieldset, ds


def test_field_sampling():
    """Test sampling fields at all particle positions.


    Returns:
        None.

    """
    def UpdateP_parcels(dx, fieldset, time):
        """Test function: sample a variable."""
        dx['P'] = fieldset.P[time, dx.z.item(), dx.lat.item(), dx.lon.item()]
        return dx

    def UpdateP_xarray(dx, fieldset):
        """Test function: sample a variable."""
        loc = dict(time=dx.time, depth=dx.z, lat=dx.lat, lon=dx.lon)
        dx['P'] = fieldset.P.sel(loc, method='nearest').load().item()
        return dx

    cs = 300
    exp = 0
    variables = cfg.bgc_name_map
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det',
                 'temp': 'temp', 'Fe': 'fe', 'N': 'no3'}
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)

    ds = plx_particle_dataset(exp, lon=165, r=0)
    if cfg.test:
        ds = ds.isel(traj=slice(10)).dropna('obs', 'all')
        ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    ds = reverse_plx_particle_dataset(ds)

    particle_vars = ['P']
    initial_vals = [np.nan]  # Fill with NaN or zero?

    for var, val in zip(particle_vars, initial_vals):
        ds[var] = xr.DataArray(np.full(ds.z.shape, np.nan, dtype=np.float32),
                               dims=ds.z.dims, coords=ds.z.coords)
        ds[var][dict(obs=0)] = 0

    n_particles = ds.traj.size
    for p in range(n_particles):
        n_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for time in range(n_obs):
            particle_loc = dict(traj=p, obs=time)
            dx = ds[particle_loc]
            fieldset.P.computeTimeChunk(time, 1)
            ds[particle_loc] = UpdateP_parcels(dx, fieldset, time)


    fieldset.computeTimeChunk(0, timedelta(hours=48).total_seconds())
    return


# fieldset, ds = felx_alt_test_simple()
