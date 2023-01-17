# -*- coding: utf-8 -*-
"""Tests functions for felx experiment (non-parcels version).

Notes:
    - Uses Python=3.9
    - Primarily uses parcels version 2.4.0

Example:

Todo:
    - Add Kd490 Fields
    - Calculate background Iron fields
    - Add background Iron fields
    - Calculate & save BGC fields at particle positions
        - Calculate OOM
        - Test fieldset vs xarray times
        - Research best method for:
            - indexing 4D data
            - chunking/opening data
            - parallelisation
    - Filter & save particle datasets
        - Spinup:
            - Calculate num of affected particles
            - Test effect of removing particles
        - Available BGC fields:
            - Figure out availble times
            - Calculate num of affected particles
            - Test effect of removing particles
    - Write UpdateIron function
        - Calculate OOM & memory
    - Run simulation
        - Add parallelisation
        - Optimise parallelisation

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 10 11:52:04 2023

"""
import math
import numpy as np
import xarray as xr

import cfg
from tools import timeit, mlogger
from datasets import (get_datetime_bounds, get_ofam_filenames, ofam3_datasets,
                      plx_particle_dataset)
from dataset_plx_reverse import reverse_plx_particle_obs
from fieldsets import (get_fe_pclass, add_fset_constants,
                       ofam3_fieldset_parcels221, add_Kd490_field_parcels221,
                       add_ofam3_bgc_fields_parcels221,
                       ofam3_fieldset_parcels240, add_Kd490_field_parcels240,
                       add_ofam3_bgc_fields_parcels240)

logger = mlogger('test_felx')


def UpdateIron(ds, fieldset, time, constants):
    """Lagrangian Iron scavenging."""
    loc = dict(time=ds.time, depth=ds.z, lat=ds.lat, lon=ds.lon)

    T = fieldset.temp.sel(loc, method='nearest')
    D = fieldset.Det.sel(loc, method='nearest')
    Z = fieldset.Z.sel(loc, method='nearest')
    P = fieldset.P.sel(loc, method='nearest')
    N = fieldset.N.sel(loc, method='nearest')

    Kd490 = fieldset.Kd490.sel(loc, method='nearest')

    Fe = ds.fe

    # IronScavenging
    ds['scav'] = ((constants.k_org * math.pow((D / constants.w_det), 0.58) * Fe)
                  + (constants.k_inorg * math.pow(Fe, 1.5)))

    # IronRemineralisation
    bcT = math.pow(constants.b, constants.c * T)
    y_2 = 0.01 * bcT
    u_P = 0.01 * bcT
    if ds.z < 180:
        u_D = 0.02 * bcT
    else:
        u_D = 0.01 * bcT

    ds['reg'] = 0.02 * (u_D * D + y_2 * Z + u_P * P)

    # IronPhytoUptake
    Frac_z = math.exp(-ds.z * Kd490)
    I = constants.PAR * constants.I_0 * Frac_z
    J_max = math.pow(constants.a * constants.b, constants.c * T)
    J = J_max * (1 - math.exp(-(constants.alpha * I) / J_max))
    J_bar = J_max * min(J / J_max, N / (N + constants.k_N), Fe / (Fe + constants.k_fe))
    ds['phy'] = 0.02 * J_bar * P

    # IronSourceInput
    ds['src'] = 0#df.Fe[time, particle.depth, particle.lat, particle.lon]

    # Iron
    ds['fe'] = ds.src + ds.reg - ds.phy - ds.scav

    return ds


def iron_model_constants():
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    constants = {}
    """Add fieldset constantss."""
    # Phytoplankton paramaters.
    # Initial slope of P-I curve.
    constants['alpha'] = 0.025
    # Photosynthetically active radiation.
    constants['PAR'] = 0.34  # Qin 0.34!!!
    # Growth rate at 0C.
    constants['I_0'] = 300  # !!!
    constants['a'] = 0.6
    constants['b'] = 1.066
    constants['c'] = 1.0
    constants['k_fe'] = 1.0
    constants['k_N'] = 1.0
    constants['k_org'] = 1.0521e-4  # !!!
    constants['k_inorg'] = 6.10e-4  # !!!

    # Detritus paramaters.
    # Detritus sinking velocity
    constants['w_det'] = 10  # 10 or 5 !!!

    # Zooplankton paramaters.
    constants['y_1'] = 0.85
    constants['g'] = 2.1
    constants['E'] = 1.1
    constants['u_Z'] = 0.06

    # # Not constant.
    # constants['Kd_490'] = 0.43
    # constants['u_D'] = 0.02  # depends on depth & not constant.
    # constants['u_D_180'] = 0.01  # depends on depth & not constant.
    # constants['u_P'] = 0.01  # Not constant.
    # constants['y_2'] = 0.01  # Not constant.
    constants = dotdict(constants)
    return constants





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

    #--------------------------------------------------------------------------------
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
    ds = reverse_plx_particle_obs(ds)

    # Add BGC particle arrays to dataset.
    particle_vars = ['u', 'prev_u']
    initial_vals = [np.nan, np.nan]  # Fill with NaN or zero?

    for var, val in zip(particle_vars, initial_vals):
        ds[var] = xr.DataArray(np.full(ds.z.shape, np.nan, dtype=np.float32),
                               dims=ds.z.dims, coords=ds.z.coords)
        ds[var][dict(obs=0)] = 0

    #--------------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------------
    # Todo: Save dataset.

    return fieldset, ds


@timeit
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

    @timeit
    def loop_particle_obs(p, ds, fieldset, update_func, variables):
        n_obs = ds.isel(traj=p).dropna('obs', 'all').obs.size
        for time in range(n_obs):
            particle_loc = dict(traj=p, obs=time)
            ds[particle_loc], fieldset = update_func(ds[particle_loc], fieldset, time, variables)
        return ds, fieldset

    logger.debug('Test field sampling ({}): {}p, 1y, {} fields.'
                 .format(method, nparticles, nfields))
    exp = 0
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det', 'U': 'u', 'V': 'v', 'W': 'w'}
    variables = dict((k, variables[k]) for k in list(variables.keys())[:nfields])

    # Particle trajectories.
    ds = plx_particle_dataset(exp, lon=165, r=0)
    ds = ds.drop({'age', 'zone', 'distance', 'unbeached', 'u'})

    # Subset number of particles.
    ds = ds.isel(traj=slice(nparticles)).dropna('obs', 'all')
    # Subset number of times.
    ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    # Reverse particleset.
    ds = reverse_plx_particle_dataset(ds)

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
    ds.to_netcdf(cfg.data / 'test/test_field_sampling_{}_{}p_{}f.nc'.format(method, nparticles, nfields))
    return ds


def check_test_field_sampling_interpolation(nparticles, nfields):
    """Check xarray and parcels particle trajectory field sampling equality."""
    logger.debug('Check xarray and parcels fields: 1y {}p {}f.'.format(nparticles, nfields))
    ds1 = xr.open_dataset(cfg.data / 'test/test_field_sampling_xarray_{}p_{}f.nc'.format(nparticles, nfields))
    ds2 = xr.open_dataset(cfg.data / 'test/test_field_sampling_parcels_{}p_{}f.nc'.format(nparticles, nfields))

    for var in ds1.data_vars:
        if not all(~np.isnan(ds1[var]) == ~np.isnan(ds2[var])):
            logger.debug('Failed check for {} field: 1y {}p {}f.'.format(var, nparticles, nfields))
    return


# fieldset, ds = felx_alt_test_simple()
nparticles = 5
nfields = 6
# ds = test_field_sampling(nparticles, nfields, method='parcels')
# ds = test_field_sampling(nparticles, nfields, method='xarray')
check_test_field_sampling_interpolation(nparticles, nfields)
