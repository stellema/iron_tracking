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
    - Run simulation+
    C
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
from cfg import paths
from tools import timeit, mlogger
from datasets import (get_ofam_filenames, ofam3_datasets,
                      plx_particle_dataset)
from inverse_plx import inverse_plx_dataset, inverse_particle_obs
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
    exp = cfg.ExpData(0, lon=165)
    constants = iron_model_constants()
    # field_vars = cfg.bgc_name_map
    field_vars = {'U': 'u', 'V': 'v'}
    fieldset = ofam3_datasets(exp, variables=field_vars)

    ds = plx_particle_dataset(exp)
    if cfg.test:
        ds = ds.isel(traj=slice(1)).dropna('obs', 'all')
        ds = ds.where(ds.time >= np.datetime64('2012-01-01'))
    ds = inverse_particle_obs(ds)

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
