# -*- coding: utf-8 -*-
"""

Example:

Notes:
    - Create Fieldset.
    - Create ParticleClass.
    - Create ParticleSet.
    - Create kernels

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu Oct 20 12:24:04 2022

"""
import math
from parcels import ErrorCode, AdvectionRK4
import math
import random
import parcels
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from parcels import (FieldSet, ParticleSet, VectorField, Variable, JITParticle, ScipyParticle)

import cfg
from kernels import (AdvectionRK4_3D, recovery_kernels, IronScavenging,
                     IronRemineralisation, IronSourceInput, IronPhytoUptake, Iron)
from fncs import (get_datetime_bounds, get_ofam_filenames, from_ofam3,
                  ofam3_fieldset, add_ofam3_bgc_fields, get_fe_pclass,
                  add_fset_constants, add_Kd490_field)



def test_fieldset_initialisation():
    cs = 300
    exp = 0
    variables = cfg.bgc_name_map
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det',
                 'temp': 'temp', 'Fe': 'fe', 'N': 'no3'}
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)
    fieldset.computeTimeChunk(0, timedelta(hours=48).seconds)
    return fieldset


def test_pset_creation():
    exp = 0
    # Simulation runtime paramaters.
    dt = timedelta(hours=48)  # Advection step (negative for backward).
    outputdt = timedelta(days=2)  # Advection steps to write.
    runtime = timedelta(days=20)

    # Create Fieldset.
    cs = 300
    variables = cfg.bgc_name_map
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det', 'temp': 'temp', 'Fe': 'fe', 'N': 'no3'}
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)
    fieldset.computeTimeChunk(0, dt.seconds)

    # Create ParticleClass.
    pclass = get_fe_pclass(fieldset)
    return fieldset, pclass


def test_forward_particle_execution():
    exp = 0
    # Simulation runtime paramaters.
    dt = timedelta(hours=48)  # Advection step (negative for backward).
    outputdt = timedelta(days=2)  # Advection steps to write.
    runtime = timedelta(days=20)

    # Create Fieldset.
    cs = 300
    variables = cfg.bgc_name_map
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)
    fieldset.computeTimeChunk(0, dt.seconds)

    # Create ParticleClass.
    pclass = get_fe_pclass(fieldset)
    # Create ParticleSet.
    pset_start = fieldset.P.grid.time[0]
    npart = 3
    times = np.repeat(pset_start, npart)
    lats = np.repeat(0, npart)
    depths = np.repeat(150, npart)
    lons = np.linspace(165, 170, npart)
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lons, lat=lats, depth=depths, time=times,
                                 lonlatdepth_dtype=np.float32)

    # Kernels.
    kernels = pset.Kernel(AdvectionRK4_3D)
    kernels += (pset.Kernel(IronSourceInput) + pset.Kernel(IronScavenging) +
                pset.Kernel(IronRemineralisation) + pset.Kernel(IronPhytoUptake)
                + pset.Kernel(Iron))

    # # Create output ParticleFile p_name and time steps to write output.
    # output_file = pset.ParticleFile(cfg.data/'test.nc', outputdt=outputdt)  # @todo

    print(pset)
    start = fieldset.time_origin.time_origin  # @todo
    # ParticleSet execution endtime.
    endtime = int(pset_start + runtime.total_seconds())  # @todo
    pset.execute(kernels, endtime=endtime, dt=dt, #output_file=output_file,
                  verbose_progress=True, recovery=recovery_kernels)
    print(pset)



def test_pset_from_file():
    exp = 0
    cs = 300
    variables = cfg.bgc_name_map
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)

    # Create ParticleClass.
    pclass = get_fe_pclass(fieldset)

    # Pset from file
    filename = cfg.data / 'plx/plx_hist_165_v1r00.nc'
    df = xr.open_dataset(str(filename), decode_cf=False)
    df = df.isel(traj=slice(10)).dropna('obs', 'all')  # !!!

    df_vars = [v for v in df.data_vars]

    vars = {}

    for v in ['trajectory', 'time', 'lat', 'lat', 'z']:
        vars[v] = np.ma.filled(df.variables[v], np.nan)

    vars['depth'] = np.ma.filled(df.variables['z'], np.nan)
    vars['id'] = np.ma.filled(df.variables['trajectory'], np.nan)

    if isinstance(vars['time'][0], np.timedelta64):
        vars['time'] = np.array([t / np.timedelta64(1, 's') for t in vars['time']])

    for v in pclass.getPType().variables:
        if v.name in df_vars:
            vars[v.name] = np.ma.filled(df.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt',
                            'depth', 'id', 'fileid', 'state'] \
                and v.to_write:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
        # to_write[v.name] = v.to_write

# fieldset = test_fieldset_initialisation()
# fieldset, pclass = test_pset_creation
# test_forward_particle_execution()
