# -*- coding: utf-8 -*-
"""Test parcels fieldset, particleset and importing plx particle trajectories.

Notes:
    - Uses Python=3.X?
    - Primarily uses parcels version 2.2.1
    - Create Fieldset.
    - Create ParticleClass.
    - Create ParticleSet.
    - Create kernels

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Thu Oct 20 12:24:04 2022

"""
import math
import parcels
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from parcels import (FieldSet, ParticleSet, VectorField, Variable, JITParticle,
                     ScipyParticle, ErrorCode, AdvectionRK4)

from cfg import paths, ExpData, bgc_name_map
from datasets import get_ofam_filenames
from kernels import (AdvectionRK4_3D, recovery_kernels, IronScavenging,
                     IronRemineralisation, IronSourceInput, IronPhytoUptake, Iron)
from fieldsets import (ofam3_fieldset_parcels221, add_Kd490_field_parcels221,
                       add_ofam3_bgc_fields_parcels221,
                       ofam3_fieldset_parcels240, add_Kd490_field_parcels240,
                       add_ofam3_bgc_fields_parcels240,
                       get_fe_pclass, add_fset_constants)


if parcels.__version__ == '2.2.1':
    ofam3_fieldset = ofam3_fieldset_parcels221
    add_ofam3_bgc_fields = add_ofam3_bgc_fields_parcels221
    add_Kd490_field = add_Kd490_field_parcels221
else:
    ofam3_fieldset = ofam3_fieldset_parcels240
    add_ofam3_bgc_fields = add_ofam3_bgc_fields_parcels240
    add_Kd490_field = add_Kd490_field_parcels240


def test_fieldset_initialisation():
    """Test creation of a parcels fieldset using OFAM3 fields."""
    cs = 300
    exp = ExpData(scenario=0)
    variables = bgc_name_map
    variables = {'P': 'phy', 'Z': 'zoo', 'Det': 'det',
                 'temp': 'temp', 'Fe': 'fe', 'N': 'no3'}
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)
    fieldset.computeTimeChunk(0, timedelta(hours=48).total_seconds())
    return fieldset


def test_pset_creation():
    """Test creation of a parcels particleset using OFAM3 fields."""
    exp = ExpData(scenario=0)
    # Simulation runtime paramaters.
    dt = timedelta(hours=48)  # Advection step (negative for backward).

    # Create Fieldset.
    cs = 300
    variables = bgc_name_map
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)
    fieldset.computeTimeChunk(0, dt.seconds)

    # Create ParticleClass.
    pclass = get_fe_pclass(fieldset)
    return fieldset, pclass


def test_forward_particle_execution():
    exp = ExpData(scenario=0)
    # Simulation runtime paramaters.
    dt = timedelta(hours=48)  # Advection step (negative for backward).
    outputdt = timedelta(days=2)  # Advection steps to write.
    runtime = timedelta(days=20)

    # Create Fieldset.
    cs = 300
    variables = bgc_name_map
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
    # @todo
    # output_file = pset.ParticleFile(cfg.paths.data/'test.nc', outputdt=outputdt)

    print(pset)
    start = fieldset.time_origin.time_origin  # @todo
    # ParticleSet execution endtime.
    endtime = int(pset_start + runtime.total_seconds())  # @todo
    pset.execute(kernels, endtime=endtime, dt=dt, #output_file=output_file,
                  verbose_progress=True, recovery=recovery_kernels)
    print(pset)


def test_pset_from_file():
    """

    Returns:
        None.

    Notes:
        - Data file time has units "hours since 2012-12-31 12:00:00"

    """
    def AgeTest(particle, fieldset, time):
        if particle.state == ErrorCode.Evaluate:
            particle.agetest = particle.agetest + math.fabs(particle.dt)

    exp = ExpData(scenario=0)
    cs = 300

    variables = bgc_name_map
    fieldset = ofam3_fieldset(exp=exp, cs=cs)
    # fieldset = add_ofam3_bgc_fields(fieldset, variables, exp)
    fieldset = add_Kd490_field(fieldset)
    fieldset = add_fset_constants(fieldset)

    restarttime = 279480 # !!! just picked lowest df.time.min('obs').min() but should be None

    # Create ParticleClass.
    class uParticle(ScipyParticle):
        agetest = Variable('agetest', initial=0, dtype=np.float32)
    pclass = uParticle

    kwargs = {}
    # Pset from file
    filename = paths.data / 'plx/plx_hist_165_v1r00.nc'
    df = xr.open_dataset(str(filename), decode_cf=False)
    df = df.isel(traj=slice(10)).dropna('obs', 'all')  # !!!

    # TODO: Convert time values in dataset
    # hours since 2012-12-31 12:00:00 --> hours since 1981-01-01 12:00:00
    td = (datetime(exp.year_bnds[1], 12, 31, 12) - datetime(exp.year_bnds[0], 1, 1, 12))
    df['time'] = df['time'] + (td.total_seconds() / (60 * 60))

    # Get particle variables.
    df_vars = [v for v in df.data_vars]
    vars = {}
    to_write = {}  # TODO: list of variables to write.

    for v in pclass.getPType().variables:
        if v.name in df_vars:
            vars[v.name] = np.ma.filled(df.variables[v.name], np.nan)
        to_write[v.name] = v.to_write

    vars['depth'] = np.ma.filled(df.variables['z'], np.nan)
    vars['id'] = np.ma.filled(df.variables['trajectory'], np.nan)

    if isinstance(vars['time'][0], np.timedelta64):
        vars['time'] = np.array([t / np.timedelta64(1, 's') for t in vars['time']])

    pclass.setLastID(0)  # reset to zero offset

    if restarttime is None:
        restarttime = np.nanmin(vars['time'])
    if callable(restarttime):
        restarttime = restarttime(vars['time'])
    else:
        restarttime = restarttime

    inds = np.where(vars['time'] >= restarttime)
    for v in vars:
        if to_write[v]:
            # vars[v] = vars[v][:, ::-1]
            vars[v] = vars[v][inds]

        elif to_write[v] == 'once':
            # vars[v] = vars[v][:, ::-1]
            vars[v] = vars[v][inds[0]]

        if v not in ['lon', 'lat', 'depth', 'time', 'id']:
            kwargs[v] = vars[v]

    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'],
                       lat=vars['lat'], depth=vars['depth'], time=vars['time'],
                       pid_orig=vars['id'],
                       lonlatdepth_dtype=np.float32, **kwargs)

    dt = timedelta(hours=48)  # Advection step (negative for backward).
    runtime = timedelta(days=5)
    kernels = pset.Kernel(AgeTest)
    pset.execute(kernels, runtime=runtime, dt=dt,
                 verbose_progress=True, recovery=recovery_kernels)


# fieldset = test_fieldset_initialisation()
# fieldset, pclass = test_pset_creation
# test_forward_particle_execution()
