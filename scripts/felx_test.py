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

# Create ParticleSet.
pset_start = fieldset.P.grid.time[0]
npart = 3
times = np.repeat(pset_start, npart)
# lats = np.repeat(-5.1, npart)
# depths = np.repeat(150, npart)
# lons = np.linspace(153, 154, npart)
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
