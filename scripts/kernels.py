# -*- coding: utf-8 -*-
"""Parcels kernels for EUC iron particle back-tracking experiment.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Fri Oct 7 09:19:11 2022

"""
import math
from parcels import ErrorCode, AdvectionRK4
import math
import random
import parcels
import numpy as np
import xarray as xr
from datetime import datetime
from parcels import (FieldSet, ParticleSet, VectorField, Variable, JITParticle)

import cfg

def AdvectionRK4_3D(particle, fieldset, time):
    """Fourth-order Runge-Kutta 3D particle advection."""
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt


def IronScavenging(particle, fieldset, time):
    """Lagrangian Iron scavenging.

    Equation:
        Fe_scav = k_Fe_org * (D_f/ω_D)^0.58 * Fe + k_Fe_inorg * Fe^1.5
        - Detritus Sinking velocity (w_D)
        - Organic Iron scavenging rate constant (k_org)
        - Inorganic Iron scavenging rate constant (k_inorg)
    """
    D = fieldset.Det[time, particle.depth, particle.lat, particle.lon]
    Fe = particle.fe

    particle.scav = ((fieldset.k_org * math.pow((D / fieldset.w_det), 0.58) * Fe)
                     + (fieldset.k_inorg * math.pow(Fe, 1.5)))


def IronRemineralisation(particle, fieldset, time):
    """Iron remineralisation Fe_reg(z, t, T, D, Z, P).

    Equation:
        Fe_reg = 0.02 * (μ_D * D + γ_2 * Z + μ_P * P)
        - Remineralisation rate (< 180 m): μ_D = 0.02 b^cT
        - Remineralisation rate (≥ 180 m): μ_D = 0.01 b^cT
        - Phytoplankton mortality (μ_P) = 0.01 b^cT
        - Excretion (γ_2) = 0.01 b^cT

    """
    T = fieldset.temp[time, particle.depth, particle.lat, particle.lon]
    D = fieldset.Det[time, particle.depth, particle.lat, particle.lon]
    Z = fieldset.Z[time, particle.depth, particle.lat, particle.lon]
    P = fieldset.P[time, particle.depth, particle.lat, particle.lon]

    bcT = math.pow(fieldset.b, fieldset.c * T)
    y_2 = 0.01 * bcT
    u_P = 0.01 * bcT
    if particle.depth < 180:
        u_D = 0.02 * bcT
    else:
        u_D = 0.01 * bcT
    particle.reg = 0.02 * (u_D * D + y_2 * Z + u_P * P)


def IronPhytoUptake(particle, fieldset, time):
    """Iron uptake for phytoplankton growth Fe_phy(z, t, T, N, Fe).

    Equation:
        Fe_phy = 0.02 * J_bar * P
        - Phytoplankton growth (J_bar):

        - Impact of light on phytoplankton growth rate (J)
        - Maximum phytoplankton growth (J_max)
    """
    Fe = particle.fe
    T = fieldset.temp[time, particle.depth, particle.lat, particle.lon]
    P = fieldset.P[time, particle.depth, particle.lat, particle.lon]
    N = fieldset.N[time, particle.depth, particle.lat, particle.lon]
    Kd490 = fieldset.Kd490[time, particle.depth, particle.lat, particle.lon]

    Frac_z = math.exp(-particle.depth * Kd490)

    I = fieldset.PAR * fieldset.I_0 * Frac_z

    J_max = math.pow(fieldset.a * fieldset.b, fieldset.c * T)
    J = J_max * (1 - math.exp(-(fieldset.alpha * I) / J_max))
    J_bar = J_max * min(J / J_max, N / (N + fieldset.k_N), Fe / (Fe + fieldset.k_fe))
    particle.phy = 0.02 * J_bar * P


def IronSourceInput(particle, fieldset, time):
    """Initialise source iron at LLWBCs."""
    particle.src = 0#fieldset.Fe[time, particle.depth, particle.lat, particle.lon]


def Iron(particle, fieldset, time):
    particle.fe = particle.src + particle.reg - particle.phy - particle.scav


def DeleteParticle(particle, fieldset, time):
    particle.delete()

recovery_kernels = {ErrorCode.ErrorOutOfBounds: DeleteParticle,
                    ErrorCode.Error: DeleteParticle}
