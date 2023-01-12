# -*- coding: utf-8 -*-
"""Parcels fieldset functions for OFAM3 velocity and biogeochemistry data.

Notes:
    - Contains OFAM3 fieldset functions for different parcels versions (2.2.1 and 2.4.0).

Example:
    times = get_datetime_bounds(0)
    var='adic'
    f = get_ofam_filenames(var, get_datetime_bounds(0))
    ds = xr.open_dataset(f[0])

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Oct 19 13:39:06 2022

"""
import math
import parcels
import numpy as np
import xarray as xr
from calendar import monthrange
from datetime import datetime, timedelta
from parcels import (FieldSet, ParticleSet, VectorField, Variable, ScipyParticle,
                     JITParticle, fieldfilebuffer)

import cfg
from datasets import get_datetime_bounds, get_ofam_filenames


def from_ofam3_parcels221(filenames, variables, dimensions, indices=None,
                          mesh='spherical', allow_time_extrapolation=None,
                          field_chunksize='auto', time_periodic=False,
                          tracer_interp_method='bgrid_tracer', **kwargs):
    """Initialise FieldSet object from NetCDF files of OFAM3 fields (parcels 2.2.1).

    Args:
        filenames (dict): Dictionary mapping variables to file(s). {var:{dim:[files]}}
        variables (dict): Dictionary mapping variables to variable names in the file(s).
        dimensions (dict): Dictionary mapping data dims to dimensions in the netCF file(s).
        indices (TYPE, optional): DESCRIPTION. Defaults to None.
        mesh (str, optional): DESCRIPTION. Defaults to 'spherical'.
        allow_time_extrapolation (TYPE, optional): DESCRIPTION. Defaults to None.
        interp_method (str, optional): DESCRIPTION. Defaults to None.
        tracer_interp_method (str, optional): Defaults to 'bgrid_tracer'.
        time_periodic (bool, optional): DESCRIPTION. Defaults to False.
        field_chunksize (optional): Size of the chunks in dask loading.
            Can be None (no chunking) or False (no chunking), 'auto', or a dict
            in the format '{parcels_varname: {netcdf_dimname :
                (parcels_dimname, chunksize_as_int)}, ...}'
        **kwargs (dict): netcdf_engine {'netcdf', 'scipy'}

    Returns:
        fieldset (parcels.Fieldset): OFAM3 Fieldset.

    """
    interp_method = {}
    for v in variables:
        if v in ['U', 'V']:
            interp_method[v] = 'bgrid_velocity'
        elif v in ['W']:
            interp_method[v] = 'bgrid_w_velocity'
        else:
            interp_method[v] = tracer_interp_method

    if 'creation_log' not in kwargs.keys():
        kwargs['creation_log'] = 'from_mom5'

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions,
                                    mesh=mesh, indices=indices,
                                    time_periodic=time_periodic,
                                    allow_time_extrapolation=allow_time_extrapolation,
                                    field_chunksize=field_chunksize,
                                    interp_method=interp_method,
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)
    return fieldset


def from_ofam3_parcels240(filenames, variables, dimensions, indices=None,
                          mesh='spherical', allow_time_extrapolation=None,
                          chunksize='auto', time_periodic=False,
                          tracer_interp_method='bgrid_tracer', **kwargs):
    """Initialise FieldSet object from NetCDF files of OFAM3 fields (parcels 2.4.0).

    Args:
        filenames (dict): Dictionary mapping variables to file(s). {var:{dim:[files]}}
        variables (dict): Dictionary mapping variables to variable names in the file(s).
        dimensions (dict): Dictionary mapping data dims to dimensions in the netCF file(s).
        indices (TYPE, optional): DESCRIPTION. Defaults to None.
        mesh (str, optional): DESCRIPTION. Defaults to 'spherical'.
        allow_time_extrapolation (TYPE, optional): DESCRIPTION. Defaults to None.
        interp_method (str, optional): DESCRIPTION. Defaults to None.
        tracer_interp_method (str, optional): Defaults to 'bgrid_tracer'.
        time_periodic (bool, optional): DESCRIPTION. Defaults to False.
        field_chunksize (optional): Size of the chunks in dask loading.
            Can be None (no chunking) or False (no chunking), 'auto', or a dict
            in the format '{parcels_varname: {netcdf_dimname :
                (parcels_dimname, chunksize_as_int)}, ...}'
        **kwargs (dict): netcdf_engine {'netcdf', 'scipy'}

    Returns:
        fieldset (parcels.Fieldset): OFAM3 Fieldset.

    """
    interp_method = {}
    for v in variables:
        if v in ['U', 'V']:
            interp_method[v] = 'bgrid_velocity'
        elif v in ['W']:
            interp_method[v] = 'bgrid_w_velocity'
        else:
            interp_method[v] = tracer_interp_method

    if 'creation_log' not in kwargs.keys():
        kwargs['creation_log'] = 'from_mom5'

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions,
                                    mesh=mesh, indices=indices,
                                    time_periodic=time_periodic,
                                    allow_time_extrapolation=allow_time_extrapolation,
                                    chunksize=chunksize,
                                    chunkdims_name_map=cfg.chunkdims_name_map,
                                    interp_method=interp_method,
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)
    return fieldset


def ofam3_fieldset_parcels221(exp=0, cs=300):
    """Create a 3D parcels fieldset from OFAM model output (parcels 2.2.1).

    Args:
        exp (int, optional): Scenario (for data file years). Defaults to 0.
        cs (int, optional): Chunk size. Defaults to 300.

    Returns:
        fieldset (parcels.Fieldset): Fieldset with U,V,W fields.

     Notes:
         - chunksize = {'Time':  1, 'sw_ocean': 1, 'st_ocean': 1,
                        'yu_ocean': cs, 'xu_ocean': cs}
         - chunksize[v] = {'Time': ('time', 1),
                           'st_ocean': ('depth', 1), 'sw_ocean': ('depth', 1),
                           'yu_ocean': ('lat', cs), 'yt_ocean': ('lat', cs),
                           'xu_ocean': ('lon', cs), 'xt_ocean': ('lon', cs)}
         - {parcels_vname: {netcdf_dimname: (parcels_dimname, chunksize_int)}
         - chunksize[v] = {'time': ('Time', 1),
                           'depth': ('sw_ocean', 1), 'depth': ('st_ocean', 1),
                           'lat': ('yu_ocean', cs), 'lat': ('yt_ocean', cs),
                           'lon': ('xu_ocean', cs), 'lon': ('xt_ocean', cs)}

    """
    # Add OFAM3 dims to chunk name map.
    # fieldfilebuffer.DaskFileBuffer.add_to_dimension_name_map_global(
    #     cfg.chunkdims_name_map)

    # Dictionary mapping variables to variable names in the netCDF file(s).
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}

    # Dictonary mapping parcels to OFAM3 dimension names.
    dimensions = {}
    for v in variables:
        dimensions[v] = {'lon': 'xu_ocean', 'lat': 'yu_ocean',
                         'depth': 'sw_ocean', 'time': 'Time'}

    # Dictionary mapping variables to file(s).
    mesh = [str(cfg.data / 'ofam_mesh_grid_part.nc')]  # OFAM3 coords dataset.
    filenames = {}
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp))
        filenames[v] = {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': f}

    # Size of the chunks in dask loading.
    if cs in ['auto', False]:
        chunks = cs
        nmap = None
    else:
        nmap = cfg.chunkdims_name_map
        # field_chunksize.
        chunks = {'Time': 1,
                  'sw_ocean': 1, 'st_ocean': 1, 'depth': 1,
                  'yt_ocean': cs, 'yu_ocean': cs, 'lat': cs,
                  'xt_ocean': cs, 'xu_ocean': cs, 'lon': cs}

    kwargs = dict(field_chunksize=chunks, chunkdims_name_map=nmap, deferred_load=True)
    fieldset = from_ofam3_parcels221(filenames, variables, dimensions, **kwargs)
    return fieldset


def ofam3_fieldset_parcels240(exp=0, cs=300):
    """Create a 3D parcels fieldset from OFAM model output (parcels 2.4.0).

    Args:
        exp (int, optional): Scenario (for data file years). Defaults to 0.
        cs (int, optional): Chunk size. Defaults to 300.

    Returns:
        fieldset (parcels.Fieldset): Fieldset with U,V,W fields.

    """
    # Add OFAM3 dims to chunk name map.
    fieldfilebuffer.DaskFileBuffer.add_to_dimension_name_map_global(
        cfg.chunkdims_name_map)

    # Dictionary mapping variables to variable names in the netCDF file(s).
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}

    # Dictonary mapping parcels to OFAM3 dimension names.
    dimensions = {}
    for v in variables:
        dimensions[v] = {'lon': 'xu_ocean', 'lat': 'yu_ocean',
                         'depth': 'sw_ocean', 'time': 'Time'}

    # Dictionary mapping variables to file(s).
    mesh = [str(cfg.data / 'ofam_mesh_grid_part.nc')]  # OFAM3 coords dataset.
    filenames = {}
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp))
        filenames[v] = {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': f}

    # Size of the chunks in dask loading.
    if cs in ['auto', False]:
        chunksizs = cs
    else:
        # chunksize.
        chunksize = {}
        for v in variables:
            chunksize[v] = {'time': ('Time', 1), 'depth': ('sw_ocean', 1),
                            'lat': ('yu_ocean', cs), 'lon': ('xu_ocean', cs)}

    kwargs = dict(chunksize=chunksize, deferred_load=True)
    fieldset = from_ofam3_parcels240(filenames, variables, dimensions, **kwargs)
    return fieldset


def add_ofam3_bgc_fields_parcels221(fieldset, variables, exp):
    """Add additional OFAM3 fields to fieldset (parcels 2.2.1).

    Args:
        fieldset (parcels.Fieldset): Parcels fieldset.
        variables (dict): Dictionary mapping variables names. {'P': 'phy'}
        exp (int): experiment integer.
        chunks (int): chunksize.

    Returns:
        fieldset (parcels.Fieldset): Parcels fieldset with added fields.

    """
    # Dictonary mapping parcels to OFAM3 dimension names.
    dimensions = {}
    for v in variables:
        dimensions[v] = {'lon': 'xu_ocean', 'lat': 'yt_ocean',
                         'depth': 'sw_ocean', 'time': 'Time'}

    # Dictionary mapping variables to file(s).
    filenames = {}
    mesh = [str(cfg.data / 'ofam_mesh_grid_part.nc')]  # OFAM3 coords dataset.
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp))
        filenames[v] = {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': f}

    # Size of the chunks in dask loading.
    chunks = fieldset.get_fields()[0].field_chunksize

    xfieldset = from_ofam3_parcels221(filenames, variables, dimensions,
                                      field_chunksize=chunks)

    # Field time origins and calander.
    xfieldset.time_origin = fieldset.time_origin
    xfieldset.time_origin.time_origin = fieldset.time_origin.time_origin
    xfieldset.time_origin.calendar = fieldset.time_origin.calendar

    # Add fields to fieldset.
    for fld in xfieldset.get_fields():
        fld.units = parcels.tools.converters.UnitConverter()
        fieldset.add_field(fld, fld.name)
    return fieldset


def add_ofam3_bgc_fields_parcels240(fieldset, variables, exp):
    """Add additional OFAM3 fields to fieldset (parcels 2.X).

    Args:
        fieldset (parcels.Fieldset): Parcels fieldset.
        variables (dict): Dictionary mapping variables names. {'P': 'phy'}
        exp (int): experiment integer.
        chunks (int): chunksize.

    Returns:
        fieldset (parcels.Fieldset): Parcels fieldset with added fields.

    """
    # Dictonary mapping parcels to OFAM3 dimension names.
    dimensions = {}
    for v in variables:
        dimensions[v] = {'lon': 'xu_ocean', 'lat': 'yt_ocean',
                         'depth': 'sw_ocean', 'time': 'Time'}

    # Dictionary mapping variables to file(s).
    filenames = {}
    mesh = [str(cfg.data / 'ofam_mesh_grid_part.nc')]  # OFAM3 coords dataset.
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp))
        filenames[v] = {'depth': mesh, 'lat': mesh, 'lon': mesh, 'data': f}

    # Size of the chunks in dask loading.
    chunks = fieldset.get_fields()[0].chunksize

    xfieldset = from_ofam3_parcels240(filenames, variables, dimensions, chunksize=chunks)

    # Field time origins and calander.
    xfieldset.time_origin = fieldset.time_origin
    xfieldset.time_origin.time_origin = fieldset.time_origin.time_origin
    xfieldset.time_origin.calendar = fieldset.time_origin.calendar

    # Add fields to fieldset.
    for fld in xfieldset.get_fields():
        fld.units = parcels.tools.converters.UnitConverter()
        fieldset.add_field(fld, fld.name)
    return fieldset


def add_Kd490_field_parcels221(fieldset):
    """Add Kd490 field to fieldset (parcels version 2.2.1).

    Args:
        fieldset (parcels.fieldset): Parcels fieldset.

    Returns:
        fieldset (parcels.fieldset): Parcels fieldset with added Kd490 field.

    """
    # Dictonary mapping parcels to OFAM3 dimension names.
    filenames = str(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc')
    variables = {'Kd490': 'Kd490'}
    dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}

    # Size of the chunks in dask loading.
    chunks = fieldset.get_fields()[0].field_chunksize
    time_periodic = timedelta(days=365).total_seconds()

    xfieldset = FieldSet.from_netcdf(filenames, variables, dimensions,
                                     field_chunksize=chunks, time_periodic=time_periodic)
    # Field time origins and calander.
    xfieldset.time_origin = fieldset.time_origin
    xfieldset.time_origin.time_origin = fieldset.time_origin.time_origin
    xfieldset.time_origin.calendar = fieldset.time_origin.calendar

    # Add fields to fieldset.
    for fld in xfieldset.get_fields():
        fld.units = parcels.tools.converters.UnitConverter()
        fieldset.add_field(fld, fld.name)
    return fieldset


def add_Kd490_field_parcels240(fieldset):
    """Add Kd490 field to fieldset (parcels version 2.4.0).

    Args:
        fieldset (parcels.fieldset): Parcels fieldset.

    Returns:
        fieldset (parcels.fieldset): Parcels fieldset with added Kd490 field.

    """
    # Dictonary mapping parcels to OFAM3 dimension names.
    filenames = str(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc')
    variables = {'Kd490': 'Kd490'}
    dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}

    # Size of the chunks in dask loading.
    chunks = fieldset.get_fields()[0].chunksize
    time_periodic = timedelta(days=365).total_seconds()

    xfieldset = FieldSet.from_netcdf(filenames, variables, dimensions,
                                     chunksize=chunks, time_periodic=time_periodic)
    # Field time origins and calander.
    xfieldset.time_origin = fieldset.time_origin
    xfieldset.time_origin.time_origin = fieldset.time_origin.time_origin
    xfieldset.time_origin.calendar = fieldset.time_origin.calendar

    # Add fields to fieldset.
    for fld in xfieldset.get_fields():
        fld.units = parcels.tools.converters.UnitConverter()
        fieldset.add_field(fld, fld.name)
    return fieldset


def add_fset_constants(fieldset):
    """Add fieldset constants."""
    # Phytoplankton paramaters.
    # Initial slope of P-I curve.
    fieldset.add_constant('alpha', 0.025)
    # Photosynthetically active radiation.
    fieldset.add_constant('PAR', 0.34)  # Qin 0.34!!!
    # Growth rate at 0C.
    fieldset.add_constant('I_0', 300)  # !!!
    fieldset.add_constant('a', 0.6)
    fieldset.add_constant('b', 1.066)
    fieldset.add_constant('c', 1.0)
    fieldset.add_constant('k_fe', 1.0)
    fieldset.add_constant('k_N', 1.0)
    fieldset.add_constant('k_org', 1.0521e-4)  # !!!
    fieldset.add_constant('k_inorg', 6.10e-4)  # !!!

    # Detritus paramaters.
    # Detritus sinking velocity
    fieldset.add_constant('w_det', 10)  # 10 or 5 !!!

    # Zooplankton paramaters.
    fieldset.add_constant('y_1', 0.85)
    fieldset.add_constant('g', 2.1)
    fieldset.add_constant('E', 1.1)
    fieldset.add_constant('u_Z', 0.06)

    # # Not constant.
    # fieldset.add_constant('Kd_490', 0.43)
    # fieldset.add_constant('u_D', 0.02)  # depends on depth & not constant.
    # fieldset.add_constant('u_D_180', 0.01)  # depends on depth & not constant.
    # fieldset.add_constant('u_P', 0.01)  # Not constant.
    # fieldset.add_constant('y_2', 0.01)  # Not constant.
    return fieldset


def get_fe_pclass(fieldset, dtype=np.float32, **kwargs):
    """Particle class."""

    class FeParticle(ScipyParticle):
        """Particle class.

        s = Variable('s', initial=0., dtype=dtype, to_write='once')
        phy = Variable('phy', initial=fieldset.P, dtype=dtype)
        """
        # Iron.
        fe = Variable('fe', initial=fieldset.Fe, dtype=dtype)
        # Uptake of iron for phytoplankton growth.
        phy = Variable('phy', initial=0., dtype=dtype)
        # Scavenging.
        scav = Variable('scav', initial=0., dtype=dtype)
        # Exogenous inputs.
        src = Variable('src', initial=0., dtype=dtype)
        # Remineralisation.
        reg = Variable('reg', initial=0., dtype=dtype)
    return FeParticle
