# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:23:10 2023

@author: a-ste
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
from tools import get_datetime_bounds, get_ofam_filenames


def from_ofam3(filenames, variables, dimensions, indices=None, mesh='spherical',
               allow_time_extrapolation=None, chunksize='auto', time_periodic=False,
               tracer_interp_method='bgrid_tracer', **kwargs):
    """Initialise FieldSet object from NetCDF files of OFAM3 fields (parcels 2.X).

    Args:
        filenames (dict): Dictionary mapping variables to file(s).
            {var:{dim:[files]}}
        variables (dict): Dictionary mapping variables to variable names in the file(s).
        dimensions (TYPE): Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
        indices (TYPE, optional): DESCRIPTION. Defaults to None.
        mesh (str, optional): DESCRIPTION. Defaults to 'spherical'.
        allow_time_extrapolation (TYPE, optional): DESCRIPTION. Defaults to None.
        interp_method (str, optional): DESCRIPTION. Defaults to None.
        tracer_interp_method (str, optional): Defaults to 'bgrid_tracer'.
        time_periodic (TYPE, optional): DESCRIPTION. Defaults to False.
        chunksize (TYPE, optional): Size of the chunks in dask loading.
            Can be None (no chunking) or False (no chunking), 'auto', or a dict
            in the format '{parcels_varname: {netcdf_dimname :
                (parcels_dimname, chunksize_as_int)}, ...}'

        **kwargs (dict): netcdf_engine {'netcdf', 'scipy'}

    Returns:
        fieldset (TYPE): DESCRIPTION.

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
                                    interp_method=interp_method,
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)
    return fieldset


def ofam3_fieldset(exp=0, cs=300):
    """Create a 3D parcels fieldset from OFAM model output (parcels 2.X).

    Returns:
        fieldset (parcels.Fieldset)

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
        chunks = cs
        nmap = None
    else:
        nmap = cfg.chunkdims_name_map
        # chunksize.
        chunks = {'Time': 1,
                  'sw_ocean': 1, 'st_ocean': 1, 'depth': 1,
                  'yt_ocean': cs, 'yu_ocean': cs, 'lat': cs,
                  'xt_ocean': cs, 'xu_ocean': cs, 'lon': cs}

    kwargs = dict(chunksize=chunks, chunkdims_name_map=nmap, deferred_load=True)
    fieldset = from_ofam3(filenames, variables, dimensions, **kwargs)
    return fieldset


def add_ofam3_bgc_fields(fieldset, variables, exp):
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

    xfieldset = from_ofam3(filenames, variables, dimensions, chunksize=chunks)

    # Field time origins and calander.
    xfieldset.time_origin = fieldset.time_origin
    xfieldset.time_origin.time_origin = fieldset.time_origin.time_origin
    xfieldset.time_origin.calendar = fieldset.time_origin.calendar

    # Add fields to fieldset.
    for fld in xfieldset.get_fields():
        fld.units = parcels.tools.converters.UnitConverter()
        fieldset.add_field(fld, fld.name)
    return fieldset


def format_Kd490_dataset():
    """Merge & add time dimension to Kd490 monthly files."""
    # Open kd490 dataset.
    files = [cfg.obs / 'GMIS_Kd490/GMIS_S_K490_{:02d}.nc'.format(m) for m in range(1, 13)]
    df = xr.open_mfdataset(files, combine='nested', concat_dim='time', decode_cf=False)

    # Create New Dataset.
    # Time bounds.
    time = np.array([np.datetime64('1981-{:02d}-16'.format(i+1)) for i in range(12)],
                    dtype='datetime64[ns]')
    time_bnds = np.array([['1981-{:02d}-01T12'.format(m),
                           '1981-{:02d}-{:02d}T12'.format(m, monthrange(1981, m)[-1])]
                          for m in range(1, 13)], dtype='datetime64[ns]')

    time_attrs = {'bounds': 'time_bnds',
                  # 'units': 'seconds since 1981-01-01 00:00:00',
                  # 'calendar': '365_day',
                  'axis': 'T',
                  'long_name': 'time',
                  'standard_name': 'time'
                  }
    coords = {'lat': df.lat, 'lon': df.lon, 'time': (['time'], time, time_attrs),
              'time_bnds': (['time', 'nv'], time_bnds), 'nv': [1., 2.]}

    attrs = {'Conventions': 'CF-1.7',
             'title': df.attrs['Title'],
             'history': df.attrs['history'],
             'institution': 'Joint Research Centre (IES)',
             'source': 'https://data.europa.eu/data/datasets/dfe3524f-7d57-486c-ac92-cfa730887c48?locale=en',
             'comment': 'Monthly climatology sea surface diffuse attenuation coefficient at 490nm in m^-1 (log10 scalling) derived from the SeaWiFS sensor.',
             'references': 'Werdell, P.J. (2005). Ocean color K490 algorithm evaluation. http://oceancolor.gsfc.nasa.gov/REPROCESSING/SeaWiFS/R5.1/k490_update.html'
             }

    ds = xr.Dataset({'Kd490': (['time', 'lat', 'lon'], df['Kd490'].values, df['Kd490'].attrs)},
                    coords=coords, attrs=attrs)

    ds.time.encoding = {'units': 'seconds since 1981-01-01 00:00:00'}

    # Save dataset.
    ds.to_netcdf(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc', format='NETCDF4',
                 encoding={'Kd490': {'shuffle': True, 'chunksizes': [1, 3503, 9577],
                                     'zlib': True, 'complevel': 5}})

    # # Interpolate lat and lon to OFAM3
    # # Open OFAM3 mesh coordinates.
    # mesh = xr.open_dataset(cfg.data / 'ofam_mesh_grid_part.nc')

    # Plot
    # ds.Kd490.plot(col='time', col_wrap=4)
    # plt.tight_layout()
    # plt.savefig(cfg.fig / 'particle_source_map.png', bbox_inches='tight',
    #             dpi=300)
    files = cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc'
    ds = xr.open_dataset(files)


def add_Kd490_field(fieldset):
    """Add Kd490 field to fieldset.

    Args:
        fieldset (Parcels fieldset): Parcels fieldset.

    Returns:
        fieldset (Parcels fieldset): Parcels fieldset with added fields.

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
