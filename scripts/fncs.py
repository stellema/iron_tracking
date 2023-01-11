# -*- coding: utf-8 -*-
"""

Example:

Notes:

Todo:
exp=0


times = get_datetime_bounds(0)
var='adic'
f = get_ofam_filenames(var, get_datetime_bounds(0))
ds = xr.open_dataset(f[0])


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
from tools import get_datetime_bounds, get_ofam_filenames


def from_ofam3(filenames, variables, dimensions, indices=None, mesh='spherical',
               allow_time_extrapolation=None, field_chunksize='auto', time_periodic=False,
               tracer_interp_method='bgrid_tracer', **kwargs):
    """Initialise FieldSet object from NetCDF files of OFAM3 fields (parcels 2.2.1).

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
                                    field_chunksize=field_chunksize,
                                    interp_method=interp_method,
                                    gridindexingtype='mom5', **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)
    return fieldset


def ofam3_fieldset(exp=0, cs=300):
    """Create a 3D parcels fieldset from OFAM model output (parcels 2.2.1).

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
    fieldset = from_ofam3(filenames, variables, dimensions, **kwargs)
    return fieldset


def add_ofam3_bgc_fields(fieldset, variables, exp):
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

    xfieldset = from_ofam3(filenames, variables, dimensions, field_chunksize=chunks)

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
    """Add Kd490 field to fieldset (parcels 2.2.1).

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


def ParticleIron(ds, fieldset, time, constants):
    """Lagrangian Iron scavenging."""
    loc = dict(time=ds.time, depth=ds.z, lat=ds.lat, lon=ds.lon)

    T = fieldset.temp.sel(loc, method='nearest')
    D = fieldset.Det.sel(loc, method='nearest')
    Z = fieldset.Z.sel(loc, method='nearest')
    P = fieldset.P.sel(loc, method='nearest')
    N = fieldset.N.sel(loc, method='nearest')

    Kd490 = df.Kd490.sel(loc, method='nearest')

    Fe = ds.fe

    # IronScavenging
    particle.scav = ((df.k_org * math.pow((D / df.w_det), 0.58) * Fe)
                     + (df.k_inorg * math.pow(Fe, 1.5)))

    # IronRemineralisation
    bcT = math.pow(df.b, df.c * T)
    y_2 = 0.01 * bcT
    u_P = 0.01 * bcT
    if particle.depth < 180:
        u_D = 0.02 * bcT
    else:
        u_D = 0.01 * bcT

    particle.reg = 0.02 * (u_D * D + y_2 * Z + u_P * P)

    # IronPhytoUptake
    Frac_z = math.exp(-particle.depth * Kd490)
    I = df.PAR * df.I_0 * Frac_z
    J_max = math.pow(df.a * df.b, df.c * T)
    J = J_max * (1 - math.exp(-(df.alpha * I) / J_max))
    J_bar = J_max * min(J / J_max, N / (N + df.k_N), Fe / (Fe + df.k_fe))
    particle.phy = 0.02 * J_bar * P

    # IronSourceInput
    particle['src'] = 0#df.Fe[time, particle.depth, particle.lat, particle.lon]

    # Iron
    particle['fe'] = particle.src + particle.reg - particle.phy - particle.scav

    return particle


def ofam3_datasets(exp, variables=cfg.bgc_name_map):
    """ Get OFAM3 BGC fields.

    Args:
        exp (int): Scenario.

    Returns:
        ds (xarray.Dataset): Dataset of BGC fields.

    Todo:
        - Chunk dataset
        - Change dtype
        - add constants
        - Add Kd490

    """
    dims_name_map = {'lon': ['xt_ocean', 'xu_ocean'],
                     'lat': ['yt_ocean', 'yu_ocean'],
                     'depth': ['st_ocean', 'sw_ocean']}

    mesh = xr.open_dataset(cfg.data / 'ofam_mesh_grid_part.nc')  # OFAM3 coords dataset.

    def rename_ofam3_coords(ds):
        rename_map = {'Time': 'time'}
        for k, v in dims_name_map.items():
            c = [i for i in v if i in ds.dims][0]
            rename_map[c] = k
            ds.coords[c] = mesh[v[0]].values
        ds = ds.rename(rename_map)
        return ds

    cs = 300
    chunks = {'Time': 1,
              'sw_ocean': 1, 'st_ocean': 1, 'depth': 1,
              'yt_ocean': cs, 'yu_ocean': cs, 'lat': cs,
              'xt_ocean': cs, 'xu_ocean': cs, 'lon': cs}

    nmonths = 2 if any([i for i in variables.keys() if i in cfg.bgc_name_map.keys()]) else 12
    # Dictionary mapping variables to file(s).
    files = []
    for v in variables:
        f = get_ofam_filenames(variables[v], get_datetime_bounds(exp, nmonths))
        files.append(f)
    files = np.concatenate(files)

    ds = xr.open_mfdataset(files, chunks=chunks, preprocess=rename_ofam3_coords,
                           compat='override', combine='by_coords', coords='minimal',
                           data_vars='minimal', parallel=False)

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
