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
from datetime import datetime, timedelta
from parcels import (FieldSet, ParticleSet, VectorField, Variable, ScipyParticle,
                     JITParticle, fieldfilebuffer)

import cfg


def get_datetime_bounds(exp):
    """Get list of experiment start and end year datetimes."""
    y = cfg.years[exp]
    times = [datetime(y[0], 1, 1), datetime(y[1], 12, 31)]

    if cfg.home.drive == 'C:':
        times[0] = datetime(y[1], 1, 1)
        times[1] = datetime(y[1], 2, 29)
    return times


def get_ofam_filenames(var, times):
    """Create OFAM3 file list based on selected times."""
    f = []
    for y in range(times[0].year, times[1].year + 1):
        for m in range(times[0].month, times[1].month + 1):
            f.append(str(cfg.ofam / 'ocean_{}_{}_{:02d}.nc'.format(var, y, m)))
    return f


def from_ofam3(filenames, variables, dimensions, indices=None, mesh='spherical',
               allow_time_extrapolation=None, time_periodic=False,
               tracer_interp_method='bgrid_tracer', chunksize=None, **kwargs):
    """Initialise FieldSet object from NetCDF files of OFAM3 fields.

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
                                    interp_method=interp_method,
                                    gridindexingtype='mom5',
                                    chunkdims_name_map=cfg.chunkdims_name_map,
                                    chunksize=chunksize, **kwargs)

    if hasattr(fieldset, 'W'):
        fieldset.W.set_scaling_factor(-1)
    return fieldset


def ofam3_fieldset(exp=0, cs=300):
    """Create a 3D parcels fieldset from OFAM model output.

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
        chunksize = cs
    else:
        chunksize = {}
        for v in variables:
            chunksize[v] = {'time': ('Time', 1), 'depth': ('sw_ocean', 1),
                            'lat': ('yu_ocean', cs), 'lon': ('xu_ocean', cs)}

    kwargs = dict(deferred_load=True)
    fieldset = from_ofam3(filenames, variables, dimensions, chunksize=chunksize,
                          **kwargs)
    return fieldset


def add_ofam3_bgc_fields(fieldset, variables, exp):
    """Add additional OFAM3 fields to fieldset.

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
    chunksize = fieldset.U.grid.chunksize

    xfieldset = from_ofam3(filenames, variables, dimensions, chunksize=chunksize)

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
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
    ds['time'] = np.array([datetime(1981, i+1, 1, 12) for i in range(12)],
                          dtype='datetime64[ns]')
    # Save dataset.
    ds.to_netcdf(cfg.obs / 'GMIS_Kd490/GMIS_S_Kd490.nc')

    # # Interpolate lat and lon to OFAM3
    # # Open OFAM3 mesh coordinates.
    # mesh = xr.open_dataset(cfg.data / 'ofam_mesh_grid_part.nc')

    # Plot
    # ds.Kd490.plot(col='time', col_wrap=4)
    # plt.tight_layout()
    # plt.savefig(cfg.fig / 'particle_source_map.png', bbox_inches='tight',
    #             dpi=300)
    files = cfg.obs / 'SEAWIFS_Kd490/SEASTAR_SEAWIFS_GAC.19980801_20100831.L3b.MC.KD.nc'
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
    chunksize = fieldset.get_fields()[0].grid.chunksize

    xfieldset = FieldSet.from_netcdf(filenames, variables, dimensions, chunksize=chunksize,
                                     time_periodic=timedelta(days=365))

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


def pset_from_plx_file(fieldset, pclass, filename, **kwargs):
    """Initialise particle data from back-tracked ParticleFile.

    This creates a new ParticleSet based on locations of all particles written
    in a netcdf ParticleFile at a certain time.
    Particle IDs are preserved if restart=True

    filename = cfg.data / 'plx/plx_hist_165_v1r00.nc'
    kwargs = dict(lonlatdepth_dtype=np.float32)
    """
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
        if v.name in pfile_vars:
            vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
        elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt',
                            'depth', 'id', 'fileid', 'state'] \
                and v.to_write:
            raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
        to_write[v.name] = v.to_write


    # if reduced:
    #     for v in vars:
    #         if v not in ['lon', 'lat', 'depth', 'time', 'id']:
    #             kwargs[v] = vars[v]
    # else:
    #     if restarttime is None:
    #         restarttime = np.nanmin(vars['time'])
    #     if callable(restarttime):
    #         restarttime = restarttime(vars['time'])
    #     else:
    #         restarttime = restarttime

    #     inds = np.where(vars['time'] <= restarttime)
    #     for v in vars:
    #         if to_write[v] is True:
    #             vars[v] = vars[v][inds]
    #         elif to_write[v] == 'once':
    #             vars[v] = vars[v][inds[0]]
    #         if v not in ['lon', 'lat', 'depth', 'time', 'id']:
    #             kwargs[v] = vars[v]

    # if restart:
    #     pclass.setLastID(0)  # reset to zero offset
    # else:
    #     vars['id'] = None

    # pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=vars['lon'],
    #                    lat=vars['lat'], depth=vars['depth'], time=vars['time'],
    #                    pid_orig=vars['id'], repeatdt=repeatdt,
    #                    lonlatdepth_dtype=lonlatdepth_dtype, **kwargs)

    # if reduced and 'nextid' in df.variables:
    #     pclass.setLastID(df.variables['nextid'].item())
    # else:
    #     pclass.setLastID(np.nanmax(df.variables['trajectory']) + 1)

    # if xlog:
    #     xlog['file'] = vars['lon'].size
    #     xlog['pset_start'] = restarttime
    #     if reduced:
    #         if 'restarttime' in df.variables:
    #             xlog['pset_start'] = df.variables['restarttime'].item()
    #         if 'runtime' in df.variables:
    #             xlog['runtime'] = df.variables['runtime'].item()
    #         if 'endtime' in df.variables:
    #             xlog['endtime'] = df.variables['endtime'].item()
    # return pset
