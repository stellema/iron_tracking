# -*- coding: utf-8 -*-
"""Calculate & save BGC fields at particle positions.

Fields: Temperature, Det, Zooplankton, Phytoplankton, NO3, Kd490, Fe (optional)
Parallelise non-parallelisable code across different longitudes/scenarios:
    - Subset plx datasets.
    - Save inverse plx dataset.
    - Create unfinished felx_bgc dataset with required formatted variables.

Parallelise saving fields across particles & variables (for each release longitude & scenario).
These functions will save output as .npy files, which need to be joined and saved.

Notes:
    - Historical: 2000-2012 (12 years = ~4383 days)
    - RCP: 2087-2101 (12 years=2089-2101)
    - Initial year is a La Nina year (for spinup)
    - 85% of particles finish

Example:

Questions:

TODO:
    - Delete temporary patch code (split file & non-MPI 'var' version).
    - Rename empty felx_bgc dataset names.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jan 14 15:03:35 2023

"""
from argparse import ArgumentParser
from datetime import datetime, timedelta  # NOQA
import matplotlib.pyplot as plt  # NOQA
from memory_profiler import profile
import numpy as np
import pandas as pd  # NOQA
import xarray as xr  # NOQA


import cfg
from cfg import ExpData
from datasets import save_dataset, BGCFields, plx_particle_dataset, convert_plx_times
from felx_dataset import FelxDataSet
from tools import timeit, mlogger
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

logger = mlogger('files_felx')


@timeit(my_logger=logger)
def update_field_AAA(ds, field_dataset, var, dim_map):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """

    def update_field_particle(traj, ds, field, dim_map, var):
        """Subset using entire array for a particle."""
        mask = ~np.isnan(traj)
        ds = ds.sel(traj=traj[mask][0].item())

        # Map dimension names to particle position arrays (ds and field dims may differ).
        loc = {}
        for dim, dim_ds in dim_map.items():
            loc[dim] = ds[dim_ds]

        ds = ds[var]

        # Version 1 (quickest because of all the padded NaNs).
        n_obs = ds.obs[mask].astype(dtype=int).values
        for i in n_obs:
            xloc = dict([(k, v[i]) for k, v in loc.items()])
            ds[dict(obs=i)] = field.sel(xloc, method='nearest', drop=True)

        # Version 2.
        # ds = field.sel(loc, method='nearest', drop=True)
        return ds

    kwargs = dict(ds=ds, field=field_dataset[var], dim_map=dim_map, var=var)
    dx = np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)
    return dx


@timeit(my_logger=logger)
def save_felx_BGC_field_subset(exp, var, n):
    """Sample OFAM3 BGC fields at particle positions.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.
        var (str): BGC variable.
        n (int): File tmp subset number.

    Returns:
        None.

    Notes:
        - First save/create plx_inverse file.
    """
    # Initialise particle dataset.
    pds = FelxDataSet(exp)

    # Create temp directory and filenames.
    tmp_file = pds.bgc_var_tmp_filenames(var)[n]

    # if pds.check_file_complete(tmp_file):
    #     return
    # pds.check_bgc_prereq_files()

    # Create fieldset for variable.
    fieldset = BGCFields(exp)

    if var in fieldset.vars_ofam:
        field_dataset = fieldset.ofam_dataset(variables=var, decode_times=True, chunks='auto')
        dim_map = fieldset.dim_map_ofam
    else:
        field_dataset = fieldset.kd490_dataset(chunks='auto')
        dim_map = fieldset.dim_map_kd

    ds = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=True)

    if var == 'kd':
        ds['month'] = ds.month.astype(dtype=np.float32)

    # Save temp file subset for variable.
    traj_bnd = pds.particle_subsets(ds, pds.n_subsets)[n]

    # Calculate & save particle subset.
    dx = ds.isel(traj=slice(*traj_bnd))

    logger.info('{}/{}: Calculating. Subset {} of {} particles={}/{}'
                .format(tmp_file.parent.stem, tmp_file.name, n, pds.n_subsets,
                        np.diff(traj_bnd)[0], ds.traj.size))

    result = update_field_AAA(dx, field_dataset, var, dim_map)
    df = xr.Dataset(coords=dx.coords)
    df[var] = (['traj', 'obs'], result)
    save_dataset(df, tmp_file, msg='')
    logger.info('{}/{}: Saved.'.format(tmp_file.parent.stem, tmp_file.name))
    ds.close()
    df.close()
    return


def save_felx_BGC_fields(exp):
    """Save OFAM3 BGC fields at particle positions by merging saved tmp subsets.

    Example:
        module use /g/data3/hh5/public/modules
        module load conda/analysis3-22.04
        python3 ./particle_BGC_fields.py -e $EXP -x $LON -v 0 -r $R -func 'save_files'

    Notes:
        * Pre-req files: felx_bgc_*_tmp.nc and all tmp subsets files.
        * Job script: felx_ds.py
        * Requires: 42GB memory for >1 hour per file. (250h: 27-35GB, 165-190h: 42GB)
        * Adds metadata as found in original parcels output, OFAM3 files and Kd490 fields.
            * N.B. formatted plx files convert particle velocity to transport [Sv].
            * Time units are already encoded.
        * Updates wrong source ids saved in felx_bgc_*_tmp.nc.
        * Merges interior source IDs.
        * Reverts to original time variable after changes for spinup years.
        * Changes dtype of BGC fields to float32.
        * After saving files, can delete felx_bgc_*_tmp.nc and all tmp subsets files.
        * Linearlly interpolates NaN fields (i.e., at positions when beached).

    """
    pds = FelxDataSet(exp)
    assert pds.check_bgc_tmp_files_complete()

    file = exp.file_felx_bgc

    ds = xr.open_dataset(pds.exp.file_felx_bgc_tmp, decode_times=True, decode_cf=True)

    # Put all temp files together.
    logger.info('{}: Setting dataset variables.'.format(file.stem))
    for var in pds.bgc_variables:
        da = xr.open_mfdataset(pds.bgc_var_tmp_filenames(var), chunks='auto')

        # Interpolate missing any missing particle observations (e.g., when beached).
        ds[var] = da[var].interpolate_na('obs', method='linear', max_gap=10)
        da.close()

        # Re-apply NaN mask.
        ds[var] = ds[var].where(ds.valid_mask)

        # Convert dtype.
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype('float32')

    # Setting source region IDs (bug fix).
    logger.debug('{}: Setting source region IDs.'.format(file.stem))
    ds_plx = plx_particle_dataset(pds.exp.file_plx)  # Bug fix (saved u instead of zone).
    ds['zone'] = ds_plx.zone
    ds_plx.close()

    # Merge South and North interior Source IDs.
    for i in range(4):
        # South interior: 7, 8, 9, 10, 11 -> 7.
        ds['zone'] = xr.where(ds.zone == 8 + i, 7, ds.zone)
    for i in range(5):
        # North Interior: 12, 13, 14, 15, 16 -> 8
        ds['zone'] = xr.where(ds.zone == 12 + i, 8, ds.zone)

    # Revert time array (due to repeat spinup sampling).
    ds['time'] = ds.time_orig

    # ds = ds.rename({k: v for k, v in pds.bgc_variables_nmap.items()})
    ds = ds.drop(['time_orig', 'valid_mask', 'month', 'fe', 'age', 'distance', 'unbeached'])

    ds = pds.add_variable_attrs(ds)

    # Change time encoding.
    logger.info('{}: Converting time encoding...'.format(file.stem))
    dt = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=False, use_cftime=True,
                         decode_cf=True)
    dt = dt.time

    # Change time encoding (errors - can't overwrite file & doesen't work when not decoded).
    attrs = dict(units=dt.units, calendar=dt.calendar)
    attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='gregorian')  # OFAM3
    times = np.apply_along_axis(convert_plx_times, 1, arr=dt, obs=dt.obs.astype(dtype=int),
                                attrs=attrs, attrs_new=attrs_new)
    times = times.astype(dtype=np.float32)
    ds['time'] = (('traj', 'obs'), times)
    ds['time'].attrs['units'] = attrs_new['units']
    ds['time'].attrs['calendar'] = attrs_new['calendar']
    dt.close()

    # basic checks.
    if np.diff(ds.traj).max() > 1:
        logger.info('Error check: Missing some particles?')
    if np.diff(ds.obs).max() > 1:
        logger.info('Error check: Missing some obs?')

    # Save file.
    save_dataset(ds, file, msg='Added BGC fields at paticle positions.')
    logger.info('{}: Felx BGC fields file saved.'.format(file.stem))
    ds.close()
    return


def parallelise_prereq_files(scenario):
    """Calculate prerequisite files for saving BGC fields using MPI.

    Args:
        scenario (int): Scenario index {0-1}.

    Example:
        module use /g/data3/hh5/public/modules
        module load conda/analysis3-unstable
        mpiexec -n $PBS_NCPUS python3 ./particle_BGC_fields.py -e $EXP -func 'prereq_files'

    Notes:
        * Checks/saves plx subset, inverse_plx and felx_bgc*_tmp files.
        * Run using 4 CPUs on gadi for each scenario.
        * Job script: felx_prereq.sh
        * Particle dataset requires ~172GB for historical & ~190GB for RCP for 4 hours.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    lon = [165, 190, 220, 250][rank]

    for r in range(8):
        name = 'felx_bgc'
        exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=r, name=name)
        pds = FelxDataSet(exp)
        pds.check_bgc_prereq_files()
    return


def parallelise_BGC_fields(exp, variable_i=0, check=False):
    """Save particle BGC fields tmp subsets using MPI.

    Args:
        exp (ExpData): Experiment information class.
        variable_i (int, optional): BGC variable integer {0-5}. Defaults to 0.
        check (bool, optional): Only log remaining files to save. Defaults to False.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pds = FelxDataSet(exp)

    # Input ([[var0, 0], [var0, 1], ..., [varn, n]).
    var_n_all = [[v, i] for v in pds.bgc_variables[variable_i:] for i in range(pds.n_subsets)]

    # Remove any finished saved subsets from list.
    for v, n in var_n_all.copy():
        files = pds.bgc_var_tmp_filenames(v)

        if files[n].exists():
            var_n_all.remove([v, n])

    var, n = var_n_all[rank]

    if rank == 0:
        logger.info('{}: Files to save={}/{}'.format(exp.file_felx_bgc.stem, len(var_n_all),
                                                     len(pds.bgc_variables) * pds.n_subsets))
        v_list = []  # Log remaining files.
        for v in np.unique(np.array(var_n_all)[:, 0]):
            inds = np.where(np.array(var_n_all)[:, 0] == v)
            n_list = list(np.array(var_n_all)[inds][:, 1])
            v_list.append([v, n_list])
        v_list = ['{} (unsaved={}): {}'.format(v[0], len(v[1]), v[1]) for v in v_list]
        logger.info('{}'.format(v_list))

    logger.info('{}: Rank={}, var={}, n={}/{}'.format(exp.file_felx_bgc.stem, rank, var, n,
                                                      pds.n_subsets - 1))
    if not check:
        save_felx_BGC_field_subset(exp, var, n)
    return


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=250, type=int,
                   help='Release longitude [165, 190, 220, 250].')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index [0, 1].')
    p.add_argument('-v', '--version', default=0, type=int, help='Felx experiment version.')
    p.add_argument('-r', '--index', default=0, type=int, help='File repeat index [0-7].')
    p.add_argument('-func', '--function', default='bgc_fields', type=str,
                   help='[bgc_fields, prereq_files, save_files]')
    # 'bgc_fields' args.
    p.add_argument('-iv', '--variable_i', default=0, type=int, help='BGC variable index [0-7].')
    p.add_argument('-c', '--check', default=False, type=bool, help='Check remaining files')
    args = p.parse_args()

    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name='felx_bgc',
                  test=cfg.test)
    func = args.function

    if cfg.test:
        func = ['bgc_fields', 'prereq_files', 'save_files'][0]

    if MPI is not None:
        if func == 'prereq_files':
            parallelise_prereq_files(scenario)

        if func == 'bgc_fields':
            parallelise_BGC_fields(exp, variable_i=args.variable_i, check=args.check)

    if func == 'save_files':
        if not exp.file_felx_bgc.exists():
            save_felx_BGC_fields(exp)
