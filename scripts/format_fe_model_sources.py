# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun 14 14:09:24 2023

"""
import dask
import numpy as np
import xarray as xr
from argparse import ArgumentParser

from cfg import paths, ExpData, zones, test
from datasets import save_dataset
from tools import unique_name, mlogger, timeit
from fe_exp import FelxDataSet

dask.config.set(**{'array.slicing.split_large_chunks': True})

logger = mlogger('source_files')


@timeit
def group_particles_by_variable(ds, var='time'):
    """Group particles by time.

    Data variables must only have one dimension: (traj) -> (time, traj).
    Select first time observation in input dataset and this will group the
    particles by release time.

    Args:
        pds (FelxDataSet): FelxDataSet instance.
        ds (xarrary.Dataset): 1D Particle data

    Returns:
        ds (xarrary.Dataset): Dataset with additional time dimension.

    """
    # Stack & unstack dims: (traj) -> (time, zone, traj).
    ds = ds.set_index(tzt=[var, 'traj'])
    ds = ds.chunk('auto')
    ds = ds.unstack('tzt')

    # Drop traj duplicates due to unstack.
    ds = ds.dropna('traj', 'all')
    return ds


@timeit
def group_euc_transport(ds, source_traj):
    """Calculate EUC transport per release time & source.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    # Group particle transport by time.
    df = ds.drop([v for v in ds.data_vars if v not in ['u', 'time']])
    df = group_particles_by_variable(df, 'time')
    df.coords['zone'] = np.arange(len(zones._all))

    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    df['u_zone'] = (['rtime', 'zone'], np.empty((df.rtime.size, df.zone.size)))

    for z in df.zone.values:
        df_zone = df.sel(traj=df.traj[df.traj.isin(source_traj[z])])

        # Sum particle transport.
        df['u_zone'][dict(zone=z)] = df_zone.u.sum('traj', keep_attrs=True)

    return df['u_zone']

@timeit
def group_fe_by_source(pds, ds, source_traj):
    """Calculate EUC iron weighted mean per release time, source and depth.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    var = 'fe'
    var_avg = 'fe_mean'
    var_new = 'fe_mean_zone'
    var_new_z = 'fe_mean_zone_depth'

    # Group particle transport by time.
    df = ds.drop([v for v in ds.data_vars if v not in [var, 'u', 'time']])

    df = group_particles_by_variable(df, 'time')
    df.coords['zone'] = np.arange(len(zones._all))
    df.coords['z'] = np.arange(25, 350 + 25, 25, dtype=int)  # !!! Check
    pid_map = pds.map_var_to_particle_ids(ds, var='z', var_array=df.z.values)

    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    df[var_new] = (['rtime', 'zone'], np.empty((df.rtime.size, df.zone.size)))
    df[var_new_z] = (['rtime', 'zone', 'z'], np.empty((df.rtime.size, df.zone.size, df.z.size)))
    df[var_avg] = (df[var] * df.u) / df.u.sum('traj')

    for zone_i, zone in enumerate(df.zone.values):
        pids = df.traj[df.traj.isin(source_traj[zone])]
        df_zone = df.sel(traj=pids)

        if df_zone.traj.size > 0:
            # Sum particle transport.
            df[var_new][dict(zone=zone_i)] = (df_zone[var] * df_zone.u).sum('traj', keep_attrs=True) / df_zone.u.sum('traj')
        else:
            df[var_new][dict(zone=zone_i)] = np.nan

        for depth_i, depth in enumerate(df.z.values):
            loc = dict(zone=zone_i, z=depth_i)
            pids = df_zone.traj[df_zone.traj.isin(pid_map[depth])]
            df_zone_z = df_zone.sel(traj=pids)
            if df_zone_z.traj.size > 0:
                # Sum particle transport.
                df[var_new_z][loc] = (df_zone_z[var] * df_zone_z.u).sum('traj', keep_attrs=True) / df_zone_z.u.sum('traj')
            else:
                df[var_new_z][loc] = np.nan
    return df


@timeit
def create_source_file(pds):
    """Create netcdf file with particle source information.

    Coordinates:
        traj: particle IDs
        source: source regions 0-10
        rtime: particle release times

    Data variables:
        trajectory   (zone, traj): Particle ID.
        time         (zone, traj): Release time.
        age          (zone, traj): Transit time.
        zone         (zone, traj): Source ID.
        distance     (zone, traj): Transit distance.
        u            (zone, traj): Initial transport.
        u_zone       (rtime, zone): EUC transport / release time & source.
        u_sum        (rtime): Total EUC transport at each release time.
        time_at_zone (zone, traj): Time at source.
        z_at_zone    (zone, traj): Depth at source.

        lat_max      (zone, traj): Max latitude.
        lat_min      (zone, traj): Min latitude.
        lon_max      (zone, traj): Max longitude.
        lon_min      (zone, traj): Min longitude
        z_max        (zone, traj): Max depth.
        z_min        (zone, traj): Min depth.

    """
    file = exp.file_source
    logger.info('{}: Creating particle source file.'.format(file.stem))

    ds = xr.open_dataset(pds.exp.file_felx)
    # if test:
    #     ds = ds.isel(traj=slice(400, 700))
    ds['zone'] = ds.zone.isel(obs=0)

    logger.info('{}: Particle information at source.'.format(file.stem))
    ds = pds.get_final_particle_obs(ds)

    logger.info('{}: Dictionary of particle IDs at source.'.format(file.stem))
    source_traj = pds.map_var_to_particle_ids(ds, var='zone', var_array=range(len(zones._all)),
                                              file=pds.exp.file_source_map)

    logger.info('{}: Sum EUC transport per source.'.format(file.stem))

    # Group variables by source.
    # EUC transport: (traj) -> (rtime, zone). Run first bc can't stack 2D.
    u_zone = group_euc_transport(ds, source_traj)
    logger.info('{}: Iron weighted mean per source.'.format(file.stem))
    fe_zone = group_fe_by_source(pds, ds, source_traj)

    logger.info('{}: Group by source.'.format(file.stem))

    # Group variables by source: (traj) -> (zone, traj).
    df = group_particles_by_variable(ds, var='zone')

    # Merge transport grouped by release time with source grouped variables.
    df.coords['zone'] = u_zone.zone.copy()  # Match 'zone' coord.
    df.coords['z'] = fe_zone.z.copy()  # Match 'zone' coord.

    for v in ['fe_mean', 'fe_mean_zone', 'fe_mean_zone_depth']:
        df[v] = fe_zone[v]
    df['u_zone'] = u_zone
    df['u_sum'] = df.u_zone.sum('zone', keep_attrs=True)  # Add total transport per release.

    # Convert age: seconds to days.
    df['age'] *= 1 / (60 * 60 * 24)
    df.age.attrs['units'] = 'days'

    # Save dataset.
    logger.info('{}: Saving...'.format(file.stem))
    df = df.chunk()
    df = df.unify_chunks()
    save_dataset(df, file, msg='Calculated grouped source statistics.')
    logger.info('{}: Saved.'.format(file.stem))


if __name__ == "__main__":
    p = ArgumentParser(description="""Format source file.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Release longitude [165, 190, 220, 250].')
    p.add_argument('-s', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-v', '--version', default=0, type=int, help='Version index.')
    p.add_argument('-r', '--index', default=7, type=int, help='File repeat index [0-7].')
    p.add_argument('-f', '--func', default='run', type=str, help='run, fix or save.')
    args = p.parse_args()

    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index

    exp = ExpData(name='fe', scenario=scenario, lon=lon, version=version, file_index=index,
                  out_subdir='v{}'.format(version))

    pds = FelxDataSet(exp)
    create_source_file(pds)

    # files_all = pds.exp.file_felx_all
    # ds = xr.open_mfdataset(files_all, combine='nested', chunks='auto', coords='minimal')
