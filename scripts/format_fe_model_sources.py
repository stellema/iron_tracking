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

from cfg import paths, ExpData, test
from datasets import save_dataset
from tools import mlogger, timeit
from fe_exp import FelxDataSet

dask.config.set(**{'array.slicing.split_large_chunks': True})
np.seterr(divide='ignore', invalid='ignore')
logger = mlogger('source_files')



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


def group_euc_transport(pds, ds, pid_source_map):
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
    df.coords['zone'] = np.arange(len(pds.zones._all))

    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    df['u_src'] = (['rtime', 'zone'], np.empty((df.rtime.size, df.zone.size)))
    df['u_src'].attrs = df.u.attrs

    for z in df.zone.values:
        df_zone = df.sel(traj=df.traj[df.traj.isin(pid_source_map[z])])

        # Sum particle transport.
        df['u_src'][dict(zone=z)] = df_zone.u.sum('traj', keep_attrs=True)

    return df['u_src']


def group_fe_by_source(pds, ds, pid_source_map, var='fe'):
    """Calculate EUC iron weighted mean per release time, source and depth.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    var_avg = var + '_mean'
    var_avg_src = var + '_mean_src'
    var_avg_depth = var + '_mean_src_depth'

    # Group particle transport by time.
    df = ds.drop([v for v in ds.data_vars if v not in [var, 'u', 'time']])

    df = group_particles_by_variable(df, 'time')
    df.coords['zone'] = pds.zone_indexes
    df.coords['z'] = pds.release_depths

    pid_depth_map = pds.map_var_to_particle_ids(ds, var='z', var_array=df.z.values)

    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    df[var_avg] = (df[var] * df.u).sum('traj') / df.u.sum('traj')  # Particle weighted mean (rtime).
    df[var_avg_src] = (['rtime', 'zone'], np.full((df.rtime.size, df.zone.size), np.nan))
    df[var_avg_src].attrs = ds[var].attrs
    df[var_avg_depth] = (['rtime', 'zone', 'z'], np.full((df.rtime.size, df.zone.size, df.z.size), np.nan))
    df[var_avg_depth].attrs = ds[var].attrs

    for zone_i, zone in enumerate(df.zone.values):
        # Subset to particles at the same source.
        pids_src = df.traj[df.traj.isin(pid_source_map[zone])]
        df_src = df.sel(traj=pids_src)

        if df_src.traj.size > 0:
            # Weighted mean grouped by source.
            loc = dict(zone=zone_i)
            df[var_avg_src][loc] = (df_src[var] * df_src.u).sum('traj') / df_src.u.sum('traj')

            for depth_i, depth in enumerate(df.z.values):
                # Subset to particles at the same source and depth.
                pids_src_z = df_src.traj[df_src.traj.isin(pid_depth_map[depth])]
                df_src_z = df_src.sel(traj=pids_src_z)

                if df_src_z.traj.size > 0:
                    # Weighted mean grouped by source and EUC release depth.
                    loc = dict(zone=zone_i, z=depth_i)
                    df[var_avg_depth][loc] = (df_src_z[var] * df_src_z.u).sum('traj') / df_src_z.u.sum('traj')
    return df


@timeit(my_logger=logger)
def create_source_file(pds):
    """Create netcdf file with particle source information.

    Coordinates:
        traj: particle ID
        zone: source region id 0-9
        rtime: particle release time

    Data variables:
        trajectory   (zone, traj): Particle ID.
        time         (zone, traj): Release time.
        lat          (zone, traj): latitude at the EUC.
        lon          (zone, traj): Longitude at the EUC.
        z            (zone, traj): Depth at the EUC.
        time_at_src  (zone, traj): Time at the source.
        z_at_src     (zone, traj): Depth at the source.
        u            (zone, traj): Initial transport.
        age          (zone, traj): Transit time.

        temp         (zone, traj): Mean temperature.
        phy          (zone, traj): Mean phytoplankton concentration.
        fe           (zone, traj): Iron at the EUC.
        fe_src       (zone, traj): Iron at the source.
        fe_scav      (zone, traj): Total iron scavenged.
        fe_reg       (zone, traj): Total iron remineralized.
        fe_phy       (zone, traj): Total phytoplankton uptake of iron.

        fe_mean      (rtime): EUC iron (weighted mean).
        fe_mean_src  (rtime, zone): EUC iron (source weighted mean).
        fe_mean_src_depth (rtime, zone, z): EUC iron (source & depth weighted mean).

        fe_src_mean  (rtime): Source iron (weighted mean).
        fe_src_mean_src (rtime, zone): Source iron (source weighted mean).
        fe_src_mean_src_depth (rtime, zone, z): Source iron (source & depth weighted mean).

        u_total     (rtime): Total EUC transport at each release time.
        u_total_src (rtime, zone): EUC transport / release time & source.

    """
    file = exp.file_source
    logger.info('{}: Creating particle source file.'.format(file.stem))

    ds = xr.open_dataset(pds.exp.file_felx)
    if test:
        ds = ds.isel(traj=slice(400, 700))
    ds['zone'] = ds.zone.isel(obs=0)

    logger.info('{}: Particle information at source.'.format(file.stem))
    ds = pds.get_final_particle_obs(ds)

    logger.info('{}: Dictionary of particle IDs at source.'.format(file.stem))
    pid_source_map = pds.map_var_to_particle_ids(ds, var='zone', var_array=pds.zone_indexes,
                                                 file=None)

    logger.info('{}: Sum EUC transport per source.'.format(file.stem))

    # Group variables by source.
    # EUC transport: (traj) -> (rtime, zone). Run first bc can't stack 2D.
    u_zone = group_euc_transport(pds, ds, pid_source_map)

    logger.info('{}: Iron weighted mean per source.'.format(file.stem))
    fe_zone = group_fe_by_source(pds, ds, pid_source_map, var='fe')
    fe_src_zone = group_fe_by_source(pds, ds, pid_source_map, var='fe_src')
    temp_zone = group_fe_by_source(pds, ds, pid_source_map, var='temp')
    phy_zone = group_fe_by_source(pds, ds, pid_source_map, var='phy')

    logger.info('{}: Group by source.'.format(file.stem))

    # Group variables by source: (traj) -> (zone, traj).
    df = group_particles_by_variable(ds, var='zone')

    # Merge transport grouped by release time with source grouped variables.
    df.coords['zone'] = u_zone.zone.copy()  # Match 'zone' coord.
    df.coords['z'] = fe_zone.z.copy()  # Match 'zone' coord.

    df['u_total_src'] = u_zone
    df['u_total'] = df['u_total_src'].sum('zone', keep_attrs=True)  # Add total transport per release.

    for v in ['_mean', '_mean_src']:
        df['temp' + v] = temp_zone['temp' + v]

    for v in ['_mean', '_mean_src']:
        df['phy' + v] = phy_zone['phy' + v]

    for v in ['_mean', '_mean_src', '_mean_src_depth']:
        df['fe' + v] = fe_zone['fe' + v]

    for v in ['_mean', '_mean_src', '_mean_src_depth']:
        df['fe_src' + v] = fe_src_zone['fe_src' + v]

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
