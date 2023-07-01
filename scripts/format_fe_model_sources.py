# -*- coding: utf-8 -*-
"""Fe model output: calculate source statistics and save as netcdf file.

Notes:
    - Calculate and save datasets for each file repeat and then merge.
    - Saved in felx/data/fe_model/v*/fe_source*.nc
    - Ignore "RuntimeWarning: invalid value encountered in divide"

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jun 14 14:09:24 2023

"""
from argparse import ArgumentParser
import dask
import numpy as np
import warnings
import xarray as xr

from cfg import ExpData, test, DXDY
from datasets import save_dataset, get_ofam3_coords
from tools import mlogger, timeit
from fe_exp import FelxDataSet

dask.config.set(**{'array.slicing.split_large_chunks': True})
warnings.filterwarnings('ignore', category=RuntimeWarning)
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


def group_var_sum_by_source(pds, ds, pid_source_map, var='u'):
    """Calculate EUC transport per release time & source.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    # Group particle transport by time.
    df = ds.drop([v for v in ds.data_vars if v not in ['z', var, 'time']])
    df = group_particles_by_variable(df, 'time')
    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    mesh = get_ofam3_coords()
    df.coords['zone'] = pds.zone_indexes
    df.coords['z'] = pds.release_depths
    df.coords['z_at_src'] = mesh.st_ocean.values

    # Initialise source-grouped transport variable.
    for v, dims in zip([var + '_sum_src', var + '_sum_depth', var + '_sum_src_depth'],
                       [['rtime', 'zone'], ['rtime', 'zone', 'z'], ['rtime', 'zone', 'z_at_src']]):
        df[v] = (dims, np.full(tuple([df[v].size for v in dims]), np.nan))
        df[v].attrs = ds[var].attrs

    df[var + '_sum'] = df[var].sum('traj', keep_attrs=True)  # Add total transport per release.

    for z in df.zone.values:
        dx = df.sel(traj=df.traj[df.traj.isin(pid_source_map[z])])

        # Sum particle transport.
        df[var + '_sum_src'][dict(zone=z)] = dx[var].sum('traj', keep_attrs=True)

        for v, depth_var in zip([var + '_sum_depth', var + '_sum_src_depth'], ['z', 'z_at_src']):
            pid_depth_map = pds.map_var_to_particle_ids(ds, var=depth_var, var_array=df[depth_var].values)

            for depth_i, depth in enumerate(df[depth_var].values):
                # Subset to particles at the same source and depth.
                dxx = dx.sel(traj=dx.traj[dx.traj.isin(pid_depth_map[depth])])

                if dxx.traj.size > 0:
                    # Weighted mean grouped by source and EUC release depth.
                    loc = {'zone': z, depth_var: depth_i}
                    df[v][loc] = dxx[var].sum('traj', keep_attrs=True)

    return df


def group_var_avg_by_source(pds, ds, pid_source_map, var='fe', depth_var='z'):
    """Calculate EUC iron weighted mean per release time, source and depth.

    Args:
        ds (xarray.Dataset): Particle data.
        source_traj (dict): Dictionary of particle IDs from each source.

    Returns:
        ds[u_zone] (xarray.DataArray): Transport from source (rtime, zone).

    """
    var_avg = var + '_avg'
    var_avg_src = var_avg + '_src'
    var_avg_depth = var_avg + '_depth'
    var_avg_src_depth = var_avg + '_src_depth'
    var_list = [var_avg, var_avg_src, var_avg_depth, var_avg_src_depth]

    # Group particle transport by time.
    df = ds.drop([v for v in ds.data_vars if v not in [var, 'z', 'u', 'time']])

    df = group_particles_by_variable(df, 'time')

    mesh = get_ofam3_coords()
    df.coords['zone'] = pds.zone_indexes
    df.coords['z'] = pds.release_depths
    df.coords['z_at_src'] = mesh.st_ocean.values

    pid_depth_map = pds.map_var_to_particle_ids(ds, var=depth_var, var_array=df[depth_var].values)

    # Rename stacked time coordinate (avoid duplicate 'time' variable).
    df = df.rename({'time': 'rtime'})

    # Initialise source-grouped transport variable.
    for v, dims in zip(var_list, [['rtime'], ['rtime', 'zone'], ['rtime', depth_var],
                                  ['rtime', 'zone', depth_var]]):
        df[v] = (dims, np.full(tuple([df[v].size for v in dims]), np.nan))
        df[v].attrs = ds[var].attrs

    # Calculate variables
    # Particle weighted mean (rtime).
    df[var_avg] = (df[var] * df.u).sum('traj') / df.u.sum('traj')

    for zone_i, zone in enumerate(df.zone.values):
        # Subset to particles at the same source.
        dx = df.sel(traj=df.traj[df.traj.isin(pid_source_map[zone])])

        if dx.traj.size > 0:
            # Weighted mean grouped by source.
            loc = dict(zone=zone_i)
            df[var_avg_src][loc] = (dx[var] * dx.u).sum('traj') / dx.u.sum('traj')

            for depth_i, depth in enumerate(df[depth_var].values):
                # Subset to particles at the same source and depth.
                dxx = dx.sel(traj=dx.traj[dx.traj.isin(pid_depth_map[depth])])

                if dxx.traj.size > 0:
                    # Weighted mean grouped by source and EUC release depth.
                    loc = {'zone': zone_i, depth_var: depth_i}
                    df[var_avg_src_depth][loc] = (dxx[var] * dxx.u).sum('traj') / dxx.u.sum('traj')

    for depth_i, depth in enumerate(df[depth_var].values):
        # Subset to particles at the same depth.
        dx = df.sel(traj=df.traj[df.traj.isin(pid_depth_map[depth])])

        if dx.traj.size > 0:
            # Weighted mean grouped by source and EUC release depth.
            loc = {depth_var: depth_i}
            df[var_avg_depth][loc] = (dx[var] * dx.u).sum('traj') / dx.u.sum('traj')
    return df


def transfer_dataset_release_times(pds, ds):
    """Ensure fe model dataset contains all particles released on the same day."""
    # Release times of particles in this dataset.
    t = ds['time'].ffill('obs').isel(obs=-1, drop=True)  # Last value (at the EUC).
    attrs = ds.attrs

    # Open previous file and check if any rtime values overlap (add to this dataset).
    if pds.exp.file_index > 0:

        ds_prev = xr.open_dataset(pds.exp.file_felx_all[pds.exp.file_index - 1])

        # Release times of particles in dataset.
        t_prev = ds_prev['time'].ffill('obs').isel(obs=-1, drop=True)  # Last value (at the EUC).

        # Check if any release times overlap between files.
        check = t.drop_duplicates('traj').isin(t_prev.drop_duplicates('traj'))

        if any(check):
            rtimes = np.unique(t[check])  # Release times that overlap.

            # Particle IDs in previous file to be transferred.
            pids = []
            for i in rtimes:
                pids.append(t_prev.where(t_prev == i, drop=True).traj)
            pids = np.concatenate(pids)

            # Subset particles to be transferred.
            ds_prev = ds_prev.sel(traj=pids)
            ds_prev = ds_prev.dropna('obs', 'all')

            # Ensure 'obs' dimension is the same size for each dataset (needed for merging)
            sizes = [ds.obs.size, ds_prev.obs.size]
            size_diff = int(np.fabs(sizes[0] - sizes[1]))

            # Pad ds
            if sizes[1] > sizes[0]:
                pad_tmp = xr.full_like(ds.isel(obs=slice(-size_diff - 1, -1)), np.nan)
                pad_tmp['obs'] = pad_tmp['obs'] + size_diff + 1
                ds = xr.concat([ds, pad_tmp], 'obs')

            # Pad ds_prev
            elif sizes[0] > sizes[1]:
                pad_tmp = xr.full_like(ds_prev.isel(obs=slice(-size_diff - 1, -1)), np.nan)
                pad_tmp['obs'] = pad_tmp['obs'] + size_diff + 1
                ds_prev = xr.concat([ds_prev, pad_tmp], 'obs')

            # Add particles from previous dataset to the dataset.
            ds = xr.concat([ds_prev, ds], 'traj')
            ds.attrs = attrs
            logger.info('{}: Add particles from previous file (p={}: {}).'
                        .format(pds.exp.file_source_tmp.stem, ds_prev.traj.size, rtimes))
        ds_prev.close()

    # Open next file and check if any rtime values overlap (remove from this dataset).
    if pds.exp.file_index < 7:
        ds_next = xr.open_dataset(pds.exp.file_felx_all[pds.exp.file_index + 1])

        # Release times of particles in dataset.
        t_next = ds_next['time'].ffill('obs').isel(obs=-1, drop=True)  # Last value (at the EUC).

        # Check if any release times overlap between files.
        check = t.drop_duplicates('traj').isin(t_next.drop_duplicates('traj'))

        if any(check):
            rtimes = np.unique(t[check])  # Release times that overlap.

            # Particle IDs in next file to be dropped from this dataset.
            pids = []
            for i in rtimes:
                pids.append(t.where(t == i, drop=True).traj)
            pids = np.concatenate(pids)

            # Drop particles.
            ds = ds.sel(traj=ds.traj[~ds.traj.isin(pids)])
            logger.info('{}: Dropped particles in next file (p={}: {}).'
                        .format(pds.exp.file_source_tmp.stem, pids.size, rtimes))
        ds_next.close()
    return ds


@timeit(my_logger=logger)
def create_source_file(pds):
    """Create netcdf file with particle source information.

    Coordinates:
        traj: particle ID
        zone: source region id 0-9
        rtime: particle release time
        z: release depths
        z_at_src: OFAM3 depths (binned)

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

        fe_avg       (rtime): EUC iron (weighted mean).
        fe_avg_src   (rtime, zone): EUC iron (source weighted mean).
        fe_avg_depth (rtime, z): EUC iron (depth weighted mean).
        fe_avg_src_depth (rtime, zone, z): EUC iron (source & depth weighted mean).

        fe_src_avg   (rtime): Source iron (weighted mean).
        fe_src_avg_src (rtime, zone): Source iron (source weighted mean).
        fe_src_avg_depth (rtime, z_at_src): Source iron (depth weighted mean).
        fe_src_avg_src_depth (rtime, zone, z_at_src): Source iron (source & depth weighted mean).

        u_sum     (rtime): Total EUC transport at each release time.
        u_sum_src (rtime, zone): EUC transport / release time & source.

    """
    file = exp.file_source_tmp
    logger.info('{}: Creating particle source file.'.format(file.stem))

    ds = xr.open_dataset(pds.exp.file_felx)

    # # !!! Save original pds.exp.file_source_map for fe_model
    # if 'obs' in ds.zone.dims:  # !!!
    #     ds['zone'] = ds.zone.isel(obs=0)  # !!!
    # _ = pds.map_var_to_particle_ids(ds, var='zone', var_array=pds.zone_indexes, file=pds.exp.file_source_map)  # !!!

    ds = transfer_dataset_release_times(pds, ds)

    if test:
        ds = ds.isel(traj=slice(0, 2000))

    if 'obs' in ds.zone.dims:
        ds['zone'] = ds.zone.isel(obs=0)
    ds = pds.add_variable_attrs(ds)

    logger.info('{}: Particle information at source.'.format(file.stem))
    ds = pds.get_final_particle_obs(ds)
    ds = ds.drop('obs')

    logger.info('{}: Dictionary of particle IDs at source.'.format(file.stem))
    pid_source_map = pds.map_var_to_particle_ids(ds, var='zone', var_array=pds.zone_indexes,
                                                 file=pds.exp.file_source_map_alt)

    ds['fe_flux'] = ds.fe * (ds.u / DXDY)
    ds = pds.add_variable_attrs(ds)

    logger.info('{}: Variable sum per source.'.format(file.stem))
    # Group variables by source.
    # EUC transport: (traj) -> (rtime, zone). Run first bc can't stack 2D.
    sum_ds_list = []
    sum_ds_list.append(group_var_sum_by_source(pds, ds, pid_source_map, var='u'))
    sum_ds_list.append(group_var_sum_by_source(pds, ds, pid_source_map, var='fe_flux'))

    logger.info('{}: Variable weighted mean per source.'.format(file.stem))
    varz = ['fe', 'fe_src', 'fe_phy', 'fe_reg', 'fe_scav', 'temp', 'phy'][::-1]

    avg_ds_list = []
    for var in varz:
        depth_var = 'z_at_src' if var == 'fe_src' else 'z'
        avg_ds_list.append(group_var_avg_by_source(pds, ds, pid_source_map, var=var, depth_var=depth_var))

    logger.info('{}: Group by source.'.format(file.stem))
    # Group variables by source: (traj) -> (zone, traj).
    df = group_particles_by_variable(ds, var='zone')

    # Merge transport grouped by release time with source grouped variables.
    # Add coordinates.
    df.coords['zone'] = avg_ds_list[0].zone.copy()
    df.coords['z'] = avg_ds_list[0].z.copy()
    df.coords['z_at_src'] = avg_ds_list[0].z_at_src.copy()

    for var, dz in zip(['u', 'fe_flux'], sum_ds_list):
        for v in ['_sum', '_sum_src', '_sum_depth', '_sum_src_depth']:
            df[var + v] = dz[var + v]

    for var, dz in zip(varz, avg_ds_list):
        for v in ['_avg', '_avg_src', '_avg_depth', '_avg_src_depth']:
            df[var + v] = dz[var + v]

    # Convert age: seconds to days.
    df['age'] *= 1 / (60 * 60 * 24)
    df.age.attrs['units'] = 'days'

    # Save dataset.
    logger.info('{}: Saving...'.format(file.stem))

    if 'obs' in df:
        df = df.drop('obs')

    df = df.chunk()
    df = df.unify_chunks()
    save_dataset(df, file, msg='Calculated grouped source statistics.')
    logger.info('{}: Saved.'.format(file.stem))
    df.close()
    ds.close()


if __name__ == "__main__":
    p = ArgumentParser(description="""Format source file.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Release longitude [165, 190, 220, 250].')
    p.add_argument('-s', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-v', '--version', default=0, type=int, help='Version index.')
    p.add_argument('-r', '--index', default=5, type=int, help='File repeat index [0-7].')
    args = p.parse_args()

    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index

    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index)
    pds = FelxDataSet(exp)

    # Create source file for file index.
    if not pds.exp.file_source_tmp.exists():
        create_source_file(pds)

    # Create combined source file.
    if not pds.exp.file_source.exists():
        files_all = pds.exp.file_source_tmp_all

        if all([f.exists() for f in files_all]):
            logger.info('{}: Saving combined source file.'.format(pds.exp.file_source.stem))
            ds = xr.open_mfdataset(files_all, data_vars='minimal')
            ds = ds.chunk()
            ds = ds.unify_chunks()
            save_dataset(ds, pds.exp.file_source, msg='Combined source files.')
            logger.info('{}: Saved combined source file.'.format(pds.exp.file_source.stem))
            ds.close()
