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
from cfg import paths, ExpData
from datasets import save_dataset, BGCFields
from felx_dataset import FelxDataSet
from tools import timeit, mlogger
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

logger = mlogger('files_felx')


@timeit(my_logger=logger)
@profile
def update_field_AAA(ds, field_dataset, var, dim_map):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """

    def update_field_particle(traj, ds, field, dim_map, var):
        """Subset using entire array for a particle."""
        mask = ~np.isnan(traj)
        p = traj[mask][0].item()
        dx = ds.sel(traj=p)

        # Map dimension names to particle position arrays (ds and field dims may differ).
        loc = {}
        for dim, dim_dx in dim_map.items():
            loc[dim] = dx[dim_dx]

        # Version 1 (quickest because of all the padded NaNs).
        n_obs = dx.obs[mask].astype(dtype=int).values
        for i in n_obs:
            xloc = dict([(k, v[i]) for k, v in loc.items()])
            dx[var][dict(obs=i)] = field.sel(xloc, method='nearest', drop=True)

        # Version 2.
        # dx[var] = field.sel(loc, method='nearest', drop=True)
        return dx[var]

    kwargs = dict(ds=ds, field=field_dataset[var], dim_map=dim_map, var=var)
    dx = np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)
    return dx


@timeit(my_logger=logger)
def save_felx_BGC_field_subset(exp, var, n, split_file=False, split_int=None):
    """Sample OFAM3 BGC fields at particle positions.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.
        var (str): DESCRIPTION.
        n (int): DESCRIPTION.
        split_file (bool, optional): Divide subset again. Defaults to False.
        split_int (None or int, optional): Integer of split (0, 1). Defaults to None.

    Returns:
        None.

    Notes:
        - First save/create plx_inverse file.
    """
    # Create fieldset for variable.
    fieldset = BGCFields(exp)

    if var in fieldset.vars_ofam:
        field_dataset = fieldset.ofam_dataset(variables=var, decode_times=True)
        dim_map = fieldset.dim_map_ofam
    else:
        field_dataset = fieldset.kd490_dataset()
        dim_map = fieldset.dim_map_kd

    # Initialise particle dataset.
    pds = FelxDataSet(exp)

    # Create temp directory and filenames.
    tmp_files = pds.bgc_var_tmp_filenames(var)
    tmp_file = tmp_files[n]

    if split_file:
        tmp_files_split = pds.bgc_var_tmp_filenames_split(var, n)
        tmp_file = tmp_files_split[split_int]

    if pds.check_file_complete(tmp_file):
        return

    pds.check_bgc_prereq_files()

    ds = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=True)

    if var == 'kd':
        ds['month'] = ds.month.astype(dtype=np.float32)

    # Save temp file subset for variable.
    traj_bnds = pds.bgc_tmp_traj_subsets(ds)
    if exp.lon == 250 and exp.scenario == 0:
        if exp.file_index == 0 and var == 'phy':
            traj_bnds = pds.bgc_tmp_traj_subsets_even(ds)
        if ((exp.file_index == 0 and var == 'zoo') or (exp.file_index == 1 and var == 'phy')):
            traj_bnds = pds.bgc_tmp_traj_subsets_even_half(ds)

    traj_bnd = traj_bnds[n]

    # Calculate & save particle subset.
    dx = ds.isel(traj=slice(*traj_bnd))

    if split_file:
        # Divide tmp subset particles (add +10 to help make one file finish quicker).
        n_obs = dx.z.where(np.isnan(dx.z), 1).sum(dim='obs')
        n_obs_grp = (n_obs.cumsum() / (n_obs.sum() / 2)).astype(dtype=int)
        mid = traj_bnd[0] + n_obs_grp.where(n_obs_grp == 0, drop=True).traj.size
        mid += 10
        traj_bnds_split = [[traj_bnd[0], mid], [mid, traj_bnd[1]]]
        traj_bnd = traj_bnds_split[split_int]
        dx = ds.isel(traj=slice(*traj_bnd))

    logger.info('{}/{}: Calculating. Subset {} of {} particles={}/{}'
                .format(tmp_file.parent.stem, tmp_file.name, n, pds.num_subsets,
                        np.diff(traj_bnd)[0], ds.traj.size))
    dx = update_field_AAA(dx, field_dataset, var, dim_map)

    np.save(tmp_file, dx[var], allow_pickle=True)
    logger.info('{}/{}: Saved.'.format(tmp_file.parent.stem, tmp_file.name))

    if split_file:
        # Concatenate & save split tmp file.
        if all([f.exists() for f in tmp_files_split]):
            data = []
            for tmp_file in tmp_files_split:
                data.append(np.load(tmp_file, allow_pickle=True))
            arr = np.concatenate(data, axis=0)

            tmp_file = pds.bgc_var_tmp_filenames(var)[n]
            np.save(tmp_file, arr, allow_pickle=True)
            logger.info('{}/{}: Saved.'.format(tmp_file.parent.stem, tmp_file.name))
    return


@profile
def save_felx_BGC_fields(exp):
    """Run and save OFAM3 BGC fields at particle positions as n temp files."""
    pds = FelxDataSet(exp)

    file = exp.file_felx_bgc
    if cfg.test:
        file = paths.data / ('test/' + file.stem + '.nc')

    ds = pds.init_bgc_felx_file(save_empty=False)
    # Put all temp files together.
    logger.info('{}: Setting dataset variables.'.format(file.stem))
    ds = pds.set_bgc_dataset_vars_from_tmps(ds)

    ds = ds.where(ds.valid_mask)
    ds['time'] = ds.time_orig
    ds = ds.drop('valid_mask')
    ds = ds.drop('time_orig')

    # Save file.
    logger.info('{}: Saving...'.format(file.stem))
    save_dataset(ds, file, msg='Added BGC fields at paticle positions.')
    logger.info('{}: Felx BGC fields file saved.'.format(file.stem))
    return


@profile
def parallelise_prereq_files(scenario):
    """Calculate plx subset and inverse_plx files.

    Notes:
        * Assumes 8 processors
        * 10 file x 16 min each = ~2.6 hours.
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


@profile
def parallelise_BGC_fields(exp, split_file=False):
    """Step 2. Parallelise saving particle BGC fields as temp files.

    Notes:
        * Assumes 35 processors
        * Arounf 7-8 hours each file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pds = FelxDataSet(exp)

    # !!! tmp bug fix.
    if split_file and exp.lon == 250 and exp.scenario == 0 and exp.file_index in [0, 1]:
        if exp.file_index == 0:
            phy_subs = [['phy', n, i] for n in [17, 57, 60, 65, 66, 69, 70, 71, 80, 94,
                                                95, 96, 97, 98, 99] for i in [0, 1]]
            zoo_subs = [['zoo', n, i] for n in [0] for i in [0, 1]]  # TODO
            var_n_all = [*phy_subs, *zoo_subs]

        if exp.file_index == 1:
            var_n_all = [['phy', n, i] for n in [19, 23, 24, 27, 28, 30, 31, 34, 35, 37, 38,
                                                 45, 46] for i in [0, 1]]

        # Remove any finished saved subsets from list.
        for v, n, i in var_n_all.copy():
            files = pds.bgc_var_tmp_filenames(v)
            if pds.check_file_complete(files[n]):
                var_n_all.remove([v, n, i])
            else:
                file = pds.bgc_var_tmp_filenames_split(v, n)[i]
                if pds.check_file_complete(file):
                    var_n_all.remove([v, n, i])

        var, n, i = var_n_all[rank]
        kwargs = dict(split_file=True, split_int=i)

    else:
        # Input ([[var0, 0], [var0, 1], ..., [varn, n]).
        var_n_all = [[v, i] for v in pds.bgc_variables for i in range(pds.num_subsets)]

        # Remove any finished saved subsets from list.
        for vn in var_n_all.copy():
            files = pds.bgc_var_tmp_filenames(vn[0])
            if pds.check_file_complete(files[vn[1]]):
                var_n_all.remove(vn)


        var, n = var_n_all[rank]
        kwargs = dict(split_file=False, split_int=None)  # !!! tmp bug fix.

    if rank == 0:
        logger.info('{}: Files to save={}/{}'.format(exp.file_felx_bgc.stem, len(var_n_all),
                                                     len(pds.bgc_variables) * pds.num_subsets))
        logger.info('{}'.format(var_n_all))

    logger.info('{}: Rank={}, var={}, n={}/{}'.format(exp.file_felx_bgc.stem, rank, var, n,
                                                      pds.num_subsets - 1))
    save_felx_BGC_field_subset(exp, var, n, **kwargs)  # !!! tmp bug fix - delete kwargs
    return


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=250, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-v', '--version', default=0, type=int, help='FeLX experiment version.')
    p.add_argument('-r', '--index', default=0, type=int, help='File repeat.')
    p.add_argument('-func', '--function', default='bgc_fields', type=str,
                   help='[bgc_fields, bgc_fields_var, prereq_files, save_files]')
    # Args for tmp 'bgc_fields_var'.
    p.add_argument('-var', '--variable', default='phy', type=str, help='BGC variable.')
    p.add_argument('-n', '--n_tmp', default=0, type=int, help='Subset index.')
    p.add_argument('-split', '--split_file', default=False, type=bool, help='Split tmp subset.')
    args = p.parse_args()

    scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name='felx_bgc',
                  test=cfg.test)

    if cfg.test:
        func = ['bgc_fields', 'prereq_files', 'save_files'][0]
    else:
        func = args.function

    if MPI is not None:
        if func == 'prereq_files':
            parallelise_prereq_files(scenario)

        elif func == 'bgc_fields':
            parallelise_BGC_fields(exp, split_file=args.split_file)

        elif func == 'save_files':
            pds = FelxDataSet(exp)
            if pds.check_bgc_tmp_files_complete():
                save_felx_BGC_fields(exp)
