# -*- coding: utf-8 -*-
"""FelxDataSet class contains constants and functions the lagrangian iron model.

Includes functions for sampling and saving BGC fields at particle positions.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sun Feb 19 14:24:37 2023

"""
import calendar
from datetime import datetime, timedelta  # NOQA
import matplotlib.pyplot as plt
from memory_profiler import profile

import numpy as np
import os
import pandas as pd  # NOQA
import xarray as xr  # NOQA

import cfg  # NOQA
from cfg import paths, ExpData
from datasets import plx_particle_dataset, save_dataset  # convert_plx_times
from tools import timeit, mlogger

logger = mlogger('files_felx')


class FelxDataSet(object):
    """Felx particle dataset."""

    def __init__(self, exp):
        """Initialise & format felx particle dataset."""
        self.exp = exp
        self.num_subsets = 100
        self.bgc_variables = ['phy', 'zoo', 'det', 'temp', 'no3', 'kd', 'fe']
        if cfg.test:
            self.bgc_variables = ['phy', 'zoo', 'det']
        self.bgc_variables_nmap = {'Phy': 'phy', 'Zoo': 'zoo', 'Det': 'det', 'Temp': 'temp',
                                   'Fe': 'fe', 'NO3': 'no3', 'kd': 'kd'}
        self.variables = ['scav', 'fe_src', 'fe_p', 'reg', 'phy_up']

    def particles_after_start_time(self, ds):
        """Select particles that have reached a source at the start of the spinup period.

        Particles criteria:
            * Be released in the actual experiment time period.
            *<NOT APPLIED> At the source at least during the spinup period.

        Example:
            pids = ds.particles_after_start_time()
            ds = ds.sel(traj=pids)

        """
        mask_at_euc = ds.time.dt.year.max('obs') >= self.exp.time_bnds[0].year
        # mask_source = self.ds.isel(obs=0, drop=True).time.dt.year >= self.exp.spinup_bnds[0].year

        pids = ds.traj.where(mask_at_euc, drop=True)
        return pids

    def empty_DataArray(self, ds, name=None, fill_value=np.nan, dtype=np.float32):
        """Add empty DataArray."""
        return xr.DataArray(xr.full_like(ds.z, fill_value, dtype=dtype), dims=('traj', 'obs'),
                            coords=ds.coords, name=name)

    def override_spinup_particle_times(self, time):
        """Change the year of particle time during spinup to spinup year.

        Example:
            ds['time'] = ds.override_spinup_particle_times(ds.time)

        """
        spinup_year = self.exp.spinup_year

        # Mask Non-Spinup & NaT times.
        mask = (time < np.datetime64(self.exp.time_bnds[0]))

        time_spinup = time.where(mask)

        # Account for leapdays in timeseries if spinup is a not a leap year.
        if not calendar.isleap(spinup_year):
            # Change Feb 29th to Mar 1st (while still in datetime64 format).
            leapday_new = np.datetime64('{}-03-01T12:00'.format(spinup_year))
            leapday_mask = ((time_spinup.dt.month == 2) & (time_spinup.dt.day == 29))
            time_spinup = time_spinup.where(~leapday_mask, leapday_new)

        # Replace year in spinup period by converting to string.
        time_spinup = time_spinup.dt.strftime('{}-%m-%dT%H:%M'.format(spinup_year))

        # Convert dtype back to 'datetime64'.
        # Replace NaNs with a number to avoid conversion error (
        time_spinup = time_spinup.where(mask, 0)
        time_spinup = time_spinup.astype('datetime64[ns]')

        # N.B. xr.where condition order important as modified times only valid for spinup & ~NaT.
        time_new = xr.where(mask, time_spinup, time)
        return time_new

    def revert_spinup_particle_times(self):
        """Change the year of particle time during spinup to spinup year.

        Notes:
            * Requires original particle times to be saved to dataset.
            * At the source at least during the spinup period.

        """
        # # Self checks.
        # mask = (self.ds_time_orig >= np.datetime64(self.exp.time_bnds[0]))
        # fill = np.datetime64('2023')
        # assert self.ds_time_orig.shape == self.ds.time.shape
        # assert all(self.ds_time_orig.where(mask, fill) == self.ds.time.where(mask, fill))

        return self.ds_time_orig

    def subset_min_time(self, time='2012-01-01'):
        """Change particle obs times during spin up to spinup year."""
        return self.ds.where(self.ds.time >= np.datetime64(time))

    def set_dataset_vars(ds, variables, arrays):
        """Set arrays as DataArrays in dataset."""
        for var, arr in zip(variables, arrays):
            ds[var] = (['traj', 'obs'], arr)
        return ds

    def save_empty_bgc_felx_file(self):
        """Save new felx file (empty BGC data variables)."""
        ds = xr.open_dataset(self.exp.file_plx_inv, decode_times=True, decode_cf=True)

        # Add data_vars
        ds['valid_mask'] = ~np.isnan(ds.trajectory)
        # convert to days since 1979 (same as ofam3)
        ds['month'] = ds.time.dt.month.astype(dtype=np.float32)
        ds['time_orig'] = ds.time.copy()  # particle times before modifying
        ds['time'] = self.override_spinup_particle_times(ds.time)

        for var in self.bgc_variables:
            ds[var] = self.empty_DataArray(ds)

        save_dataset(ds, str(self.exp.file_felx_bgc), msg='')
        ds.close()

        # ds = xr.open_dataset(self.exp.file_felx_bgc, decode_times=False, use_cftime=True,
        #                       decode_cf=True)
        # # Change time encoding (errors - can't overwrite file & doesen't work when not decoded).
        # attrs = dict(units=ds.time.units, calendar=ds.time.calendar)
        # attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='gregorian')  # OFAM3
        # times = np.apply_along_axis(convert_plx_times, 1, arr=ds.time,
        #                             obs=ds.obs.astype(dtype=int), attrs=attrs,
        #                             attrs_new=attrs_new)
        # ds['time'] = (('traj', 'obs'), times)
        # ds['time'].attrs['units'] = attrs_new['units']
        # ds['time'].attrs['calendar'] = attrs_new['calendar']

        # for var in ds.data_vars:
        #     if ds[var].dtype == np.float64:
        #         ds[var] = ds[var].astype(dtype=np.float32)
        # ds.close()
        # save_dataset(ds, self.exp.file_plx_inv, msg='Converted time calendar.')

        # # alt
        # ds = xr.open_dataset(exp.file_felx_bgc, decode_times=False, use_cftime=True,
        #                       decode_cf=True)
        # attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='gregorian')  # OFAM3
        # ds['time'].attrs['units'] = attrs_new['units']
        # ds['time'].attrs['calendar'] = attrs_new['calendar']
        # save_dataset(ds, 'test.nc', msg='Converted time calendar.')
        return

    @timeit(my_logger=logger)
    def save_plx_file_particle_subset(self):
        """Divide orig plx dataset & subset FelX time period."""
        ds = plx_particle_dataset(self.exp.file_plx_orig)

        size = ds.traj.size
        logger.info('{}: Getting Felx particle subset...'.format(self.exp.file_plx_orig.stem))
        pids = self.particles_after_start_time(ds)

        ds = ds.sel(traj=pids)
        logger.info('{}: Subset felx particles size={}->{}.'.format(self.exp.file_plx_orig.stem,
                                                                    size, ds.traj.size))

        ntraj = int(np.floor(ds.traj.size / 2))

        files = [paths.data / 'plx/{}{:02d}.nc'.format(self.exp.file_plx.stem[:-2],
                                                       (self.exp.file_index_orig * 2) + i)
                 for i in range(2)]
        for file, subset in zip(files, [slice(0, ntraj), slice(ntraj, ds.traj.size)]):
            dx = ds.isel(traj=subset)
            logger.info('{}: Divided particles size={}->{}.'.format(file.stem, size, dx.traj.size))
            dx = dx.dropna('obs', 'all')
            logger.debug('{}: Saving subset...'.format(file.stem))
            save_dataset(dx, str(file), msg='Subset plx file.')
            logger.debug('{}: Saved subset.'.format(file.stem))
            dx.close()
        ds.close()

    @timeit(my_logger=logger)
    def save_inverse_plx_dataset(self):
        """Open, reverse and save plx dataset.

        Args:
            exp (cfg.ExpData): Experiment dataclass instance.
            open_kwargs (dict): xarray.open_dataset keywords.

        Notes:
            - Saves file as './data/felx/plx_hist_165_v1r00.nc'
            - Drops 'unbeached' variable.
            - Runs in 5-10 mins per file.

        """
        def inverse_particle_obs(ds):
            """Reverse obs order for plx particles & remove leading NaNs.

            Args:
                ds (xarray.Dataset): Dataset particle trajectories.

            Returns:
                ds (xarray.Dataset): dataset particle trajectories with reversed coord 'obs'.

            Notes:
                - Dataset requires data variable 'trajectory'.

            Alternate:
                for p in range(ds.traj.size):
                    # Index of first non-NaN value.
                    N = np.isnan(ds.isel(traj=p).trajectory).sum().item()
                    ds[dict(traj=p)] = ds.isel(traj=p).roll(obs=-N)

            """
            def align_ragged_array(ds):
                """Roll non-NaN 1D data array to start."""
                def roll_nan_data(ds):
                    return np.roll(ds, -np.sum(pd.isnull(ds)), axis=0)
                return np.apply_along_axis(roll_nan_data, -1, ds)

            # Reverse order of obs indexes (forward).
            ds = ds.isel(obs=slice(None, None, -1))

            for var in ds.data_vars:
                ds[var] = (['traj', 'obs'], align_ragged_array(ds[var]))

            # Reorder coord 'obs'.
            ds.coords['obs'] = np.arange(ds.obs.size, dtype=ds.obs.dtype)
            return ds

        logger.info('{}: Calculating inverse plx file...'.format(self.exp.file_plx.stem))
        # Particle trajectories.
        ds = plx_particle_dataset(self.exp.file_plx)

        # Copy & drop 1D data vars.
        u = ds.u.copy()
        zone = ds.u.copy()
        ds = ds.drop({'u', 'zone'})

        # Reverse particleset.
        ds = inverse_particle_obs(ds)

        # Replace 1D data vars.
        ds['u'] = u
        ds['zone'] = zone

        # Save dataset.
        save_dataset(ds, str(self.exp.file_plx_inv), msg='Inverse particle obs dimension.')
        logger.info('{}: Saved inverse plx file.'.format(self.exp.file_plx_inv.stem))

    def check_file_complete(self, file):
        """Check file exists and can be opened without error."""
        file_complete = False
        if file.exists():
            try:
                if file.suffix == '.nc':
                    df = xr.open_dataset(str(file), chunks='auto')
                    df.close()
                elif file.suffix == '.npy':
                    df = np.load(file, allow_pickle=True)
                file_complete = True

            except Exception as err:
                file_complete = False
                os.remove(file)
                logger.info('{}: File check failed. {}: {}. Deleted.'
                            .format(file.stem, type(err), err))
        return file_complete

    def check_bgc_prereq_files(self):
        """Check needed files saved and can be opened without error."""
        file = self.exp.file_plx
        file_complete = self.check_file_complete(file)
        if not file_complete:
            self.save_plx_file_particle_subset()

        file = self.exp.file_plx_inv
        file_complete = self.check_file_complete(file)
        if not file_complete:
            self.save_inverse_plx_dataset()

        file = self.exp.file_felx_bgc
        file_complete = self.check_file_complete(file)
        if not file_complete:
            self.save_empty_bgc_felx_file()

    def bgc_var_tmp_filenames(self, var):
        """Get tmp filenames for BGC tmp subsets."""
        tmp_dir = paths.data / 'felx/tmp_{}'.format(self.exp.file_felx_bgc.stem)
        if not tmp_dir.is_dir():
            os.makedirs(tmp_dir, exist_ok=True)
        tmp_files = [tmp_dir / '{}_{}.npy'.format(var, i) for i in range(self.num_subsets)]
        return tmp_files

    def bgc_tmp_traj_subsets(self, ds):
        """Get traj slices for BGC tmp subsets."""
        traj_bnds = np.linspace(0, ds.traj.size + 1, self.num_subsets + 1, dtype=int)
        traj_slices = [[traj_bnds[i], traj_bnds[i+1] - 1] for i in range(self.num_subsets)]
        return traj_slices

    def check_bgc_tmp_files_complete(self):
        """Check all variable tmp file subsets complete."""
        complete = True
        for var in self.bgc_variables:
            for tmp_file in self.bgc_var_tmp_filenames(var):
                file_complete = self.check_file_complete(tmp_file)
                if not file_complete:
                    complete = False
        return complete

    def rename_bgc_tmp_files(self):
        """Rename error in variable tmp filenames."""
        for var in self.bgc_variables:
            tmp_dir = paths.data / 'felx/tmp_{}_{}'.format(self.exp.file_felx_bgc.stem, var)
            tmp_files = [tmp_dir / '{}.np.npy'.format(i) for i in range(self.num_subsets)]
            for file in tmp_files:
                if self.check_file_complete(file):
                    if '.np.npy' in file.name:
                        file.rename(file.parent / file.name.replace('.np.npy', '.npy'))
        return

    def set_bgc_dataset_vars_from_tmps(self, ds):
        """Set arrays as DataArrays in dataset (N.B. renames saved vairables)."""
        for name, var in self.bgc_variables_nmap.items():
            data = []
            for tmp_file in self.bgc_var_tmp_filenames(var):
                data.append(np.load(tmp_file, allow_pickle=True))
            arr = np.concatenate(data, axis=0)

            ds[name] = (['traj', 'obs'], arr)
            ds[name] = ds[name].interpolate_na('obs', method='slinear', limit=10)
        return ds

    def init_felx_bgc_dataset(self):
        """Get finished felx BGC dataset and format."""
        file = self.file_felx_bgc
        ds = xr.open_dataset(file)

        variables = ['scav', 'src', 'iron', 'reg', 'phyup']
        for var in variables:
            ds[var] = self.empty_DataArray(ds)
        self.add_iron_model_constants()
        return ds

    def add_iron_model_constants(self):
        """Add a constants to the FieldSet."""
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
        for name, value in constants.items():
            setattr(self, name, value)

    def test_plot_variable(self, var='det'):
        """Line plot of variable as a function of obs for each particle."""
        fig, ax = plt.subplots(figsize=(10, 7))
        self.ds[var].plot.line(ax=ax, x='obs', add_legend=False, yincrease=True)
        plt.show()

    def test_plot_particle_4D(self, pid, var='det'):
        """Plot 3D trajectory with contoured variable."""
        # 3D Plot.
        dx = self.pds.ds.sel(traj=pid)
        c = dx[var]
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        ax.plot3D(dx.lon, dx.lat, dx.z, c='k')
        sp = ax.scatter3D(dx.lon, dx.lat, dx.z, c=c, cmap=plt.cm.magma)
        fig.colorbar(sp, shrink=0.6)
        plt.show()
