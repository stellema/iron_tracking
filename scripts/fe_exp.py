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

import numpy as np
import os
import pandas as pd  # NOQA
import xarray as xr  # NOQA

import cfg
from cfg import paths
from datasets import plx_particle_dataset, save_dataset
from tools import timeit, mlogger, unique_name

logger = mlogger('files_felx')


class FelxDataSet(object):
    """Iron model functions and parameters."""

    def __init__(self, exp):
        """Initialise & format felx particle dataset."""
        self.exp = exp
        self.n_subsets = 100
        self.bgc_variables = ['phy', 'zoo', 'det', 'temp', 'no3', 'kd']
        self.bgc_variables_nmap = {'Phy': 'phy', 'Zoo': 'zoo', 'Det': 'det', 'Temp': 'temp',
                                   'NO3': 'no3', 'kd': 'kd'}
        self.variables = ['scav', 'fe_src', 'fe_p', 'reg', 'phy_up']
        self.add_iron_model_params()

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
        return xr.DataArray(fill_value, dims=('traj', 'obs'), coords=ds.coords, name=name)

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
            - Saves file as './data/plx/plx_hist_165_v1r00.nc'
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
        zone = ds.zone.copy()  # Error - wrong variable in saved files.
        ds = ds.drop({'u', 'zone'})

        # Reverse particleset.
        ds = inverse_particle_obs(ds)

        # Replace 1D data vars.
        ds['u'] = u
        ds['zone'] = zone

        # Save dataset.
        save_dataset(ds, str(self.exp.file_plx_inv), msg='Inverse particle obs dimension.')
        logger.info('{}: Saved inverse plx file.'.format(self.exp.file_plx_inv.stem))

    def init_felx_bgc_tmp_dataset(self):
        """Create felx_bgc file (empty BGC data variables)."""
        ds = xr.open_dataset(self.exp.file_plx_inv, decode_times=True, decode_cf=True)

        # Add data_vars.
        ds['valid_mask'] = ~np.isnan(ds.trajectory)
        # convert to days since 1979 (same as ofam3).
        ds['month'] = ds.time.dt.month.astype(dtype=np.float32)
        ds['time_orig'] = ds.time.copy()  # particle times before modifying
        ds['time'] = self.override_spinup_particle_times(ds.time)

        for var in self.bgc_variables:
            ds[var] = self.empty_DataArray(ds)

        save_dataset(ds, str(self.exp.file_felx_bgc_tmp), msg='')

        # ds = xr.open_dataset(self.exp.file_felx_bgc_tmp, decode_times=False, use_cftime=True,
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
        # ds = xr.open_dataset(exp.file_felx_bgc_tmp, decode_times=False, use_cftime=True,
        #                       decode_cf=True)
        # attrs_new = dict(units='days since 1979-01-01 00:00:00', calendar='gregorian')  # OFAM3
        # ds['time'].attrs['units'] = attrs_new['units']
        # ds['time'].attrs['calendar'] = attrs_new['calendar']
        # save_dataset(ds, 'test.nc', msg='Converted time calendar.')
        return ds

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
                logger.info('{}: File check failed. {}: {}.'
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

        file = self.exp.file_felx_bgc_tmp
        file_complete = self.check_file_complete(file)
        if not file_complete:
            ds = self.init_felx_bgc_tmp_dataset()
            ds.close()

    def bgc_var_tmp_filenames(self, var, suffix='.nc'):
        """Get tmp filenames for BGC tmp subsets."""
        tmp_dir = self.exp.out_dir / 'tmp_{}'.format(self.exp.file_felx_bgc.stem)
        if not tmp_dir.is_dir():
            os.makedirs(tmp_dir, exist_ok=True)
        tmp_files = [tmp_dir / '{}_{:03d}{}'.format(var, i, suffix) for i in range(self.n_subsets)]
        return tmp_files

    def particle_subsets(self, ds, size):
        """Particle subsets for BGC tmp files."""
        # Divide particles into subsets with roughly the same number of total obs per subset.
        # Count number of obs per particle & group into N subsets.
        n_obs = ds.z.where(np.isnan(ds.z), 1).sum(dim='obs')
        n_obs_grp = (n_obs.cumsum() / (n_obs.sum() / size)).astype(dtype=int)
        # Number of particles per subset.
        n_traj = [n_obs_grp.where(n_obs_grp == i, drop=True).traj.size for i in range(size)]
        # Particle index [first, last+1] in each subset.
        n_traj = np.array([0, *n_traj])
        traj_bnds = [[np.cumsum(n_traj)[i], np.cumsum(n_traj)[i + 1]] for i in range(size)]
        if traj_bnds[-1][-1] != ds.traj.size:
            traj_bnds[-1][-1] = ds.traj.size
        return traj_bnds

    def check_bgc_tmp_files_complete(self):
        """Check all variable tmp file subsets complete."""
        complete = True
        f_list = []
        for var in self.bgc_variables:
            for tmp_file in self.bgc_var_tmp_filenames(var):
                file_complete = self.check_file_complete(tmp_file)
                if not file_complete:
                    f_list.append(tmp_file.stem)
                    complete = False
        print(f_list)
        return complete

    def test_set_bgc_dataset_vars_from_tmps(self, ds, n_tmps=20):
        """Set arrays as DataArrays in dataset (N.B. renames saved vairables)."""
        ntraj = self.bgc_tmp_traj_subsets(ds)[n_tmps][-1]
        ds = ds.isel(traj=slice(ntraj))
        for var in self.bgc_variables:
            tmp_files = self.bgc_var_tmp_filenames(var)
            da = xr.open_mfdataset(tmp_files[:n_tmps])
            ds[var] = da[var].interpolate_na('obs', method='slinear', limit=10)
        return ds

    def get_updated_traj(self):
        """Get and or save trajectory IDs after applying new EUC definition."""
        file = self.exp.file_felx_init

        if file.exists():
            ds = xr.open_dataset(file)

        else:
            ds = xr.open_dataset(self.exp.file_felx_bgc)

            # Apply new EUC definition (u > 0.1 m/s)
            traj = ds.u.where((ds.u / cfg.DXDY) > 0.1, drop=True).traj
            ds = ds.sel(traj=traj)
            ds = ds.drop([v for v in ds.data_vars if v not in ['trajectory']])

            logger.debug('{}: Saving...'.format(file.stem))
            save_dataset(ds, file, msg='Saved init felx dataset')
            logger.debug('{}: Saved.'.format(file.stem))

        traj = ds.traj
        ds.close()
        return traj

    def init_felx_dataset(self):
        """Get finished felx BGC dataset and format."""
        ds = xr.open_dataset(self.exp.file_felx_bgc)

        # Apply new EUC definition (u > 0.1 m/s)
        traj = ds.u.where((ds.u / cfg.DXDY) > 0.1, drop=True).traj
        ds = ds.sel(traj=traj)

        variables = ['fe_scav', 'fe_reg', 'fe_phy', 'fe']
        for var in variables:
            # ds[var] = self.empty_DataArray(ds)
            ds[var] = xr.DataArray(np.nan, dims=('traj', 'obs'), coords=ds.coords)
        return ds

    def save_felx_dataset(self):
        """Save & format finished felx tmp datasets."""
        file = self.exp.file_felx

        # Merge tmp files
        tmp_files = self.felx_tmp_filenames(48)
        ds_fe = xr.open_mfdataset(tmp_files)

        ds = self.init_felx_dataset()

        variables = ['fe_scav', 'fe_reg', 'fe_phy', 'fe']
        for var in variables:
            ds[var] = ds_fe[var]

        if 'age' not in ds.data_vars:
            logger.debug('{}: Copying age and distance.'.format(file.stem))
            ds_inv = xr.open_dataset(self.exp.file_plx_inv)
            for var in ['age', 'distance']:
                ds[var] = ds_inv[var]
            ds_inv.close()

        # Add metadata
        ds.attrs['constants'] = self.params.copy()
        ds = self.add_variable_attrs(ds)

        logger.info('{}: Saving...'.format(file.stem))
        save_dataset(ds, file, msg='Created from Lagrangian iron model.')
        logger.info('{}: Saved.'.format(file.stem))
        ds_fe.close()
        return ds

    def init_felx_optimise_dataset(self):
        """Initialise empty felx dataset subset to 1 year and close to equator."""
        file = cfg.paths.data / 'felx/{}_tmp_optimise.nc'.format(self.exp.file_base)
        if file.exists():
            ds = xr.open_dataset(file)

        else:
            ds = self.init_felx_dataset()
            # Subset particles to one year and close to the equator.
            target = np.datetime64('{}-01-01T12'.format(self.exp.year_bnds[-1]))
            ds_f = xr.open_dataset(self.exp.file_plx).isel(obs=0, drop=True)
            traj = ds_f.where((ds_f.time >= target) & ((ds.u / cfg.DXDY) > 0.1)
                              & (ds_f.lat >= -0.25) & (ds_f.lat <= 0.25), drop=True).traj
            ds = ds.sel(traj=traj)
            ds_f.close()
        return ds

    def felx_tmp_filenames(self, size):
        """Get tmp filenames for felx tmp subsets."""
        tmp_dir = paths.data / 'felx/tmp_{}'.format(self.exp.file_felx.stem)
        if not tmp_dir.is_dir():
            os.makedirs(tmp_dir, exist_ok=True)
        tmp_files = [tmp_dir / '{:02d}.nc'.format(i) for i in range(size)]
        tmp_files = [unique_name(tmp_files[i]) for i in range(size)]
        return tmp_files

    def add_iron_model_params(self):
        """Add a constants to the FieldSet.

        Redfield ratio of 1 P: 16 N: 106 5C: −172 O2: 3.2
        """
        params = {}
        # Scavenging paramaters.
        params['c_scav'] = 2.5  # Scavenging rate constant [no units]
        params['k_org'] = 1e-3  # Organic iron scavenging rate constant [(nM Fe)^-0.58 day^-1] (Qin: 1.0521e-4, Galbraith: 4e-4)
        params['k_inorg'] = 1e-3  # Inorganic iron scavenging rate constant [(nM m Fe)^-0.5 day^-1] (Qin: 6.10e-4, Galbraith: 6e-4)
        params['tau'] = 1.24e-2  # Scavenging rate [day^-1] (OFAM3 equation) (Oke: 1, other: 1.24e-2)

        # Detritus paramaters.
        params['w_D'] = 10  # Detritus sinking velocity [m day^-1] (Qin: 10 or Oke: 5)
        # params['w_D'] = lambda z: 16 + max(0, (z - 80)) * 0.05  # [m day^-1] linearly increases by 0.5 below 80m
        params['mu_D'] = 0.005  # 0.02 b**cT
        params['mu_D_180'] = 0.03  # 0.01 b**cT

        # Phytoplankton paramaters.
        params['I_0'] = 280  # Surface incident solar radiation [W/m^2] (not actually constant)
        params['alpha'] = 0.025  # Initial slope of P-I curve [day^-1 / (Wm^-2)]. (Qin: 0.256)
        params['PAR'] = 0.43  # Photosynthetically active radiation (0.34/0.43) [no unit]
        params['a'] = 0.6  # 0.6 Growth rate at 0C [day^-1] (Qin: 0.27?)
        params['b'] = 1.066  # 1.066 Temperature sensitivity of growth [no units]
        params['c'] = 1.0  # Growth rate reference for light limitation [C^-1]
        params['k_fe'] = 1.0  # Half saturation constant for Fe uptake [mmol N m^-3] [needs converting to mmol Fe m^-3]
        params['k_N'] = 1.0  # Half saturation constant for N uptake [mmol N m^-3]
        params['mu_P'] = 0.01  # (*b^cT) Phytoplankton mortality [day^-1]

        # Zooplankton paramaters.
        params['gamma_1'] = 0.85  # Assimilation efficiency [no units]
        params['g'] = 2.1  # Maximum grazing rate [day^-1]
        params['epsilon'] = 1.1  # Prey capture rate [(mmol N m^-3)^-1 day^-1]
        params['mu_Z'] = 0.06  # Quadratic mortality [(mmol N m^-3)^-1 day^-1]
        params['gamma_2'] = 0.0059699  # (*b^cT) Excretion [day^-1]

        for name, value in params.items():
            setattr(self, name, value)
        setattr(self, 'params', params)

    def update_params(self, new_dict):
        """Update params dictionary."""
        self.params.update(new_dict)
        for key, value in new_dict.items():
            setattr(self, key, value)
            self.params[key] = value

    def add_variable_attrs(self, ds):
        """Add attributes metadata to dataset variables."""
        attrs = {'trajectory': {'long_name': 'Unique identifier for each particle', 'cf_role': 'trajectory_id'},
                 'time': {'long_name': 'time', 'standard_name': 'time', 'axis': 'T'},
                 'lat': {'long_name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'},
                 'lon': {'long_name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'},
                 'z': {'long_name': 'depth', 'standard_name': 'depth', 'units': 'm', 'axis': 'Z', 'positive': 'down'},
                 'age': {'long_name': 'Particle transit time', 'standard_name': 'age', 'units': 's'},
                 'u': {'long_name': 'Transport', 'standard_name': 'u', 'units': 'Sv'},
                 'zone': {'long_name': 'Source location ID', 'standard_name': 'zone'},
                 'temp': {'long_name': 'Potential temperature', 'standard_name': 'sea_water_potential_temperature', 'units': 'degrees C', 'source': 'OFAM3-WOMBAT'},
                 'phy': {'long_name': 'Phytoplankton', 'standard_name': 'phy', 'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'zoo': {'long_name': 'Zooplankton', 'standard_name': 'zoo', 'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'det': {'long_name': 'Detritus', 'standard_name': 'det', 'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'no3': {'long_name': 'Nitrate', 'standard_name': 'no3', 'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'kd': {'long_name': 'Diffuse Attenuation Coefficient at 490 nm', 'standard_name': 'Kd490', 'units': 'm^-1', 'scaling': 'log10', 'source': 'SeaWiFS-GMIS'},
                 'fe_scav': {'long_name': 'Scavenged dFe', 'standard_name': 'dFe scav', 'units': 'μmol/m^3 Fe'},
                 'fe_reg': {'long_name': 'Remineralised dFe', 'standard_name': 'dFe remin', 'units': 'μmol m^-3 Fe'},
                 'fe_phy': {'long_name': 'Phytoplankton dFe Uptake', 'standard_name': 'dFe phyto', 'units': 'μmol m^-3 Fe'},
                 'fe': {'long_name': 'Dissolved Iron', 'standard_name': 'dFe', 'units': 'μmol m^-3 Fe'}
                 }

        for k, v in attrs.items():
            if k in ds.data_vars:
                ds[k].attrs = v
        return ds
