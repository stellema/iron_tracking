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
import pandas as pd
import xarray as xr

from cfg import paths, DXDY, ExpData, zones, test
from datasets import (plx_particle_dataset, save_dataset, append_dataset_history,
                      concat_exp_dimension)
from tools import timeit, mlogger

logger = mlogger('files_felx')


class FelxDataSet(object):
    """Iron model functions and parameters."""

    def __init__(self, exp=ExpData()):
        """Initialise & format felx particle dataset."""
        self.exp = exp
        self.zones = zones
        self.zone_indexes = np.arange(len(zones._all), dtype=int)
        self.n_subsets = 100  # For felx_bgc tmp subsets.
        self.bgc_variables = ['phy', 'zoo', 'det', 'temp', 'no3', 'kd']
        self.bgc_variables_nmap = {'Phy': 'phy', 'Zoo': 'zoo', 'Det': 'det', 'Temp': 'temp',
                                   'NO3': 'no3', 'kd': 'kd'}
        self.variables = ['fe_scav', 'fe_reg', 'fe_phy', 'fe']
        self.release_depths = np.arange(25, 350 + 25, 25, dtype=int)
        self.release_lons = np.array([165, 190, 220, 250], dtype=int)
        self.add_iron_model_params()

    def add_iron_model_params(self):
        """Add a constants to the FieldSet.

        Redfield ratio of 1 P: 16 N: 106 5C: −172 O2: 3.2
        """
        params = {}
        # Scavenging paramaters.
        params['c_scav'] = 2.5  # Scavenging rate constant [no units]
        params['k_org'] = 0.000618  # Organic iron scavenging rate constant [(uM N)^-0.58 day^-1] (Qin: 1.0521e-4, Galbraith: 4e-4)
        params['k_inorg'] = 0.00117  # Inorganic iron scavenging rate constant [(nM m Fe)^-0.5 day^-1] (Qin: 6.10e-4, Galbraith: 6e-4)
        # params['tau'] = 1.24e-2  # Scavenging rate [day^-1] (OFAM3 equation) (Oke: 1, other: 1.24e-2)

        # Detritus paramaters.
        params['mu_D'] = 0.001  # 0.02 b**cT
        params['mu_D_180'] = 0.05  # 0.01 b**cT
        params['w_D'] = 10  # Detritus sinking velocity [m day^-1] (Qin: 10 or Oke: 5)
        # params['w_D'] = lambda z: 16 + max(0, (z - 80)) * 0.05  # [m day^-1] linearly increases by 0.5 below 80m

        # Phytoplankton paramaters.
        params['I_0'] = 300  # Surface incident solar radiation [W/m^2] (not actually constant)
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
        params['gamma_2'] = 0.01  # (*b^cT) Excretion [day^-1]
        params['g'] = 2.1  # Maximum grazing rate [day^-1]
        params['epsilon'] = 1.1  # Prey capture rate [(mmol N m^-3)^-1 day^-1]
        params['mu_Z'] = 0.06  # Quadratic mortality [(mmol N m^-3)^-1 day^-1]

        for name, value in params.items():
            setattr(self, name, value)
        setattr(self, 'params', params)

    def update_params(self, new_dict):
        """Update params dictionary."""
        self.params.update(new_dict)
        for key, value in new_dict.items():
            setattr(self, key, value)
            self.params[key] = value

    def empty_DataArray(self, ds, name=None, fill_value=np.nan, dtype=np.float32):
        """Add empty DataArray."""
        return xr.DataArray(fill_value, dims=('traj', 'obs'), coords=ds.coords, name=name)

    def set_dataset_vars(ds, variables, arrays):
        """Set arrays as DataArrays in dataset."""
        for var, arr in zip(variables, arrays):
            ds[var] = (['traj', 'obs'], arr)
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

    def override_spinup_particle_times(self, time):
        """Change the year of particle time during spinup to spinup year.

        Example:
            ds['time'] = pds.override_spinup_particle_times(ds.time)

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
            dx = save_dataset(dx, str(file), msg='Subset plx file.')
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
        ds = save_dataset(ds, str(self.exp.file_plx_inv), msg='Inverse particle obs dimension.')
        logger.info('{}: Saved inverse plx file.'.format(self.exp.file_plx_inv.stem))

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

        ds = save_dataset(ds, str(self.exp.file_felx_bgc_tmp), msg='')
        return ds

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

    def felx_tmp_filenames(self, size):
        """Get tmp filenames for felx tmp subsets."""
        tmp_dir = self.exp.out_subdir / 'tmp_{}'.format(self.exp.file_felx.stem)
        if not tmp_dir.is_dir():
            os.makedirs(tmp_dir, exist_ok=True)
        tmp_files = [tmp_dir / 'tmp_{}_{:02d}.nc'.format(self.exp.id, i) for i in range(size)]
        return tmp_files

    def init_felx_dataset(self):
        """Get finished felx BGC dataset and format."""
        ds = xr.open_dataset(self.exp.file_felx_bgc)

        # Apply new EUC definition (u > 0.1 m/s)
        traj = ds.u.where((ds.u / DXDY) > 0.1, drop=True).traj
        ds = ds.sel(traj=traj)

        for var in self.variables:
            ds[var] = xr.DataArray(None, dims=('traj', 'obs'), coords=ds.coords)
        return ds

    def init_felx_optimise_dataset(self):
        """Initialise empty felx dataset subset to 1 year and close to equator."""
        file = paths.data / 'felx/{}_tmp_optimise.nc'.format(self.exp.id)
        if file.exists():
            ds = xr.open_dataset(file)

        else:
            ds = self.init_felx_dataset()
            # Subset particles to one year and close to the equator.
            target = np.datetime64('{}-01-01T12'.format(self.exp.year_bnds[-1]))
            ds_f = xr.open_dataset(self.exp.file_plx).isel(obs=0, drop=True)
            traj = ds_f.where((ds_f.time >= target) & ((ds.u / DXDY) > 0.1)
                              & (ds_f.lat >= -0.25) & (ds_f.lat <= 0.25), drop=True).traj
            ds = ds.sel(traj=traj)
            ds_f.close()
        return ds

    def save_felx_dataset(self, size=80):
        """Save & format finished felx tmp datasets."""
        file = self.exp.file_felx

        # Merge temp subsets of fe model output.
        tmp_files = self.felx_tmp_filenames(size)

        ds_fe = xr.open_mfdataset([f for f in tmp_files if f.exists()])

        # Create initial dataset
        logger.info('{}: Get initial dataset.'.format(file.stem))
        # ds = self.init_felx_dataset()
        ds = xr.open_dataset(self.exp.file_felx_bgc, chunks='auto')
        ds = ds.sel(traj=ds_fe.traj.load())
        ds = ds.drop(['kd', 'no3', 'det', 'zoo'])

        # Add extra variables that aren't in dataset (age and distance)
        logger.info('{}: Add age.'.format(file.stem))
        ds_inv = xr.open_dataset(self.exp.file_plx_inv, chunks={'traj': ds.traj.size, 'obs': ds.obs.size})
        ds['age'] = ds_inv['age'].sel(traj=ds.traj)

        # Add fe model output from temp files
        logger.info('{}: Add fe variables.'.format(file.stem))
        for var in self.variables:
            ds[var] = ds_fe[var]

        ds = ds.unify_chunks()

        logger.info('{}: Drop trailing NaNs (pre={}).'.format(file.stem, ds.obs.size))
        ds = ds.where(~np.isnan(ds.z).load(), drop=True)
        ds['u'] = ds.u.isel(obs=0, drop=True)
        if 'obs' in ds.zone.dims:
            ds['zone'] = ds.zone.isel(obs=0)  # Not run in v0
        logger.info('{}: Dropped trailing NaNs (post={}).'.format(file.stem, ds.obs.size))

        # Add metadata
        ds = self.add_variable_attrs(ds)

        # Save dataset.
        logger.info('{}: Saving...'.format(file.stem))
        ds = append_dataset_history(ds, msg='Created from Lagrangian iron model.')

        comp = dict(zlib=True, complevel=5, contiguous=False)
        for var in ds:
            ds[var].encoding.update(comp)

        ds = ds.chunk()
        ds.to_netcdf(file)
        logger.info('{}: Saved.'.format(file.stem))

        ds_fe.close()
        ds_inv.close()
        ds_fe.close()
        return ds

    def get_particle_IDs_of_fe_errors(self, ds):
        """Get particle IDs of particles with negative iron variables."""
        pids = ds.where(ds.fe < 0, drop=True).traj
        return pids

    def save_felx_dataset_fixed(self, size=80):
        """Save & format finished felx tmp datasets."""
        file = self.exp.file_felx

        # Merge temp subsets of fe model output.
        tmp_files = self.felx_tmp_filenames(size)
        tmp_files = [f for f in tmp_files if f.exists()]
        ds_new = xr.open_mfdataset(tmp_files)

        felx_file_v0 = self.exp.out_subdir / 'tmp_err/fe_{}.nc'.format(self.exp.id)

        ds = xr.open_dataset(felx_file_v0)  # Open v0 complete felx dataset.

        ds_new = ds_new.isel(obs=slice(ds.obs.size))  # Other file dropped trailing NaNs
        pids_new = self.get_particle_IDs_of_fe_errors(ds)  # Particles to re-run
        pids_old = ds.traj[~ds.traj.isin(pids_new)]

        assert all(ds_new.traj == pids_new)

        # Add fe model output from temp files
        logger.debug('{}: Add fe variables.'.format(file.stem))
        for var in self.variables:
            logger.debug('{}: Add {}.'.format(file.stem, var))
            ds[var] = xr.merge([ds[var].sel(traj=pids_old), ds_new[var]])[var]

        # Add metadata
        ds = self.add_variable_attrs(ds)

        # Save dataset.
        logger.info('{}: Saving...'.format(file.stem))
        ds = append_dataset_history(ds, msg='Saved merged iron model output.')

        comp = dict(zlib=True, complevel=5, contiguous=False)
        for var in ds:
            ds[var].encoding.update(comp)

        ds = ds.chunk()
        ds = ds.unify_chunks()
        ds.to_netcdf(file)
        logger.info('{}: Saved.'.format(file.stem))

        ds_new.close()
        return ds

    def save_felx_dataset_particle_subset(self, size=80, v0_error_fix=False):
        """Save & format finished felx tmp datasets."""
        file = self.exp.file_felx

        # Merge temp subsets of fe model output.
        tmp_files = self.felx_tmp_filenames(size)
        tmp_files = [f for f in tmp_files if f.exists()]
        ds_new_list = [xr.open_dataset(f) for f in tmp_files]

        felx_file_v0 = self.exp.out_dir / 'v4/fe_{}.nc'.format(self.exp.id_v0)
        ds = xr.open_dataset(felx_file_v0)  # Open v0 complete felx dataset.

        # Add fe model output from temp files
        logger.debug('{}: Add fe variables.'.format(file.stem))
        for ds_new in ds_new_list:
            ds_new = ds_new.reindex(traj=sorted(ds_new.traj))
            ds_new = ds_new.isel(obs=slice(ds.obs.size)) # Other file dropped trailing NaNs

            pids_new = ds_new.traj.values  # Particles to merge
            pids_old = ds.traj[~ds.traj.isin(pids_new)]

            for var in self.variables:
                logger.debug('{}: Add {}.'.format(file.stem, var))
                ds[var] = xr.merge([ds[var].sel(traj=pids_old), ds_new[var]])[var]

        # Add metadata
        ds = self.add_variable_attrs(ds)

        # Save dataset.
        logger.info('{}: Saving...'.format(file.stem))
        ds = append_dataset_history(ds, msg='Saved merged iron model output.')

        comp = dict(zlib=True, complevel=5, contiguous=False)
        for var in ds:
            ds[var].encoding.update(comp)

        ds = ds.chunk()
        ds = ds.unify_chunks()
        ds.to_netcdf(file)
        logger.info('{}: Saved.'.format(file.stem))

        return ds

    def add_variable_attrs(self, ds, post=False):
        """Add attributes metadata to dataset variables."""
        attrs = {'trajectory': {'long_name': 'Unique identifier for each particle',
                                'cf_role': 'trajectory_id'},
                 'time': {'long_name': 'time', 'standard_name': 'time', 'axis': 'T'},
                 'lat': {'long_name': 'latitude', 'standard_name': 'latitude',
                         'units': 'degrees_north', 'axis': 'Y'},
                 'lon': {'long_name': 'longitude', 'standard_name': 'longitude',
                         'units': 'degrees_east', 'axis': 'X'},
                 'z': {'long_name': 'depth', 'standard_name': 'depth', 'units': 'm', 'axis': 'Z',
                       'positive': 'down'},
                 'age': {'long_name': 'Transit time', 'standard_name': 'age'},
                 'distance': {'long_name': 'Distance', 'standard_name': 'distance', 'units': 'm'},
                 'u': {'long_name': 'Transport', 'standard_name': 'transport', 'units': 'Sv'},
                 'zone': {'long_name': 'Source location ID', 'standard_name': 'zone'},
                 'temp': {'long_name': 'Potential temperature',
                          'standard_name': 'sea_water_potential_temperature', 'units': 'degrees C',
                          'source': 'OFAM3-WOMBAT'},
                 'phy': {'long_name': 'Phytoplankton', 'standard_name': 'phytplankton',
                         'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'zoo': {'long_name': 'Zooplankton', 'standard_name': 'zooplankton',
                         'units': 'mmol/m^3 N', 'source': 'OFAM3-WOMBAT'},
                 'det': {'long_name': 'Detritus', 'standard_name': 'detritus', 'units': 'mmol/m^3 N',
                         'source': 'OFAM3-WOMBAT'},
                 'no3': {'long_name': 'Nitrate', 'standard_name': 'NO3', 'units': 'mmol/m^3 N',
                         'source': 'OFAM3-WOMBAT'},
                 'kd': {'long_name': 'Diffuse Attenuation Coefficient at 490 nm',
                        'standard_name': 'Kd490', 'units': 'm^-1', 'source': 'SeaWiFS-GMIS'},
                 'fe': {'long_name': 'Dissolved Iron', 'standard_name': 'dFe', 'units': 'umol/m^3 Fe'},
                 'fe_flux': {'long_name': 'Dissolved Iron Flux', 'standard_name': 'dFe flux',
                             'units': 'umol m^-2 s^-1 Fe'},
                 'fe_scav': {'long_name': 'Scavenged Dissolved Iron', 'standard_name': 'dFe',
                             'units': 'umol/m^3 Fe'},
                 'fe_reg': {'long_name': 'Remineralised Dissolved Iron', 'standard_name': 'dFe',
                            'units': 'umol/m^3 Fe'},
                 'fe_phy': {'long_name': 'Phytoplankton Dissolved Iron Uptake',
                            'standard_name': 'dFe', 'units': 'umol/m^3 Fe'}
                 }

        for k, v in attrs.items():
            if k in ds.data_vars:
                ds[k].attrs = v

        if post:
            # Update attributes and units for post-analysis
            if 'age' in ds.data_vars:
                ds['age'].attrs['units'] = 'days'  # Tries to be decoded if saved.

            for v in ds.data_vars:
                if 'units' in ds[v].attrs:
                    if 'fe' in str(v) and 'fe_flux' not in str(v):
                        # Replace umol/m^3 with nM
                        ds[v].attrs['units'] = 'nM'

                    if 'mmol' in str(ds[v].attrs['units']):
                        ds[v].attrs['units'] = 'μM N'

                for n in ['long_name', 'standard_name']:
                    if n in ds[v].attrs:
                        # Replace 'Dissolved Iron' with 'dFe'
                        ds[v].attrs[n] = ds[v].attrs[n].replace('Dissolved Iron', 'dFe')
                        # if 'dFe' not in str(ds[v].attrs[n]):
                        ds[v].attrs[n] = ds[v].attrs[n].title()
                        ds[v].attrs[n] = str(ds[v].attrs[n]).replace('Dfe', 'dFe')
        return ds

    def weighted_mean(self, ds, var='fe', dim='traj', groupby_depth=False):
        """Calculate particle weighted mean."""
        # Particle data
        if 'obs' in ds[var].dims:
            # ds_i = ds.isel(obs=0)  # at source
            ds = ds.ffill('obs').isel(obs=-1, drop=True)  # at EUC

        weights = ds.u

        if groupby_depth:
            var_mean = (ds[var] * weights).groupby(ds.z).sum() / (weights).groupby(ds.z).sum()
        else:
            var_mean = ds[var].weighted(weights).mean(dim).load()
        return var_mean

    def get_final_particle_obs(self, ds):
        """Reduce particle dataset variables to first/last observation."""
        df = xr.Dataset()
        df = df.assign_attrs(ds.attrs)

        df['trajectory'] = ds['trajectory'].isel(obs=0, drop=True)  # First value (at the source).

        # Location variables.
        for var in ['time', 'lat', 'lon', 'z']:
            df[var] = ds[var].ffill('obs').isel(obs=-1)  # Last value (at the EUC).
            df[var].attrs['long_name'] += ' at the EUC'

        df['time_at_src'] = ds['time'].isel(obs=0, drop=True)  # First value (at the source).
        df['time_at_src'].attrs['long_name'] = 'Source Time'
        df['z_at_src'] = ds['z'].isel(obs=0, drop=True)  # First value (at the source).
        df['z_at_src'].attrs['long_name'] = 'Source Depth'

        df['depth'] = df['z'].copy()
        df['depth_at_src'] = df['z_at_src'].copy()

        # Physical variables.
        df['u'] = ds['u'].isel(obs=0, drop=True) if 'obs' in ds['u'].dims else ds['u']
        df['age'] = ds['age'].max('obs', skipna=True, keep_attrs=True)  # Maximum value.
        df['zone'] = ds['zone'].isel(obs=0, drop=True) if 'obs' in ds['zone'].dims else ds['zone']

        # Biogeochemical variables.
        for var in ['temp', 'phy']:
            df[var] = ds[var].mean('obs', skipna=True, keep_attrs=True)  # Particle mean.
            df[var].attrs['long_name'] = 'Mean ' + df[var].attrs['long_name']

        # Iron variables.
        # Iron at the EUC.
        df['fe'] = ds['fe'].ffill('obs').isel(obs=-1)  # Last value (at the EUC).
        df['fe'].attrs['long_name'] = 'EUC ' + df['fe'].attrs['long_name']

        # Iron at the source.
        df['fe_src'] = ds['fe'].isel(obs=0, drop=True)  # First value (at the source).
        df['fe_src'].attrs['long_name'] = 'Source ' + df['fe_src'].attrs['long_name']

        # Calculate the sum.
        for var in ['fe_scav', 'fe_reg', 'fe_phy']:
            df[var] = ds[var].sum('obs', skipna=True, keep_attrs=True)  # Particle sum.
            df[var].attrs['long_name'] = 'Total ' + df[var].attrs['long_name']

        return df

    def map_var_to_particle_ids(self, ds, var='zone', var_array=range(len(zones._all)), file=None):
        """Create dictionary of particle IDs from each source.

        Args:
            ds (xarray.Dataset): Fe model particle dataset.

        Returns:
            pid_map (dict of lists): Dictionary of particle IDs.
                dict[zone] = list(traj)

        Example:
            source_traj = map_var_to_particle_ids(ds, var='zone', var_array=range(len(zones._all)),
                                                  file=pds.exp.file_source_map)
            pid_map = map_var_to_particle_ids(ds, var='z', var_array=range(25, 350 + 25, 25))

        Notes:
            Generic version of source_particle_ID_dict

        """
        if file is not None:
            if file.exists():
                return np.load(file, allow_pickle=True).item()

        # Dict[var[i]]: [Particle IDs where var <= to var[i] and > var[i-1]].
        pid_map = dict()

        # First value in var_array.
        traj = ds.traj.where((ds[var] <= var_array[0]), drop=True)
        traj = traj.values.astype(dtype=int).tolist()  # Convert to list.
        pid_map[var_array[0]] = traj

        # Remaining values in var_array.
        for i in range(1, len(var_array)):
            traj = ds.traj.where((ds[var] > var_array[i - 1]) & (ds[var] <= var_array[i]), drop=True)
            traj = traj.values.astype(dtype=int).tolist()  # Convert to list.
            pid_map[var_array[i]] = traj

        if file is not None and not test:
            np.save(file, pid_map)  # Save dictionary as numpy file.
        return pid_map

    def fe_model_sources(self, lon=None, version=None, add_diff=False):
        """Open fe model source dataset for a longitude (historical & RCP8.5)."""
        if lon is None:
            lon = self.exp.lon
        if version == None:
            version = self.exp.version
        # Open and concat data for each scenario.
        version = [version] * 2
        if version[0] == 2:
            version[0] = 0  # Use version 0 for historical comparison
        if version[0] == 5:
            version[0] = 4  # Use version 0 for historical comparison
        exps = [ExpData(scenario=i, lon=lon, version=version[i]) for i in range(2)]
        ds = [xr.open_dataset(exp.file_source, decode_timedelta=False) for exp in exps]
        ds = concat_exp_dimension(ds, add_diff=add_diff)

        ds['names'] = ('zone', zones.names)
        ds['colors'] = ('zone', zones.colors)

        # Update attributes for plots
        ds = self.add_variable_attrs(ds, post=True)
        for v in ds.data_vars:
            if 'fe_flux' in str(v):
                # Replace umol/m^2 s with nM Sv.
                ds[v] *= DXDY  # Convert from mass flux to volume flux.
                ds[v].attrs['units'] = 'nM Sv'

            for var in ['fe_scav_avg', 'fe_phy_avg', 'fe_reg_avg']:
                if var in str(v):
                    ds[v] *= 2  # Convert from nM per day to nM.
        return ds

    def fe_model_sources_all(self, version=None, add_diff=False):
        """Open fe model source dataset for all longitudes."""
        ds = [self.fe_model_sources(lon=i, version=version, add_diff=False) for i in self.release_lons]
        ds = [ds[i].expand_dims(dict(x=[x]), axis=1) for i, x in enumerate(self.release_lons)]
        ds = xr.concat(ds, 'x')

        ds = ds.drop(['names', 'colors'])
        for k in ds.data_vars:
            if 'long_name' in ds[k].attrs:
                ds[k].attrs['long_name'] = ds[k].attrs['long_name'].replace('Dissolved Iron', 'dFe')

        if add_diff:
            ds = xr.concat([ds, (ds.isel(exp=1, drop=True) - ds.isel(exp=0, drop=True)
                                 ).expand_dims(dict(exp=[2]))], 'exp')

        ds['names'] = ('zone', zones.names)
        ds['colors'] = ('zone', zones.colors)
        return ds
