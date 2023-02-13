# -*- coding: utf-8 -*-
"""Calculate BGC fields at particle positions.

Fields: T, Det, Z, P, N, Kd490, Fe (optional)

Notes:
    - Could sample background iron (OFAM or Huang et al.)
    - Saved reversed felx dataset.
    - Historical: 2000-2012 (12 years = ~4383 days)
    - RCP: 2087-2101 (12 years=2089-2101)
    - Initial year is a La Nina year (for spinup)
    - 85% of particles finish

Example:

Questions:
    - How to interpolate kd490 lon fields.

Runtime:
    - 100 particles:
        -update_ofam3_fields_AAA: 00h:04m:38.58s (total=278.58 seconds).
        update_kd490_field_AAA: 00h:15m:06.79s (total=906.79 seconds).
        get_felx_BGC_fields: 00h:19m:52.34s (total=1192.34 seconds).
    - 500 particles:
        - update_ofam3_fields_AAA: 00h:23m:33.38s (total=1413.38 seconds).
        - update_kd490_field_AAA: 01h:11m:27.42s (total=4287.42 seconds).
        - get_felx_BGC_fields: 01h:35m:10.64s (total=5710.64 seconds).
    - 900 particles:
        -
    - 1000 particles:
        - update_ofam3_fields_AAA: 00h:54m:15.27s (total=3255.27 seconds)
        ntraj = [100, 550, 900]
        t_ofam_update = [278.58, 1413.38, ]
        t_kd_update = [906.79, 5710.64, ]
        t_total = [1192.34, 5710.64, ]

TODO:
    * Parallelise openmfdataset
    * Parallelise file index
    * Estimate computational resources



@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Sat Jan 14 15:03:35 2023

"""
from argparse import ArgumentParser
import calendar
from datetime import datetime, timedelta  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # NOQA
import xarray as xr  # NOQA

import cfg
from cfg import paths, ExpData
from datasets import ofam3_datasets, save_dataset, BGCFields
from inverse_plx import inverse_plx_dataset
from tools import timeit, mlogger

logger = mlogger('particle_BGC_field_files')
test = True


class InversePlxDataSet(object):
    """Felx particle dataset."""

    def __init__(self, exp, variables):
        """Initialise & format felx particle dataset."""
        # self.filter_particles_by_start_time()
        self.exp = exp
        # self.ds = self.from_plx_inverse(self.exp)
        self.ds = inverse_plx_dataset(self.exp, decode_times=True, decode_cf=None)

        if self.exp.test:
            self.ds = self.ds.isel(traj=slice(550))
            # ds = ds.subset_min_time(time='2012-01-01'):
            self.ds = self.ds.dropna('obs', 'all')

        self.pids = self.particles_after_start_time()

        self.ds = self.ds.sel(traj=self.pids)
        self.valid_mask = ~np.isnan(self.ds.trajectory)
        # convert to days since 1979 (same as ofam3)
        self.ds['month'] = self.ds.time.dt.month
        # self.ds = self.override_spinup_particle_times(self.exp, self.ds)
        self.ds_time_orig = self.ds.time.copy()  # particle times before modifying
        self.ds['time'] = self.override_spinup_particle_times(self.ds.time)

        for var in variables:
            self.ds[var] = self.empty_DataArray()

    def empty_DataArray(self, name=None, fill_value=np.nan, dtype=np.float32):
        """Add empty DataArray."""
        return xr.DataArray(np.full((self.ds.traj.size, self.ds.obs.size), fill_value, dtype),
                            dims=('traj', 'obs'), coords=self.ds.coords, name=name)

    def particles_after_start_time(self):
        """Select particles that have reached a source at the start of the spinup period.

        Particles criteria:
            * Be released in the actual experiment time period.
            * At the source at least during the spinup period.

        Example:
            pids = ds.particles_after_start_time()
            ds = ds.sel(traj=pids)

        """
        mask_at_euc = self.ds.time.dt.year.max('obs') >= self.exp.time_bnds[0].year
        mask_at_source = self.ds.isel(obs=0, drop=True).time.dt.year >= self.exp.spinup_bnds[0].year
        mask = mask_at_source & mask_at_euc  # Both 1D mask along traj dim.

        pids = self.ds.traj.where(mask, drop=True)
        # ds = ds.sel(traj=keep_traj)
        return pids

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


@timeit(my_logger=logger)
def update_ofam3_fields_AAA(ds, fieldset, variables):
    """Calculate fields at plx particle positions (using apply_along_axis and xarray).

    Notes:
        - doesn't fail when wrong time selected.

    """
    def update_ofam3_field(ds, field):
        def update_field_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[~np.isnan(traj)][0].item())
            loc = dict(depth=dx.z, lat=dx.lat, lon=dx.lon)
            x = field.sel(loc, method='nearest').sel(time=dx.time, method='nearest')
            return x

        kwargs = dict(ds=ds, field=field)
        return np.apply_along_axis(update_field_particle, 1, arr=ds.trajectory, **kwargs)

    dims = ['traj', 'obs']
    for var in variables:
        # da = xr.DataArray(update_ofam3_field(ds, field=fieldset[var]), dims=dims)
        # ds[var] = da.pad(obs=(0, ds[var].obs.size - da.obs.size))
        ds[var] = (dims, update_ofam3_field(ds, field=fieldset[var]))
    return ds


@timeit(my_logger=logger)
def update_kd490_field_AAA(ds, fieldset, variables):
    """Calculate fields at plx particle positions dims=(month, lat, lon)."""
    def update_kd490_field(ds, field):
        def sample_field_particle(traj, ds, field):
            """Subset using entire array for a particle."""
            dx = ds.sel(traj=traj[~np.isnan(traj)][0].item())
            loc = dict(time=dx.month, lat=dx.lat, lon=dx.lon)
            return field.sel(loc, method='nearest', drop=True)
        return np.apply_along_axis(sample_field_particle, 1, ds.trajectory, ds=ds,
                                   field=field)

    for var in variables:
        ds[var] = (['traj', 'obs'], update_kd490_field(ds, field=fieldset[var]))
    return ds


@timeit(my_logger=logger)
def get_felx_BGC_fields(exp):
    """Sample OFAM3 BGC fields at particle positions.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.

    Returns:
        ds (xarray.Dataset): Particle trajectory dataset with BGC fields.

    Notes:
        - First save/create plx_inverse file.

    Todo:
        - Unzip OFAM3 fields [in progress]
        - Upload formatted Kd490 fields [in progress]a
        - Save plx reversed datasets
        - parallelise
        - add datset encoding [done]
        - Filter spinup?
        - Add kd490 fields
    """
    file = exp.file_felx_bgc
    if test:
        file = paths.data / ('test/' + file.stem + '.nc')

    logger.info('{}: Getting OFAM3 BGC fields.'.format(file.stem))

    # Select vairables
    fieldset = BGCFields(exp)
    fieldset.kd = fieldset.kd490_dataset()

    # Initialise particle dataset.
    pds = InversePlxDataSet(exp, variables=fieldset.variables)

    pds.ds = update_ofam3_fields_AAA(pds.ds, fieldset.ofam, fieldset.vars_ofam)
    pds.ds = update_kd490_field_AAA(pds.ds, fieldset.kd, fieldset.vars_clim)
    pds.ds = pds.ds.where(pds.valid_mask)

    pds.ds['time'] = pds.revert_spinup_particle_times()

    # Save file.
    # if not test:
    save_dataset(pds.ds, file, msg='Added BGC fields at paticle positions.')
    logger.info('felx BGC fields file saved: {}'.format(file.stem))

    return


if __name__ == '__main__':
    p = ArgumentParser(description="""Particle BGC fields.""")
    p.add_argument('-x', '--lon', default=165, type=int, help='Particle start longitude(s).')
    p.add_argument('-e', '--scenario', default=0, type=int, help='Scenario index.')
    p.add_argument('-r', '--index', default=0, type=int, help='File repeat.')
    p.add_argument('-v', '--version', default=0, type=int, help='PLX experiment version.')
    args = p.parse_args()

    if not cfg.test:
        scenario, lon, version, index = args.scenario, args.lon, args.version, args.index
    else:
        scenario, lon, version, index = 0, 190, 0, 0

    name = 'felx_bgc'
    exp = ExpData(scenario=scenario, lon=lon, version=version, file_index=index, name=name,
                  out_subdir=name, test=cfg.test)
    get_felx_BGC_fields(exp)
