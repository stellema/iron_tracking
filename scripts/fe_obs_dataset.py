# -*- coding: utf-8 -*-
"""Iron observational dataset and plot functions.

Opens and formats iron observational data from:
    - GEOTRACES IDP2021
    - Tagliabue et al. (2012) [Updated 2015]
    - Huang et al. (2022) data-driven climatology dataset (Machine Learning)

Notes:
    - Only works for pandas

Todo:
    - Replace/repeat dFe surface depths
    - Replace/repeat dFe lower
    - Remove coastal obs

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Nov  7 17:05:56 2022

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter
from dataclasses import dataclass, field
from datetime import datetime
import gsw
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import re
import xarray as xr

import cfg
from cfg import paths, mon
from tools import timeit, mlogger
from datasets import get_ofam_filenames, add_coord_attrs, save_dataset

logger = mlogger('iron_observations')


class FeObsDatasets():
    def __init__(self):
        self.lons = [110, 290]
        self.lats = [-30, 30]
        self.mesh = xr.open_dataset(str(paths.data / 'ofam_mesh_grid.nc'))

    def __repr__(self):
        """Return a representation of the object."""
        return ('{}'.format(self.__class__.__name__))

    def GEOTRACES_iron_dataset(self):
        """Open GEOTRACES idp2021 dataset.

        Todo:
            - Drop bad data np.unique(ds.var5)
            - Bin model depths?

        Notes:
            - renamed t, z, y, x
            - Renamed cruise to reference
            - removed var for kwargs
            - removed Round coords.
            - convert nmol/kg to nmol/m3
            - get density from pressure [1 dbar = 10 kPa], temp and salinity [PSS-1978 scale]

        """
        var = 'fe'
        folder = paths.obs / 'GEOTRACES/idp2021/seawater/'
        file = folder / 'GEOTRACES_IDP2021_Seawater_Discrete_Sample_Data_v1.nc'
        file_new = paths.obs / '{}_{}_formatted.nc'.format(file.stem, var)

        if file_new.exists():
            ds = xr.open_dataset(file_new)
            ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x'))
            return ds

        ds = xr.open_dataset(file)

        # Rename variables and add metadata from nc_variables.txt
        info = pd.read_csv(folder / 'nc_variables.txt', sep='\t').to_xarray()
        for v in info.data_vars:
            # Convert dtype to string and replace NaNs with ''.
            info[v] = xr.where(pd.notnull(info[v]), info[v], '').astype(str)
        info = info.where(info['nc Variable'] != '', drop=1)
        nmap2 = dict(date_time='time', var2='depth', latitude='lat', longitude='lon',
                     metavar1='ref', var88=var, var1='p', var15='temp',
                     var16='salt', var17='salt_bottle', var5='bottle_flag')
        for k, v in nmap2.items():
            info['Variable'] = info['Variable'].where(info['nc Variable'] != k, v)
        nmap = dict([[k, v] for k, v in zip(info['nc Variable'].values, info['Variable'].values)])
        ds = ds.rename(nmap)

        # Drop extra data variables.
        data_vars = [var, 'time', 'depth', 'lat', 'lon', 'ref', 'p', 'temp', 'salt']
        ds = ds.drop([v for v in ds.data_vars if v not in data_vars])

        # Format data types.
        ds['time'] = ds.time.astype(dtype='datetime64[D]')  # Remove H,M,S values from time.
        ds['ref'] = ds.ref.astype(dtype=str)

        # Drop NaNs.
        ds = ds.where(~np.isnan(ds[var]), drop=True)

        # Stack 'N_STATIONS' and 'N_SAMPLES'.
        ds = ds.stack({'index': ('N_STATIONS', 'N_SAMPLES')})
        ds = ds.sel(index=ds[var].dropna('index', 'all').index)
        ds = ds.drop_vars(['N_SAMPLES', 'N_STATIONS', 'index'])
        ds.coords['index'] = ds.index

        # Save to new filename.
        save_dataset(ds, file_new, msg='Modified {}'.format(file.name))
        return ds

    def GEOTRACES_iron_dataset_4D(self):
        """Open geotraces data, subset location and convert to multi-dim dataset."""
        # TODO: delete _SI from 'GEOTRACES_IDP2021_fe_trop_pacific_SI.nc'
        file = paths.obs / 'GEOTRACES_IDP2021_fe_trop_pacific.nc'
        if file.exists():
            ds = xr.open_dataset(file)
            ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x'))
            ds = ds.drop(['rho'])
            return ds

        var = 'fe'
        ds = self.GEOTRACES_iron_dataset()
        ds = ds.drop(['ref'])
        lons = [110, 290]
        lats = [-30, 30]

        # Subset lat/lons.
        ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                      (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)
        ds = ds.where(~np.isnan(ds[var]), drop=True)


        # Convert to multi-dim array.
        coords = [ds.t.values, ds.z.values, ds.y.values, ds.x.values]
        index = pd.MultiIndex.from_arrays(coords, names=['t', 'z', 'y', 'x'],)
        ds = ds.drop_vars({'index', 'x', 'y', 't', 'z'})
        ds = ds.assign_coords({'index': index})
        ds = ds.unstack('index')

        # Calculate density to convert fe units to nM from nmol/kg.
        for v in ['p', 'temp', 'salt']:
            # Fill NaNs with x,y,t mean (some temp and salt areas missing in west pacific.)
            ds[v] = ds[v].where(~np.isnan(ds[v]), ds[v].mean(['t', 'x', 'y']).ffill('z'))

        ds = ds.where(~np.isnan(ds[var]), drop=True)
        # Sea pressure (absolute pressure minus 10.1325 dbar)
        p = ds.p - 10.1325
        # Absolute Salinity from Practical Salinity [g/kg].
        SA = gsw.SA_from_SP(ds.salt, p, ds.x, ds.y)
        # Conservative Temperature of seawater from in-situ temperature [C].
        CT = gsw.CT_from_t(SA, ds.temp, p)
        # Density [kg/m]
        rho = gsw.rho(SA, CT, p)
        rho = rho.assign_attrs(long_name='Density', units='kg/m', comment='in-situ density')
        ds['rho'] = rho

        ds[var] = ds.rho.values * 1e-3 * ds[var]
        ds[var].attrs['units'] = 'nM'
        ds = ds.drop(['rho', 'p', 'temp', 'salt'])

        # Add data var attrs.
        ds = ds.rename(dict(t='time', z='depth', y='lat', x='lon'))
        ds = add_coord_attrs(ds)
        save_dataset(ds, file, msg='Created multi-dimensional tropical Pacific subset.')
        return ds

    def Tagliabue_iron_dataset(self):
        """Open Tagliabue 2015 iron [nM] dataset."""
        file = paths.obs / 'tagliabue_fe_database_jun2015_public_formatted.csv'
        file_new = paths.obs / '{}.nc'.format(file.stem)

        if file_new.exists():
            ds = xr.open_dataset(file_new)
            ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x', dFe='fe'))
            return ds

        ds = pd.read_csv(file)
        ds = pd.DataFrame(ds)
        ds = ds.to_xarray()

        # Rename & format coords.
        # Convert longituge to degrees east (0-360).
        ds['lon'] = ds.lon % 360
        ds = ds.where(ds.month > 0, drop=True)  # Remove single -99 month.

        ds = ds.where(~np.isnan(ds.dFe), drop=True)

        # Format data types.
        ds['Reference'] = ds['Reference'].astype(dtype=str)
        ds['time'] = ('index', pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(y, m)
                                               for y, m in zip(ds.year.values, ds.month.values)]))
        for var in ds.data_vars:
            if ds[var].dtype == 'float64':
                ds[var] = ds[var].astype(dtype=np.float32)

        # Convert full reference to in-text citation style.
        ds['ref'] = ('index', ds.Reference.values.copy())
        ds['ref'] = ds['ref'].astype(dtype=str)

        for i, r in enumerate(ds.Reference.values):
            ref = [s.replace(',', '').replace('.', '') for s in r.split(' ')]
            author = ref[0]
            year = re.findall('\d+', r)[0]  #

            if len(year) < 4:
                if author == 'Slemons':
                    year = 2009
                if author == 'Obata':
                    year = 2008

            if ref[3].isdigit():
                ds['ref'][i] = '{} ({})'.format(author, year)

            else:
                ds['ref'][i] = '{} et al. ({})'.format(author, year)

        ds = add_coord_attrs(ds)
        ds['dFe'].attrs['standard_name'] = 'dFe'
        ds['dFe'].attrs['long_name'] = 'Dissolved Iron'
        ds['dFe'].attrs['units'] = 'nM'

        # Save to new filename.
        save_dataset(ds, file_new, msg='Modified {}'.format(file.name))
        return ds

    def Tagliabue_iron_dataset_4D(self):
        """Open Tagliabue dataset, subset location and convert to multi-dim dataset."""
        file = paths.obs / 'tagliabue_fe_database_jun2015_trop_pacific.nc'
        if file.exists():
            ds = xr.open_dataset(file)
            ds = ds.rename(dict(time='t', depth='z', lat='y', lon='x', dFe='fe'))
            return ds

        # Subset to tropical ocean.
        ds = self.Tagliabue_iron_dataset()

        ds = ds.drop_vars(['month', 'year', 'ref', 'Reference'])
        lons = [110, 290]
        lats = [-30, 30]
        ds = ds.where((ds.y >= lats[0]) & (ds.y <= lats[1]) &
                      (ds.x >= lons[0]) & (ds.x <= lons[1]), drop=True)

        # Convert to multi-dimensional array.
        coords = [ds.t.values, ds.z.values, ds.y.values, ds.x.values]
        index = pd.MultiIndex.from_arrays(coords, names=['t', 'z', 'y', 'x'], )
        ds = ds.drop_vars(['t', 'z', 'y', 'x'])
        ds = ds.assign_coords({'index': index})
        # ds = ds.dropna('index')
        ds = ds.unstack('index')

        ds = ds.rename(dict(t='time', z='depth', y='lat', x='lon'))
        ds = add_coord_attrs(ds)
        save_dataset(ds, file, msg='Created multi-dimensional tropical Pacific subset.')
        return ds

    def log_iron_obs_dataset_information(self, name='GEOTRACES'):
        """Open Tagliabue dataset and get obs information."""
        # Subset to tropical ocean & format references.
        if name == 'GEOTRACES':
            ds = self.GEOTRACES_iron_dataset()
            ds = ds.where((ds.y >= self.lats[0]) & (ds.y <= self.lats[1]) &
                          (ds.x >= self.lons[0]) & (ds.x <= self.lons[1]), drop=True)
            refs = np.unique(ds.ref)

        elif name == 'Tagliabue':
            ds = self.Tagliabue_iron_dataset()
            ds = ds.where((ds.y >= self.lats[0]) & (ds.y <= self.lats[1]) &
                          (ds.x >= self.lons[0]) & (ds.x <= self.lons[1]), drop=True)
            t = pd.to_datetime(['{:.0f}-{:02.0f}-01'.format(ds.year[i].item(), ds.month[i].item())
                                for i in range(ds.index.size)])
            ds['t'] = ('index', t)
            refs = refs[np.argsort([re.findall('\d+', r)[0] for r in refs])]  # NOQA

        # Log observation information for each reference (time, lat, lon, depth ranges).
        for i, ref in enumerate(refs):
            dr = ds.where(ds.ref == ref, drop=True)

            # Reference location and time range.
            t = ', '.join(np.unique(dr.t.dt.strftime('%Y-%m')))
            z = '{:.0f}-{:.0f} m (N={})'.format(dr.z.min(), dr.z.max(), np.unique(dr.z).size)
            y = '{:.1f}°-{:.1f}° (N={})'.format(dr.y.min(), dr.y.max(), np.unique(dr.y).size)
            x = '{:.1f}-{:.1f}°E (N={})'.format(dr.x.min(), dr.x.max(), np.unique(dr.x).size)

            logger.info('{: >26}: N={: >3}, y={}, x={}, z={}, t={}'.format(ref, dr.index.size,
                                                                           y, x, z, t))

            # Lat, lon pairs and corresponding depth ranges.
            crd = np.zeros((dr.index.size, 2))
            for j in range(dr.index.size):
                dj = dr.isel(index=j)
                crd[j] = [dj.y, dj.x]
            crd = np.unique(crd, axis=0)

            crds = crd[:, 0].astype(dtype=str)
            zz = crd[:, 0].astype(dtype=str)
            tt = crd[:, 0].astype(dtype=str)

            for j in range(crd.shape[0]):
                dj = dr.where((dr.y == crd[j, 0]) & (dr.x == crd[j, 1]), drop=True)
                d = 'S' if crd[j, 0] < 0 else 'N'
                crds[j] = ('{:.1f}°{}, {:.1f}°E'
                           .format(np.fabs(crd[j, 0]), d, crd[j, 1]))

                zz[j] = ('{:.0f}-{:.0f} m'.format(dj.z.min(), dj.z.max()))

                tt[j] = ', '.join(np.unique(dj.t.dt.strftime('%b/%Y')))

            if crds.size < 20:
                logger.info('\n'.join(crds))
                logger.info('\n'.join(zz))
                logger.info('\n'.join(tt))
        return ds.dFe

    @timeit(my_logger=logger)
    def interpolate_depths(self, ds, method='slinear'):
        """Interpolate observation depths to OFAM3 depths."""
        def valid_coord_indexes(dx, dim):
            """Dimension indexes with non NaN elements."""
            dim_valid = dx.dropna(dim, 'all')[dim]
            indexes = dx[dim].searchsorted(dim_valid)
            return indexes

        def interpolate_depth_array(dx, ds_interp, loc):
            """Interpolate along depths indexes with non NaN elements."""
            if dx.z.size >= 3:
                z_new_subset = ds_interp.z.copy().sel(z=slice(*[dx.z[a].item() for a in [0, -1]]))
                dx = dx.interp(z=z_new_subset, method=method)
            else:
                z_new_subset = ds_interp.z.copy().sel(z=dx.z.values, method='nearest')
                dx.coords['z'] = z_new_subset

            ds_interp[loc] = dx.broadcast_like(ds_interp[loc])
            return ds_interp

        # Drop any non-NaN dims (speeds up performance if data was subset).
        for v in ['t', 'z', 'y', 'x']:
            ds = ds.dropna(v, how='all')

        # Create all-NaN obs data array with OFAM3 depths.
        ds_interp = (ds.copy() * np.nan).interp(z=self.mesh.st_edges_ocean.values)

        # Interpolate depths at each point (only iterates through non-NaN dim elements).
        for t in range(ds.t.size):
            loc = dict(t=t)
            for y in valid_coord_indexes(ds.isel(loc), dim='y'):
                loc = dict(t=t, y=y)
                for x in valid_coord_indexes(ds.isel(loc), dim='x'):
                    loc = dict(t=t, y=y, x=x)
                    dx = ds.isel(loc).dropna('z')

                    if dx.z.size > 0:
                        # Interpolate depths (unless all NaN).
                        ds_interp = interpolate_depth_array(dx, ds_interp, loc)
        return ds_interp

    def Huang_iron_dataset(self):
        """Huang et al. (2022).

        dFe climatology based on compilation of dFe observations + environmental predictors
        from satellite observations and reanalysis products using machine-learning approaches
        (random forest).

        """
        file = paths.obs / 'Huang_et_al_2022_monthly_dFe_V2.nc'
        ds = xr.open_dataset(file)
        ds = ds.rename({'Longitude': 'x', 'Latitude': 'y', 'Depth': 'z', 'Month': 't',
                        'dFe_RF': 'fe'})
        ds.coords['x'] = xr.where(ds.x < 0, ds.x + 360, ds.x, keep_attrs=True)
        ds = ds.sortby(ds.x)
        ds.t.attrs['units'] = 'Month'

        for v, n, u in zip(['y', 'x', 'z'], ['Latitude', 'Longitude', 'Depth'],
                           ['°', '°', 'm']):
            ds = ds.sortby(ds[v])
            ds[v].attrs['long_name'] = n
            ds[v].attrs['standard_name'] = n.lower()
            ds[v].attrs['units'] = u
        return ds

    @timeit(my_logger=logger)
    def combined_iron_obs_datasets(self, add_Huang=False, interp_z=True,
                                   pad_uniform_grid=False, lats=[-15, 15], lons=[110, 290]):
        """Create combined multi-dim iron observation datasets.

        Merged datasets:
            * GEOTRACES
            * Tagliabue et al. (2012)
            * [Optional] Huange et al. (2022)
            * [Optional] Evenly spaced (0.5 deg lat x 0.5 deg lon)

        Args:
            lats (list, optional): Latitude subset range. Defaults to [-15, 15].
            lons (list, optional): Longitude subset range. Defaults to [110, 290].
            pad_uniform_grid (bool, optional): Create evenly spaced NaN grid. Defaults to False.
            add_Huang (bool, optional): Merge data with the Huang et al. . Defaults to False.

        Returns:
            ds_merged (xarray.Dataset): Dataset with merged

        Notes:
            * Interpolates to OFAM3 depth grid.
            * Slice depth AFTER interpolating depths.
            * Interpolating depths takes around takes ~8 mins for Huang dataset
            * 00h:08m:38.18s: using pad_uniform_grid=False, add_Huang=True
            * Resulting grid won't be totally even (requires 0.1 deg spacing - too big).

        """
        def format_obs_dataset(ds, name):
            """Format (subset & interpolate) observation dataset."""
            # Subset to tropical ocean.
            ds = ds.sel(y=slice(*lats), x=slice(*lons))

            # Interpolate depths.
            if interp_z:
                ds = self.interpolate_depths(ds)

            # Drop/rename vars and convert to Datasets (must be after interpolate).
            if not isinstance(ds, xr.Dataset):
                ds = ds.to_dataset(name=name)

            # Drop any non-NaN dims (speeds up performance if data was subset).
            for v in ['t', 'z', 'y', 'x']:
                ds = ds.dropna(v, how='all')

            # Convert dtype to float32 (except for times).
            for var in list(ds.variables):
                if ds[var].dtype not in ['float32', 'datetime64[ns]']:
                    ds[var] = ds[var].astype('float32')
            return ds

        file = paths.obs / 'fe_dataset_combined{}.nc'.format('_all' if add_Huang else '')
        if file.exists():
            ds = xr.open_dataset(file)
            return ds

        name = 'fe'

        dss = []
        dss.append(self.GEOTRACES_iron_dataset_4D().fe)
        dss.append(self.Tagliabue_iron_dataset_4D().fe)

        if add_Huang:
            dss.append(self.Huang_iron_dataset().fe)
            dss[-1] = dss[-1].isel(t=slice(12))
            dss[-1]['t'] = pd.to_datetime(['2022-{:02d}-01'.format(i + 1) for i in range(12)])

        dss = [format_obs_dataset(ds_, name) for ds_ in dss]

        ds_merged = xr.merge(dss)

        if pad_uniform_grid:
            # Evenly spaced lat and lon (single level time and depth arrays).
            y = np.arange(lats[0], lats[-1] + 0.5, 0.5, dtype=np.float32)
            x = np.arange(lons[0], lons[-1] + 0.5, 0.5, dtype=np.float32)
            z = np.array([dss[1].z.values[0]])
            t = np.array([dss[1].t.values[0]])

            coords = dict(t=('t', t), z=('z', z), y=('y', y), x=('x', x))
            temp = xr.DataArray(np.full((1, 1, y.size, x.size), np.nan),
                                dims=list(coords.keys()), coords=coords)
            temp = temp.to_dataset(name=name)

            ds_merged = xr.merge([ds_merged, temp])
        save_dataset(ds_merged, file, msg='Not finished version')

        return ds_merged


class FeObsDataset(object):
    """Iron Observation Dataset class."""
    def __init__(self, ds):

        self.init_plot_params()
        self.init_obs_site_dict()

        # Set datasets.
        self.ds = ds
        if 't' in self.ds.dims:
            self.ds.mean('t')
        self.ds_avg = self.obs_sites_avg()

    def __repr__(self):
        """Return a representation of the object."""
        return ('{}'.format(self.__class__.__name__))

    def init_plot_params(self):
        """Add various plot paramaters."""
        # Plot colors.
        self.colors = ['wheat', 'rosybrown', 'darkred', 'peru', 'red', 'darkorange', 'salmon',
                       'gold', 'olive', 'lime', 'g', 'aqua', 'b', 'teal', 'dodgerblue', 'indigo',
                       'm', 'hotpink', 'grey']

        # Plot linestyles.
        self.lss = ['-', '--', ':', '-.', (0, (1, 10)), (0, (5, 10)), (0, (1, 1)), (0, (5, 1)),
                    (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
                    (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]

    def init_obs_site_dict(self):
        """Add dict of lat lon ranges that contain observations."""
        c = 0.2
        obs_site = {}
        obs_site['mc'] = dict(y=[4, 9.2], x=[118, 135], name='Mindanao Current', c='mediumspringgreen')
        obs_site['ngcu'] = dict(y=[-6, -3], x=[140, 150], name='VS+PNG', c='darkorange')
        obs_site['nicu'] = dict(y=[-5.5, -3], x=[152, 157], name='SS+NICU', c='pink')

        obs_site['vs'] = dict(y=[-6, -5], x=[145, 150], name='Vitiaz Strait', c='darkorange')
        obs_site['ss'] = dict(y=[-5.5, -3], x=[151, 156], name='Solomon Strait', c='deeppink')
        obs_site['ni'] = dict(y=[-4, -3], x=[152, 156], name='New Ireland', c='pink')
        obs_site['png'] = dict(y=[-6.6, -6.1], x=[152, 153], name='PNG', c='r')
        obs_site['ssea'] = dict(y=[-4.5, -3], x=[140, 145], name='Solomon Sea', c='y')

        # Specific observation sites.
        obs_site['mcx'] = dict(y=[7-c, 7+c], x=[130-c, 130+c], name='Mindanao Current', c='mediumspringgreen')
        obs_site['mc5x'] = dict(y=[5.2-c, 5.2+c], x=[125-c, 127+c], name='Mindanao Current', c='mediumspringgreen')
        obs_site['vsx'] = dict(y=[-3.3-c, -3.3+c], x=[144-c, 144+c], name='Vitiaz Strait', c='darkorange')
        obs_site['ssx'] = dict(y=[-5-c, -5+c], x=[155-c, 155+c], name='Solomon Strait', c='deeppink')

        obs_site['int_s'] = dict(y=[-10, -2.5], x=[170, 290], name='South Interior', c='darkviolet')
        obs_site['int_n'] = dict(y=[2.5, 10], x=[138, 295], name='North Interior', c='b')
        obs_site['eq'] = dict(y=[-0.5, 0.5], x=[145, 295], name='Equator', c='k')
        obs_site['euc'] = dict(y=[-2.6, 2.6], x=[145, 295], name='Equatorial Undercurrent', c='k')
        # obs_site['int'] = dict(y=[obs_site['int_s']['y'], obs_site['int_n']['y']],
        #                             x=[obs_site['int_s']['x'], obs_site['int_n']['x']],
        #                             name='Interior', c='seagreen')

        self.obs_site = obs_site

    def get_obs_locations(self, name):
        """Get iron observation lat & lon positions in observation data array.

        Args:
            name (str): name of obs_site region to slice.

        Returns:
            dx (xarray.DataArray): Data subset in region.
            obs (list): List of [lat, lon] pairs where there is an obs.
            coords (list): List of formatted "(lat°S, lon°E)" coords of obs.

        """
        # Define lists of lat & lon subsets based on given name.
        if name in self.obs_site.keys():
            lats, lons = [[self.obs_site[name][i]] for i in ['y', 'x']]

        elif name in ['int']:
            lats, lons = [[self.obs_site[n][i] for n in ['int_n', 'int_s']] for i in ['y', 'x']]

        elif name in ['llwbcs']:
            lats, lons = [[self.obs_site[n][i] for n in ['mc', 'vs', 'ss']] for i in ['y', 'x']]

        # Subset & merge dataset.
        dx = []
        for y, x in zip(lats, lons):
            dx.append(self.ds.fe.sel(y=slice(*y), x=slice(*x)))

        dx = xr.merge(dx).fe

        if 't' in dx.dims:
            dx = dx.mean('t')

        # Search subset for lat, lons where there are observations.
        obs = []
        for x in dx.x.values:
            for y in dx.y.values:
                dxx = dx.sel(x=x, y=y).dropna('z', 'all')
                if dxx.size > 0:
                    obs.append([y, x])

        # Format string of lat, lon coordinates.
        coords = ['({:.1f}°E, {:.1f}°{})'.format(x, np.fabs(y), 'N' if y > 0 else 'S')
                  for x, y in obs]

        return dx, obs, coords

    def sel_site(self, names):
        """Get X-Y location of obs."""
        dx = []
        for n in names:
            dx.append(self.ds.fe.sel(y=slice(*self.obs_site[n]['y']),
                                     x=slice(*self.obs_site[n]['x'])))
            return xr.merge(dx).fe

    def obs_sites_avg(self):
        """Create dataset based on the average of dFe at each obs sites as a function of depth."""
        df = xr.Dataset()
        for n in self.obs_site.keys():
            df[n] = self.sel_site([n]).mean(['x', 'y'])

            # Fix top depths
            z_inds = np.arange(df[n].mean('t').z.size, dtype=int)
            n_obs = (~np.isnan(self.sel_site([n]).mean('t'))).sum('x').sum('y')
            try:
                z_valid = z_inds[n_obs > 2][0]
            except IndexError:
                z_valid = z_inds[n_obs >= 1][0]
            for z in range(z_valid):
                if n_obs[z] < 1:
                    df[n][dict(z=z)] = df[n][dict(z=z_valid)]

        for n, nn in zip(['interior', 'llwbcs'], [['int_n', 'int_s'], ['mc', 'vs', 'ss']]):
            df[n] = self.sel_site(nn).mean(['x', 'y'])
            # Fix top depths
            z_inds = np.arange(df[n].mean('t').z.size, dtype=int)
            n_obs = (~np.isnan(self.sel_site(nn).mean('t'))).sum('x').sum('y')
            try:
                z_valid = z_inds[n_obs > 2][0]
            except IndexError:
                z_valid = z_inds[n_obs >= 1][0]

            for z in range(z_valid):
                if n_obs[z] < 1:
                    df[n][dict(z=z)] = df[n][dict(z=z_valid)]

        df['EUC'] = self.sel_site(['euc']).mean(['y'])
        df = df.dropna('z', 'all')
        return df


def get_merged_FeObsDataset():
    dfs = FeObsDatasets()
    setattr(dfs, 'ds', dfs.combined_iron_obs_datasets(add_Huang=False, interp_z=True))
    setattr(dfs, 'ds_all', dfs.combined_iron_obs_datasets(add_Huang=True, interp_z=True))
    setattr(dfs, 'dfe', FeObsDataset(dfs.ds))
    setattr(dfs, 'dfe_all', FeObsDataset(dfs.ds_all))

    interior_vars = ['int_s', 'int_n', 'eq', 'euc', 'interior']
    for n in ['eq', 'euc']:
        var = 'euc'
        dfs.dfe.ds_avg[n] = xr.where(~np.isnan(dfs.dfe.ds_avg[n]), dfs.dfe.ds_avg[n],
                                     dfs.dfe_all.ds_avg[var].mean('t'))
    for n in [n for n in dfs.dfe.ds_avg.data_vars if n not in interior_vars]:
        var = 'llwbcs'
        dfs.dfe.ds_avg[n] = xr.where(~np.isnan(dfs.dfe.ds_avg[n]), dfs.dfe.ds_avg[n],
                                     dfs.dfe_all.ds_avg[var].mean('t'))
    for n in interior_vars:
        var = 'interior'
        dfs.dfe.ds_avg[n] = xr.where(~np.isnan(dfs.dfe.ds_avg[n]), dfs.dfe.ds_avg[n],
                                     dfs.dfe_all.ds_avg[var].mean('t'))
    return dfs
