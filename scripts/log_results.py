# -*- coding: utf-8 -*-
"""Print source transport, etc.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue May  3 23:21:39 2022

"""
import scipy
import numpy as np
import xarray as xr
# import seaborn as sns

import cfg
from cfg import paths, ExpData, ltr, release_lons, scenario_name, scenario_abbr, zones
from fe_exp import FelxDataSet
from fe_obs_dataset import iron_source_profiles
from tools import mlogger
from stats import test_signifiance


def log_source_var(pds, ds, var, lon):
    """Log source variable (hist, change, etc) at longitude."""
    # Header.
    head = '{:>18}: '.format('{} v={} {}E'.format(var, pds.exp.version, str(lon)))
    names = ['HIST', 'D', '(D%)', 'p', 'sum_H%', 'sum_P%', 'pp']
    for i, n in enumerate(names):
        head += '{:^{s}}'.format(n, s=9 + min([0, 3 - i]))  # Add space between column names.
    logger.info(head)

    # Total (HIST, PROJ, Δ, Δ%, p-value).
    total = ds[var].sel(x=lon)
    pvalue = test_signifiance(total.isel(exp=0), total.isel(exp=1))
    total = total.mean('rtime')
    s = '{:>18}: '.format('total')
    s += '{:>6.3f} {: >8.4f} ({: >7.1%}) {:>9}'.format(*total[::2], total[2] / total[0], pvalue)
    logger.info(s)

    # Source (HIST, Δ, Δ%, p-value, HIST/HIST_total%, RCP/RCP_total%, diff percentage points).
    for z in ds.zone.values:
        dx = ds[var + '_src'].sel(x=lon, zone=z)
        pv = test_signifiance(dx[0], dx[1])
        dx = dx.mean('rtime')

        # Source percent of total EUC (HIST, PROJ, Δ percentage points).
        pct = [(dx.isel(exp=i) / total.isel(exp=i)) * 100 for i in [0, 1]]

        s = '{:>18}: '.format(ds.names.sel(zone=z).item())
        # Source Transport (HIST, HIST%, Δ, Δ%, pvalue).
        s += '{:>6.3f} {: >8.4f} ({: >7.1%}) {:>9}'.format(*dx[::2], dx[2] / dx[0], pv)

        # Source contribution percent hist, proj and change (i.e. makes up xpp more of total).
        s += ' {: >5.1f}% {: >4.0f}% {: >5.1f}%'.format(*pct, pct[1] - pct[0])
        logger.info(s)
    logger.info('')


if __name__ == '__main__':
    # Print lagrangian source transport values.
    for v in [0, 2, 3]:
        pds = FelxDataSet(ExpData(version=v))
        ds = pds.fe_model_sources_all(add_diff=True)
        logger = mlogger('results_v{}'.format(pds.exp.version))

        for var in ['u_sum', 'fe_avg', 'fe_src_avg', 'fe_flux_sum', 'fe_reg_avg', 'fe_scav_avg', 'fe_phy_avg']:
            for lon in release_lons:
                log_source_var(pds, ds, var, lon)
