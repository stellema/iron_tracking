# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:


@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Jan 18 08:01:16 2023

"""
import numpy as np
import xarray as xr
import pandas as pd

from cfg import ExpData
from tools import mlogger
from datasets import (ofam3_datasets, merge_interior_sources, concat_exp_dimension)
from inverse_plx import inverse_plx_dataset

logger = mlogger('files_plx')


def source_dataset(exp):
    """Get source datasets.

    Args:
        exp (cfg.ExpData): Experiment dataclass instance.

    Returns:
        ds (xarray.Dataset): Source dataset.

    Notes:
        - Stack scenario dimension
        - Add attributes
        - Change order of source dimension
        - merge sources
        - Changed ztime to time_at_zone (source time)
        - Changed z0 to z_f (source depth)

    """
    ds = xr.open_dataset(exp.file_plx_source)
    ds = merge_interior_sources(ds)
    # Reorder zones.
    inds = np.array([1, 2, 6, 7, 8, 3, 4, 5, 0])
    ds = ds.isel(zone=inds)
    return ds


scenario, lon = 1, 220
exp = ExpData(scenario=scenario, lon=lon)
ds = source_dataset(exp)
times = ds.time_at_zone.values[pd.notnull(ds.time_at_zone)]
traj = ds.traj[times >= np.datetime64(['2000-01-01', '2089-01-01'][exp])]
dx = ds.sel(traj=traj)

ni = ds.traj.size
nf = dx.traj.size
logger.info('lon={} exp={}: n_traj initial={}, remaining={} %={:.1%}'.format(lon, exp, ni, nf, (nf-ni)/ni))

# lon=165 exp=0: n_traj initial=537823, remaining=195704 %=-63.6%
# lon=165 exp=1: n_traj initial=623943, remaining=221389 %=-64.5%
# lon=190 exp=0: n_traj initial=559848, remaining=172132 %=-69.3%
# lon=190 exp=1: n_traj initial=596258, remaining=181411 %=-69.6%
# lon=220 exp=0: n_traj initial=493538, remaining=143472 %=-70.9%
# lon=220 exp=1: n_traj initial=505505, remaining=146257 %=-71.1%
# lon=250 exp=0: n_traj initial=408710, remaining=108892 %=-73.4%
# lon=250 exp=1: n_traj initial=401104, remaining=106889 %=-73.4%
