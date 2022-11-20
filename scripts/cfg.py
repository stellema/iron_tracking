# -*- coding: utf-8 -*-
"""

Example:

Notes:
    - WOMBAT variables: http://cosima.org.au/index.php/working-groups/bgc/
    - adic, anthropogenic+pre-industrial DIC*, mmol m-32
    - dic, pre-industrial DIC*, mmol m-3
    - alk, total alkalinity, mmol m-3
    - o2, dissolved oxygen, mmol m-3
    - no3, nitrate, mmol m-3
    - fe, dissolved iron, umol m-3
    - phy, phytoplankton biomass, mmol N m-3
    - zoo, zooplankton biomass, mml N m-3
    - det, detritus, mmol N m-3
    - caco3, calcium carbonate, mmol m-3

Todo:
    - unzip zoo, det

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Oct 19 13:40:14 2022

"""
import sys
import string
import calendar
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple

np.set_printoptions(suppress=True)

# Setup directories.
if Path.home().drive == 'C:':
    home = Path.home()
    repo = home / 'Projects/felx/'
    ofam = home / 'datasets/OFAM/trop_pac'
    obs = home / 'datasets/obs'
else:
    home = Path('/g/data/e14/as3189')
    repo = home / 'stellema/plx'
    ofam = home / 'OFAM/trop_pac'
    obs = home / 'obs'

data, fig, log = [repo / s for s in ['data', 'plots', 'logs']]
sys.path.append(repo / 'scripts')

years = [[1981, 2012], [2070, 2101]]

# OFAM3 dimensions for NetcdfFileBuffer namemaps (chunkdims_name_map).
# chunkdims_name_map (opt.): gives a name map to the FieldFileBuffer that
# declared a mapping between chunksize name, NetCDF dimension and Parcels dimension;
# required only if currently incompatible OCM field is loaded and chunking is used by 'chunksize' (which is the default)
chunkdims_name_map = {'time': ['Time', 'time'],
                      'lon': ['lon', 'xu_ocean', 'xt_ocean'],
                      'lat': ['lat', 'yu_ocean', 'yt_ocean'],
                      'depth': ['depth', 'st_ocean', 'sw_ocean']}

bgc_vars = ['adic', 'alk', 'caco3', 'det', 'dic', 'fe', 'no3', 'o2', 'phy',
            'zoo']
bgc_name_map = {'P': 'phy', 'Z': 'zoo', 'Det': 'det', 'temp': 'temp',
                'Fe': 'fe', 'N': 'no3', 'DIC': 'dic'}

