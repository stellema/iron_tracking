# -*- coding: utf-8 -*-
"""Iron Lagrangian Experiment (FeLX) configuration file.

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

Example:

Todo:
    - unzip zoo, det
    - replace exp with era
    @ TODO: replace args in plx_particle_dataset

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Wed Oct 19 13:40:14 2022

"""
import calendar
import dask
import pathlib
import string
import sys
import warnings
import xarray as xr

# from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np

xr.set_options(keep_attrs=True)
dask.config.set({"array.slicing.split_large_chunks": True})

# Filter warnings.
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# warnings.filterwarnings('ignore', message='SerializationWarning')
warnings.filterwarnings('ignore', message='RuntimeWarning')
np.set_printoptions(suppress=True)
# np.seterr(invalid='ignore')


# Dictonaries and Constants.
test = True if pathlib.Path.home().drive == 'C:' else False
loggers = {}
# exp_abr = ['hist', 'rcp']
ltr = [i + ')' for i in list(string.ascii_lowercase)]
mon = list(calendar.month_abbr)[1:]  # Month abbreviations.


# Sverdrup.
SV = 1e6

# Radius of Earth [m].
EARTH_RADIUS = 6378137

# Metres in 1 degree of latitude [m].
LAT_DEG = (2 * np.pi / 360) * EARTH_RADIUS


def LON_DEG(lat):
    """Metres in 1 degree of latitude [m]."""
    return LAT_DEG * np.cos(np.radians(lat))


DXDY = 25 * 0.1 * LAT_DEG / 1e6

# OFAM3 dimensions for NetcdfFileBuffer namemaps (chunkdims_name_map).
# chunkdims_name_map (opt.): gives a name map to the FieldFileBuffer that
# declared a mapping between chunksize name, NetCDF dimension and Parcels dimension;
# required only if currently incompatible field is loaded and 'chunksize' is used.
chunkdims_name_map = {'time': ['Time', 'time'],
                      'lon': ['lon', 'xu_ocean', 'xt_ocean'],
                      'lat': ['lat', 'yu_ocean', 'yt_ocean'],
                      'depth': ['depth', 'st_ocean', 'sw_ocean']}
bgc_name_map = {'P': 'phy', 'Z': 'zoo', 'Det': 'det', 'temp': 'temp',
                'Fe': 'fe', 'N': 'no3', 'DIC': 'dic'}

bgc_vars = ['adic', 'alk', 'caco3', 'det', 'dic', 'fe', 'no3', 'o2', 'phy', 'zoo']


@dataclass
class ProjectPaths():
    """Define paths to find data or save output."""

    home: pathlib.Path

    def __post_init__(self):
        """Set path names for home PC or Gadi."""
        if self.home.drive == 'C:':
            self.repo = self.home / 'Projects/felx/'
            self.plx = self.home / 'Projects/plx/data/'
            self.ofam = pathlib.Path('D:') / 'datasets/OFAM/trop_pac'
            self.obs = pathlib.Path('D:') / 'datasets/obs'
        else:
            self.home = pathlib.Path('/g/data/e14/as3189')
            self.repo = self.home / 'stellema/felx'
            self.plx = self.home / 'stellema/plx/data/'
            self.ofam = self.home / 'OFAM/trop_pac'
            self.obs = self.home / 'obs'

        self.data = self.repo / 'data'
        self.figs = self.repo / 'plots'
        self.logs = self.repo / 'logs'


paths = ProjectPaths(pathlib.Path.home())
sys.path.append(paths.repo / 'scripts')


@dataclass
class ExpData:
    """Data class for experiment paramaters.

    Attributes:
        scenario (int): Scenario integer (0, 1).
        lon (int): Release longitude (165, 190, 220, 250).
        name (str, optional): Experiment name. Defaults to felx.
        test (bool, optional). Defaults to False.
        spinup_year_offset (int, optional): Years after start for spinup. Defaults to 4.
        spinup_num_years (int, optional): Number of years to spinup. Defaults to 10.
        out_subdir (pathlib.Path, optional): Defaults to 'felx'.
        test_month_bnds (int, optional): Test months of OFAM3 fields. Defaults to 3.

        Generated
        scenario_name (str):
        year_bnds (list of int):
        time_bnds (list of datetime.datetime):
        test_time_bnds (list of datetime.datetime):
        spinup_year (int):
        spinup_bnds (list of datetime.datetime):

        out_dir (pathlib.Path):
        file_base (str): e.g., '<exp>_<lon>_v<version>_<file_index>'
        file_felx (pathlib.Path): data/<out_subdir>/<name>_<exp>_<lon>_v<version>_<file_index>.nc
        file_felx_bgc (pathlib.Path): data/felx/felx_bgc_<exp>_<lon>_v<version>_<file_index>.nc
        file_plx_inv (pathlib.Path): 'data/felx/felx_<exp>_<lon>_v<version>_<file_index>_inverse.nc
        file_plx (pathlib.Path): ../plx/data/plx/plx_<exp>_<lon>_v1r<file_index>.nc
        file_plx_source (pathlib.Path): ../plx/data/source/plx_<exp>_<lon>_v1.nc

    """

    scenario: int
    lon: int = field(default=165)
    name: str = field(default='fe')
    version: int = field(default=0)
    file_index: int = field(default=0)
    test: bool = field(default=False)
    spinup_year_offset: int = field(default=4)
    spinup_num_years: int = field(default=10)
    out_subdir: str = field(default='')
    test_month_bnds: int = field(default=3)
    # Iron model params
    scav_eq: str = field(default='Galibraith')
    ntraj: int = field(default=100)

    def __post_init__(self):
        """Modify or change property values after the object has been initialized."""
        self.lon_str = '{}°E'.format(self.lon)
        self.scenario_name = ['Historical', 'RCP8.5'][self.scenario]
        self.scenario_abbr = ['hist', 'rcp'][self.scenario]

        # Time period.
        self.year_bnds = [[2000, 2012], [2089, 2101]][self.scenario]
        self.time_bnds = [datetime(self.year_bnds[0], 1, 1),
                          datetime(self.year_bnds[1], 12, 31)]
        self.test_time_bnds = [datetime(2012, 1, 1), datetime(2012, min(12, self.test_month_bnds), 28)]
        if self.test_month_bnds > 12:
            self.test_time_bnds[0] = self.test_time_bnds[1] - timedelta(days=30) * self.test_month_bnds
        if self.test:
            self.time_bnds = self.test_time_bnds

        self.spinup_year = self.year_bnds[0] + self.spinup_year_offset
        self.spinup_bnds = [datetime(self.year_bnds[0] - self.spinup_num_years, 1, 1),
                            datetime(self.year_bnds[1] - self.spinup_num_years, 12, 31)]
        # Data file names.
        self.file_index_orig = int(np.ceil((self.file_index - 1) / 2))
        self.out_dir = paths.data / 'fe_model'

        self.out_subdir = self.out_dir / self.out_subdir

        self.file_base = '{}_{}_v{}_{:02d}'.format(self.scenario_abbr, self.lon, self.version,
                                                   self.file_index)

        self.file_felx = self.out_subdir / '{}_{}.nc'.format(self.name, self.file_base)
        self.file_felx_tmp_dir = self.out_subdir / 'tmp_{}_{}/'.format(self.name, self.file_base)
        self.file_felx_bgc = self.out_dir / 'felx_bgc_{}.nc'.format(self.file_base)
        self.file_felx_bgc_tmp = self.out_dir / 'felx_bgc_{}_tmp.nc'.format(self.file_base)
        self.file_plx = paths.data / 'plx/plx_{}_{}_v1_{:02d}.nc'.format(self.scenario_abbr,
                                                                         self.lon, self.file_index)
        self.file_plx_inv = paths.data / 'plx/{}_inverse.nc'.format(self.file_plx.stem)
        self.file_plx_orig = paths.plx / 'plx/{}r{:02d}.nc'.format(self.file_plx.stem[:-3],
                                                                   self.file_index_orig)
        self.file_plx_source = paths.plx / 'sources/{}.nc'.format(self.file_plx.stem[:-3])


@dataclass
class ZoneData:
    """Pacific Ocean Zones."""

    Zone = namedtuple("Zone", "id name name_full loc")
    Zone.__new__.__defaults__ = (None,) * len(Zone._fields)
    nz = Zone(0, 'nz', 'None')
    vs = Zone(1, 'vs', 'Vitiaz Strait', [-6.1, -6.1, 147, 149.7])
    ss = Zone(2, 'ss', 'Solomon Strait', [-5.1, -5.1, 151.3, 154.7])
    mc = Zone(3, 'mc', 'Mindanao Current', [8, 8, 125.9, 129.1])
    cs = Zone(4, 'cs', 'Celebes Sea', [[0.45, 8, 120.1, 120.1],
                                       [8, 8, 120.1, 125]])
    idn = Zone(5, 'idn', 'Indonesian Seas', [[-8.5, 0.4, 120.1, 120.1],
                                             [-8.7, -8.7, 120.1, 142]])
    sc = Zone(6, 'sc', 'Solomon Islands', [-6.1, -6.1, 155.2, 158])
    sth = Zone(7, 'sth', 'South Interior', [-6.1, -6.1, 158, 283])
    nth = Zone(8, 'nth', 'North Interior', [8, 8, 129.1, 278.5])

    sgc = Zone(99, 'sgc', 'St Georges Channel', [-4.6, -4.6, 152.3, 152.7])
    ssx = Zone(99, 'ssx', 'Solomon Strait', [-4.8, -4.8, 153, 154.7])

    def __post_init__(self):
        """Source data."""
        self.lons = [165, 190, 220, 250]
        self.inner_lons = [[158, *self.lons, 280], [128.5, *self.lons, 278.5], ]
        self._all = [self.nz, self.vs, self.ss, self.mc, self.cs, self.idn,
                     self.sc, self.sth, self.nth]

        self.colors = ['silver', 'darkorange', 'deeppink', 'mediumspringgreen',
                       'seagreen', 'y', 'red', 'darkviolet', 'blue']
        self.names = np.array([z.name_full for z in self._all])

        self.inner_names = ['{} ({}-{}°E)'.format(z.name_full, *self.inner_lons[i][x:x+2])
                            for i, z in enumerate([self.sth, self.nth]) for x in range(5)]

        self.names_all = np.concatenate([self.names[:-2], self.inner_names])

        self.colors_all = np.concatenate([self.colors[:-2],
                                          *[[self.colors[i]] * 5 for i in [-2, -1]]])


zones = ZoneData()
