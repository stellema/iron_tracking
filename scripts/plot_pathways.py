# -*- coding: utf-8 -*-
"""

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Jul 31 01:12:48 2023

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
import warnings

from cfg import paths, release_lons, ExpData, zones
from fe_exp import FelxDataSet
from stats import weighted_bins_fd, get_min_weighted_bins, get_source_transit_mode
from tools import mlogger, timeit

warnings.filterwarnings('ignore')


def subset_array(a, N):
    """Subset array to N linearly arranged unique elements."""
    return np.unique(np.array(a)[np.linspace(0, len(a) - 1, N, dtype=int)])


def create_map_axis(fig, ax, extent=[120, 285, -10, 10], ocean_color='azure',
                    land_color=cfeature.COLORS['land_alt1'], xticks=None, yticks=None):
    """Create a figure and axis with cartopy."""
    zorder = 10  # Placement of features (reduce to move to bottom.)
    proj = ccrs.PlateCarree()
    proj._threshold /= 20.

    # Set map extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent(extent, crs=proj)

    # Features.
    if land_color is not None:
        land = cfeature.NaturalEarthFeature('physical', 'land', '10m', ec='k',
                                            fc=cfeature.COLORS['land'])
        ax.add_feature(land, fc=land_color, lw=0.2, zorder=zorder + 1)

    # ax.add_feature(cfeature.LAND, color=cfeature.COLORS['land_alt1'], zorder=zorder + 1)
    # ax.add_feature(cfeature.COASTLINE, zorder=zorder + 1)
    if ocean_color is not None:
        ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=1)

    try:
        # Make edge frame on top.
        ax.outline_patch.set_zorder(zorder + 1)  # Depreciated.
    except AttributeError:
        ax.spines['geo'].set_zorder(zorder + 1)  # Updated for Cartopy

    # Set ticks.
    if xticks is None and yticks is None:
        xticks = np.arange(*extent[:2], 20)
        yticks = ax.get_yticks()

    ax.set_xticks(xticks, crs=proj)
    ax.set_xticklabels(['{}°E'.format(i) for i in xticks])
    ax.set_yticks(yticks, crs=proj)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # # Minor ticks.
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.subplots_adjust(bottom=0.2, top=0.8)
    ax.set_aspect('auto')
    return fig, ax, proj


def plot_particle_source_map(add_labels=True, savefig=True,
                             add_lon_lines=release_lons, **kwargs):
    """Plot a map of particle source boundaries.

    Args:
        add_labels (bool, optional): Add source names. Defaults to True.
        savefig (bool, optional): Save the figure. Defaults to True.

    Returns:
        fig, ax, proj

    """
    def get_source_coords():
        """Latitude and longitude points defining source region lines.

        Returns:
            lons (array): Longitude coord pairs (start/end).
            lats (array): Latitude coord pairs (start/end).
            z_index (array): Source IDs cooresponding to coord pairs.

        """
        lons = np.zeros((10, 2))
        lats = lons.copy()
        z_index = np.zeros(lons.shape[0], dtype=int)
        i = 0
        for iz, z in enumerate(zones._all[1:]):
            loc = [z.loc] if type(z.loc[0]) != list else z.loc
            for c in loc:
                z_index[i] = iz + 1
                lats[i] = c[0:2]  # Lon (west, east).
                lons[i] = c[2:4]  # Lat (south, north).
                i += 1
        return lons, lats, z_index

    def add_source_labels(ax, z_index, labels, proj):
        """Add source location names to map."""
        # Source names to plot (skips 'None').
        text = [z.name_full for z in zones._all[1:]]

        # Split names into two lines & centre align (excluding interior).
        text[0] = text[0][:-3]  # Strait -> Str.
        text[1] = text[1][:-3]  # Strait -> Str.
        text[2] = 'Mindanao\n Current'
        text[3] = 'Celebes\n   Sea'
        text[4] = 'Indonesian Seas'
        text[5] = 'Solomon Is.'

        # Locations to plot text.
        loc = np.array([[-6.4, 147.75], [-5.6, 152], [9.2, 127],  # VS, SS, MC.
                        [3.7, 120.7], [-9, 126], [-6.2, 157],  # CS, IDN, SC.
                        [-6.6, 192], [8.9, 192]])  # South & north interior.

        phi = np.full(loc.size, 0.)  # rotation.
        for i in [0, 1]:
            phi[i] = 282
        phi[5] = 300

        # Add labels to plot.
        for i, z in enumerate(np.unique(z_index)):
            ax.text(loc[i][1], loc[i][0], text[i], fontsize=8, rotation=phi[i],
                    weight='bold', ha='left', va='top', zorder=12,
                    transform=proj)
            # bbox=dict(fc='w', edgecolor='k', boxstyle='round', alpha=0.7)
        return ax

    # Draw map.
    extent = [117, 285, -10, 10]
    yticks = np.arange(-8, 9, 4)
    xticks = np.arange(120, 290, 20)

    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    fig, ax, proj = create_map_axis(fig, ax, extent=extent, xticks=xticks, yticks=yticks, **kwargs)

    # Plot lines between each lat & lon pair coloured by source.
    lons, lats, z_index = get_source_coords()
    for i, z in enumerate(z_index):
        ax.plot(lons[i], lats[i], zones.colors[z], lw=3, label=zones.names[z],
                zorder=10, transform=proj)  # zorder above ocean & below land

    if add_lon_lines is not False:
        for i, x in enumerate(add_lon_lines):
            ax.plot([x, x], [-2.6, 2.6], 'k', lw=2, zorder=10, transform=proj)

    for i, z in enumerate(z_index):
        ax.plot(lons[i], lats[i], zones.colors[z], lw=3, label=zones.names[z],
                zorder=10, transform=proj)  # zorder above ocean & below land
    if add_labels:
        ax = add_source_labels(ax, z_index, zones.names, proj)
    plt.tight_layout()

    # Save.
    if savefig:
        plt.savefig(paths.figs / 'particle_source_map.png', bbox_inches='tight', dpi=300)

    return fig, ax, proj


@timeit
def plot_source_pathways(exp, N_total=300, age='mode', add_lon_lines=release_lons):
    """Plot a subset of pathways on a map, colored by source.

    Args:
        exp (int): Scenario Index.
        lon (int): Release longitude.
        v (int): version.
        r (int): File number.
        N_total (int, optional): Number of pathways to plot. Defaults to 300.
        age (str, optional): Particle age {'mode', 'min', 'any', 'max'}.
        xlines (list, optional): Plot EUC lines at lon. Defaults to False.

    """
    print('Plot x={} exp{} N={} r={} age={}'.format(exp.lon, exp.scenario, N_total,
                                                    exp.file_index, age))

    reps = list(range(min(3, 8 - exp.file_index)))
    zids = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    pdss = [FelxDataSet(ExpData(scenario=exp.scenario, lon=exp.lon, version=exp.version,
                                file_index=exp.file_index + i)) for i in reps]
    pds = pdss[0]
    ds = xr.open_mfdataset([pdss[i].exp.file_felx for i in reps])

    # Source dataset (hist & RCP for bins).
    dss = pds.fe_model_sources_all(add_diff=0).sel(x=pds.exp.lon)
    ds['zone'] = ds.zone.isel(obs=0, drop=True).load()

    # Divide by source based on percent of total.
    dn = dss.isel(exp=exp.scenario).dropna('traj', 'all')
    dn = dn.sel(zone=np.array(zids))
    pct = dn.u_sum_src.mean('rtime') / dn.u_sum.mean('rtime')
    N = np.ceil(pct * 100) / 100 * N_total
    N = np.ceil(N).values.astype(dtype=int)

    # Particle IDs dict.
    pid_dict = pds.map_var_to_particle_ids(ds, var='zone', var_array=zids)
    # Subset particles based on age method.
    if age == 'any':
        # Get any N particle IDs per source region.
        traj = np.concatenate([subset_array(pid_dict[z], n)
                               for z, n in zip(zids, N)])

    # Find mode transit time pids.
    elif age in ['mode', 'min', 'max']:
        traj = []
        for z, n in zip(zids, N):
            mode = get_source_transit_mode(dss, 'age', exp.scenario, z)
            dz = dn.sel(zone=z).dropna('traj', 'all')
            dz = dz.sel(traj=dz.traj[dz.traj.isin(pid_dict[z])])

            if age == 'mode':
                mask = (dz.age >= mode[0]) & (dz.age <= mode[-1])
                pids = dz.traj.where(mask, drop=True)
                if pids.size < n and z == 5:
                    mode = [get_source_transit_mode(dss, 'age', exp.scenario, z, 0.4),
                            get_source_transit_mode(dss, 'age', exp.scenario, z, 0.6)]
                    mask = (dz.age >= mode[0]) & (dz.age <= mode[-1])
                    pids = dz.traj.where(mask, drop=True)

            elif age == 'min':
                # lim = dz.age.min('traj').item() + 2 * np.diff(mode)
                lim = get_source_transit_mode(dss, 'age', exp.scenario, z, 0.1)
                pids = dz.traj.where((dz.age <= lim), drop=True)

            elif age == 'max':
                lim = get_source_transit_mode(dss, 'age', exp.scenario, z, 0.9)
                pids = dz.traj.where((dz.age >= lim), drop=True)

            # Get N particle IDs per source region.
            traj.append(subset_array(pids, n))

        if not all([t.size for t in traj] == N):
            print('Particle size error', [t.size for t in traj], N)
        traj = np.concatenate(traj)

    # Particle trajectory data.

    # Define line colors for each source region & broadcast to ID array size.
    c = np.array(zones.colors)[zids]
    c = np.concatenate([np.repeat(c[i], N[i]) for i in range(N.size)])

    # Indexes to shuffle particle pathway plotting order (less overlap bias).
    shuffle = np.random.permutation(len(traj))

    # Plot particles.
    fig, ax, proj = plot_particle_source_map(add_lon_lines=add_lon_lines, savefig=False,
                                             ocean_color='lightcyan', land_color='darkgrey')
    for i in shuffle:
        dx = ds.sel(traj=traj[i])
        ax.plot(dx.lon, dx.lat, c[i], lw=0.5, alpha=0.2, zorder=10, transform=proj)

    plt.tight_layout()
    plt.savefig(paths.figs / 'pathway_{}_{}_n{}.png'.format(pds.exp.id, age, N_total),
                bbox_inches='tight', dpi=350)
    plt.show()
    return


if __name__ == "__main__":
    exp = ExpData(scenario=0, lon=250, version=0, file_index=5)
    pds = FelxDataSet(exp)
    plot_source_pathways(exp, N_total=1000, age='mode', add_lon_lines=release_lons)
