# -*- coding: utf-8 -*-
"""Create animated scatter plot.

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Nov 5 16:00:08 2019

"""
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cfg
from tools import (get_unique_file, timeit)
from fncs import get_plx_id, get_index_of_last_obs
from plots import (plot_particle_source_map, update_title_time, create_map_axis,
                   get_data_source_colors)
from create_source_files import source_particle_ID_dict

# Fixes AttributeError: 'GeoAxesSubplot' object has no attribute 'set_offsets'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

plt.rc('animation', html='html5')
logger = logging.getLogger('my-logger')
logger.propagate = False


@timeit
def init_particle_data(ds, ntraj, ndays, method='thin', start=None, zone=None,
                       partial_paths=True):
    """Subset dataset by number of particles, observations and start date.

    Args:
        ds (xarray.Dataset): plx dataset.
        ntraj (int): Number of particles to subset.
        ndays (int or None): Number of particle observations (None=all obs).
        method (str, optional): Subset method ('slice', 'thin'). Defaults to 'thin'.
        start (str, optional): Start date. Defaults to '2012-12-31T12'.
        zone (None or list, optional): Subset by a source id. Defaults to None.
        partial_paths (bool, optional): Only return completed paths. Defaults to False.

    Returns:
        ds (xarray.Dataset): Subset plx dataset.
        plottimes (TYPE): DESCRIPTION.

    Notes:
        - ds requires dataset filename stem (i.e., ds.attrs['file']).
        - Method='slice': ntraj is the total particles (e.g., slice(1000))
        - Method='thin': thin by 'ntraj' particles (e.g., ds.traj[::4])
        - time will be subset as ndays from start date (possible bug for r > 0)

    Todo:
        - select start time

    """
    # Update dataset attrs with subset info.
    ds.attrs['zone'] = zone
    ds.attrs['ntraj'] = ntraj
    ds.attrs['ndays'] = ndays
    ds.attrs['file'] = '{}_p{}_t{}'.format(ds.attrs['file'], ntraj, ndays)

    # Drop unnecessary data variables.
    ds = ds.drop({'age', 'distance', 'unbeached'})

    # Time coordinate.
    if start is None:
        start = ds.time.isel(obs=0, traj=0).values
    else:
        start = np.datetime64(start)
    days = ds.obs.size - 1 if ndays is None else ndays
    runtime = np.timedelta64(days, 'D')  # Time span to animate.
    end = start - runtime

    # Subset particles by source.
    if zone is not None:
        print('Zone')
        pids_all = []
        pids = [source_particle_ID_dict(ds, ds.attrs['exp'], ds.attrs['lon'], 1, R)
                for R in range(ds.attrs['r'], 10)]
        for z in list(zone):
            pids_all.append(np.concatenate([p[z] for p in pids]))
        ds = ds.where(ds.traj.isin(np.concatenate(pids_all)), drop=True)
        if 'obs' in ds['zone'].dims:
            ds['zone'] = ds.zone.isel(obs=0)

    if not partial_paths:
        # want trajectories that wil have finished before animation ends.
        # exclude partciles with non-nan positions earlier than animation end
        # find time of last position & exclude if time is before cutoff
        dt = ds.time.to_pandas().min(axis=1)
        traj_keep = ds.traj[(dt > end).values]
        ds = ds.sel(traj=traj_keep)

    # Subset the amount of particles.
    if isinstance(ntraj, int):
        if method == 'slice':
            ds = ds.isel(traj=slice(ntraj))
        elif method == 'thin':
            ds = ds.thin(dict(traj=int(ntraj)))

    # Subset times from start to  subset (slow).
    if ndays is not None:
        ds = ds.where((ds.time <= start) & (ds.time > end), drop=True)

    # Create array of timesteps to plot.
    dt = 2  # Days between frames (output saved every two days in files).
    dt = -np.timedelta64(int(dt*24*3600*1e9), 'ns')
    plottimes = np.arange(start, end, dt)
    return ds, plottimes


def update_lines_2D(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (current & last few positions).

    Simulaneous particle release:
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    """
    title.set_text(update_title_time('', plottimes[t]))

    # Find particle indexes at t & last few t (older=greater time).
    inds = (times >= plottimes[t]) & (times < plottimes[t - dt])

    # Indexes of particles/line to plot.
    P = np.arange(lats.shape[0], dtype=int)[np.any(inds, axis=1)]

    # Iterate through each particle & plot.
    for p in P:
        # Plot line segment (current & last few positions).
        # Delayed particle release.
        lines[p].set_data(lons[p][inds[p]], lats[p][inds[p]])

    return lines


def update_lines_no_delay(t, lines, lons, lats, times, plottimes, title, dt):
    """Update trajectory line segments (current & last few positions).

    Simultaneous particle release:
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    """
    title.set_text(update_title_time('', plottimes[t]))

    # Find particle indexes at t & last few t (older=greater time).
    inds = (times >= plottimes[t]) & (times < plottimes[t - dt])

    # Indexes of particles/line to plot.
    P = np.arange(lats.shape[0], dtype=int)[np.any(inds, axis=1)]

    # Iterate through each particle & plot.
    for p in P:
        # Plot line segment (current & last few positions).
        # Simulaneous particle release.
        lines[p].set_data(lons[p, t-dt:t], lats[p, t-dt:t])
    return lines


def animate_particle_scatter(ds, plottimes, delay=True):
    """Animate particle trajectories as black dots.

    Args:
        ds (xarray.Dataset): DESCRIPTION.
        plottimes (TYPE): DESCRIPTION.
        delay (bool, optional): Delayed or instant particle release. Defaults to True.

    Notes:
        - Add release longitude line add_lon_lines=[ds.attrs['lon']]
        - Alternate map:
            fig, ax, proj = create_map_axis((12, 5), [120, 288, -12, 12])

    """
    # Define particle data variables (slow).
    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)

    # Setup figure.
    fig, ax, proj = plot_particle_source_map(add_lon_lines=False)

    # Plot first frame.
    t = 0
    b = plottimes[t] == times

    if delay:
        X, Y = lons[b], lats[b]
    else:  # Non-delay release.
        X, Y = lons[:, t], lats[:, t]

    img = ax.scatter(X, Y, c='k', s=5, marker='o', zorder=20, transform=proj)

    title_str = ''
    title = plt.title(update_title_time(title_str, plottimes[t]))
    plt.tight_layout()

    def animate(t):
        title.set_text(update_title_time(title_str, plottimes[t]))
        b = plottimes[t] == times

        if delay:
            X, Y = lons[b], lats[b]

        else:
            # Non-delay release.
            X, Y = lons[:, t], lats[:, t]

        img.set_offsets(np.c_[X, Y])
        # fig.canvas.draw()
        return img,

    frames = np.arange(1, len(plottimes))

    anim = animation.FuncAnimation(fig, animate,  frames=frames, interval=1200,
                                   blit=True, repeat=False)

    # Save.
    file = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(ds.attrs['file']))
    writer = animation.writers['ffmpeg'](fps=18)
    anim.save(str(file), writer=writer, dpi=300)


@timeit
def animate_particle_lines(ds, plottimes, dt=4, delay=True, forward=False, source_map=False):
    """Animate trajectories as line segments (current & last few positions).

    Args:
        ds (TYPE): DESCRIPTION.
        plottimes (numpy.ndarray(dtype='datetime64[ns]'): Array of times.
        dt (int, optional): Number of trailing positions for particle line. Defaults to 4.
        delay (bool, optional): Release at time (delay) or obs index. Defaults to True.
        forward (bool, optional): Animate particles forward in time. Defaults to False.

    Notes:
        - Backgroup map without boundaries:
            fig, ax, proj = create_map_axis((12, 5), extent=[120, 288, -12, 12])

        - Simulaneous release:
            lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0]
                     for p in N]
        - bug saving files with zone
    """
    # Number of particles.
    N = range(ds.traj.size)

    # Data variables (slow).
    lons = np.ma.filled(ds.variables['lon'], np.nan)
    lats = np.ma.filled(ds.variables['lat'], np.nan)
    times = np.ma.filled(ds.variables['time'], np.nan)

    # Particle line colours.
    colors = get_data_source_colors(ds)
    if all(colors == colors[0]):
        # Make all lines black.
        colors = np.full_like(colors, 'b')

    # Index of plottimes to animate as frames (accounting for trailing positions).
    frames = np.arange(dt, len(plottimes) - dt)

    # Reverse frames for forward-tracked particles.
    if forward:
        # lons = np.flip(lons, axis=1)
        # lats = np.flip(lats, axis=1)
        # times = np.flip(times, axis=1)
        frames = frames[::-1]
        dt *= -1

    # Setup figure.    figsize = (13, 5.8)
    if source_map:
        fig, ax, proj = plot_particle_source_map(savefig=False, add_lon_lines=False)
    else:
        fig, ax, proj = create_map_axis((13, 5.8), extent=[117, 285, -11, 11],
                                        yticks=np.arange(-10, 11, 5),
                                        xticks=np.arange(120, 290, 20))

    kwargs = dict(lw=0.45, alpha=0.5, transform=proj)

    # Plot first frame.
    t = 0
    if delay:
        func = update_lines_2D
        lines = [ax.plot(lons[p, t], lats[p, t], c=colors[p], **kwargs)[0] for p in N]
    else:
        func = update_lines_no_delay
        lines = [ax.plot(lons[p, :dt], lats[p, :dt], c=colors[p], **kwargs)[0] for p in N]

    title = plt.title(update_title_time('', plottimes[t]), fontsize=16)
    plt.tight_layout()

    # Animate.
    fargs = (lines, lons, lats, times, plottimes, title, dt)
    anim = animation.FuncAnimation(fig, func, frames=frames, fargs=fargs, blit=True,
                                   interval=800, repeat=False)

    # Save animation.
    file = get_unique_file(cfg.fig / 'vids/{}.mp4'.format(ds.attrs['file']))
    writer = animation.writers['ffmpeg'](fps=12)
    anim.save(str(file), writer=writer, dpi=300)


rlon = 250
exp, v, r = 0, 1, 6
file = [get_plx_id(exp, rlon, v, r, 'plx') for r in range(r, r+2)]
ds = xr.open_mfdataset(file)
ds.attrs['lon'] = rlon
ds.attrs['exp'] = exp
ds.attrs['r'] = r
ds.attrs['file'] = file[0].stem

# Default plot: (ndays=2400 ntraj=3).
ds, plottimes = init_particle_data(ds, ntraj=2, ndays=2500, zone=None)
animate_particle_lines(ds, plottimes, delay=True, forward=False, source_map=False)

# ds, plottimes = init_particle_data(ds, ntraj=1, ndays=2500, zone=[1, 2, 3], partial_paths=False)
# animate_particle_lines(ds, plottimes, dt=4, delay=True, forward=False)

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=3600, zone=None)
# animate_particle_scatter(ds, plottimes, delay=True)

# ds, plottimes = init_particle_data(ds, ntraj=None, ndays=1200, zone=[2])
# animate_particle_lines(ds, plottimes, forward=False)
