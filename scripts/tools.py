# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:45:46 2023

@author: a-ste
"""
import sys
import math
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta


import cfg


def get_datetime_bounds(exp, test_months=2):
    """Get list of experiment start and end year datetimes."""
    y = cfg.years[exp]
    times = [datetime(y[0], 1, 1), datetime(y[1], 12, 31)]

    if cfg.home.drive == 'C:':
        times[0] = datetime(y[1], 1, 1)
        times[1] = datetime(y[1], test_months, 29)
    return times


def get_ofam_filenames(var, times):
    """Create OFAM3 file list based on selected times."""
    f = []
    for y in range(times[0].year, times[1].year + 1):
        for m in range(times[0].month, times[1].month + 1):
            f.append(str(cfg.ofam / 'ocean_{}_{}_{:02d}.nc'.format(var, y, m)))
    return f


def mlogger(name):
    """Create a logger.

    Args:
        name (str): Log file name.
        parcels (bool, optional): Import parcels logger. Defaults to False.

    Returns:
        logger: logger.

    """
    global loggers

    loggers = cfg.loggers

    class NoWarnFilter(logging.Filter):
        def filter(self, record):
            show = True
            for key in ['Casting', 'Trying', 'Particle init', 'Did not find',
                        'Plot saved', 'field_chunksize', 'Compiled zParticle']:
                if record.getMessage().startswith(key):
                    show = False
            return show

    # logger = logging.getLogger(__name__)
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(cfg.log / '{}.log'.format(name))
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(message)s')
        f_format = logging.Formatter('%(asctime)s:%(message)s',
                                     "%Y-%m-%d %H:%M:%S")

        if len(logger.handlers) != 0:
            logger.handlers.clear()

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.setLevel(logging.DEBUG)
        logger.addFilter(NoWarnFilter())

        # logger.propagate = False
        loggers[name] = logger
    return logger


def timeit(method):
    """Wrap function to time method execution time."""
    @wraps(method)
    def timed(*args, **kw):
        logger = mlogger(Path(sys.argv[0]).stem)
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        h, rem = divmod((te - ts).total_seconds(), 3600)
        m, s = divmod(rem, 60)

        logger.info('{}: {:}:{:}:{:05.2f} total: {:.2f} seconds.'
                    .format(method.__name__, int(h), int(m), s,
                            (te - ts).total_seconds()))

        return result

    return timed
