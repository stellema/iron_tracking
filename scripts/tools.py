# -*- coding: utf-8 -*-
"""Useful functions.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 10 19:45:46 2023

"""
from datetime import datetime
from functools import wraps
import logging
import numpy as np
from pathlib import Path
import sys
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

import cfg


def mlogger(filename, level=logging.DEBUG):
    """Initialise logger that writes to a stream and file.

    Args:
        filename (str): File name for log output.
        level (int, optional): Logging level. Defaults to logging.DEBUG.

    Returns:
        logger (logging.Logger): Logger.

    Notes:
        - If filename is None, it uses the script/function name that called it.
        - Requires 'loggers = {}' in cfg.py
        - Creates log file in 'projectdir/logs/'
        - Should work with multiple current loggers.
        - Filters the logging of common warnings from parcels version 2.2.1

    """

    class NoWarnFilter(logging.Filter):
        """Filter logging of common warnings from parcels version 2.2.1."""

        def filter(self, record):
            show = True
            for key in ['Casting', 'Trying', 'Particle init', 'Did not find',
                        'Plot saved', 'field_chunksize', 'Compiled zParticle']:
                if record.getMessage().startswith(key):
                    show = False
            return show

    global loggers

    loggers = cfg.loggers

    # Name log outout file using calling function name.
    if filename is None:
        filename = __name__  # Function name.

    # Return existing logger if possible.
    if loggers.get(filename):
        return loggers.get(filename)

    # Create logger.
    logger = logging.getLogger(filename)

    # Create stream and file handlers.
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(cfg.paths.logs / '{}.log'.format(filename))

    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers.
    if MPI is None:
        c_format = logging.Formatter('{message}', style='{')
        f_format = logging.Formatter('{asctime}: {message}', '%Y-%m-%d %H:%M:%S', style='{')
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        fmt = '{asctime}:' + ': rank={: 2d}: '.format(rank) + '{message}'
        c_format = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S", style='{')
        f_format = c_format

        if rank > 0:
            # levels below 10 ignored (show debug)
            # change to 20 for other ranks (shows info not debug)
            level = logging.INFO

    if len(logger.handlers) != 0:
        logger.handlers.clear()

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger.
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(level)
    logger.addFilter(NoWarnFilter())

    # logger.propagate = False
    loggers[filename] = logger
    return logger


def timeit(_method=None, *, my_logger=None):
    """Timer decorator that logs elapsed time."""
    def decorator_timeit(method):
        """Wrap function to time method execution time."""
        @wraps(method)
        def timed(*args, **kwargs):
            # Timer.
            t_s = datetime.now()
            result = method(*args, **kwargs)
            t_e = datetime.now() - t_s

            # Format elpased time string.
            h, rem = divmod(t_e.total_seconds(), 3600)
            m, s = divmod(rem, 60)

            # Log elapsed time.
            if my_logger is None:
                logger = mlogger(method.__name__)
                logger = mlogger(Path(sys.argv[0]).stem)
            else:
                logger = my_logger
            logger.info('{}: {:02.0f}h:{:02.0f}m:{:05.2f}s (total={:.2f} seconds).'
                        .format(method.__name__, h, m, s, t_e.total_seconds()))
            return result
        return timed

    if _method is None:
        return decorator_timeit
    else:
        return decorator_timeit(_method)


def unique_name(name):
    """Create unique filename by appending numbers."""
    def append_file(name, i):
        return name.parent / '{}_{:02d}{}'.format(name.stem, i, name.suffix)
    i = 0
    while append_file(name, i).exists():
        i += 1
    return append_file(name, i)
