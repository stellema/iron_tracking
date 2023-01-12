# -*- coding: utf-8 -*-
"""Useful functions.

Notes:

Example:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Tue Jan 10 19:45:46 2023

"""
import inspect
import sys
import math
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta

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
        """"Filter logging of common warnings from parcels version 2.2.1."""
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
    f_handler = logging.FileHandler(cfg.log / '{}.log'.format(filename))

    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers.
    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s:%(message)s',
                                 "%Y-%m-%d %H:%M:%S")

    if len(logger.handlers) != 0:
        logger.handlers.clear()

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger.
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(NoWarnFilter())

    # logger.propagate = False
    loggers[filename] = logger
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
