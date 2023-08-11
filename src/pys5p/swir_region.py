# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""Return the usable area on the SWIR detector.

There are two definitions::

 'illuminated':
    Detector area illuminated by external sources, defined as
    a rectangular area where the signal is at least 50% of the
    maximum signal. Coordinates: rows [11:228], columns [16:991].

 'level2':
    A smaller area used in official SWIR level 1B (ir)radiance
    products. Coordinates: rows [12:227], columns [20:980].

Notes
-----
Row 257 of the SWIR detector is neglected.
"""
__all__ = ['coords', 'mask']

import numpy as np


def coords(mode: str = 'illuminated', band: str = '78') -> tuple[slice, slice]:
    """Return slice defining the illuminated region on the SWIR detector.

    Parameters
    ----------
    mode  :   {'illuminated', 'level2'}, optional
       default is 'illuminated'
    band  :   str, optional
       select band 7 or 8, default is both bands
    """
    if mode == 'level2':
        if band == '7':
            return np.s_[12:227, 20:500]
        if band == '8':
            return np.s_[12:227, :480]
        # else
        return np.s_[12:227, 20:980]

    if band == '7':
        return np.s_[11:228, 16:500]
    if band == '8':
        return np.s_[11:228, :491]
    # else
    return np.s_[11:228, 16:991]


def mask(mode: str = 'illuminated', band: str = '78') -> np.ndarray:
    """Return mask of the illuminated region, where the value of
    the illuminated pixels are set to True.

    Parameters
    ----------
    mode  :   {'illuminated', 'level2'}, optional
       default is 'illuminated'
    band  :   str, optional
       select band 7 or 8, default is both bands
    """
    if band in ('7', '8'):
        res = np.full((256, 500), False)
    else:
        res = np.full((256, 1000), False)

    res[coords(mode, band)] = True

    return res
