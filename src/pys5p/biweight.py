"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implementation of the Tukey's biweight algorithm

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import warnings

import numpy as np


# ----- local functions -------------------------
def __biweight(data, spread=False):
    """
    Calculate biweight median and spread, for whole dataset
    """
    mask = np.isfinite(data)
    if np.all(~mask):
        if spread:
            return (np.nan, 0.)

        return np.nan

    # calculate biweight median
    data_median = np.nanmedian(data)
    delta = data - data_median
    delta_median = np.nanmedian(np.abs(delta))
    if delta_median == 0.:
        if spread:
            return (data_median, 0.)

        return data_median

    wmx = np.clip(1 - (delta / (6 * delta_median)) ** 2, 0, None) ** 2
    biweight_median = data_median + np.nansum(wmx * delta) / np.nansum(wmx)
    if not spread:
        return biweight_median

    # calculate biweight spread
    umn = np.clip((delta / (9 * delta_median)) ** 2, None, 1)
    buff = np.nansum(delta ** 2 * (1 - umn) ** 4)
    buff /= np.nansum((1 - umn) * (1 - 5 * umn)) ** 2
    biweight_spread = np.sqrt(np.sum(mask) * buff)

    return (biweight_median, biweight_spread)


def __biweight_axis(data, axis, spread=False):
    """
    Calculate only biweight median, along a given axis
    """
    mask = np.isfinite(data)
    all_nan = np.all(np.isnan(data), axis=axis)

    # calculate biweight median
    data_median = np.nanmedian(data, axis=axis, keepdims=True)
    delta = data - data_median
    delta_median = np.nanmedian(np.abs(delta), axis=axis, keepdims=True)
    _mm = delta_median != 0.
    delta_median[~_mm] = np.nan

    wmx = np.clip(1 - (delta / (6 * delta_median)) ** 2, 0, None) ** 2
    biweight_median = np.squeeze(data_median)
    _mm = np.squeeze(_mm) & ~all_nan
    biweight_median[_mm] += \
        np.nansum(wmx * delta, axis=axis)[_mm] / np.nansum(wmx, axis=axis)[_mm]
    if not spread:
        return biweight_median

    # calculate biweight spread
    umn = np.clip((delta / (9 * delta_median)) ** 2, None, 1)
    buff = np.nansum(delta ** 2 * (1 - umn) ** 4, axis=axis)[_mm]
    buff /= np.nansum((1 - umn) * (1 - 5 * umn), axis=axis)[_mm] ** 2
    biweight_spread = np.full(biweight_median.shape, np.nan)
    biweight_spread[_mm] = np.sqrt(np.sum(mask, axis=axis)[_mm] * buff)

    return (biweight_median, biweight_spread)


# ----- main function -------------------------
def biweight(data, axis=None, spread=False):
    """
    Calculate Tukey's biweight.
    Implementation based on Eqn. 7.6 and 7.7 in the SWIR OCAL ATBD.

    Parameters
    ----------
    data   :   array_like
       input array
    axis   :   int, optional
       axis along which the biweight medians are computed.
        - Note that axis will be ignored when data is a 1-D array.
    spread :   bool, optional
       if True, then return also the biweight spread.

    Returns
    -------
    out    :   ndarray
       biweight median and biweight spread if function argument "spread" is True
    """
    if axis is None or data.ndim == 1:
        return __biweight(data, spread)

    # only a single axis!
    if not isinstance(axis, int):
        raise TypeError('axis not an integer')
    if not 0 <= axis < data.ndim:
        raise ValueError('axis out-of-range')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return __biweight_axis(data, axis, spread)
