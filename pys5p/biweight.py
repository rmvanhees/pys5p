"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implement Tukey's biweight algorithm

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import division

import numpy as np

def __corr_std(nval):
    """
    Corrects the biweight spread to an unbaised estimator for the standard
    deviation, unther the assumption of an uncontaminated normal distribution.
    """
    return 0.9909 + (0.5645 + 2.805 / nval) / nval

def biweight(data, axis=None, spread=False, std=False):
    """
    Calculate Tukey's biweight.
    Implementation based on Eqn. 7.6 and 7.7 in the SWIR OCAL ATBD.

    Parameters
    ----------
    data   :   array_like
       Input array.
    axis   :   int, optional
       Axis along which the biweight medians are computed.
    spread :   bool
       If True, then return the biweight spread
    std :   bool
       If True, then return an approximation of the unbiased standard deviation
    Returns
    -------
    out    :   ndarray
       biweight median and biweight spread if function argument "spread" is True
    """
    sbi = 0.
    if std:
        spread = True

    # define lambda function to return only median or (median, spread)
    out_parms = lambda xbi, sbi, both: (xbi, sbi) if both else xbi

    if axis is None:
        xx = data[np.isfinite(data)]
        if xx.size == 0:
            return out_parms(np.nan, 0., spread)
        med_xx = np.median(xx)
        deltas = xx - med_xx
        med_dd = np.median(np.abs(deltas))
        if med_dd == 0:
            return out_parms(med_xx, 0., spread)

        wmx = np.maximum(0, 1 - (deltas / (6 * med_dd)) ** 2) ** 2
        xbi = med_xx + np.sum(wmx * deltas) / np.sum(wmx)
        if spread:
            umn = np.minimum(1, (deltas / (9 * med_dd)) ** 2)
            sbi = np.sum(deltas ** 2 * (1 - umn) ** 4)
            sbi /= np.sum((1 - umn) *  (1 - 5 * umn)) ** 2
            sbi = np.sqrt(len(xx) * sbi)
            if std:
                sbi *= __corr_std(len(xx))
    else:
        if np.all(np.isnan(data)):
            shape = np.isfinite(data, axis=axis).shape
            return out_parms(np.full(shape, np.nan),
                             np.zeros(shape, np.nan),
                             spread)

        med_xx = np.nanmedian(data, axis=axis)
        xbi = med_xx
        sbi = np.zeros_like(med_xx)

        deltas = data - np.expand_dims(med_xx, axis=axis)
        med_dd = np.nanmedian(np.abs(deltas), axis=axis)

        indices = (med_dd == 0)
        if not np.all(indices):
            med_dd[indices] = 1.  # dummy value
            med_dd = np.expand_dims(med_dd, axis=axis)
            wmx = np.maximum(0, 1 - (deltas / (6 * med_dd)) ** 2) ** 2
            xbi[~indices] = (med_xx + np.sum(wmx * deltas, axis=axis)
                             / np.sum(wmx, axis=axis))[~indices]
            if spread:
                umn = np.minimum(1, (deltas / (9 * med_dd)) ** 2)
                len_xx = np.sum(np.isfinite(data), axis=axis)
                buff = np.sum(deltas ** 2 * (1 - umn) ** 4, axis=axis)
                buff /= np.sum((1 - umn) *  (1 - 5 * umn), axis=axis) ** 2
                if std:
                    buff = __corr_std(len_xx) * np.sqrt(len_xx * buff)
                else:
                    buff = np.sqrt(len_xx * buff)
                sbi[~indices] = buff[~indices]

    return out_parms(xbi, sbi, spread)
