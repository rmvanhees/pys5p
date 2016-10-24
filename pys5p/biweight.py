'''
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

Implement Tukey's biweight algorithm

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

'''
from __future__ import division

import numpy as np

def biweight(data, axis=None, spread=False):
    """
    Calculate Tukey's biweight.
    Implementation based on Eqn. 7.6 and 7.7 in the SWIR OCAL ATBD.

    """
    if axis is None:
        xx = data[data.isfinite()]
        med_xx = np.median(xx)
        deltas = xx - med_xx
        med_dd = np.median(np.abs(deltas))
        if med_dd == 0:
            xbi = med_xx
            if spread:
                sbi = 0.
        else:
            wmx = np.maximum(0, 1 - (deltas / (6 * med_dd)) ** 2) ** 2
            xbi = med_xx + np.sum(wmx * deltas) / np.sum(wmx)
            if spread:
                umn = np.minimum(1, (deltas / (9 * med_dd)) ** 2)
                sbi = np.sum(deltas ** 2 * (1 - umn) ** 4)
                sbi /= np.sum((1 - umn) *  (1 - 5 * umn)) ** 2
                sbi = np.sqrt(len(xx) * sbi)
    else:
        med_xx = np.nanmedian(data, axis=axis)
        xbi = med_xx
        sbi = np.zeros( med_xx.shape, dtype=np.float64 )

        deltas = data - np.expand_dims( med_xx, axis=axis )

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
                sbi[~indices] = np.sqrt(len_xx * buff)[~indices]

    if spread:
        return (xbi, sbi)
    else:
        return xbi
