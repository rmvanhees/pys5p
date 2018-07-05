"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implement Tukey's biweight algorithm

Copyright (c) 2016--2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import division

import numpy as np


# ----- local functions -------------------------
def __biweight_median(data):
    """
    Calculate only biweight median
    """
    mask = np.isfinite(data)
    if np.all(~mask):
        return np.nan

    xx = data[mask]
    med_xx = np.median(xx)
    deltas = xx - med_xx
    med_dd = np.median(np.abs(deltas))
    if med_dd == 0:
        return med_xx

    wmx = np.maximum(0, 1 - (deltas / (6 * med_dd)) ** 2) ** 2

    return med_xx + np.sum(wmx * deltas) / np.sum(wmx)


def __biweight(data):
    """
    Calculate biweight median and spread
    """
    mask = np.isfinite(data)
    if np.all(~mask):
        return (np.nan, 0.)

    xx = data[mask]
    med_xx = np.median(xx)
    deltas = xx - med_xx
    med_dd = np.median(np.abs(deltas))
    if med_dd == 0:
        return (med_xx, 0.)

    wmx = np.maximum(0, 1 - (deltas / (6 * med_dd)) ** 2) ** 2
    xbi = med_xx + np.sum(wmx * deltas) / np.sum(wmx)
    umn = np.minimum(1, (deltas / (9 * med_dd)) ** 2)
    sbi = np.sum(deltas ** 2 * (1 - umn) ** 4)
    sbi /= np.sum((1 - umn) * (1 - 5 * umn)) ** 2
    sbi = np.sqrt(len(xx) * sbi)
    return (xbi, sbi)


# ----- main function -------------------------
def biweight(data, axis=None, cpu_count=None, spread=False):
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
    cpu_count : int, optional
       specify number of threads to be used.
        - Multiprocessing is used only when axis is not None.
    spread :   bool, optional
       if True, then return the biweight spread.

    Returns
    -------
    out    :   ndarray
       biweight median and biweight spread if function argument "spread" is True
    """
    import multiprocessing as mp

    if axis is None or data.ndim == 1:
        if spread:
            return __biweight(data)

        return __biweight_median(data)

    # only a single axis!
    assert isinstance(axis, int)
    assert 0 <= axis < data.ndim

    if cpu_count is None:
        pool = mp.Pool()
    else:
        pool = mp.Pool(cpu_count)

    res_size = data.size // data.shape[axis]
    tmp = np.moveaxis(data, axis, 0)        # returns a numpy view
    shape = tmp.shape
    buff = tmp.reshape(shape[0], res_size)  # returns a numpy view
    ins = [buff[:, ii] for ii in range(res_size)]
    if spread:
        outs = pool.map(__biweight, ins)
    else:
        outs = pool.map(__biweight_median, ins)
    pool.close()
    pool.join()

    if spread:
        res = np.array(outs).reshape(shape[1:] + (2,))
        return (res[..., 0], res[..., 1])

    return np.array(outs).reshape(shape[1:])
