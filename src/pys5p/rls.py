"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implementation of the Relative Least-Squares regression (RLS).

The RLS regression is used to find the linear dependence y(x) = c0 + c1 * x
that best describes the data before and after correction, using absolute
residuals y_i - (c0 + c1 * x_i) divided by the expected signals c1 * x_i in
the least-squares sum. Offset c0 has an arbitrary size and should not affect
the fit result. Weight factors are determined such to effectively spread the
data points evenly over the whole range of x, making the result less
sensitive to the actual spacing between the data points.

Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np
from numpy import ma


def calc_ma_weights(xdata: np.ndarray, masked: np.ndarray) -> ma.MaskedArray:
    """
    Generate weight factor per pixel

    Notes
    -----
    It might be that np.apply_along_axis() is slightly faster, however, the
    for-loop 'row in buff' is also very efficient when using the specially
    designed MaskedArray 'buff' which we eventually use to store the weight
    values per pixel.
    """
    buff = ma.array(np.repeat([xdata], masked.shape[0], axis=0), mask=masked)
    for row in buff:
        valid = ma.compressed(row)
        if len(valid) < 2:
            continue
        wght = np.concatenate(([2 * (valid[1] - valid[0])],
                               valid[2:] - valid[0:-2],
                               [2 * (valid[-1] - valid[-2])]))
        row[~row.mask] = wght

    return buff


def rls_fit(xdata: np.ndarray, ydata) -> tuple:
    """
    Perform RLS regression finding linear dependence y(x) = c0 + c1 * x

    Parameters
    ----------
    xdata :  ndarray, shape (M,)
       X-coordinates of the M sample points (xdata[i], ydata[..., i])
       The array values have to be positive and increasing
    ydata :  MaskedArray or ndarray, shape (..., M)
       Y-coordinates of the sample points

    Returns
    -------
    c0, c1, std_c0, std_c1 : tuple of ndarrays
       coefficients of the linear dependence and their standard deviations

    Raises
    ------
    RuntimeError
       if M < 2: too few points for a fit
       if M_xx != M_yy: number of samples not equal for xdata, ydata
       if ydata.shape[:-1] != samples_not_saturated.shape:
          arrays ydata and mask do not have equal shapes

    Notes
    -----
    Calling a rls-function with MaskedArrays is much slower than with
    plain ndarrays.

    The coefficients are set to NaN when the number of samples are less than 2.

    The standard deviations can only be calculated when the number of samples
    are larger than two, else the standard deviations are equal to zero.
    """
    # pylint: disable=too-many-locals
    if xdata.size < 2:
        raise RuntimeError('too few sample points for a fit')
    if xdata.size != ydata.shape[-1]:
        raise RuntimeError('number of samples not equal for xdata, ydata')

    # perform all computations on 2 dimensional arrays
    img_shape = ydata.shape[:-1]
    yy1 = ydata.reshape(-1, xdata.size)

    # calculate weights
    if ma.isMaskedArray(ydata):
        wghts = calc_ma_weights(xdata, ma.getmaskarray(yy1))
    else:
        buff = np.concatenate(([2 * (xdata[1] - xdata[0])],
                               xdata[2:] - xdata[0:-2],
                               [2 * (xdata[-1] - xdata[-2])]))
        wghts = np.repeat([buff], yy1.shape[0], axis=0)
    wx1 = wghts / xdata
    wx2 = wghts / xdata ** 2   # is wx1 / xdata faster? 

    # calculate the Q elements
    q00 = wghts.sum(axis=1)
    q01 = wx1.sum(axis=1)
    q02 = wx2.sum(axis=1)

    q11 = (wx1 * yy1).sum(axis=1)
    q12 = (wx2 * yy1).sum(axis=1)
    q22 = (wx2 * yy1 ** 2).sum(axis=1)

    # calculate the Z elements
    zz1 = q00 * q02 - q01 ** 2
    zz2 = q00 * q12 - q01 * q11
    zz3 = q02 * q11 - q01 * q12

    # calculate fit parameters and their uncertainties
    num = yy1.count(axis=1) if ma.isMaskedArray(ydata) else len(xdata)
    cc0 = zz2 / zz1
    cc1 = zz3 / zz1
    if ma.isMaskedArray(ydata):
        chi2 = ma.abs(q22 - q12 * cc0 - q11 * cc1) / np.clip(num - 2, 1, None)
        chi2[num <= 2] = 0
        sc0 = ma.sqrt(q00 * chi2 / zz1)
        sc1 = ma.sqrt(q02 * chi2 / zz1)

        return (cc0.reshape(img_shape).filled(np.nan),
                cc1.reshape(img_shape).filled(np.nan),
                sc0.reshape(img_shape).filled(np.nan),
                sc1.reshape(img_shape).filled(np.nan))

    # using only non-MaskedArray functions
    chi2 = np.abs(q22 - q12 * cc0 - q11 * cc1) / np.clip(num - 2, 1, None)
    chi2[num <= 2] = 0
    sc0 = np.sqrt(q00 * chi2 / zz1)
    sc1 = np.sqrt(q02 * chi2 / zz1)

    return (cc0.reshape(img_shape), cc1.reshape(img_shape),
            sc0.reshape(img_shape), sc1.reshape(img_shape))




def rls_fit0(xdata: np.ndarray, ydata) -> tuple:
    """
    Perform RLS regression finding linear dependence y(x) = c1 * x

    Parameters
    ----------
    xdata :  ndarray, shape (M,)
       X-coordinates of the M sample points (xdata[i], ydata[..., i])
       The array values have to be positive and increasing
    ydata :  MaskedArray or ndarray, shape (..., M)
       Y-coordinates of the sample points

    Returns
    -------
    c1, std_c1 : tuple of ndarrays
       coefficients of the linear dependence and their standard deviations

    Raises
    ------
    RuntimeError
       if M < 2: too few points for a fit
       if M_xx != M_yy: number of samples not equal for xdata, ydata
       if ydata.shape[:-1] != samples_not_saturated.shape:
          arrays ydata and mask do not have equal shapes

    Notes
    -----
    The coefficients are set to NaN when the number of samples are less than 2.

    The standard deviations can only be calculated when the number of samples
    are larger than two, else the standard deviations are equal to zero.
    """
    if xdata.size < 2:
        raise RuntimeError('too few points for a fit')
    if xdata.size != ydata.shape[-1]:
        raise RuntimeError('number of samples not equal for xdata, ydata')

    # perform all computations on 2 dimensional arrays
    img_shape = ydata.shape[:-1]
    yy1 = ydata.reshape(-1, xdata.size)

    # calculate weights
    if ma.isMaskedArray(ydata):
        wghts = calc_ma_weights(xdata, ma.getmaskarray(yy1))
    else:
        buff = np.concatenate(([2 * (xdata[1] - xdata[0])],
                               xdata[2:] - xdata[0:-2],
                               [2 * (xdata[-1] - xdata[-2])]))
        wghts = np.repeat([buff], yy1.shape[0], axis=0)
    wx1 = wghts / xdata
    wx2 = wghts / xdata ** 2

    # calculate the Q elements
    q00 = wghts.sum(axis=1)
    q11 = (wx1 * yy1).sum(axis=1)
    q22 = (wx2 * yy1 ** 2).sum(axis=1)

    # calculate fit parameter and its variance
    num = yy1.count(axis=1) if ma.isMaskedArray(ydata) else len(xdata)
    cc1 = q11 / q00
    if ma.isMaskedArray(ydata):
        cc1[num < 1] = ma.masked
        chi2 = ma.abs(q22 - q00 * cc1 ** 2) / np.clip(num - 1, 1, None)
        chi2[num <= 1] = ma.masked
        sc1 = ma.sqrt(chi2 / q00)
        return (cc1.reshape(img_shape).filled(np.nan),
                sc1.reshape(img_shape).filled(np.nan))

    # using only non-MaskedArray functions
    cc1[num < 1] = np.nan
    chi2 = np.abs(q22 - q00 * cc1 ** 2) / np.clip(num - 1, 1, None)
    chi2[num <= 1] = np.nan
    sc1 = np.sqrt(chi2 / q00)
    return (cc1.reshape(img_shape), sc1.reshape(img_shape))
