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


def rls_fit(xx, yy):
    """
    Perform RLS regression finding linear dependence y(x) = c0 + c1 * x

    Parameters
    ----------
    xx :  ndarray, shape (M,)
       X-coordinates of the M sample points (xx[i], yy[..., i])
       The array values have to be positive and increasing
    yy :  MaskedArray or ndarray, shape (..., M)
       Y-coordinates of the sample points

    Returns
    -------
    c0, c1, std_c0, std_c1 : tuple of ndarrays
       coefficients of the linear dependence and their standard deviations

    Raises
    ------
    RuntimeError
       if M < 2: too few points for a fit
       if M_xx != M_yy: number of samples not equal for xx, yy
       if yy.shape[:-1] != samples_not_saturated.shape:
          arrays yy and mask do not have equal shapes

    Notes
    -----
    The coefficients are set to NaN when the number of samples are less than 2.

    The standard deviations can only be calculated when the number of samples
    are larger than two, else the standard deviations are equal to zero.
    """
    if xx.size < 2:
        raise RuntimeError('too few sample points for a fit')
    if xx.size != yy.shape[-1]:
        raise RuntimeError('number of samples not equal for xx, yy')

    # perform all computations on 2 dimensional arrays
    img_shape = yy.shape[:-1]
    if isinstance(yy, ma.MaskedArray):
        yy1 = yy.reshape(-1, xx.size)
    else:
        yy1 = ma.array(yy).reshape(-1, xx.size)

    # generate weight factor per pixel
    wght = np.empty(yy1.shape, dtype=float)
    wght[:] = np.concatenate(([2 * (xx[1] - xx[0])],
                              xx[2:] - xx[0:-2],
                              [2 * (xx[-1] - xx[-2])]))
    wght = ma.array(wght, mask=yy1.mask, hard_mask=True)
    wx1 = wght / xx
    wx2 = wght / xx ** 2

    # calculate the Q elements
    q00 = wght.sum(axis=1)
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
    num = yy1.count(axis=1)
    cc0 = zz2 / zz1
    cc0[num < 2] = np.nan
    cc1 = zz3 / zz1
    cc1[num < 2] = np.nan
    chi2 = np.abs(q22 - q12 * cc0 - q11 * cc1) / np.clip(num - 2, 1, None)
    chi2[num <= 2] = 0
    sc0 = np.sqrt(q00 * chi2 / zz1)
    sc1 = np.sqrt(q02 * chi2 / zz1)

    return (cc0.reshape(img_shape), cc1.reshape(img_shape),
            sc0.reshape(img_shape), sc1.reshape(img_shape))


def rls_fit0(xx, yy):
    """
    Perform RLS regression finding linear dependence y(x) = c1 * x

    Parameters
    ----------
    xx :  ndarray, shape (M,)
       X-coordinates of the M sample points (xx[i], yy[..., i])
       The array values have to be positive and increasing
    yy :  MaskedArray or ndarray, shape (..., M)
       Y-coordinates of the sample points

    Returns
    -------
    c1, std_c1 : tuple of ndarrays
       coefficients of the linear dependence and their standard deviations

    Raises
    ------
    RuntimeError
       if M < 2: too few points for a fit
       if M_xx != M_yy: number of samples not equal for xx, yy
       if yy.shape[:-1] != samples_not_saturated.shape:
          arrays yy and mask do not have equal shapes

    Notes
    -----
    The coefficients are set to NaN when the number of samples are less than 2.

    The standard deviations can only be calculated when the number of samples
    are larger than two, else the standard deviations are equal to zero.
    """
    if xx.size < 2:
        raise RuntimeError('too few points for a fit')
    if xx.size != yy.shape[-1]:
        raise RuntimeError('number of samples not equal for xx, yy')

    # perform all computations on 2 dimensional arrays
    img_shape = yy.shape[:-1]
    if isinstance(yy, np.ma.MaskedArray):
        yy1 = yy.reshape(-1, xx.size)
    else:
        yy1 = ma.array(yy).reshape(-1, xx.size)

    # generate weight factor per pixel
    wght = np.empty(yy1.shape, dtype=float)
    wght[:] = np.concatenate(([2 * (xx[1] - xx[0])],
                              xx[2:] - xx[0:-2],
                              [2 * (xx[-1] - xx[-2])]))
    wght = ma.array(wght, mask=yy1.mask, hard_mask=True)
    wx1 = wght / xx
    wx2 = wght / xx ** 2

    # calculate the Q elements
    q00 = wght.sum(axis=1)
    q11 = (wx1 * yy1).sum(axis=1)
    q22 = (wx2 * yy1 ** 2).sum(axis=1)

    # calculate fit parameter and its variance
    num = yy1.count(axis=1)
    cc1 = q11 / q00
    cc1[num < 1] = np.nan
    chi2 = np.abs(q22 - q00 * cc1 ** 2) / np.clip(num - 1, 1, None)
    chi2[num <= 1] = 0
    sc1 = np.sqrt(chi2 / q00)

    return (cc1.reshape(img_shape), sc1.reshape(img_shape))
