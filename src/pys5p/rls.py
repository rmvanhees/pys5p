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

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np


# pylint: disable=invalid-unary-operand-type
# - local functions --------------------------------
def __rls_fit_all(xx, yy):
    """
    RLS regression without masked pixels
    """
    wght = np.concatenate(([2 * (xx[1] - xx[0])],
                           xx[2:] - xx[0:-2],
                           [2 * (xx[-1] - xx[-2])]))
    wx1 = wght / xx
    wx2 = wght / xx ** 2

    # calculate the Q elements
    q00 = np.sum(wght)
    q01 = np.sum(wx1)
    q02 = np.sum(wx2)

    q11 = yy @ wx1
    q12 = yy @ wx2
    q22 = yy ** 2 @ wx2

    # calculate the Z elements
    zz1 = q00 * q02 - q01 ** 2
    zz2 = q00 * q12 - q01 * q11
    zz3 = q02 * q11 - q01 * q12

    cc0 = zz2 / zz1
    cc1 = zz3 / zz1

    if xx.size == 2:
        chi2 = np.zeros(cc0.shape[0], dtype=float)
        sc0 = np.zeros(cc0.shape[0], dtype=float)
        sc1 = np.zeros(cc1.shape[0], dtype=float)
    else:
        chi2 = np.abs(q22 - q12 * cc0 - q11 * cc1) / (xx.size - 2)
        sc0 = np.sqrt(q00 * chi2 / zz1)
        sc1 = np.sqrt(q02 * chi2 / zz1)

    return (cc0, cc1, sc0, sc1, chi2)


def __rls_fit_masked(xx, yy, mask, num):
    """
    RLS regression with masked pixels
    """
    # generate weight factor per pixel
    wght = np.empty(yy.shape, dtype=float)
    wght[:] = np.concatenate(([2 * (xx[1] - xx[0])],
                              xx[2:] - xx[0:-2],
                              [2 * (xx[-1] - xx[-2])]))
    wght[~mask] = 0.
    wx1 = wght / xx
    wx2 = wght / xx ** 2

    # calculate the Q elements
    q00 = np.sum(wght, axis=1)
    q01 = np.sum(wx1, axis=1)
    q02 = np.sum(wx2, axis=1)

    q11 = np.sum(wx1 * yy, axis=1)
    q12 = np.sum(wx2 * yy, axis=1)
    q22 = np.sum(wx2 * yy ** 2, axis=1)

    # calculate the Z elements
    zz1 = q00 * q02 - q01 ** 2
    zz2 = q00 * q12 - q01 * q11
    zz3 = q02 * q11 - q01 * q12

    zz1[zz1 == 0] = float.fromhex('0x1.ep-122')
    cc0 = zz2 / zz1
    cc0[num < 2] = np.nan
    cc1 = zz3 / zz1
    cc1[num < 2] = np.nan

    chi2 = np.abs(q22 - q12 * cc0 - q11 * cc1) / np.clip(num - 2, 1, None)
    chi2[num <= 2] = 0
    sc0 = np.sqrt(q00 * chi2 / zz1)
    sc1 = np.sqrt(q02 * chi2 / zz1)

    return (cc0, cc1, sc0, sc1, chi2)


# - main functions _--------------------------------
def rls_fit(xx, yy, samples_not_saturated=None):
    """
    Fast implementation of the RLS regression finding linear dependence
      y(x) = c0 + c1 * x

    Parameters
    ----------
    xx :  ndarray, shape (M,)
       X-coordinates of the M sample points (xx[i], yy[..., i])
       The array values have to be positive and increasing
    yy :  ndarray, shape (..., M)
       Y-coordinates of the sample points
    samples_not_saturated : ndarray, optional
       number of sample points to include in the fit, startinge from the first

    Returns
    -------
    c0, c1, std_c0, std_c1, chi2 : tuple of ndarrays
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
    The standard deviations can only be calculated when the number of samples
    are larger than two, else the standard deviations are equal to zero.

    The coefficients are set to NaN when the number of samples are less than 2.

    Examples
    --------
    How to use this function:
    >>> dimx = dimy = 2048
    >>> texp = np.array([1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 5e-2, 1e-1, .5, 1, 5])
    >>> alpha = np.random.random_sample((dimy, dimx)) + 11
    >>> beta = np.random.random_sample((dimy, dimx)) + 3
    >>> data = alpha[..., np.newaxis] + texp * beta[..., np.newaxis]
    >>> cc0, cc1, sc0, sc1, chi2 = rls_fit(texp, data)
    >>> mask_nan = np.isfinite(cc0)
    >>> print('cc0 ', np.nanmean(cc0),
    ...       np.abs(cc0[mask_nan] - alpha[mask_nan]).max())
    cc0  11.49983030605162 7.105427357601002e-15
    >>> print('sc0 ', np.nanmean(sc0))
    sc0  5.0509648555957586e-08
    """
    if xx.size < 2:
        raise RuntimeError('too few sample points for a fit')
    if xx.size != yy.shape[-1]:
        raise RuntimeError('number of samples not equal for xx, yy')

    # perform all computations on 2 dimensional arrays
    data = yy.reshape(-1, xx.size)

    # ---------- include all data samples ----------
    if samples_not_saturated is None:
        cc0, cc1, sc0, sc1, chi2 = __rls_fit_all(xx, data)

    # ---------- exclude several of the last data samples ----------
    else:
        if yy.shape[:-1] != samples_not_saturated.shape:
            raise RuntimeError('arrays yy and mask do not have equal shapes')
        buff = (np.arange(yy.size) % xx.size).reshape(yy.shape)
        mask = buff <= samples_not_saturated[:, :, np.newaxis]
        del buff
        if np.isnan(yy).any():
            mask[np.isnan(yy)] = False
        mask = mask.reshape(-1, xx.size)

        cc0, cc1, sc0, sc1, chi2 = __rls_fit_masked(
            xx, data, mask, samples_not_saturated.reshape(-1))

    return (cc0.reshape(yy.shape[:-1]), cc1.reshape(yy.shape[:-1]),
            sc0.reshape(yy.shape[:-1]), sc1.reshape(yy.shape[:-1]),
            chi2.reshape(yy.shape[:-1]))


# --------------------------------------------------
def rls_fit0(xx, yy, samples_not_saturated=None):
    """
    Fast implementation of RLS regression finding linear dependence
      y(x) = c1 * x

    Parameters
    ----------
    xx :  ndarray, shape (M,)
       X-coordinates of the M sample points (xx[i], yy[..., i])
       The array values have to be positive and increasing
    yy :  ndarray, shape (..., M)
       Y-coordinates of the sample points
    samples_not_saturated : ndarray, optional
       number of sample points to include in the fit, startinge from the first

    Returns
    -------
    c1, std_c1 : tuple of ndarrays
       coefficient of the linear dependence and its standard deviation

    Raises
    ------
    RuntimeError
       if M < 2: too few points for a fit
       if M_xx != M_yy: number of samples not equal for xx, yy
       if yy.shape != mask.shape: arrays yy and mask do not have equal shapes

    See also
    --------
       pys5p.rls.rls_fit
    """
    if xx.size < 2:
        raise RuntimeError('too few points for a fit')
    if xx.size != yy.shape[-1]:
        raise RuntimeError('number of samples not equal for xx, yy')

    # perform all computations on 2 dimensional arrays
    data = yy.reshape(-1, xx.size)

    # ---------- include all data samples ----------
    if samples_not_saturated is None:
        wght = np.concatenate(([2 * (xx[1] - xx[0])],
                               xx[2:] - xx[0:-2],
                               [2 * (xx[-1] - xx[-2])]))
        wx1 = wght / xx
        wx2 = wght / xx ** 2

        # calculate the Q elements
        q00 = np.sum(wght)
        q11 = data @ wx1
        q22 = data ** 2 @ wx2

        # calculate fit parameter and its variance
        cc1 = q11 / q00
        chi2 = np.abs(q22 - q00 * cc1 ** 2) / (xx.size - 1)
        sc1 = np.sqrt(chi2 / q00)

    # ---------- exclude several of the last data samples ----------
    else:
        if yy.shape[:-1] != samples_not_saturated.shape:
            raise RuntimeError('arrays yy and mask do not have equal shapes')
        buff = (np.arange(yy.size) % xx.size).reshape(yy.shape)
        mask = buff <= samples_not_saturated[:, :, np.newaxis]
        del buff
        mask = mask.reshape(-1, xx.size)
        num = samples_not_saturated.reshape(-1)

        # generate weight factor per pixel
        wght = np.empty(data.shape, dtype=float)
        wght[:] = np.concatenate(([2 * (xx[1] - xx[0])],
                                  xx[2:] - xx[0:-2],
                                  [2 * (xx[-1] - xx[-2])]))
        wght[~mask] = 0.
        wx1 = wght / xx
        wx2 = wght / xx ** 2

        # calculate the Q elements
        q00 = np.sum(wght, axis=1)
        q11 = np.sum(wx1 * data, axis=1)
        q22 = np.sum(wx2 * data ** 2, axis=1)

        # calculate fit parameter and its variance
        cc1 = q11 / q00
        cc1[num < 2] = np.nan
        chi2 = np.abs(q22 - q00 * cc1 ** 2) / np.clip(num - 1, 1, None)
        chi2[num <= 2] = 0
        sc1 = np.sqrt(chi2 / q00)

    return (cc1.reshape(yy.shape[:-1]),
            sc1.reshape(yy.shape[:-1]),
            chi2.reshape(yy.shape[:-1]))
