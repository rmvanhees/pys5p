#
# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""This module contains routines to divide or add uncertainties."""

__all__ = ['unc_div', 'unc_sum']

import numpy as np


def unc_div(value_a: np.ndarray, sigma_a: np.ndarray,
            value_b: np.ndarray, sigma_b: np.ndarray) -> np.ndarray:
    r"""Absolute error for parameter `a` divided by `b`.

    .. math:: (a / b) * \sqrt{(\sigma_a / a)^2 + (\sigma_b / b)^2}
    """
    if not value_a.shape == value_b.shape == sigma_a.shape == sigma_b.shape:
        raise TypeError('dimensions of input arrays are not the same')

    mask = (np.isfinite(value_a) & np.isfinite(sigma_a)
            & np.isfinite(value_b) & np.isfinite(sigma_b))

    if np.sum(mask) == sigma_a.size:
        return (value_a / value_b) * np.sqrt((sigma_a / value_a) ** 2
                                             + (sigma_b / value_b) ** 2)

    res = np.full(sigma_a.shape, np.nan)
    res[mask] = ((value_a[mask] / value_b[mask])
                 * np.sqrt((sigma_a[mask] / value_a[mask]) ** 2
                           + (sigma_b[mask] / value_b[mask]) ** 2))
    return res


def unc_sum(sigma_a: np.ndarray, sigma_b: np.ndarray) -> np.ndarray:
    r"""Absolute error for the sum of the parameters `a` and `b`.

    .. math:: \sqrt{\sigma_a^2 + \sigma_b^2}
    """
    if sigma_a.shape != sigma_b.shape:
        raise TypeError('dimensions of sigma_a and sigma are not the same')

    mask = np.isfinite(sigma_a) & np.isfinite(sigma_b)

    if np.sum(mask) == sigma_a.size:
        return np.sqrt(sigma_a ** 2 + sigma_b ** 2)

    res = np.full(sigma_a.shape, np.nan)
    res[mask] = np.sqrt(sigma_a[mask] ** 2 + sigma_b[mask] ** 2)

    return res
