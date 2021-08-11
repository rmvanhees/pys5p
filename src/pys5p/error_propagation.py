"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

Implement some function for error propagation

Copyright (c) 2017-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np


def unc_sum(sigma_a, sigma_b):
    """
    absolute error for a +/- b is sqrt(sigma_a^2 + sigma_b^2)
    """
    if sigma_a.shape != sigma_b.shape:
        raise TypeError('dimensions of sigma_a and sigma are not the same')

    mask = np.isfinite(sigma_a) & np.isfinite(sigma_b)

    if np.sum(mask) == sigma_a.size:
        return np.sqrt(sigma_a ** 2 + sigma_b ** 2)

    res = np.full(sigma_a.shape, np.nan)
    res[mask] = np.sqrt(sigma_a[mask] ** 2 + sigma_b[mask] ** 2)

    return res


def unc_div(value_a, sigma_a, value_b, sigma_b):
    """
    absolute error for a / b is sqrt((sigma_a/a)^2 + (sigma_b/b)^2) * a/b
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
