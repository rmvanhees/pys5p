"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest

Note
----
Please use the code as tutorial

Copyright (c) 2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

import numpy as np


#-------------------------
def relirr_factor(bands, azi=2., elev=4.):
    """
    generate RELIRR correcting factors (azimuth, elevation)
    """
    from pys5p.ckd_io import CKDio
    from numpy.polynomial.chebyshev import chebval2d
    from scipy import interpolate

    with CKDio() as ckd:
        ckd_bands = ckd.relirr(bands=bands)
        if len(ckd_bands) != len(bands):
            raise ValueError

    if len(bands) == 2:
        _sz = ckd_bands[0]['cheb_coefs'].shape
        if _sz != ckd_bands[1]['cheb_coefs'].shape:
            raise ValueError

        buff1 = np.empty((_sz[0], _sz[1]), dtype=float)
        buff2 = np.empty((_sz[0], _sz[1]), dtype=float)
        for iy in range(_sz[0]):
            for ix in range(_sz[1]):
                buff1[iy, ix] = chebval2d(
                    elev/6, azi/12, ckd_bands[0]['cheb_coefs'][iy, ix, :, :])
                buff2[iy, ix] = chebval2d(
                    elev/6, azi/12, ckd_bands[1]['cheb_coefs'][iy, ix, :, :])
        # first spectral band
        print(ckd_bands[0]['mapping_cols'].shape,
              ckd_bands[0]['mapping_cols'])
        f_interpol = interpolate.interp2d(ckd_bands[0]['mapping_cols'],
                                          ckd_bands[0]['mapping_rows'],
                                          buff1, kind='linear')
        buff1_fine = f_interpol(np.arange(500), np.arange(256))
        # second spectral band
        print(ckd_bands[1]['mapping_cols'].shape,
              ckd_bands[1]['mapping_cols'])
        f_interpol = interpolate.interp2d(ckd_bands[1]['mapping_cols'],
                                          ckd_bands[1]['mapping_rows'],
                                          buff2, kind='linear')
        buff2_fine = f_interpol(np.arange(500), np.arange(256))
        # combine both spectral bands
        relirr_fine = np.hstack((buff1_fine, buff2_fine))
        for ix in range(1000):
            print(ix, relirr_fine[100, ix])
    elif len(bands) == 1:
        _sz = ckd_bands[0]['cheb_coefs'].shape
        buff1 = np.empty((_sz[0], _sz[1]), dtype=float)

        for iy in range(_sz[0]):
            for ix in range(_sz[1]):
                buff1[iy, ix] = chebval2d(
                    elev/6, azi/12, ckd_bands[0]['cheb_coefs'][iy, ix, :, :])

        f_interpol = interpolate.interp2d(ckd_bands[0]['mapping_cols'],
                                          ckd_bands[0]['mapping_rows'],
                                          buff1, kind='linear')
        relirr_fine = f_interpol(np.arange(500), np.arange(256))
    else:
        raise ValueError

    print(relirr_fine.shape, relirr_fine[100, 200])

    return relirr_fine


# - main function --------------------------------------
def main():
    """
    main function when called from the command-line
    """
    from pys5p.s5p_plot import S5Pplot

    res = relirr_factor('78')
    plot = S5Pplot('test_ckd_relirr.pdf')
    plot.draw_signal(res)
    plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
