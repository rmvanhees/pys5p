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

from pathlib import Path

import numpy as np

#-------------------------
def relirr_factor(azi=2., elev=4.):
    """
    """
    from pys5p.ckd_io import CKDio
    from numpy.polynomial.chebyshev import chebval2d
    from scipy import interpolate

    with CKDio() as ckd:
        ckd_relirr = ckd.relirr()

    _sz = ckd_relirr['cheb_qvd']['coefs'].shape
    relirr_coarse = np.empty((_sz[0], _sz[1]), dtype=float)
    for iy in range(_sz[0]):
        for ix in range(_sz[1]):
            coeffs = ckd_relirr['cheb_qvd']['coefs'][iy, ix, :, :]
            relirr_coarse[iy, ix] = chebval2d(elev/6, azi/12, coeffs)

    f_interpol = interpolate.interp2d(ckd_relirr['mapping_cols'],
                                      ckd_relirr['mapping_rows'],
                                      relirr_coarse, kind='linear')
    relirr_fine = f_interpol(np.arange(1000), np.arange(256))
    print(relirr_coarse.shape, relirr_fine.shape, relirr_fine[100, 200])

    return relirr_fine
                         

# - main function --------------------------------------
def main():
    """
    main function when called from the command-line
    """
    from pys5p.s5p_plot import S5Pplot

    res = relirr_factor()
    plot = S5Pplot('test_ckd_relirr.pdf')
    plot.draw_signal(res)
    plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
