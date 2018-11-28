"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on pys5p.sron_colormaps

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')


# --------------------
def test_cmaps():
    """
    Show the SRON Matplotlib color-maps
    """
    import matplotlib.pyplot as plt

    from scipy.stats import multivariate_normal

    from pys5p.sron_colormaps import SRONcmaps, sron_cmap

    # Our 2-dimensional distribution will be over variables X and Y
    xx = np.linspace(-3, 3, 60)
    yy = np.linspace(-3, 4, 60)
    xxm, yym = np.meshgrid(xx, yy)

    # Pack X and Y into a single 3-dimensional array
    xypos = np.empty(xxm.shape + (2,))
    xypos[:, :, 0] = xxm
    xypos[:, :, 1] = yym

    # Obtain the multivariate Gaussian probability distribution function
    rv1 = multivariate_normal([0., 0.], [[1.0, 0.], [0., 1.0]])
    rv2 = multivariate_normal([1., 1.], [[1.5, 0.], [0., 1.5]])
    zz = rv2.pdf(xypos) - rv1.pdf(xypos)
    zz[10, 10] = np.nan
    zz[30, 40] = np.nan

    obj = SRONcmaps()
    print(obj.namelist)

    _, axarr = plt.subplots(3, 3)
    for ii, name in enumerate(obj.namelist):
        axarr[ii % 3, ii // 3].set_title(name)
        axarr[ii % 3, ii // 3].imshow(zz, interpolation='nearest',
                                      extent=[-3, 3, -3, 3],
                                      cmap=sron_cmap(name))
    plt.show()


if __name__ == "__main__":
    test_cmaps()
