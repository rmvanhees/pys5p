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

License:  Standard 3-clause BSD
"""
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

#--------------------
def test_cmaps():
    """
    Show the SRON Matplotlib color-maps

    """
    import matplotlib.pyplot as plt

    from matplotlib.mlab import bivariate_normal

    from pys5p.sron_colormaps import SRONcmaps, sron_cmap

    xx = yy = np.arange(-3.0, 3.0, 0.1)
    xxm, yym = np.meshgrid(xx, yy)
    zz1 = bivariate_normal(xxm, yym, 1.0, 1.0, 0.0, 0.0)
    zz2 = bivariate_normal(xxm, yym, 1.5, 0.5, 1, 1)
    zz = zz2 - zz1
    zz[10, 10] = np.nan
    zz[30, 40] = np.nan

    obj = SRONcmaps()

    _, axarr = plt.subplots(2, 4)
    ii = 0
    for name in obj.namelist:
        axarr[ii % 2, ii // 2].set_title(obj.namelist[ii])
        axarr[ii % 2, ii // 2].imshow(zz, interpolation='nearest',
                                      extent=[-3, 3, -3, 3],
                                      cmap=sron_cmap(name))
        ii += 1
    plt.show()

if __name__ == "__main__":
    test_cmaps()
