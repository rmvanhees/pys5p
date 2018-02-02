"""
This file is part of pyS5pMon

https://git.sron.nl/tropomi_icm.git

Return SWIR detector mask with the 'illuminated' or 'level2' region

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""

import numpy as np


def mask(region='illuminated'):
    """
    Return mask with the illuminated region has boolean value True

    Parameters
    ----------
    region  :   string, optional
      'illuminated' - detector area illuminated by external sources,
      defined as the area where the signal is at least 50% of the main signal
      'level2' - a smaller area used for (ir)radiance products given to level 2
    """
    res = np.zeros((256, 1000), dtype=np.bool)
    if region == 'level2':
        res[12:227, 20:980] = True
    else:
        res[11:228, 16:991] = True

    return res

def coords(region='illuminated'):
    """
    Return slice defining the illuminated region

    Parameters
    ----------
    region  :   string, optional
      'illuminated' - detector area illuminated by external sources,
      defined as the area where the signal is at least 50% of the main signal
      'level2' - a smaller area used for (ir)radiance products given to level 2
    """
    if region == 'level2':
        return np.s_[12:227, 20:980]

    return np.s_[11:228, 16:991]
