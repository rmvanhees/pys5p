"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on S5Pplot.draw_geo

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os.path

from glob import glob
#from unittest import TestCase

#-------------------------
def test_geo():
    """
    Check classes L1BioCAL, L1BioRAD and S5Pplot.draw_geo

    """
    from ..get_data_dir import get_data_dir
    from ..l1b_io import L1BioCAL, L1BioRAD
    from ..s5p_plot import S5Pplot

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_RA_*.nc'))
    if len(filelist) == 0:
        return

    # test footprint mode
    l1b = L1BioRAD(filelist[-1])
    print(l1b, file=sys.stderr)
    l1b.select()
    geo = l1b.get_geo_data(icid=4)
    del l1b

    plot = S5Pplot('test_plot_geo.pdf')
    plot.draw_geolocation(geo['latitude'], geo['longitude'],
                          sequence=geo['sequence'])

    # test subsatellite mode
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_CA_*.nc'))
    l1b = L1BioCAL(filelist[-1])
    print(l1b, file=sys.stderr)
    l1b.select('BACKGROUND_RADIANCE_MODE_0005')
    geo = l1b.get_geo_data()
    del l1b

    plot.draw_geolocation(geo['satellite_latitude'],
                          geo['satellite_longitude'],
                          sequence=geo['sequence'],
                          subsatellite=True)
    del plot

if __name__ == '__main__':
    test_geo()
