from __future__ import absolute_import
from __future__ import print_function

import os.path

from glob import glob
from unittest import TestCase

import matplotlib

matplotlib.use('TkAgg')

#-------------------------
def test_geo():
    """
    Let the user test the software!!!

    Please use the code as tutorial
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
    l1b = L1BioRAD( os.path.join(data_dir, 'L1B', filelist[-1]) )
    l1b.select()
    geo = l1b.get_geo_data( icid=4 )
    del l1b

    plot = S5Pplot('test_plot_geo.pdf')
    plot.draw_geolocation( geo['latitude'], geo['longitude'],
                           sequence=geo['sequence'] )

    # test subsatellite mode
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_CA_*.nc'))
    l1b = L1BioCAL( os.path.join(data_dir, filelist[-1]) )
    l1b.select('BACKGROUND_RADIANCE_MODE_0005')
    geo = l1b.get_geo_data()
    del l1b

    plot.draw_geolocation( geo['satellite_latitude'],
                           geo['satellite_longitude'],
                           sequence=geo['sequence'],
                           subsatellite=True )

    del plot
    

class TestCmd(TestCase):
    def test_basic(self):
        test_geo()
