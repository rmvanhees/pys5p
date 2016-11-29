from __future__ import absolute_import
from __future__ import print_function

import os.path

import matplotlib

from unittest import TestCase

matplotlib.use('TkAgg')

#-------------------------
def test_geo():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from pys5p.l1b_io import L1BioCAL, L1BioRAD
    from pys5p.s5p_plot import S5Pplot

    plot = S5Pplot( 'test_geo.pdf' )

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/L1B-CALIBRATION'
    else:
        data_dir = '/stage/EPSstorage/jochen/TROPOMI_ODA_DATA/data_set_05102016/OFFL/L1B/2015/08/21/00058'

    # test footprint mode
    fl_name = 'S5P_OFFL_L1B_RA_BD7_20150821T012540_20150821T030710_00058_01_000000_20160721T173643.nc'
    l1b = L1BioRAD( os.path.join(data_dir, 'L1B-RADIANCE', fl_name) )
    l1b.select()
    geo = l1b.get_geo_data( icid=4 )
    del l1b

    plot.draw_geolocation( geo['latitude'], geo['longitude'],
                           sequence=geo['sequence'] )

    # test subsatellite mode
    fl_name = 'S5P_OFFL_L1B_CA_SIR_20150821T012540_20150821T030710_00058_01_000000_20160721T173643.nc'
    l1b = L1BioCAL( os.path.join(data_dir, 'L1B-CALIBRATION', fl_name) )
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

#--------------------------------------------------
if __name__ == '__main__':
    print( '*** Info: call function test_geo()')
    test_geo()
