from __future__ import absolute_import
from __future__ import print_function

import shutil

import os.path

from unittest import TestCase

#--------------------------------------------------
def test_rd_icm():
    """
    Perform some simple test to check the ICM_io class

    Please use the code as tutorial

    """
    from pys5p.icm_io import ICMio

    if os.path.isdir('/Users/richardh'):
        fl_path = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    elif os.path.isdir('/nfs/TROPOMI/ical/'):
        fl_path = '/nfs/TROPOMI/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    else:
        fl_path = '/data/richardh/Tropomi/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    fl_name = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001100_20151002T140000.h5'

    icm = ICMio( os.path.join(fl_path, fl_name) )
    print( icm.get_processor_version() )
    print( icm.get_creation_time() )
    print( icm.get_coverage_time() )

    if len(icm.select('ANALOG_OFFSET_SWIR')) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'analog_offset_swir_value' )
        print( 'analog_offset_swir_value: ', res.shape )
        print(icm.get_msm_attr( 'analog_offset_swir_value', 'units' ))

    if len(icm.select('BACKGROUND_MODE_1063',
                      msm_path='BAND%_CALIBRATION')) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'signal_avg' )
        print( 'signal_avg: ', res.shape )
        print(icm.get_msm_attr( 'signal_avg', 'units' ))

        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'biweight_value' )
        print( 'biweight_value: ', res.shape )
        print(icm.get_msm_attr( 'biweight_value', 'units' ))

    if len(icm.select( 'SOLAR_IRRADIANCE_MODE_0202' )) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'irradiance_avg' )
        print( 'irradiance_avg: ', res.shape )
        print(icm.get_msm_attr( 'irradiance_avg', 'units' ))

    if len(icm.select( 'EARTH_RADIANCE_MODE_0004' )) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'radiance_avg_row' )
        print( 'radiance_avg_row: ', res.shape )
        print(icm.get_msm_attr( 'radiance_avg_row', 'units' ))

    if os.path.isdir('/Users/richardh'):
        fl_path2 = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    else:
        fl_path2 = '/data/richardh/Tropomi'
    fl_name2 = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001101_20151002T140000.h5'
    shutil.copy( os.path.join(fl_path, fl_name),
                 os.path.join(fl_path2, fl_name2) )
    icm = ICMio( os.path.join(fl_path2, fl_name2), readwrite=True )
    icm.select( 'BACKGROUND_MODE_1063' )
    res = icm.get_msm_data( 'signal_avg', '7' )
    print( 'signal_avg[7]: ', res.shape )
    res[:,:] = 2
    icm.set_msm_data( 'signal_avg', res, band='7' )
    res = icm.get_msm_data( 'signal_avg', '8' )
    print( 'signal_avg[8]: ', res.shape )
    res[:,:] = 3
    icm.set_msm_data( 'signal_avg', res, band='8' )

    del icm

class TestCmd(TestCase):
    def test_basic(self):
        test_rd_icm()

#--------------------------------------------------
if __name__ == '__main__':
    test_rd_icm()
