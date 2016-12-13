"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class ICMio

Note
----
Please use the code as tutorial

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os.path
import shutil

from glob import glob
#from unittest import TestCase

#--------------------------------------------------
def test_rd_icm():
    """
    Perform some simple checks on the ICMio class

    """
    from ..get_data_dir import get_data_dir
    from ..icm_io import ICMio

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'ICM', 'S5P_TEST_ICM_CA_SIR_*.h5'))
    if len(filelist) == 0:
        return
    
    icm = ICMio(filelist[0])
    print(icm, file=sys.stderr)
    print(icm.get_processor_version())
    print(icm.get_creation_time())
    print(icm.get_coverage_time())

    if len(icm.select('ANALOG_OFFSET_SWIR')) > 0:
        #print(icm.get_ref_time())
        #print(icm.get_delta_time())
        #print(icm.get_instrument_settings())
        #print(icm.get_housekeeping_data())
        print('GEO: ', icm.get_geo_data().shape)
        res = icm.get_msm_data('analog_offset_swir_value')
        print('analog_offset_swir_value: ', res.shape)
        print(icm.get_msm_attr('analog_offset_swir_value', 'units'))

    if len(icm.select('BACKGROUND_MODE_1063',
                      msm_path='BAND%_CALIBRATION')) > 0:
        #print(icm.get_ref_time())
        #print(icm.get_delta_time())
        #print(icm.get_instrument_settings())
        #print(icm.get_housekeeping_data())
        print('GEO: ', icm.get_geo_data().shape)
        res = icm.get_msm_data('signal_avg')
        print('signal_avg: ', res.shape)
        print(icm.get_msm_attr('signal_avg', 'units'))

        print('GEO: ', icm.get_geo_data().shape)
        res = icm.get_msm_data('biweight_value')
        print('biweight_value: ', res.shape)
        print(icm.get_msm_attr('biweight_value', 'units'))

    if len(icm.select('SOLAR_IRRADIANCE_MODE_0202')) > 0:
        #print(icm.get_ref_time())
        #print(icm.get_delta_time())
        #print(icm.get_instrument_settings())
        #print(icm.get_housekeeping_data())
        print('GEO: ', icm.get_geo_data().shape)
        res = icm.get_msm_data('irradiance_avg')
        print('irradiance_avg: ', res.shape)
        print(icm.get_msm_attr('irradiance_avg', 'units'))

    if len(icm.select('EARTH_RADIANCE_MODE_0004')) > 0:
        #print(icm.get_ref_time())
        #print(icm.get_delta_time())
        #print(icm.get_instrument_settings())
        #print(icm.get_housekeeping_data())
        print('GEO: ', icm.get_geo_data().shape)
        res = icm.get_msm_data('radiance_avg_row')
        print('radiance_avg_row: ', res.shape)
        print(icm.get_msm_attr('radiance_avg_row', 'units'))

    fl_patch = filelist[0].replace('_TEST_', '_PATCH_')
    print(os.path.join(data_dir, filelist[0]))
    print(os.path.join(data_dir, fl_patch))
    shutil.copy(os.path.join(data_dir, filelist[0]),
                os.path.join(data_dir, fl_patch))
    icm = ICMio(os.path.join(data_dir, fl_patch), readwrite=True)
    icm.select('BACKGROUND_MODE_1063')
    res = icm.get_msm_data('signal_avg', '7')
    print('signal_avg[7]: ', res.shape)
    res[:,:] = 2
    icm.set_msm_data('signal_avg', res, band='7')
    res = icm.get_msm_data('signal_avg', '8')
    print('signal_avg[8]: ', res.shape)
    res[:,:] = 3
    icm.set_msm_data('signal_avg', res, band='8')

    del icm

if __name__ == '__main__':
    test_rd_icm()