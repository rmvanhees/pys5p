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

License:  BSD-3-Clause
"""
import shutil

from pathlib import Path

import numpy as np

from pys5p.get_data_dir import get_data_dir
from pys5p.icm_io import ICMio

#--------------------------------------------------
def test_rd_icm():
    """
    Perform some simple checks on the ICMio class

    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return

    filelist = list(Path(data_dir, 'ICM').glob('S5P_TEST_ICM_CA_SIR_*.h5'))
    if not filelist:
        return

    for flname in sorted(filelist):
        icm = ICMio(flname)
        print(icm)
        print(icm.get_processor_version())
        print(icm.get_creation_time())
        print(icm.get_coverage_time())

        if icm.select('ANALOG_OFFSET_SWIR'):
            print(icm.get_ref_time())
            print(icm.get_delta_time())
            print(icm.get_instrument_settings())
            print(icm.get_housekeeping_data())
            for key in icm.get_geo_data():
                print('GEO[{}]: '.format(key), icm.get_geo_data()[key].shape)
            res = icm.get_msm_data('analog_offset_swir_value')
            print('analog_offset_swir_value: ', res.shape)
            print(icm.get_msm_attr('analog_offset_swir_value', 'units'))

        if icm.select('BACKGROUND_MODE_1063',
                      msm_path='BAND%_CALIBRATION'):
            print(icm.get_ref_time())
            print(icm.get_delta_time())
            print(icm.get_instrument_settings())
            print(icm.get_housekeeping_data())
            for key in icm.get_geo_data():
                print('GEO[{}]: '.format(key), icm.get_geo_data()[key].shape)
            res = icm.get_msm_data('signal_avg')
            print('signal_avg: ', res.shape)
            print(icm.get_msm_attr('signal_avg', 'units'))
            res = icm.get_msm_data('biweight_value')
            print('biweight_value: ', res.shape)
            print(icm.get_msm_attr('biweight_value', 'units'))
            res = icm.get_msm_data('measurement_quality')
            print('measurement_quality: ', res[0].shape, np.sum(res == 0))

        if icm.select('SOLAR_IRRADIANCE_MODE_0202'):
            #print(icm.get_ref_time())
            #print(icm.get_delta_time())
            #print(icm.get_instrument_settings())
            #print(icm.get_housekeeping_data())
            for key in icm.get_geo_data():
                print('GEO[{}]: '.format(key), icm.get_geo_data()[key].shape)
            res = icm.get_msm_data('irradiance_avg')
            print('irradiance_avg: ', res.shape)
            print(icm.get_msm_attr('irradiance_avg', 'units'))
            res = icm.get_msm_data('measurement_quality')
            print('measurement_quality: ', res[0].shape, np.sum(res == 0))

        if icm.select('EARTH_RADIANCE_MODE_0004'):
            print(icm.get_ref_time())
            print(icm.get_delta_time())
            print(icm.get_instrument_settings())
            print(icm.get_housekeeping_data())
            for key in icm.get_geo_data():
                print('GEO[{}]: '.format(key), icm.get_geo_data()[key].shape)
            res = icm.get_msm_data('radiance_avg_row')
            print('radiance_avg_row: ', res.shape)
            print(icm.get_msm_attr('radiance_avg_row', 'units'))
            res = icm.get_msm_data('measurement_quality')
            print('measurement_quality: ', res[0].shape, np.sum(res == 0))

    ## check patching of ICM product
    for flname in sorted(filelist):
        icm = ICMio(flname)
        if not icm.select('BACKGROUND_MODE_1063'):
            continue

        fl_patch = Path(flname).name
        fl_patch = str(Path('/tmp', fl_patch.replace('_TEST_', '_PATCH_')))
        print(fl_patch)
        shutil.copy(flname, fl_patch)
        icm = ICMio(fl_patch, readwrite=True)
        bands = icm.select('BACKGROUND_MODE_1063')
        print('bands: ', bands)
        res = icm.get_msm_data('signal_avg', band=bands[0])
        print('signal_avg[7]: ', res.shape)
        res[:, :] = 2
        icm.set_msm_data('signal_avg', res, band='7')
        res = icm.get_msm_data('signal_avg', '8')
        print('signal_avg[8]: ', res.shape)
        res[:, :] = 3
        icm.set_msm_data('signal_avg', res, band='8')

if __name__ == '__main__':
    test_rd_icm()
