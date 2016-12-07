from __future__ import absolute_import
from __future__ import print_function

import re
import os.path

from glob import glob
from unittest import TestCase

def test_rd_icm(msm_dset=None):
    """
    Perform a full read-test a ICM product using the ICMio class

    Please use the code as tutorial

    """
    from ..get_data_dir import get_data_dir
    from ..icm_io import ICMio

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'ICM', 'S5P_TEST_ICM_CA_*.h5'))
    if len(filelist) == 0:
        return

    for name in filelist:
        icm = ICMio(name)
        print( icm )
        print('version: ', icm.get_processor_version())
        print('creation_time', icm.get_creation_time())
        print('coverage_time', icm.get_coverage_time())
        for key1 in icm.fid:
            if not key1.startswith('BAND'):
                continue
            print(key1)
            for key2 in icm.fid[key1]:
                print('-->', key2)
                icm.select( key2 )
                res1 = icm.get_ref_time()
                res2 = icm.get_delta_time()
                print('\t delta time: ', res2.shape)
                res3 = icm.get_instrument_settings()
                print('\t instrument settings [{}]: '.format(res3.size),
                      res3.shape)
                res4 = icm.get_housekeeping_data()
                print('\t housekeeping data [{}]: '.format(res4.size),
                      res4.shape) 

                if msm_dset is None:
                    if key1.endswith('_RADIANCE'):
                        geo = icm.get_geo_data(band=icm.bands[0],
                                               geo_dset='latitude,longitude')
                        print('\t geodata: ', geo.shape)
                        dset_name = 'radiance_avg'
                    elif key1.endswith('_IRRADIANCE'):
                        geo = icm.get_geo_data(band=icm.bands[0])
                        print('\t geodata: ', geo.shape)
                        dset_name = 'irradiance_avg'
                    elif key1.endswith('_ANALYSIS'):
                        if key2 == 'ANALOG_OFFSET_SWIR':
                            dset_name = 'analog_offset_swir_value'
                        elif key2 == 'DPQF_MAP':
                            dset_name = 'dpqf_map'
                        elif key2 == 'LONG_TERM_SWIR':
                            dset_name = 'long_term_swir_value'
                        elif key2 == 'NOISE':
                            dset_name = 'noise'
                        else:
                            dset_name = 'signal_avg'
                    else:
                        geo = icm.get_geo_data(band=icm.bands[0])
                        print('\t geodata: ', geo.shape)
                        dset_name = 'signal_avg'
                else:
                    dset_name = msm_dset

                for ib in icm.bands:
                    dset = icm.get_msm_data( dset_name, band=ib )
                    print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))

                for ib in re.findall('..', icm.bands):
                    dset = icm.get_msm_data( dset_name, band=ib )
                    print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))

        del icm

if __name__ == '__main__':
    test_rd_icm()
