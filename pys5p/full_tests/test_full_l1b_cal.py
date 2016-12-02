from __future__ import absolute_import
from __future__ import print_function

import re
import os.path

from glob import glob
from unittest import TestCase

def test_rd_calib(msm_dset=None):
    """
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    """
    from pys5p.l1b_io import L1BioCAL

    data_dir = '/data/richardh/pys5p-data/L1B'
    cal_prod = 'S5P_OFFL_L1B_CA_UVN_20150821T012540_20150821T030710_00058_01_000000_20160721T173643.nc'

    l1b = L1BioCAL(os.path.join(data_dir, cal_prod))
    print( l1b )
    print('orbit:   ', l1b.get_orbit())
    print('version: ', l1b.get_processor_version())
    for key1 in l1b.fid:
        if not key1.startswith('BAND'):
            continue
        print(key1)
        for key2 in l1b.fid[key1]:
            print('-->', key2)
            l1b.select( key2 )
            res1 = l1b.get_ref_time()
            res2 = l1b.get_delta_time()
            print('\t delta time: ', res2.shape)
            res3 = l1b.get_instrument_settings()
            print('\t instrument settings [{}]: '.format(res3.size), res3.shape)
            res4 = l1b.get_housekeeping_data()
            print('\t housekeeping data [{}]: '.format(res4.size), res4.shape) 
            geo = l1b.get_geo_data()
            print('\t geodata: ', geo.dtype.names, geo.shape)
            #for ii in range(geo.size):
            #    print( '\t', ii, geo['sequence'][ii],
            #           geo['satellite_latitude'][ii],
            #           geo['satellite_longitude'][ii] )
            if msm_dset is None:
                if key1.endswith('_RADIANCE'):
                    dset_name = 'radiance_avg'
                elif key1.endswith('_IRRADIANCE'):
                    dset_name = 'irradiance_avg'
                else:
                    dset_name = 'signal'
            else:
                dset_name = msm_dset

            #for ib in l1b.bands:
            #    dset = l1b.get_msm_data( dset_name, band=ib )
            #    print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))

            for ib in re.findall('..', l1b.bands):
                dset = l1b.get_msm_data( dset_name, band=ib )
                print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))
                
    del l1b


if __name__ == '__main__':
    test_rd_calib()
