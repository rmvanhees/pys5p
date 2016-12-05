from __future__ import absolute_import
from __future__ import print_function

import re
import os.path

from glob import glob
from unittest import TestCase

def test_rd_radiance(msm_dset=None):
    """
    Perform a full read-test a L1B radiance product using the L1BioRAD class

    Please use the code as tutorial

    """
    from pys5p.get_data_dir import get_data_dir
    from pys5p.l1b_io import L1BioRAD

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_RA_*.nc'))
    if len(filelist) == 0:
        return

    for name in filelist:
        l1b = L1BioRAD(os.path.join(data_dir, name))
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
                print('\t instrument settings [{}]: '.format(res3.size),
                      res3.shape)
                res4 = l1b.get_housekeeping_data()
                print('\t housekeeping data [{}]: '.format(res4.size),
                      res4.shape)
                geo = l1b.get_geo_data()
                print('\t geodata[:]: {}'.format(geo.shape))

                if msm_dset is None:
                    dset_name = 'radiance'
                else:
                    dset_name = msm_dset

                dset = l1b.get_msm_data( dset_name )
                print('\t {}[:]: {}'.format(dset_name, dset.shape))
                print()
                for icid in sorted(res3['ic_id']):
                    geo = l1b.get_geo_data(icid=icid)
                    print('\t geodata[{}]: {}'.format(icid, geo.shape))
                    dset = l1b.get_msm_data(dset_name, icid=icid)
                    print('\t {}[{}]: {}'.format(dset_name, icid, dset.shape))

    del l1b

if __name__ == '__main__':
    test_rd_radiance()
