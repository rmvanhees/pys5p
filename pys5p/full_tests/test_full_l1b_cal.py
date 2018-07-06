"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on L1BioCAL

Note
----
Please use the code as tutorial

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
import sys
import re

from pathlib import Path

def test_rd_calib(msm_dset=None):
    """
    Perform a full read-test a L1B calibration product using the L1BioCAL class

    """
    from ..get_data_dir import get_data_dir
    from ..l1b_io import L1BioCAL

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_CA_*.nc'))
    if not filelist:
        return

    for name in sorted(filelist):
        print(name, file=sys.stderr)
        l1b = L1BioCAL(name)
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
                _ = l1b.get_ref_time()
                res2 = l1b.get_delta_time()
                print('\t delta time: ', res2.shape)
                res3 = l1b.get_instrument_settings()
                print('\t instrument settings [{}]: '.format(res3.size),
                      res3.shape)
                res4 = l1b.get_housekeeping_data()
                print('\t housekeeping data [{}]: '.format(res4.size),
                      res4.shape)
                geo = l1b.get_geo_data()
                print('\t geodata: ', geo.dtype.names, geo.shape)

                if msm_dset is None:
                    if key1.endswith('_RADIANCE'):
                        dset_name = 'radiance_avg'
                    elif key1.endswith('_IRRADIANCE'):
                        dset_name = 'irradiance_avg'
                    else:
                        if key2.endswith('_AFTER_DN2V'):
                            dset_name = 'signal_avg'
                        else:
                            dset_name = 'signal'
                else:
                    dset_name = msm_dset

                for ib in l1b.bands:
                    dset = l1b.get_msm_data( dset_name, band=ib )
                    print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))

                for ib in re.findall('..', l1b.bands):
                    dset = l1b.get_msm_data( dset_name, band=ib )
                    print('\t {}[{}]: {}'.format(dset_name, ib, dset.shape))

    del l1b

if __name__ == '__main__':
    test_rd_calib()
