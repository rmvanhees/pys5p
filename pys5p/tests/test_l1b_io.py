"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class L1Bio

Note
----
Please use the code as tutorial

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import sys
from pathlib import Path

from pys5p.get_data_dir import get_data_dir

#------------------------- TEST-modules and Tutorials -------------------------
def test_rd_calib(msm_type='BACKGROUND_RADIANCE_MODE_0005', msm_dset='signal'):
    """
    Perform some simple checks on the L1BioCAL class

    """
    from pys5p.l1b_io import L1Bio

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_CA_*.nc'))
    if not filelist:
        return

    with L1Bio(filelist[-1]) as l1b:
        print(l1b, file=sys.stderr)
        print('orbit:   ', l1b.get_orbit())
        print('version: ', l1b.get_processor_version())
        l1b.select(msm_type)
        for key in l1b:
            print('{}: {!r}'.format(key, l1b.__getattribute__(key)))

        print('reference time: ', l1b.get_ref_time())
        print('delta time: ', l1b.get_delta_time())
        res = l1b.get_instrument_settings()
        print('instrument settings [{}]: '.format(res.size), res.shape)
        res = l1b.get_exposure_time()
        print('exposure time [{}]: '.format(len(res)), res)
        res = l1b.get_housekeeping_data()
        print('housekeeping data [{}]: '.format(res.size), res.shape)
        geo = l1b.get_geo_data()
        print('geodata: ', geo.dtype.names, geo.shape)
        #for ii in range(geo.size):
        #    print(ii, geo['sequence'][ii], geo['satellite_latitude'][ii],
        #           geo['satellite_longitude'][ii])
        dset = l1b.get_msm_data(msm_dset, band=l1b.bands[0:2])
        print('{}: {}'.format(msm_dset, dset.shape))

def test_rd_irrad(msm_type='STANDARD_MODE', msm_dset='irradiance'):
    """
    Perform some simple checks on the L1BioIRR class

    """
    from pys5p.l1b_io import L1BioIRR

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_IR_*.nc'))
    if not filelist:
        return

    with L1BioIRR(filelist[-1]) as l1b:
        print(l1b, file=sys.stderr)
        print('orbit:   ', l1b.get_orbit())
        print('version: ', l1b.get_processor_version())
        print(l1b.select(msm_type))
        for key in l1b:
            print('{}: {!r}'.format(key, l1b.__getattribute__(key)))

        print('reference time: ', l1b.get_ref_time())
        print('delta time: ', l1b.get_delta_time())
        res = l1b.get_instrument_settings()
        print('instrument settings [{}]: '.format(res.size), res.shape)
        res = l1b.get_exposure_time()
        print('exposure time [{}]: '.format(len(res)), res)
        res = l1b.get_housekeeping_data()
        print('housekeeping data [{}]: '.format(res.size), res.shape)
        dset = l1b.get_msm_data(msm_dset, band=l1b.bands[0:2])
        print('{}: {}'.format(msm_dset, dset.shape))

def test_rd_rad(msm_type='STANDARD_MODE', icid=4, msm_dset='radiance'):
    """
    Perform some simple checks on the L01BioRAD class

    """
    from pys5p.l1b_io import L1BioRAD

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_RA_*.nc'))
    if not filelist:
        return

    l1b = L1BioRAD(filelist[-1])
    print(l1b, file=sys.stderr)
    print('orbit:   ', l1b.get_orbit())
    print('version: ', l1b.get_processor_version())
    l1b.select(msm_type)
    for key in l1b:
        print('{}: {!r}'.format(key, l1b.__getattribute__(key)))

    print('reference time: ', l1b.get_ref_time())
    print('delta time: ', l1b.get_delta_time())
    res = l1b.get_instrument_settings()
    print('instrument settings [{}]: '.format(res.size), res.shape)
    res = l1b.get_exposure_time()
    print('exposure time [{}]: '.format(len(res)), res)
    # res = l1b.get_housekeeping_data(icid=icid)
    res = l1b.get_housekeeping_data()
    print('housekeeping data [{}]: '.format(res.size), res.shape)
    # geo = l1b.get_geo_data(icid=icid)
    geo = l1b.get_geo_data()
    print('geodata: ', geo.dtype.names, geo.shape)
    #for ii in range(geo.shape[0]):
    #    print(ii, geo['sequence'][ii, 117], geo['latitude'][ii, 117],
    #           geo['longitude'][ii, 117])
    # print(msm_dset, l1b.get_msm_data(msm_dset, icid=icid).shape)
    print(msm_dset, l1b.get_msm_data(msm_dset).shape)
    l1b.close()

def test_rd_eng():
    """
    Perform some simple checks on the L1BioENG class

    """
    from pys5p.l1b_io import L1BioENG

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_ENG_DB_*.nc'))
    if not filelist:
        return

    with L1BioENG(filelist[-1]) as l1b:
        print(l1b, file=sys.stderr)
        print('orbit:   ', l1b.get_orbit())
        print('version: ', l1b.get_processor_version())
        for key in l1b:
            print('{}: {!r}'.format(key, l1b.__getattribute__(key)))

        print('reference time: ', l1b.get_ref_time())
        print('delta time: ', l1b.get_delta_time().size)
        print('msmtset: ', l1b.get_msmtset().size) 
        print('msmtset_db: ', l1b.get_msmtset_db())
        print('swir_hk_db: ', l1b.get_swir_hk_db())


if __name__ == '__main__':
    test_rd_calib()
    test_rd_irrad()
    test_rd_rad()
    test_rd_eng()
