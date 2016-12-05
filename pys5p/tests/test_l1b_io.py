from __future__ import absolute_import
from __future__ import print_function

import os.path

from glob import glob
from unittest import TestCase

#------------------------- TEST-modules and Tutorials -------------------------
def test_rd_calib(msm_type='BACKGROUND_RADIANCE_MODE_0005', msm_dset='signal'):
    """
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    """
    from ..get_data_dir import get_data_dir
    from ..l1b_io import L1BioCAL

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_CA_*.nc'))
    if len(filelist) == 0:
        return

    l1b = L1BioCAL(filelist[-1])
    print( l1b )
    print('orbit:   ', l1b.get_orbit())
    print('version: ', l1b.get_processor_version())
    l1b.select( msm_type )
    for key in l1b:
        print('{}: {!r}'.format(key, l1b.__getattribute__(key)))

    print('reference time: ', l1b.get_ref_time())
    print('delta time: ', l1b.get_delta_time())
    res = l1b.get_instrument_settings()
    print('instrument settings [{}]: '.format(res.size), res.shape)
    res = l1b.get_housekeeping_data()
    print('housekeeping data [{}]: '.format(res.size), res.shape) 
    geo = l1b.get_geo_data()
    print('geodata: ', geo.dtype.names, geo.shape)
    #for ii in range(geo.size):
    #    print( ii, geo['sequence'][ii], geo['satellite_latitude'][ii],
    #           geo['satellite_longitude'][ii] )
    dset = l1b.get_msm_data( msm_dset, band=l1b.bands[0:2] )
    print('{}: {}'.format(msm_dset, dset.shape))
    del l1b

def test_rd_irrad(msm_type='STANDARD_MODE', msm_dset='irradiance'):
    """
    Perform some simple tests to check the L1BioIRR classes

    Please use the code as tutorial

    """
    from ..get_data_dir import get_data_dir
    from ..l1b_io import L1BioIRR

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_IR_*.nc'))
    if len(filelist) == 0:
        return

    l1b = L1BioIRR(filelist[-1])
    print( l1b )
    print('orbit:   ', l1b.get_orbit())
    print('version: ', l1b.get_processor_version())
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    res = l1b.get_instrument_settings()
    print( 'instrument settings [{}]: '.format(res.size), res.shape )
    res = l1b.get_housekeeping_data()
    print( 'housekeeping data [{}]: '.format(res.size), res.shape )
    dset = l1b.get_msm_data( msm_dset, band=l1b.bands[0:2] )
    print( '{}: {}'.format(msm_dset, dset.shape) )
    del l1b

def test_rd_rad(icid=4, msm_dset='radiance'):
    """
    Perform some simple tests to check the L01BioRAD classes

    Please use the code as tutorial
    """
    from ..get_data_dir import get_data_dir
    from ..l1b_io import L1BioRAD

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'L1B', 'S5P_OFFL_L1B_RA_*.nc'))
    if len(filelist) == 0:
        return

    l1b = L1BioRAD( filelist[-1] )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select()
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    res = l1b.get_instrument_settings()
    print( 'instrument settings [{}]: '.format(res.size), res.shape )
    res = l1b.get_housekeeping_data( icid=icid )
    print( 'housekeeping data [{}]: '.format(res.size), res.shape )
    geo = l1b.get_geo_data( icid=icid )
    print( 'geodata: ', geo.dtype.names, geo.shape )
    #for ii in range(geo.shape[0]):
    #    print( ii, geo['sequence'][ii, 117], geo['latitude'][ii, 117],
    #           geo['longitude'][ii, 117] )
    print( msm_dset, l1b.get_msm_data( msm_dset, icid=icid ).shape )
    del l1b

class TestCmd(TestCase):
    def test_basic(self):
        test_rd_calib()
        test_rd_irrad()
        test_rd_rad()
