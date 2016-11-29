from __future__ import absolute_import
from __future__ import print_function

import os.path

from unittest import TestCase

from pys5p.l1b_io import L1BioCAL, L1BioIRR, L1BioRAD

#------------------------- TEST-modules and Tutorials -------------------------
def test_rd_calib( l1b_product, msm_type, msm_dset, verbose ):
    """
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    """
    l1b = L1BioCAL( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    res = l1b.get_instrument_settings()
    print( 'instrument settings [{}]: '.format(res.size), res )
    res = l1b.get_housekeeping_data()
    print( 'housekeeping data [{}]: '.format(res.size), res ) 
    geo = l1b.get_geo_data()
    print( 'geodata: ', geo.dtype.names, geo.shape )
    for ii in range(geo.size):
        print( ii, geo['sequence'][ii], geo['satellite_latitude'][ii],
               geo['satellite_longitude'][ii] )
    dset = l1b.get_msm_data( msm_dset )
    print( '{}: {}'.format(msm_dset, dset.shape) )
    del l1b

def test_rd_irrad( l1b_product, msm_type, msm_dset, verbose ):
    """
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    """
    l1b = L1BioIRR( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    res = l1b.get_instrument_settings()
    print( 'instrument settings [{}]: '.format(res.size), res )
    res = l1b.get_housekeeping_data()
    print( 'housekeeping data [{}]: '.format(res.size), res ) 
    dset = l1b.get_msm_data( msm_dset )
    print( '{}: {}'.format(msm_dset, dset.shape) )
    del l1b

def test_rd_rad( l1b_product, icid, msm_dset, verbose ):
    """
    Perform some simple tests to check the L01BioRAD classes

    Please use the code as tutorial
    """
    l1b = L1BioRAD( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select()
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    res = l1b.get_instrument_settings()
    print( 'instrument settings [{}]: '.format(res.size), res )
    res = l1b.get_housekeeping_data( icid=icid )
    print( 'housekeeping data [{}]: '.format(res.size), res ) 
    geo = l1b.get_geo_data( icid=icid )
    print( 'geodata: ', geo.dtype.names, geo.shape )
    for ii in range(geo.shape[0]):
        print( ii, geo['sequence'][ii, 117], geo['latitude'][ii, 117],
               geo['longitude'][ii, 117] )
    print( msm_dset, l1b.get_msm_data( msm_dset, icid=icid ).shape )
    del l1b

def _main():
    """
    Let the user test the software!!!
    """
    import argparse

    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='run test-routines to check class L1BioXXX' )
    parser.add_argument( 'l1b_product', default=None,
                         help='name of L1B product (full path)' )
    parser.add_argument( '--msm_type', default=None,
                         help='define measurement type as <processing class>_<ic_id>' )
    parser.add_argument( '--icid', default=None,
                         type=int, choices=[2, 4, 6, 8, 10],
                         help='define ic_id, only for radiance measurements' )
    parser.add_argument( '--msm_dset', default=None,
                         help='define measurement dataset to be read/patched' )
    parser.add_argument( '--quiet', dest='verbose', action='store_false',
                         default=True, help='only show error messages' )
    args = parser.parse_args()
    if args.verbose:
        print( args )
    if args.l1b_product is None:
        parser.print_usage()
        parser.exit()

    prod_type = os.path.basename(args.l1b_product)[0:15]
    if prod_type == 'S5P_OFFL_L1B_CA' or prod_type == 'S5P_TEST_L1B_CA':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'BACKGROUND_RADIANCE_MODE_0005'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'signal'
        print('calib: ', msm_type, msm_dset)
        test_rd_calib( args.l1b_product, msm_type, msm_dset, args.verbose )
    elif prod_type == 'S5P_OFFL_L1B_IR' or prod_type == 'S5P_TEST_L1B_IR':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'STANDARD_MODE'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'irradiance'
        print('irrad: ', msm_type, msm_dset)
        test_rd_irrad( args.l1b_product, msm_type, msm_dset,  args.verbose )
    elif prod_type == 'S5P_OFFL_L1B_RA' or prod_type == 'S5P_TEST_L1B_RA':
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'radiance'
        print('rad: ', args.icid, msm_dset)
        test_rd_rad( args.l1b_product, args.icid, msm_dset, args.verbose )
    else:
        print( ' *** FATAL: unknown product type' )

#-------------------------
if __name__ == '__main__':
    _main()
