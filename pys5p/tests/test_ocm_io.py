from __future__ import absolute_import
from __future__ import print_function

import shutil

import os.path

from unittest import TestCase

#--------------------------------------------------
def test_rd_ocm( ocm_product, msm_icid, msm_dset, print_data=False ):
    """
    Perform some simple test to check the OCM_io class

    Please use the code as tutorial

    """
    from pys5p.ocm_io import OCMio

    # open OCAL Lx poduct
    ocm = OCMio( ocm_product )
    res = ocm.get_processor_version()
    if print_data:
        print( res )
    else:
        print('*** INFO: successfully obtained processor version')
    res = ocm.get_coverage_time()
    if print_data:
        print( res )
    else:
        print('*** INFO: successfully obtained coverage-period of measurements')

    # select data of measurement(s) with given ICID
    if ocm.select( msm_icid ) > 0:
        res = ocm.get_ref_time()
        if print_data:
            print( res )
        else:
            print('*** INFO: successfully obtained reference time')
        res = ocm.get_delta_time()
        if print_data:
            print( res )
        else:
            print('*** INFO: successfully obtained delta time')
        res = ocm.get_instrument_settings()
        if print_data:
            print( res )
        else:
            print('*** INFO: successfully obtained instrument settings')
        res = ocm.get_housekeeping_data()
        if print_data:
            print( res )
        else:
            print('*** INFO: successfully obtained housekeeping data')

        res = ocm.get_msm_data( msm_dset )
        if print_data:
            print( res )
        else:
            print('*** INFO: successfully obtained msm_data')

    del ocm

def _main():
    """
    Let the user test the software!!!
    """
    import argparse

    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='run test-routines to check class OCMio' )
    parser.add_argument( 'ocm_product', default=None,
                         help='name of OCAL Lx product (full path)' )
    parser.add_argument( '--msm_icid', default=None, type=int,
                         help='define measurement by ICID' )
    parser.add_argument( '--msm_dset', default=None,
                         help='define measurement dataset to be read' )
    parser.add_argument( '-d', '--data', action='store_true',
                         default=False, help='Print the values of datasets' )
    args = parser.parse_args()

    if args.ocm_product is None:
        parser.print_usage()
        parser.exit()

    test_rd_ocm( args.ocm_product, args.msm_icid, args.msm_dset, args.data )

#--------------------------------------------------
if __name__ == '__main__':
    _main()
