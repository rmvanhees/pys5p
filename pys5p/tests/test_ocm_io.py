from __future__ import absolute_import
from __future__ import print_function

import os.path

from glob import glob
from unittest import TestCase

#--------------------------------------------------
def test_rd_ocm( msm_dset='signal', print_data=False ):
    """
    Perform some simple test to check the OCM_io class

    Please use the code as tutorial

    """
    from ..get_data_dir import get_data_dir
    from ..ocm_io import OCMio

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    if not os.path.isdir(os.path.join(data_dir, 'OCM')):
        return
    msmlist = glob(os.path.join(data_dir, 'OCM', '*'))
    sdirlist = glob(os.path.join(msmlist[0], '*'))

    # read background measurements
    msm_icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = os.path.join( sdirlist[0], product_b7 )

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

class TestCmd(TestCase):
    def test_basic(self):
        test_rd_ocm()
