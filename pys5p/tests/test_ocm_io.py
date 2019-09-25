"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class OCMio

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
from pys5p.ocm_io import OCMio


#--------------------------------------------------
def test_rd_ocm(msm_dset='signal', print_data=False):
    """
    Perform some simple checks on the OCMio class

    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    if not Path(data_dir, 'OCM').is_dir():
        return
    msmlist = list(Path(data_dir, 'OCM').glob('*'))
    sdirlist = list(Path(msmlist[0]).glob('*'))

    # read background measurements
    msm_icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = str(Path(sdirlist[0], product_b7))

    # open OCAL Lx poduct
    ocm = OCMio(ocm_product)
    print(ocm, file=sys.stderr)
    res = ocm.get_processor_version()
    if print_data:
        print(res)
    else:
        print('*** INFO: successfully obtained processor version')
    res = ocm.get_coverage_time()
    if print_data:
        print(res)
    else:
        print('*** INFO: successfully obtained coverage-period of measurements')

    # select data of measurement(s) with given ICID
    if ocm.select(msm_icid) > 0:
        res = ocm.get_ref_time()
        if print_data:
            print(res)
        else:
            print('*** INFO: successfully obtained reference time')
        res = ocm.get_delta_time()
        if print_data:
            print(res)
        else:
            print('*** INFO: successfully obtained delta time')
        res = ocm.get_instrument_settings()
        if print_data:
            print(res)
        else:
            print('*** INFO: successfully obtained instrument settings')
        res = ocm.get_housekeeping_data()
        if print_data:
            print(res)
        else:
            print('*** INFO: successfully obtained housekeeping data')

        res = ocm.get_msm_data(msm_dset)
        if print_data:
            print(res)
        else:
            print('*** INFO: successfully obtained msm_data')

    ocm.close()

if __name__ == '__main__':
    test_rd_ocm()
