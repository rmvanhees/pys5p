'''
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

'''
from __future__ import print_function

import numpy as np
#import h5py

#--------------------------------------------------
def fill_constant( array, value ):
    '''
    Basic test-operation: replace values in ndarray with a constant value.

    Parameters
    ----------
    array  :  array-like
       The data to be patched
    value  :  scalar
       Fill value

    Returns
    -------
    out    :  ndarray
       Return an array with each element set to value
    '''
    return np.full_like( array, value )

def nonlinearity():
    '''
    Patch non-linearity correction.

    Low priority, small effect, hard to implement and validate

    Requires (naive approach):
     * reverse applied radiance calibration
     * reverse applied stray-light correction
     * reverse applied PRNU correction
     * reverse applied dark-current correction
     * apply alternative non-linearity correction
     * apply (alternative) PRNU correction
     * apply (alternative) stray-light correction
     * apply (alternative) radiance calibration

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def prnu():
    '''
    Patch pixel response non-uniformity correction.

    High priority, introduces potentially spectral-features (swath dependent)

    Requires (naive approach):
     * reverse applied radiance calibration
     * reverse applied stray-light correction
     * reverse applied PRNU correction
     * reverse applied dark-current correction
     * apply (alternative) dark-current correction
     * apply (alternative) PRNU correction
     * apply (alternative) stray-light correction
     * apply (alternative) radiance calibration

    Alternative: neglect impact stray-light, but apply patch to correct for 
       spectral features

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def straylight():
    '''
    Patch in-of-band stray-light correction.

    High priority, correction implemented for the most important measured 
    features, however, at low resolution. Met requirement within factor 3-6.
    Spatial stray-light seems to be the biggest problem. Unclear how hard it
    is to correct

    Requires (naive approach):
     * reverse applied radiance calibration
     * reverse applied stray-light correction
     * apply alternative stray-light correction
     * apply (alternative) radiance calibration

    Alternative: add correction of spatial stray-light

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def stray_oob():
    '''
    Patch out-of-band stray-light correction.

    Low priority, but relatively easy to implement and verify

    Requires:
     * reverse applied radiance calibration
     * apply alternative OOB stray-light correction
     * apply (alternative) radiance calibration

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def nir_oob():
    '''
    Patch NIR out-of-band stray-light correction.

    Low priority, task for KNMI

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def radiance():
    '''
    Patch radiance calibration.

    Low priority: not discussed

    Requires:
     * reverse applied radiance calibration
     * apply alternative radiance calibration

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

def irradiance():
    '''
    Patch irradiance calibration.

    Low priority: not discussed

    Requires:
     * reverse applied irradiance calibration
     * apply alternative irradiance calibration

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    '''
    pass

#--------------------------------------------------
def _main():
    '''
    Let the user test the software!!!
    '''
    import argparse
    from pathlib import Path

    from l1b_io import L1BioRAD

    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='run test-routines to check class L1BioXXX' )
    parser.add_argument( 'l1b_product', default=None,
                         help='name of L1B product (full path)' )
    parser.add_argument( '--msm_type', default=None,
                         help='define measurement type as <processing class>_<ic_id>' )
    parser.add_argument( '--msm_dset', default=None,
                         help='define measurement dataset to be read/patched' )
    parser.add_argument( '-o', '--output', default='/tmp',
                         help='directory to store patched product' )
    parser.add_argument( '--quiet', dest='verbose', action='store_false',
                         default=True, help='only show error messages' )
    args = parser.parse_args()
    if args.verbose:
        print( args )
    if args.l1b_product is None:
        parser.print_usage()
        parser.exit()

    if not Path(args.output).is_dir():
        Path(args.output).mkdir(mode=0o755)

    l1b_patch = str(Path(args.output,
                         Path(args.l1b_product.replace('_01_', '_02_')).name))
    print( args.l1b_product, l1b_patch )
    #shutil.copy( args.l1b_product, l1b_patch )

    prod_type = Path(l1b_patch).name[0:15]
    if prod_type == 'S5P_OFFL_L1B_RA':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'STANDARD_MODE'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'radiance'
        print('rad: ', msm_type, msm_dset)

        # open L1B product and read dataset
        l1b = L1BioRAD( l1b_patch, readwrite=True )
        l1b.select( msm_type )
        data = l1b.get_msm_data( msm_dset )

        # patch dataset
        data = fill_constant( data, 7 )

        # write patched data to L1B product
        l1b.set_msm_data( msm_dset, data )

        # update meta-data of product and flush all changes to disk
        del l1b

if __name__ == '__main__':
    _main()
