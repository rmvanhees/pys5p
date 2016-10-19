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
import h5py

#--------------------------------------------------
class L1Bpatch( object ):
    '''
    '''
    def fill_constant( self, data, value ):
        print( data.shape, value )
        data[:] = value
        return data
    
    def background( self ):
        '''
        '''
        pass

    def nonlinearity( self ):
        '''
        '''
        pass

    def straylight( self ):
        '''
        '''
        pass

    def radiance( self ):
        '''
        '''
        pass
    
#--------------------------------------------------
def main():
    '''
    Let the user test the software!!!
    '''
    import argparse
    import shutil
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
        l1b = L1BioRAD( l1b_patch, readwrite=True )
        l1b.select( msm_type )
        data = l1b.get_msm_data( msm_dset )
        print( data.shape )
        patch = L1Bpatch()
        data = patch.fill_constant( data, 7 )
        print( data.shape )
        l1b.set_msm_data( msm_dset, data )
        print( data.shape )
        del l1b
        del patch

if __name__ == '__main__':
    main()
