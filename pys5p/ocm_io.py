'''
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The classe OCMio provide read access to Tropomi on-ground calibration products
(Lx)

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

'''
from __future__ import print_function
from __future__ import division

import os.path

import numpy as np
import h5py

#--------------------------------------------------
class OCMio( object ):
    '''
    This class should offer all the necessary functionality to read Tropomi
    on-ground calibration products (Lx)
    '''
    def __init__( self, ocm_product ):
        '''
        Initialize access to an OCAL Lx product

        Parameters
        ----------
        ocm_product :  string
           Full path to on-ground calibration measurement

        '''
        assert os.path.isfile( ocm_product ), \
            '*** Fatal, can not find OCAL Lx product: {}'.format(ocm_product)

        # initialize class-attributes
        self.__product = ocm_product
        self.__msm_path = None
        self.__patched_msm = []
        self.band = None

        # open OCM product as HDF5 file
        self.fid = h5py.File( ocm_product, "r" )

    def __repr__( self ):
        class_name = type(self).__name__
        return '{}({!r})'.format( class_name, self.__product )

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __del__( self ):
        self.band = None
        self.fid.close()

    # ---------- RETURN VERSION of the S/W ----------
    @staticmethod
    def pynadc_version():
        '''
        Return S/W version
        '''
        from importlib import util

        version_spec = util.find_spec( "pynadc.version" )
        assert version_spec is not None

        from pynadc import version
        return version.__version__

    # ---------- Functions that work before MSM selection ----------
    def get_processor_version( self ):
        '''
        Returns version of the L01b processor
        '''
        return self.fid.attrs['processor_version'].decode('ascii')

    def get_coverage_time( self ):
        '''
        Returns start and end of the measurement coverage time
        '''
        return (self.fid.attrs['time_coverage_start'].decode('ascii'),
                self.fid.attrs['time_coverage_end'].decode('ascii'))

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time( self ):
        '''
        Returns reference start time of measurements
        '''
        from datetime import datetime, timedelta

        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        for msm in sorted(self.__msm_path):
            sgrp = grp[os.path.join(msm, 'GEODATA')]
            res[msm] = datetime(2010,1,1,0,0,0) \
                       + timedelta(seconds=int(sgrp['time'][0]))

        return res

    def get_delta_time( self ):
        '''
        Returns offset from the reference start time of measurement
        '''
        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        for msm in sorted(self.__msm_path):
            sgrp = grp[os.path.join(msm, 'GEODATA')]
            res[msm] = sgrp['delta_time'][:].astype(int)

        return res

    def get_instrument_settings( self ):
        '''
        Returns instrument settings of measurement
        '''
        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        for msm in sorted(self.__msm_path):
            sgrp = grp[os.path.join(msm, 'INSTRUMENT')]
            res[msm] = np.squeeze(sgrp['instrument_settings'])

        return res

    def get_housekeeping_data( self ):
        '''
        Returns housekeeping data of measurements
        '''
        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        for msm in sorted(self.__msm_path):
            sgrp = grp[os.path.join(msm, 'INSTRUMENT')]
            res[msm] = np.squeeze(sgrp['housekeeping_data'])

        return res

    #-------------------------
    def select( self, ic_id ):
        '''
        Select a measurement as BAND%_<ic_id>_GROUP_%

        Parameters
        ----------
        ic_id  :  integer
          used as "BAND%/ICID_{}_GROUP_%".format(ic_id)
          if ic_id is None then show available ICID's

        Returns
        -------
        Number of measurements found

        Updated object attributes:
          - bands               : available spectral bands
        '''
        self.band = ''
        self.__msm_path = []
        for ii in '12345678':
            if 'BAND{}'.format(ii) in self.fid:
                self.band = ii
                break

        if len(self.band) > 0:
            gid = self.fid['BAND{}'.format(self.band)]
            if ic_id is None:
                grp_name = 'ICID_'
                for kk in gid:
                    if kk.startswith(grp_name):
                        print(kk)
            else:
                grp_name = 'ICID_{:05}_GROUP'.format(ic_id)
                self.__msm_path = [s for s in gid if s.startswith(grp_name)]

        return len(self.__msm_path)

    #-------------------------
    def get_msm_data( self, msm_dset, fill_as_nan=False ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset    :  string
            name of measurement dataset
            if msm_dset is None then show names of available datasets

        fill_as_nan :  boolean
            replace (float) FillValues with Nan's

        Returns
        -------
        Python dictionary with msm_names as keys and their values

        '''
        fillvalue = float.fromhex('0x1.ep+122')

        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        if msm_dset is None:
            ds_path = os.path.join(self.__msm_path[0], 'OBSERVATIONS')
            for kk in grp[ds_path]:
                print(kk)
        else:
            for msm in sorted(self.__msm_path):
                ds_path = os.path.join(msm, 'OBSERVATIONS', msm_dset)

                data = np.squeeze(grp[ds_path])
                if fill_as_nan \
                   and grp[ds_path].attrs['_FillValue'] == fillvalue:
                    data[(data == fillvalue)] = np.nan

                res[msm] = data

        return res

#--------------------------------------------------
def test_rd_ocm( ocm_product, msm_icid, msm_dset, print_data=False ):
    '''
    Perform some simple test to check the OCM_io class
    '''
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

def main():
    '''
    Let the user test the software!!!
    '''
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
    main()
