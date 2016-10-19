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

    Usage:
    1) open file (new class initiated)
    2) select group of a particular measurement
    3) read data (median/averaged full frame(s), only)
    .
    . <user actions>
    .
    * back to step 2) or
    5) close file
    '''
    def __init__( self, ocm_product, verbose=False ):
        '''
        Initialize access to an ICM products

        Parameters
        ----------
        ocm_product :  string
           Patch to on-ground calibration measurement

        Note that each band is stored in a seperate product: trl1brb?g.lx.nc
        '''
        assert os.path.isfile( ocm_product ), \
            '*** Fatal, can not find OCAL Lx product: {}'.format(ocm_product)

        # initialize class-attributes
        self.__product = ocm_product
        self.__verbose = verbose
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
        Parameters
        ----------
        ic_id  :  integer
          used as "BAND%/ICID_{}_GROUP_%".format(ic_id)

        Returns
        -------
        Number of measurements found

        Updated object attributes:
         - ref_time            : reference time of measurement (datetime-object)
         - delta_time          : offset w.r.t. reference time (milli-seconds)
         - instrument_settings : copy of instrument settings
         - housekeeping_data   : copy of housekeeping data
        '''
        self.band = ''
        self.__msm_path = []
        for ii in '12345678':
            if 'BAND{}'.format(ii) in self.fid:
                self.band = ii
                break

        if len(self.band) > 0:
            grp_name = 'ICID_{:05}_GROUP'.format(ic_id)
            gid = self.fid['BAND{}'.format(self.band)]
            self.__msm_path = [s for s in gid if s.startswith(grp_name)]

        return len(self.__msm_path)

    #-------------------------
    def get_msm_data( self, msm_dset, fill_as_nan=False ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset

        Returns
        -------
        Python dictionary with msm_names as keys and their values

        '''
        fillvalue = float.fromhex('0x1.ep+122')

        res = {}
        if len(self.__msm_path) == 0:
            return res

        grp = self.fid['BAND{}'.format(self.band)]
        for msm in sorted(self.__msm_path):
            ds_path = os.path.join(msm, 'OBSERVATIONS', msm_dset)

            data = np.squeeze(grp[ds_path])
            if fill_as_nan and grp[ds_path].attrs['_FillValue'] == fillvalue:
                data[(data == fillvalue)] = np.nan
            res[msm] = data

        return res

#--------------------------------------------------
def test():
    '''
    Perform some simple test to check the OCM_io class
    '''
    if os.path.isdir('/Users/richardh'):
        fl_path = '/Users/richardh/Data/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'
    elif os.path.isdir('/nfs/TROPOMI/ocal/proc_knmi'):
        fl_path = '/nfs/TROPOMI/ocal/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'
    else:
        fl_path = '/data/richardh/Tropomi/ISRF/2015_02_23T01_36_51_svn4709_CellEarth_CH4'
    ocm_dir = 'after_strayl_l1b_val_SWIR_2'
    ocm_product = os.path.join(fl_path, ocm_dir, 'trl1brb7g.lx.nc')

    ocm = OCMio( ocm_product, verbose=True )
    print( ocm.get_processor_version() )
    print( ocm.get_coverage_time() )

    ocm.select( 31524 )
    for key in ocm:
        print( '{}: {!r}'.format(key, ocm.__getattribute__(key)) )
    print( ocm.get_ref_time() )
    print( ocm.get_delta_time() )
    instrument_settings = ocm.get_instrument_settings()
    print( instrument_settings.keys() )
    housekeeping_data = ocm.get_housekeeping_data()
    print( housekeeping_data.keys() )
    msm_data = ocm.get_msm_data( 'signal' )
    print( msm_data.values() )
    del ocm

#--------------------------------------------------
if __name__ == '__main__':
    test()
