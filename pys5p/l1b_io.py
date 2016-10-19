'''
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The classes L1BioCAL, L1BioIRR, L1BioRAD provide read access to
offline level 1b products, resp. calibration, irradiance and radiance.

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

'''
from __future__ import print_function

import os.path

import numpy as np
import h5py

#--------------------------------------------------
class L1Bio( object ):
    '''
    super class with general function to access Tropomi offline L1b products

    inherited by the classes L1BioCAL, L1BioIRR and L1BioRAD
    '''
    def __init__( self, l1b_product, readwrite=False ):
        '''
        Initialize access to a Tropomi offline L1b product
        '''
        assert os.path.isfile( l1b_product ), \
            '*** Fatal, can not find S5p L1b product: {}'.format(l1b_product)

        # initialize private class-attributes
        self.__product = l1b_product
        self.__rw = readwrite
        self.__patched_msm = []

        # open L1b product as HDF5 file
        if readwrite:
            self.fid = h5py.File( l1b_product, "r+" )
        else:
            self.fid = h5py.File( l1b_product, "r" )

    def __repr__( self ):
        class_name = type(self).__name__
        return '{}({!r}, readwrite={!r})'.format( class_name,
                                                  self.__product, self.__rw )

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __del__( self ):
        '''
        Before closing the product, we make sure that the output product
        describes what has been altered by the S/W. To keep any change
        traceable.

        In case the L1b product is altered, the attributes listed below are
        added to the group: "/METADATA/SRON_METADATA":
             - dateStamp ('now')
             - Git-version of S/W
             - list of patched datasets
             - auxiliary datasets used by patch-routines
        '''
        if len(self.__patched_msm) > 0:
            from datetime import datetime

            sgrp = self.fid.create_group( "/METADATA/SRON_METADATA" )
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = self.pys5p_version()
            dtype = h5py.special_dtype(vlen=str)
            dset = sgrp.create_dataset('patched_datasets',
                                       (len(self.__patched_msm),),
                                       dtype=dtype)

            dset[:] = np.asarray(self.__patched_msm)

        self.fid.close()

    # ---------- PUBLIC FUNCTIONS ----------
    @staticmethod
    def pys5p_version():
        '''
        Returns S/W version
        '''
        from setuptools_scm import get_version

        return get_version()

    def get_orbit( self ):
        '''
        Returns absolute orbit number
        '''
        return int(self.fid.attrs['orbit'])

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

    def get_creation_time( self ):
        '''
        Returns datetime when the L1b product was created
        '''
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].decode('ascii')

    def ref_time( self, msm_path ):
        '''
        Returns reference start time of measurements
        '''
        from datetime import datetime, timedelta

        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path,'OBSERVATIONS')]
        return datetime(2010,1,1,0,0,0) \
            + timedelta(seconds=int(grp['time'][0]))

    def delta_time( self, msm_path ):
        '''
        Returns offset from the reference start time of measurement
        '''
        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path,'OBSERVATIONS')]
        return grp['delta_time'][0,:].astype(int)

    def instrument_settings( self, msm_path ):
        '''
        Returns instrument settings of measurement

        FIXME: h5py crashes on reading instrument_settings from UVN
        '''
        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path,'INSTRUMENT')]
        if grp['instrument_settings'].shape[0] == 1:
            return grp['instrument_settings'][0]
        else:
            return grp['instrument_settings'][:]

    def housekeeping_data( self, msm_path ):
        '''
        Returns housekeeping data of measurements
        '''
        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path,'INSTRUMENT')]
        return np.squeeze(grp['housekeeping_data'])

    def msm_data( self, msm_path, msm_dset, write_data=None ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset

        write_data : array-like
            data to be written with same dimensions as dataset "msm_dset"
            if None then
               data of measurement dataset "msm_dset" is returned
            else
               data of measurement dataset "msm_dset" is overwritten
        '''
        if msm_path is None:
            return None

        ds_path = os.path.join(msm_path, 'OBSERVATIONS', msm_dset)
        if write_data is None:
            return np.squeeze(self.fid[ds_path])

        assert self.__rw
        if self.fid[ds_path].shape[1:] != write_data.shape:
            print( '*** Fatal: patch data has not same shape as original' )
            return None

        self.fid[ds_path][0,...] = write_data
        self.__patched_msm.append(ds_path)
        return write_data

#--------------------------------------------------
class L1BioCAL( L1Bio ):
    '''
    class with function to access Tropomi offline L1b calibration products
    '''
    def __init__( self, l1b_product, readwrite=False, verbose=False ):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select( self, msm_type ):
        '''
        Select a measurement as <processing class>_<ic_id>

        Parameters
        ----------
        msm_type :  string
          name of measurement group as <processing class>_<ic_id>

        Returns
        -------
        String with spectral bands found in product

        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        grp_list = [ 'ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE' ]
        for name in grp_list:
            for ii in '12345678':
                grp_path = os.path.join('BAND{}_{}'.format(ii, name), msm_type)
                if grp_path in self.fid:
                    if self.__verbose:
                        print( grp_path, grp_path in self.fid )
                    self.bands += ii

            if len(self.bands) > 0:
                grp_path = os.path.join('BAND%_{}'.format(name), msm_type)
                break

        if len(self.bands) > 0:
            self.__msm_path = grp_path

        return self.bands

    #-------------------------
    def get_ref_time( self, band=None ):
        '''
        Returns reference start time of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    def get_delta_time( self, band=None ):
        '''
        Returns offset from the reference start time of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%',self.bands[0]))

    def get_instrument_settings( self, band=None ):
        '''
        Returns instrument settings of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    def get_housekeeping_data( self, band=None ):
        '''
        Returns housekeeping data of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    def get_msm_data( self, msm_dset, band=None ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset

        band       :  None or {'1', '2', '3', ..., '8'}
            select data from one spectral band or all (Band is None)
        '''
        if band is None:
            res = {}
            for ii in self.bands:
                res[ii] = super().msm_data( self.__msm_path.replace('%', ii),
                                            msm_dset )
            return res
        else:
            return super().msm_data( self.__msm_path.replace('%', band),
                                     msm_dset )

    def set_msm_data( self, msm_dset, data_dict ):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset
        data_dict  :  dictionary with band ID as key and data to be written
             holds data to be written with same dimensions as dataset "msm_dset"
        '''
        for ii in data_dict:
            return super().msm_data( self.__msm_path.replace('%', ii),
                                     msm_dset, write_data=data_dict[ii] )

#--------------------------------------------------
class L1BioIRR( L1Bio ):
    '''
    class with function to access Tropomi offline L1b irradiance products
    '''
    def __init__( self, l1b_product, readwrite=False, verbose=False ):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select( self, msm_type ):
        '''
        Select a measurement as <processing class>_<ic_id>

        Parameters
        ----------
        msm_type :  string
          name of measurement group as <processing class>_<ic_id>

        Returns
        -------
        String with spectral bands found in product

        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        for ii in '12345678':
            grp_path = os.path.join('BAND{}_IRRADIANCE'.format(ii), msm_type)
            if grp_path in self.fid:
                if self.__verbose:
                    print( grp_path, grp_path in self.fid )
                self.bands += ii

        if len(self.bands) > 0:
            grp_path = os.path.join('BAND%_IRRADIANCE', msm_type)
            self.__msm_path = grp_path

        return self.bands

    #-------------------------
    def get_ref_time( self, band=None ):
        '''
        Returns reference start time of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    def get_delta_time( self, band=None ):
        '''
        Returns offset from the reference start time of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%',self.bands[0]))

    def get_instrument_settings( self, band=None ):
        '''
        Returns instrument settings of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    def get_housekeeping_data( self, band=None ):
        '''
        Returns housekeeping data of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    def get_msm_data( self, msm_dset, band=None ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset

        band       :  None or {'1', '2', '3', ..., '8'}
            select data from one spectral band or all (Band is None)
        '''
        if band is None:
            res = {}
            for ii in self.bands:
                res[ii] = super().msm_data( self.__msm_path.replace('%', ii),
                                            msm_dset )
            return res
        else:
            return super().msm_data( self.__msm_path.replace('%', band),
                                     msm_dset )

    def set_msm_data( self, msm_dset, data_dict ):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset
        data_dict  :  dictionary with band ID as key and data to be written
             holds data to be written with same dimensions as dataset "msm_dset"
        '''
        for ii in data_dict:
            return super().msm_data( self.__msm_path.replace('%', ii),
                                     msm_dset, write_data=data_dict[ii] )

#--------------------------------------------------
class L1BioRAD( L1Bio ):
    '''
    class with function to access Tropomi offline L1b radiance products
    '''
    def __init__( self, l1b_product, readwrite=False, verbose=False ):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select( self, msm_type ):
        '''
        Select a measurement as <processing class>_<ic_id>

        Parameters
        ----------
        msm_type :  string
          name of measurement group as <processing class>_<ic_id>

        Returns
        -------
        String with spectral bands found in product

        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        for ii in '12345678':
            grp_path = os.path.join('BAND{}_RADIANCE'.format(ii), msm_type)
            if grp_path in self.fid:
                self.bands += ii
                break

        if len(self.bands) > 0:
            self.__msm_path = grp_path

        return self.bands

    #-------------------------
    def get_ref_time( self ):
        '''
        Returns reference start time of measurements
        '''
        return super().ref_time( self.__msm_path )

    def get_delta_time( self ):
        '''
        Returns offset from the reference start time of measurement
        '''
        return super().delta_time( self.__msm_path )

    def get_instrument_settings( self ):
        '''
        Returns instrument settings of measurement
        '''
        return super().instrument_settings( self.__msm_path )

    def get_housekeeping_data( self ):
        '''
        Returns housekeeping data of measurements
        '''
        return super().housekeeping_data( self.__msm_path )

    def get_msm_data( self, msm_dset ):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           name of measurement dataset
        '''
        return super().msm_data( self.__msm_path, msm_dset )

    def set_msm_data( self, msm_dset, data ):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           name of measurement dataset
        data      :  array-like
           data to be written with same dimensions as dataset "msm_dset"
        '''
        return super().msm_data( self.__msm_path, msm_dset, write_data=data )

#------------------------- TEST-modules and Tutorials -------------------------
def test_rd_calib( l1b_product, msm_type, msm_dset, verbose ):
    '''
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    '''
    l1b = L1BioCAL( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( l1b.get_ref_time() )
    print( l1b.get_delta_time() )
    #print( l1b.get_instrument_settings() )
    print( l1b.get_housekeeping_data() )
    dset = l1b.get_msm_data( msm_dset )
    for key in sorted(dset):
        print( key, dset[key].shape )
    del l1b

def test_rd_irrad( l1b_product, msm_type, msm_dset, verbose ):
    '''
    Perform some simple tests to check the L1BioCAL classes

    Please use the code as tutorial

    '''
    l1b = L1BioIRR( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( l1b.get_ref_time() )
    print( l1b.get_delta_time() )
    print( l1b.get_instrument_settings() )
    print( l1b.get_housekeeping_data() )
    dset = l1b.get_msm_data( msm_dset )
    for key in sorted(dset):
        print( key, dset[key].shape )
    del l1b

def test_rd_rad( l1b_product, msm_type, msm_dset, verbose ):
    '''
    Perform some simple tests to check the L01BioRAD classes

    Please use the code as tutorial
    '''
    l1b = L1BioRAD( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select( msm_type )
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( l1b.get_ref_time() )
    print( l1b.get_delta_time() )
    #print( l1b.get_instrument_settings() )
    print( l1b.get_housekeeping_data() )
    print( msm_dset, l1b.get_msm_data( msm_dset ).shape )
    del l1b

def main():
    '''
    Let the user test the software!!!
    '''
    import argparse

    # parse command-line parameters
    parser = argparse.ArgumentParser( 
        description='run test-routines to check class L1BioXXX' )
    parser.add_argument( 'l1b_product', default=None,
                         help='name of L1B product (full path)' )
    parser.add_argument( '--msm_type', default=None,
                         help='define measurement type as <processing class>_<ic_id>' )
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
    if prod_type == 'S5P_OFFL_L1B_CA':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'DARK_MODE_1607'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'signal'
        print('calib: ', msm_type, msm_dset)
        test_rd_calib( args.l1b_product, msm_type, msm_dset, args.verbose )
    elif prod_type == 'S5P_OFFL_L1B_IR':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'STANDARD_MODE'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'irradiance'
        print('irrad: ', msm_type, msm_dset)
        test_rd_irrad( args.l1b_product, msm_type, msm_dset,  args.verbose )
    elif prod_type == 'S5P_OFFL_L1B_RA':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'STANDARD_MODE'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'radiance'
        print('rad: ', msm_type, msm_dset)
        test_rd_rad( args.l1b_product, msm_type, msm_dset, args.verbose )
    else:
        print( ' *** FATAL: unknown product type' )

#-------------------------
if __name__ == '__main__':
    main()
