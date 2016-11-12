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
class L1Bio(object):
    '''
    super class with general function to access Tropomi offline L1b products

    inherited by the classes L1BioCAL, L1BioIRR and L1BioRAD
    '''
    def __init__(self, l1b_product, readwrite=False):
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

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, readwrite={!r})'.format( class_name,
                                                  self.__product, self.__rw )

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __del__(self):
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
        from pys5p import version

        return version.__version__

    def get_orbit(self):
        '''
        Returns absolute orbit number
        '''
        return int(self.fid.attrs['orbit'])

    def get_processor_version(self):
        '''
        Returns version of the L01b processor
        '''
        return self.fid.attrs['processor_version'].decode('ascii')

    def get_coverage_time(self):
        '''
        Returns start and end of the measurement coverage time
        '''
        return (self.fid.attrs['time_coverage_start'].decode('ascii'),
                self.fid.attrs['time_coverage_end'].decode('ascii'))

    def get_creation_time(self):
        '''
        Returns datetime when the L1b product was created
        '''
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].decode('ascii')

    def get_attr(self, attr_name):
        '''
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name :  string
           Name of the attribute
        '''
        if attr_name in self.fid.attrs.keys():
            return self.fid.attrs[attr_name]

        return None

    def ref_time(self, msm_path):
        '''
        Returns reference start time of measurements

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        '''
        from datetime import datetime, timedelta

        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path, 'OBSERVATIONS')]
        return datetime(2010,1,1,0,0,0) \
            + timedelta(seconds=int(grp['time'][0]))

    def delta_time(self, msm_path):
        '''
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        '''
        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path, 'OBSERVATIONS')]
        return grp['delta_time'][0,:].astype(int)

    def instrument_settings(self, msm_path):
        '''
        Returns instrument settings of measurement

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        '''
        if msm_path is None:
            return None
        #
        # Due to a bug in python module h5py (v2.6.0), it fails to read
        # the UVN instrument settings directy, with exception:
        #    KeyError: 'Unable to open object (Component not found)'.
        # This is my workaround
        #
        grp = self.fid[os.path.join(msm_path, 'INSTRUMENT')]
        instr = np.empty( grp['instrument_settings'].shape,
                          dtype=grp['instrument_settings'].dtype )
        grp['instrument_settings'].read_direct(instr)
        #for name in grp['instrument_settings'].dtype.names:
        #    instr[name][:] = grp['instrument_settings'][name]

        return instr

    def housekeeping_data(self, msm_path):
        '''
        Returns housekeeping data of measurements

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        '''
        if msm_path is None:
            return None

        grp = self.fid[os.path.join(msm_path, 'INSTRUMENT')]
        return np.squeeze(grp['housekeeping_data'])

    def msm_dims(self, msm_path, msm_dset):
        '''
        Return dimensions of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        msm_dset  :  string
            Name of measurement dataset

        Returns
        -------
        out   :   array-like
            Dimensions of msm_dset
        '''
        if msm_path is None:
            return None

        ds_path = os.path.join(msm_path, 'OBSERVATIONS', msm_dset)
        return self.fid[ds_path].shape

    def msm_attr(self, msm_path, msm_dset, attr_name):
        '''
        Returns value attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        msm_dset  :  string
            Name of measurement dataset
        attr_name : string
            Name of the attribute

        Returns
        -------
        out   :   scalar or numpy array
            Value of attribute "attr_name"
        '''
        if msm_path is None:
            return None

        ds_path = os.path.join(msm_path, 'OBSERVATIONS', msm_dset)
        if attr_name in self.fid[ds_path].attrs.keys():
            attr = self.fid[ds_path].attrs['units']
            if isinstance( attr, bytes ):
                return attr.decode('ascii')
            else:
                return attr
        return None

    def msm_data(self, msm_path, msm_dset, scan_index=None, write_data=None):
        '''
        Reads or writes data from dataset "msm_dset" in group "msm_path"

        Parameters
        ----------
        msm_path   :  string
           Full path to measurement group
        msm_dset   :  string
            Name of measurement dataset.
        scan_index :  array-like
            Indices ot the scanlines to be read or written. If scan_index is
            None, then all data is read.
        write_data :  array-like
            Data to be written with same dimensions as dataset "msm_dset". If
            write_data is None, then no date is written.

        Returns
        -------
        out   :  values read from or written to dataset "msm_dset"

        '''
        if msm_path is None:
            return None

        ds_path = os.path.join(msm_path, 'OBSERVATIONS', msm_dset)
        dset = self.fid[ds_path]
        if write_data is None:
            if scan_index is None:
                return np.squeeze(dset)
            else:
                buff = np.concatenate(([scan_index[0]-10],
                                       scan_index,
                                       [scan_index[-1]+10]))
                indx = np.where((buff[1:] - buff[0:-1]) != 1)[0]

                res = None
                for ii in range(len(indx)-1):
                    data = dset[0,indx[ii]:indx[ii+1],:,:]
                    if res is None:
                        res = data
                    else:
                        res = np.append(res, data, axis=0)
                return res

        # we will overwrite existing data, thus readwrite access is required
        assert self.__rw

        # overwrite the data
        if scan_index is None:
            if dset.shape[1:] != write_data.shape:
                print( '*** Fatal: patch data has not same shape as original' )
                return None

            dset[0,...] = write_data
        else:
            buff = np.concatenate(([scan_index[0]-10],
                                   scan_index,
                                   [scan_index[-1]+10]))
            indx = np.where((buff[1:] - buff[0:-1]) != 1)[0]

            for ii in range(len(indx)-1):
                dset[0,indx[ii]:indx[ii+1],:,:] = \
                                            write_data[indx[ii]:indx[ii+1],:,:]

        # update patch logging
        self.__patched_msm.append(ds_path)
        return write_data

#--------------------------------------------------
class L1BioCAL(L1Bio):
    '''
    class with function to access Tropomi offline L1b calibration products

    The L1b calibration products are available for UVN (band 1-6) 
    and SWIR (band 7-8).
    '''
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select(self, msm_type):
        '''
        Select a calibration measurement as <processing class>_<ic_id>

        Parameters
        ----------
        msm_type :  string
          Name of calibration measurement group as <processing class>_<ic_id>

        Returns
        -------
        out  :   string
           String with spectral bands found in product

        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        grp_list = [ 'CALIBRATION', 'IRRADIANCE', 'RADIANCE' ]
        for name in grp_list:
            for ii in '12345678':
                grp_path = os.path.join('BAND{}_{}'.format(ii, name), msm_type)
                if grp_path in self.fid:
                    if self.__verbose:
                        print( '*** INFO: found: ', grp_path )
                    self.bands += ii

            if len(self.bands) > 0:
                grp_path = os.path.join('BAND%_{}'.format(name), msm_type)
                break

        if len(self.bands) > 0:
            self.__msm_path = grp_path

        return self.bands

    #-------------------------
    def get_ref_time(self, band=None):
        '''
        Returns reference start time of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        '''
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    def get_delta_time(self, band=None):
        '''
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        '''
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%',self.bands[0]))

    def get_instrument_settings(self, band=None):
        '''
        Returns instrument settings of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        '''
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    def get_housekeeping_data(self, band=None):
        '''
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        '''
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    def get_msm_attr(self, msm_dset, attr_name, band=None):
        '''
        Returns value attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
            Name of measurement dataset
        attr_name :  string
            Name of the attribute
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
           value of attribute "attr_name"
        '''
        if band is None:
            band = self.bands[0]

        return super().msm_attr( self.__msm_path.replace('%', band),
                                 msm_dset, attr_name )

    def get_geo_data(self,  band=None,
                     geo_dset='satellite_latitude,satellite_longitude'):
        '''
        Returns data of selected datasets from the GEODATA group

        Parameters
        ----------
        geo_dset  :  string
            Name(s) of datasets in the GEODATA group, comma separated
            Default is 'satellite_latitude,satellite_longitude'
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   array-like
           Compound array with data of selected datasets from the GEODATA group
        '''
        if band is None:
            band = self.bands[0]

        nscans = self.fid[self.__msm_path.replace('%', band)]['scanline'].size

        dtype = []
        for name in geo_dset.split(','):
            dtype.append((name, 'f4'))
        res = np.empty( (nscans,), dtype=dtype )
        
        grp = self.fid[os.path.join(self.__msm_path.replace('%', band),
                                    'GEODATA')]
        for name in geo_dset.split(','):
            res[name][...] = grp[name][0, :]

        return res

    def get_msm_data(self, msm_dset, band='78'):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout

        Returns
        -------
        out   :    array-like
           Data of measurement dataset "msm_dset"
        '''
        assert len(band) > 0 and len(band) <= 2
        if len(band) == 2:
            assert band == '12' or band == '34' or band == '56' or band == '78'

        res = None
        for ii in band:
            data = super().msm_data( self.__msm_path.replace('%', ii),
                                     msm_dset )
            if res is None:
                res = data
            else:
                res = np.concatenate((res, data), axis=data.ndim-1)

        return res

    def set_msm_data(self, msm_dset, data, band='78'):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        data       :  array-like
            data to be written with same dimensions as dataset "msm_dset"
        '''
        col = 0
        for ii in band:
            dim = super().msm_dims( self.__msm_path.replace('%', ii), msm_dset )
            super().msm_data( self.__msm_path.replace('%', ii),
                              msm_dset, write_data=data[...,col:col+dim[-1]] )
            col += dim[-1]

#--------------------------------------------------
class L1BioIRR(L1Bio):
    '''
    class with function to access Tropomi offline L1b irradiance products
    '''
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select(self, msm_type='STANDARD_MODE'):
        '''
        Select an irradiance measurement group

        Parameters
        ----------
        msm_type :  string
           Name of irradiance measurement group, default: "STANDARD_MODE"

        Returns
        -------
        out  :  string
           String with spectral bands found in product

        Notes
        -----
        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        for ii in '12345678':
            grp_path = os.path.join('BAND{}_IRRADIANCE'.format(ii), msm_type)
            if grp_path in self.fid:
                if self.__verbose:
                    print( '*** INFO: found: ', grp_path )
                self.bands += ii

        if len(self.bands) > 0:
            self.__msm_path = os.path.join('BAND%_IRRADIANCE', msm_type)

        return self.bands

    #-------------------------
    def get_ref_time(self, band=None):
        '''
        Returns reference start time of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    def get_delta_time(self, band=None):
        '''
        Returns offset from the reference start time of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%',self.bands[0]))

    def get_instrument_settings(self, band=None):
        '''
        Returns instrument settings of measurement
        '''
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    def get_housekeeping_data(self, band=None):
        '''
        Returns housekeeping data of measurements
        '''
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    def get_msm_attr(self, msm_dset, attr_name, band=None):
        '''
        Returns value attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
            Name of measurement dataset
        attr_name :  string
            Name of the attribute
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
            Value of attribute "attr_name"
        '''
        if band is None:
            band = self.bands[0]

        return super().msm_attr( self.__msm_path.replace('%', band),
                                 msm_dset, attr_name )

    def get_msm_data(self, msm_dset, band='78'):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout

        Returns
        -------
        out   :    array-like
            Data of measurement dataset "msm_dset"
        '''
        assert len(band) > 0 and len(band) <= 2
        if len(band) == 2:
            assert band == '12' or band == '34' or band == '56' or band == '78'

        res = None
        for ii in band:
            data = super().msm_data( self.__msm_path.replace('%', ii),
                                     msm_dset )
            if res is None:
                res = data
            else:
                res = np.concatenate((res, data), axis=data.ndim-1)

        return res

    def set_msm_data(self, msm_dset, data, band='78'):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        data       :  array-like
            data to be written with same dimensions as dataset "msm_dset"
        '''
        col = 0
        for ii in band:
            dim = super().msm_dims( self.__msm_path.replace('%', ii), msm_dset )
            super().msm_data( self.__msm_path.replace('%', ii),
                              msm_dset, write_data=data[...,col:col+dim[-1]] )
            col += dim[-1]

#--------------------------------------------------
class L1BioRAD(L1Bio):
    '''
    class with function to access Tropomi offline L1b radiance products
    '''
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__( l1b_product, readwrite=readwrite )

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.__msm_mode = None
        self.bands = ''

    #-------------------------
    def select(self, msm_type='STANDARD_MODE'):
        '''
        Select a radiance measurement group

        Parameters
        ----------
        msm_type :  string
          name of radiance measurement group, default: "STANDARD_MODE"

        Returns
        -------
        out   :   string
           String with spectral bands found in product

        Notes
        -----
        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        for ii in '12345678':
            grp_path = os.path.join('BAND{}_RADIANCE'.format(ii), msm_type)
            if grp_path in self.fid:
                if self.__verbose:
                    print( '*** INFO: found: ', grp_path )
                self.bands = ii
                break              # only one band per product

        if len(self.bands) > 0:
            self.__msm_path = grp_path

        return self.bands

    #-------------------------
    def get_ref_time(self):
        '''
        Returns reference start time of measurements
        '''
        return super().ref_time( self.__msm_path )

    def get_delta_time(self):
        '''
        Returns offset from the reference start time of measurement
        '''
        return super().delta_time( self.__msm_path )

    def get_instrument_settings(self):
        '''
        Returns instrument settings of measurement
        '''
        return super().instrument_settings( self.__msm_path )

    def get_housekeeping_data(self):
        '''
        Returns housekeeping data of measurements
        '''
        return super().housekeeping_data( self.__msm_path )

    def get_msm_attr(self, msm_dset, attr_name):
        '''
        Returns value attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
            Name of measurement dataset
        attr_name :  string
            Name of the attribute

        Returns
        -------
        out   :   scalar or numpy array
            Value of attribute "attr_name"

        '''
        return super().msm_attr( self.__msm_path, msm_dset, attr_name )

    def get_scan_index(self, icid):
        '''
        Select Radiance measurements on their ICID from instrument_configuration

        Parameters
        ----------
        icid  :  integer {2, 4, 6, 8, 10}
           Valid radiance ICIDs are (2, 4, 6, 8, 10)

        Returns
        -------
        out   :  array-like
           A ndarray with indices to selected scanline(s)
        '''
        if icid is None:
            return None

        grp = self.fid[os.path.join(self.__msm_path, 'INSTRUMENT')]
        icid_list = np.squeeze(grp['instrument_configuration']['ic_id'])

        return np.where(icid_list == icid)[0]

    def get_geo_data(self, geo_dset='latitude,longitude', scan_index=None):
        '''
        Returns data of selected datasets from the GEODATA group

        Parameters
        ----------
        geo_dset  :  string
           Name(s) of datasets in the GEODATA group, comma separated

        Returns
        -------
        out   :   array-like
           Compound array with data of selected datasets from the GEODATA group
        '''
        nrows = self.fid[self.__msm_path]['ground_pixel'].size
        if scan_index is None:
            nscans = self.fid[self.__msm_path]['scanline'].size
        else:
            nscans = len(scan_index)

        dtype = []
        for name in geo_dset.split(','):
            dtype.append((name, 'f4'))
        res = np.empty( (nscans, nrows), dtype=dtype )
        
        grp = self.fid[os.path.join(self.__msm_path, 'GEODATA')]
        for name in geo_dset.split(','):
            if scan_index is None:
                res[name][...] = grp[name][0, :, :]
            else:
                res[name][...] = grp[name][0, scan_index, :]

        return res

    def get_msm_data(self, msm_dset, scan_index=None):
        '''
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           Name of measurement dataset

        Returns
        -------
        out   :    array-like
           Data of measurement dataset "msm_dset"
        '''
        return super().msm_data( self.__msm_path, msm_dset, scan_index )

    def set_msm_data(self, msm_dset, data):
        '''
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           Name of measurement dataset
        data      :  array-like
           Data to be written with same dimensions as dataset "msm_dset"
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

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    print( 'instrument settings: ', l1b.get_instrument_settings() )
    print( 'housekeeping data: ', l1b.get_housekeeping_data() )
    geo = l1b.get_geo_data()
    print( 'geodata: ', geo.dtype.names, geo.shape )
    dset = l1b.get_msm_data( msm_dset )
    print( '{}: {}'.format(msm_dset, dset.shape) )
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

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    print( 'instrument settings: ', l1b.get_instrument_settings() )
    print( 'housekeeping data: ', l1b.get_housekeeping_data() )
    dset = l1b.get_msm_data( msm_dset )
    print( '{}: {}'.format(msm_dset, dset.shape) )
    del l1b

def test_rd_rad( l1b_product, icid, msm_dset, verbose ):
    '''
    Perform some simple tests to check the L01BioRAD classes

    Please use the code as tutorial
    '''
    l1b = L1BioRAD( l1b_product, verbose=verbose )
    print( l1b )
    print( 'orbit:   ', l1b.get_orbit() )
    print( 'version: ', l1b.get_processor_version() )
    l1b.select()
    for key in l1b:
        print( '{}: {!r}'.format(key, l1b.__getattribute__(key)) )

    print( 'reference time: ', l1b.get_ref_time() )
    print( 'delta time: ', l1b.get_delta_time() )
    print( 'instrument settings: ', l1b.get_instrument_settings() )
    print( 'housekeeping data: ', l1b.get_housekeeping_data() )
    scan_index = l1b.get_scan_index( icid )
    geo = l1b.get_geo_data( scan_index=scan_index )
    print( 'geodata: ', geo.dtype.names, geo.shape )
    print( msm_dset, l1b.get_msm_data( msm_dset, scan_index=scan_index ).shape )
    del l1b

def _main():
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
            msm_type = 'DARK_MODE_1607'
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
