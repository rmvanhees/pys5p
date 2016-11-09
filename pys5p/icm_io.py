'''
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The class ICMio provides read access to S5p Tropomi ICM_CA_SIR products

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
class ICMio(object):
    '''
    This class should offer all the necessary functionality to read Tropomi
    ICM_CA_SIR products
    '''
    def __init__(self, icm_product, readwrite=False):
        '''
        Initialize access to an ICM product

        Parameters
        ----------
        icm_product :  string
           full path to in-flight calibration measurement product
        readwrite   :  boolean
           open product in read-write mode (default is False)
        '''
        assert os.path.isfile( icm_product ), \
            '*** Fatal, can not find ICM_CA_SIR file: {}'.format(icm_product)

        # initialize class-attributes
        self.__product = icm_product
        self.__rw = readwrite
        self.__msm_path = None
        self.__patched_msm = []
        self.bands = None

        # open ICM product as HDF5 file
        if readwrite:
            self.fid = h5py.File( icm_product, "r+" )
        else:
            self.fid = h5py.File( icm_product, "r" )

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

        as attributes of this group, we write:
         - dateStamp ('now')
         - Git-version of S/W
         - list of patched datasets
         - auxiliary datasets used by patch-routines
        '''
        if len(self.__patched_msm) > 0:
            from datetime import datetime

            sgrp = self.fid.create_group( "METADATA/SRON_METADATA" )
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = self.pys5p_version()
            dtype = h5py.special_dtype(vlen=str)
            dset = sgrp.create_dataset( 'patched_datasets',
                                        (len(self.__patched_msm),), dtype=dtype)
            dset[:] = np.asarray(self.__patched_msm)

        self.bands = None
        self.fid.close()

    # ---------- RETURN VERSION of the S/W ----------
    @staticmethod
    def pys5p_version():
        '''
        Returns S/W version
        '''
        from pys5p import version

        return version.__version__

    # ---------- Functions that work before MSM selection ----------
    def get_orbit(self):
        '''
        Returns reference orbit number
        '''
        return int(self.fid.attrs['reference_orbit'])

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
        Returns version of the L01b processor
        '''
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset  = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].split(b'=')[1].decode('ascii')

    def get_attr(self, attr_name):
        '''
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        '''
        if attr_name in self.fid.attrs.keys():
            return self.fid.attrs[attr_name]

        return None

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time(self, band=None):
        '''
        Returns reference start time of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        '''
        from datetime import datetime, timedelta

        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                ref_time = (datetime(2010,1,1,0,0,0) \
                            + timedelta(seconds=int(sgrp['time'][0])))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR' )
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                ref_time = (datetime(2010,1,1,0,0,0) \
                            + timedelta(seconds=int(sgrp['time'][0])))
        else:
            grp = self.fid[msm_path]
            sgrp = grp['OBSERVATIONS']
            ref_time = (datetime(2010,1,1,0,0,0) \
                        + timedelta(seconds=int(sgrp['time'][0])))
        return ref_time

    def get_delta_time(self, band=None):
        '''
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        '''
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                if res is None:
                    res = sgrp['delta_time'][0,:].astype(int)
                else:
                    res = np.append(res, sgrp['delta_time'][0,:].astype(int))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR' )
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                if res is None:
                    res = sgrp['delta_time'][0,:].astype(int)
                else:
                    res = np.append(res, sgrp['delta_time'][0,:].astype(int))
        else:
            grp = self.fid[msm_path]
            sgrp = grp['OBSERVATIONS']
            res = sgrp['delta_time'][0,:].astype(int)

        return res

    def get_instrument_settings(self, band=None):
        '''
        Returns instrument settings of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        '''
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['instrument_settings'][:]
                else:
                    res = np.append(res, sgrp['instrument_settings'][:])
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR' )
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['instrument_settings'][:]
                else:
                    res = np.append(res, sgrp['instrument_settings'][:])
        else:
            grp = self.fid[os.path.join(msm_path, 'INSTRUMENT')]
            res = grp['instrument_settings'][:]

        return res

    def get_housekeeping_data(self, band=None):
        '''
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        '''
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['housekeeping_data'][:]
                else:
                    res = np.append(res, sgrp['housekeeping_data'][:])
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR' )
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['housekeeping_data'][:]
                else:
                    res = np.append(res, sgrp['housekeeping_data'][:])
        else:
            grp = self.fid[os.path.join(msm_path, 'INSTRUMENT')]
            res = grp['housekeeping_data'][:]

        return res

    #-------------------------
    def select(self, msm_type, msm_path=None):
        '''
        Select a measurement as <processing class>_<ic_id>

        Parameters
        ----------
        msm_type :  string
          name of measurement group
        msm_path : {'BAND%_ANALYSIS', 'BAND%_CALIBRATION',
                   'BAND%_IRRADIANCE', 'BAND%_RADIANCE'}
          name of path in HDF5 file to measurement group

        Returns
        -------
        out  :  string
           String with spectral bands found in product

        Updated object attributes:
         - bands               : available spectral bands
        '''
        self.bands = ''
        self.__msm_path = None

        # if path is given, then only determine avaialble spectral bands
        # else determine path and avaialble spectral bands
        if msm_path is not None:
            assert msm_path.startswith('BAND%') > 0, \
                '*** Fatal: msm_path should start with BAND%'

            for ii in '12345678':
                grp_path = os.path.join( msm_path.replace('%', ii), msm_type )
                if grp_path in self.fid:
                    self.bands += ii
        else:
            grp_list = [ 'ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE' ]
            for ii in '12345678':
                for name in grp_list:
                    grp_path = os.path.join( 'BAND{}_{}'.format(ii, name),
                                             msm_type )
                    if grp_path in self.fid:
                        msm_path = 'BAND{}_{}'.format('%', name)
                        self.bands += ii

        # return in case no data was found
        if len(self.bands) > 0:
            self.__msm_path = os.path.join(msm_path, msm_type)

        return self.bands

    #-------------------------
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
            Select one of the band present in the product.
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
           value of attribute "attr_name"
        '''
        if len(self.__msm_path) == 0:
            return None

        if band is None:
            band = self.bands[0]

        for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
            ds_path = os.path.join( self.__msm_path.replace('%', band),
                                    dset_grp, msm_dset )
            if ds_path not in self.fid:
                continue

            if attr_name in self.fid[ds_path].attrs.keys():
                attr = self.fid[ds_path].attrs[attr_name]
                if isinstance( attr, bytes ):
                    return attr.decode('ascii')
                else:
                    return attr

        return None

    def get_geo_data(self, band=None,
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
        if self.__msm_path is None:
            return None

        if band is None:
            band = str(self.bands[0])
        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['GEODATA']
                for key in geo_dset.split(','):
                    if res is None:
                        res = np.squeeze(sgrp[key])
                    else:
                        res = np.append(res, np.squeeze(sgrp[key]))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR' )
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join( 'BAND{}_CALIBRATION'.format(band),
                                         name.decode('ascii') )
                grp = self.fid[grp_path]
                sgrp = grp['GEODATA']
                for key in geo_dset.split(','):
                    if res is None:
                        res = np.squeeze(sgrp[key])
                    else:
                        res = np.append(res, np.squeeze(sgrp[key]))
        else:
            grp = self.fid[os.path.join(msm_path, 'GEODATA')]
            for key in geo_dset.split(','):
                if res is None:
                    res = np.squeeze(grp[key])
                else:
                    res = np.append(res, np.squeeze(grp[key]))

        return res

    def get_msm_data(self, msm_dset, band=None, fill_as_nan=False):
        '''
        Read datasets from a measurement selected by class-method "select"

        Parameters
        ----------
        msm_dset    :  string
            name of measurement dataset
            if msm_dset is None then show names of available datasets

        band       :  None or scalar {'1', '2', '3', ..., '8'}
            Select data from one spectral band
            Default is 'None' which returns the data of all available bands
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Returns
        -------
        out  :  dictionary
           Data of measurement dataset "msm_dset", band index is dictionary key
        '''
        if self.__msm_path is None:
            return None

        fillvalue = float.fromhex('0x1.ep+122')

        res = {}
        if band is None:
            for ii in self.bands:
                for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                    ds_path = os.path.join( self.__msm_path.replace('%', ii),
                                            dset_grp, msm_dset )
                    if ds_path not in self.fid:
                        continue

                    data = np.squeeze(self.fid[ds_path])
                    if fill_as_nan \
                       and self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                        data[(data == fillvalue)] = np.nan
                    res[ii] = data
        else:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                ds_path = os.path.join( self.__msm_path.replace('%', band),
                                        dset_grp, msm_dset )
                if ds_path not in self.fid:
                    continue

                data = np.squeeze(self.fid[ds_path])
                if fill_as_nan \
                   and self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                    data[(data == fillvalue)] = np.nan
                res[band] = data

        return res

    #-------------------------
    def set_msm_data(self, msm_dset, data_dict):
        '''
        Alter dataset from a measurement selected using function "select"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset

        data-dict  :  dictionary
            data to be written with same dimensions as dataset "msm_dset"
            requires a list with ndarrays alike the one returned by
            function "get_msm_data"
        '''
        if self.__msm_path is None:
            return None

        assert self.__rw

        fillvalue = float.fromhex('0x1.ep+122')

        for ii in data_dict:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                ds_path = os.path.join( self.__msm_path.replace('%', ii),
                                        dset_grp, msm_dset )
                if ds_path not in self.fid:
                    continue

                if self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                    data_dict[ii][np.isnan(data_dict[ii])] = fillvalue

                if self.fid[ds_path].shape[1:] != data_dict[ii].shape:
                    print( '*** Fatal: patch data has not same shape as original' )
                    return None

                self.fid[ds_path][0,...] = data_dict[ii]
                self.__patched_msm.append(ds_path)

#--------------------------------------------------
def test_rd_icm():
    '''
    Perform some simple test to check the ICM_io class

    Please use the code as tutorial

    '''
    import shutil

    if os.path.isdir('/Users/richardh'):
        fl_path = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    elif os.path.isdir('/nfs/TROPOMI/ical/'):
        fl_path = '/nfs/TROPOMI/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    else:
        fl_path = '/data/richardh/Tropomi/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    fl_name = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001100_20151002T140000.h5'

    icm = ICMio( os.path.join(fl_path, fl_name) )
    print( icm.get_processor_version() )
    print( icm.get_creation_time() )
    print( icm.get_coverage_time() )

    if len(icm.select('ANALOG_OFFSET_SWIR')) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'analog_offset_swir_value' )
        print( len(res), res.keys(), res['7'].shape )
        print(icm.get_msm_attr( 'analog_offset_swir_value', 'units' ))

    if len(icm.select('BACKGROUND_MODE_1063',
                      msm_path='BAND%_CALIBRATION')) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'signal_avg' )
        print( len(res), res['8'].shape )
        print(icm.get_msm_attr( 'signal_avg', 'units' ))

        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'biweight_value' )
        print( len(res), res['7'].shape )
        print(icm.get_msm_attr( 'biweight_value', 'units' ))

    if len(icm.select( 'SOLAR_IRRADIANCE_MODE_0202' )) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'irradiance_avg' )
        print( len(res), res['8'].shape )
        print(icm.get_msm_attr( 'irradiance_avg', 'units' ))

    if len(icm.select( 'EARTH_RADIANCE_MODE_0004' )) > 0:
        #print( icm.get_ref_time() )
        #print( icm.get_delta_time() )
        #print( icm.get_instrument_settings() )
        #print( icm.get_housekeeping_data() )
        print( 'GEO: ', icm.get_geo_data().shape )
        res = icm.get_msm_data( 'radiance_avg_row' )
        print( len(res), res['7'].shape )
        print(icm.get_msm_attr( 'radiance_avg_row', 'units' ))

    if os.path.isdir('/Users/richardh'):
        fl_path2 = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    else:
        fl_path2 = '/data/richardh/Tropomi'
    fl_name2 = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001101_20151002T140000.h5'
    shutil.copy( os.path.join(fl_path, fl_name),
                 os.path.join(fl_path2, fl_name2) )
    icm = ICMio( os.path.join(fl_path2, fl_name2), readwrite=True )
    icm.select( 'BACKGROUND_MODE_1063' )
    res = icm.get_msm_data( 'signal_avg' )
    res['7'][:,:] = 2
    res['8'][:,:] = 3
    icm.set_msm_data( 'signal_avg', res )

    del icm

#--------------------------------------------------
if __name__ == '__main__':
    test_rd_icm()
