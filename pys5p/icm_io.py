"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class ICMio provides read access to S5p Tropomi ICM_CA_SIR products

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import os.path

import numpy as np
import h5py

def pad_rows(arr1, arr2):
    """
    Pad the array with the least numer of rows with NaN's
    """
    if arr2.ndim == 2:
        if arr1.shape[0] < arr2.shape[0]:
            buff = arr1.copy()
            arr1 = np.full_like(arr2, np.nan)
            arr1[0:buff.shape[0], :] = buff
        elif arr1.shape[0] > arr2.shape[0]:
            buff = arr2.copy()
            arr2 = np.full_like(arr1, np.nan)
            arr2[0:buff.shape[0], :] = buff
    else:
        if arr1.shape[1] < arr2.shape[1]:
            buff = arr1.copy()
            arr1 = np.full_like(arr2, np.nan)
            arr1[:, 0:buff.shape[1], :] = buff
        elif arr1.shape[1] > arr2.shape[1]:
            buff = arr2.copy()
            arr2 = np.full_like(arr1, np.nan)
            arr2[:, 0:buff.shape[1], :] = buff

    return (arr1, arr2)

#--------------------------------------------------
class ICMio(object):
    """
    This class should offer all the necessary functionality to read Tropomi
    ICM_CA_SIR products
    """
    def __init__(self, icm_product, readwrite=False):
        """
        Initialize access to an ICM product

        Parameters
        ----------
        icm_product :  string
           full path to in-flight calibration measurement product
        readwrite   :  boolean
           open product in read-write mode (default is False)
        """
        # initialize class-attributes
        self.filename = icm_product
        self.__rw = readwrite
        self.__msm_path = None
        self.__patched_msm = []
        self.bands = None
        self.fid = None

        assert os.path.isfile(icm_product), \
            '*** Fatal, can not find ICM_CA_SIR file: {}'.format(icm_product)

        # open ICM product as HDF5 file
        if readwrite:
            self.fid = h5py.File(icm_product, "r+")
        else:
            self.fid = h5py.File(icm_product, "r")

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, readwrite={!r})'.format(class_name,
                                                 self.filename, self.__rw)

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __del__(self):
        """
        Before closing the product, we make sure that the output product
        describes what has been altered by the S/W. To keep any change
        traceable.

        as attributes of this group, we write:
         - dateStamp ('now')
         - Git-version of S/W
         - list of patched datasets
         - auxiliary datasets used by patch-routines
        """
        if len(self.__patched_msm) > 0:
            from datetime import datetime

            sgrp = self.fid.require_group("METADATA/SRON_METADATA")
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = self.pys5p_version()
            if 'patched_datasets' not in sgrp:
                dtype = h5py.special_dtype(vlen=str)
                dset = sgrp.create_dataset('patched_datasets',
                                           (len(self.__patched_msm),),
                                           maxshape=(None,), dtype=dtype)
                dset[:] = np.asarray(self.__patched_msm)
            else:
                dset = sgrp['patched_datasets']
                dset.resize(dset.shape[0] + len(self.__patched_msm), axis=0)
                dset[dset.shape[0]-1:] = np.asarray(self.__patched_msm)

        self.bands = None
        if self.fid is not None:
            self.fid.close()

    # ---------- RETURN VERSION of the S/W ----------
    @staticmethod
    def pys5p_version():
        """
        Returns S/W version
        """
        from . import version

        return version.__version__

    #-------------------------
    def find(self, msm_class):
        """
        find a measurement as <processing-class name>

        Parameters
        ----------
        msm_class :  string
          processing-class name without ICID

        Returns
        -------
        out  :  list of strings
           String with msm_type as used by ICMio.select
        """
        res = []

        grp_list = ['ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE']
        for ii in '12345678':
            for name in grp_list:
                grp_name = 'BAND{}_{}'.format(ii, name)
                if grp_name in self.fid:
                    gid = self.fid[grp_name]
                    res += [s for s in gid if s.startswith(msm_class)]

        return list(set(res))

    #-------------------------
    def select(self, msm_type, msm_path=None):
        """
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
        """
        self.bands = ''
        self.__msm_path = None

        # if path is given, then only determine avaialble spectral bands
        # else determine path and avaialble spectral bands
        if msm_path is not None:
            assert msm_path.startswith('BAND%') > 0, \
                '*** Fatal: msm_path should start with BAND%'

            for ii in '12345678':
                grp_path = os.path.join(msm_path.replace('%', ii), msm_type)
                if grp_path in self.fid:
                    self.bands += ii
        else:
            grp_list = ['ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE']
            for ii in '12345678':
                for name in grp_list:
                    grp_path = os.path.join('BAND{}_{}'.format(ii, name),
                                            msm_type)
                    if grp_path in self.fid:
                        msm_path = 'BAND{}_{}'.format('%', name)
                        self.bands += ii

        # return in case no data was found
        if len(self.bands) > 0:
            self.__msm_path = os.path.join(msm_path, msm_type)

        return self.bands

    # ---------- Functions that work before MSM selection ----------
    def get_orbit(self):
        """
        Returns reference orbit number
        """
        return int(self.fid.attrs['reference_orbit'])

    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        return self.fid.attrs['processor_version'].decode('ascii')

    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        return (self.fid.attrs['time_coverage_start'].decode('ascii'),
                self.fid.attrs['time_coverage_end'].decode('ascii'))

    def get_creation_time(self):
        """
        Returns version of the L01b processor
        """
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset  = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].split(b'=')[1].decode('ascii')

    def get_attr(self, attr_name):
        """
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        """
        if attr_name in self.fid.attrs.keys():
            return self.fid.attrs[attr_name]

        return None

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time(self, band=None):
        """
        Returns reference start time of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        from datetime import datetime, timedelta

        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                ref_time = (datetime(2010,1,1,0,0,0) \
                            + timedelta(seconds=int(sgrp['time'][0])))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
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
        """
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                if res is None:
                    res = sgrp['delta_time'][0,:].astype(int)
                else:
                    res = np.append(res, sgrp['delta_time'][0,:].astype(int))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
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
        """
        Returns instrument settings of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['instrument_settings'][:]
                else:
                    res = np.append(res, sgrp['instrument_settings'][:])
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
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
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = np.squeeze(sgrp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(sgrp['housekeeping_data']))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = np.squeeze(sgrp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(sgrp['housekeeping_data']))
        else:
            grp = self.fid[os.path.join(msm_path, 'INSTRUMENT')]
            res = np.squeeze(grp['housekeeping_data'])

        return res

    #-------------------------
    def get_msm_attr(self, msm_dset, attr_name, band=None):
        """
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
        """
        if len(self.__msm_path) == 0:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
            ds_path = os.path.join(self.__msm_path.replace('%', band),
                                   dset_grp, msm_dset)
            if ds_path not in self.fid:
                continue

            if attr_name in self.fid[ds_path].attrs.keys():
                attr = self.fid[ds_path].attrs[attr_name]
                if isinstance(attr, bytes):
                    return attr.decode('ascii')
                else:
                    return attr

        return None

    def get_geo_data(self, band=None,
                     geo_dset='satellite_latitude,satellite_longitude'):
        """
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
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = str(self.bands[0])
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        res = None
        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
                grp = self.fid[grp_path]
                sgrp = grp['GEODATA']
                for key in geo_dset.split(','):
                    if res is None:
                        res = np.squeeze(sgrp[key])
                    else:
                        res = np.append(res, np.squeeze(sgrp[key]))
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            grp_path = os.path.join(os.path.dirname(msm_path),
                                    'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = os.path.join('BAND{}_CALIBRATION'.format(band),
                                        name.decode('ascii'))
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

    def get_msm_data(self, msm_dset, band='78', columns=None,
                     msm_to_row=None, fill_as_nan=False):
        """
        Read datasets from a measurement selected by class-method "select"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset
            if msm_dset is None then show names of available datasets
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        msm_to_row : {None, 'padding', 'rebin'}
            Return the measurement data as stored in the product (None), padded
            with NaN's for the spectral band with largest number of rows, or
            rebinned according 'measurement_to_detector_row_table'
            - Default for spectral bands is to return the data as stored.
            - Default for spectral channels is to apply padding.
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Returns
        -------
        out  :  array
           Data of measurement dataset "msm_dset"
        """
        if self.__msm_path is None:
            return None

        assert len(band) > 0 and len(band) <= 2
        if len(band) == 2:
            assert band == '12' or band == '34' or band == '56' or band == '78'
            if msm_to_row is None:
                msm_to_row = 'padding'
        assert self.bands.find(band) >= 0

        fillvalue = float.fromhex('0x1.ep+122')

        res = None
        for ii in band:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                ds_path = os.path.join(self.__msm_path.replace('%', ii),
                                       dset_grp, msm_dset)
                if ds_path not in self.fid:
                    continue

                if columns is None:
                    data = np.squeeze(self.fid[ds_path])
                else:
                    data = self.fid[ds_path][...,columns[0]:columns[1]]
                    data = np.squeeze(data)
                if fill_as_nan \
                   and self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                    data[(data == fillvalue)] = np.nan

                if res is None:
                    res = data
                else:
                    if msm_to_row == 'padding':
                        (res, data) = pad_rows(res, data)

                    res = np.concatenate((res, data), axis=data.ndim-1)

        return res

    #-------------------------
    def set_housekeeping_data(self, data, band=None):
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        assert self.__rw

        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = self.__msm_path.replace('%', band)
        msm_type = os.path.basename(self.__msm_path)

        if msm_type == 'ANALOG_OFFSET_SWIR' or msm_type == 'LONG_TERM_SWIR':
            pass
        elif msm_type == 'DPQF_MAP' or msm_type == 'NOISE':
            pass
        else:
            ds_path = os.path.join(msm_path, 'INSTRUMENT', 'housekeeping_data')
            self.fid[ds_path][0,:] = data

            self.__patched_msm.append(ds_path)

    def set_msm_data(self, msm_dset, data, band='78'):
        """
        Alter dataset from a measurement selected using function "select"

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        data       :  array-like
            data to be written with same dimensions as dataset "msm_dset"

        """
        assert self.__rw
        assert self.bands.find(band) >= 0

        if self.__msm_path is None:
            return None

        init = True
        fillvalue = float.fromhex('0x1.ep+122')

        col = 0
        for ii in band:
            ds_path = os.path.join(self.__msm_path.replace('%', ii),
                                   'ANALYSIS', msm_dset)
            if ds_path in self.fid:
                dim = self.fid[ds_path].shape

                if init:
                    if self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                        data[np.isnan(data)] = fillvalue
                    self.fid[ds_path][...] = data[...,col:col+dim[-1]]
                    col += dim[-1]
                    init = False
                else:
                    self.fid[ds_path][...] = data[...,col:col+dim[-1]]

                self.__patched_msm.append(ds_path)

            ds_path = os.path.join(self.__msm_path.replace('%', ii),
                                   'OBSERVATIONS', msm_dset)
            if ds_path in self.fid:
                dim = self.fid[ds_path].shape

                if init:
                    if self.fid[ds_path].attrs['_FillValue'] == fillvalue:
                        data[np.isnan(data)] = fillvalue
                    self.fid[ds_path][0,...] = data[...,col:col+dim[-1]]
                    col += dim[-1]
                    init = False
                else:
                    self.fid[ds_path][0,...] = data[...,col:col+dim[-1]]

                self.__patched_msm.append(ds_path)

        # display warning when no dataset is updated
        if init:
            print('WARNING: dataset {} not found'.format(msm_dset))
