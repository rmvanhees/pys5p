"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The classes L1BioCAL, L1BioIRR, L1BioRAD provide read access to
offline level 1b products, resp. calibration, irradiance and radiance.

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import h5py
import numpy as np

from .biweight import biweight
from .version import version as __version__

# - global parameters ------------------------------


# - local functions --------------------------------
def pad_rows(arr1, arr2):
    """
    Pad the array with the least numer of rows with NaN's
    """
    if arr2.ndim == 1:
        pass
    elif arr2.ndim == 2:
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


def swir_exp_time(int_delay, int_hold):
    """
    Returns the exact pixel exposure time of the measurements
    """
    return 1.25e-6 * (65540 - int_delay + int_hold)


# - class definition -------------------------------
class L1Bio():
    """
    super class with general function to access Tropomi offline L1b products

    inherited by the classes L1BioCAL, L1BioIRR and L1BioRAD
    """
    def __init__(self, l1b_product, readwrite=False):
        """
        Initialize access to a Tropomi offline L1b product
        """
        # initialize private class-attributes
        self.filename = l1b_product
        self.__rw = readwrite
        self.__patched_msm = []
        self.fid = None
        self.imsm = None

        # open L1b product as HDF5 file
        if not Path(l1b_product).is_file():
            raise FileNotFoundError('{} does not exist'.format(l1b_product))

        if readwrite:
            self.fid = h5py.File(l1b_product, "r+")
        else:
            self.fid = h5py.File(l1b_product, "r")

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
        called when the object is destroyed
        """
        self.close()

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self):
        """
        Before closing the product, we make sure that the output product
        describes what has been altered by the S/W. To keep any change
        traceable.

        In case the L1b product is altered, the attributes listed below are
        added to the group: "/METADATA/SRON_METADATA":
             - dateStamp ('now')
             - Git-version of S/W
             - list of patched datasets
             - auxiliary datasets used by patch-routines
        """
        if self.fid is None:
            return

        if self.__patched_msm:
            from datetime import datetime

            sgrp = self.fid.create_group("/METADATA/SRON_METADATA")
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = __version__
            dtype = h5py.special_dtype(vlen=str)
            dset = sgrp.create_dataset('patched_datasets',
                                       (len(self.__patched_msm),),
                                       dtype=dtype)

            dset[:] = np.asarray(self.__patched_msm)

        self.fid.close()

    # ---------- PUBLIC FUNCTIONS ----------
    # ---------- class L1Bio::
    def get_attr(self, attr_name):
        """
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name :  string
           Name of the attribute
        """
        if attr_name not in self.fid.attrs.keys():
            return None

        attr = self.fid.attrs[attr_name]
        if attr.shape is None:
            return None

        return attr

    # ---------- class L1Bio::
    def get_orbit(self):
        """
        Returns absolute orbit number
        """
        res = self.get_attr('orbit')
        if res is None:
            return None

        return int(res)

    # ---------- class L1Bio::
    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        attr = self.get_attr('processor_version')
        if attr is None:
            return None

        return attr.decode('ascii')

    # ---------- class L1Bio::
    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        attr_start = self.get_attr('time_coverage_start')
        if attr_start is None:
            return None

        attr_end = self.get_attr('time_coverage_end')
        if attr_end is None:
            return None

        return (attr_start.decode('ascii'), attr_end.decode('ascii'))

    # ---------- class L1Bio::
    def get_creation_time(self):
        """
        Returns datetime when the L1b product was created
        """
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset = grp['fixed_header/source']
        if 'Creation_Date' in self.fid.attrs.keys():
            attr = dset.attrs['Creation_Date']
            if isinstance(attr, bytes):
                return attr.decode('ascii')

            return attr

        return None

    # ---------- class L1Bio::
    def ref_time(self, msm_path):
        """
        Returns reference start time of measurements

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        """
        from datetime import datetime, timedelta

        if msm_path is None:
            return None

        grp = self.fid[str(Path(msm_path, 'OBSERVATIONS'))]
        return datetime(2010, 1, 1, 0, 0, 0) \
            + timedelta(seconds=int(grp['time'][0]))

    # ---------- class L1Bio::
    def delta_time(self, msm_path):
        """
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        """
        if msm_path is None:
            return None

        grp = self.fid[str(Path(msm_path, 'OBSERVATIONS'))]
        return grp['delta_time'][0, :].astype(int)

    # ---------- class L1Bio::
    def instrument_settings(self, msm_path):
        """
        Returns instrument settings of measurement

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        """
        if msm_path is None:
            return None
        #
        # Due to a bug in python module h5py (v2.6.0), it fails to read
        # the UVN instrument settings directy, with exception:
        #    KeyError: 'Unable to open object (Component not found)'.
        # This is my workaround
        #
        grp = self.fid[str(Path(msm_path, 'INSTRUMENT'))]
        instr = np.empty(grp['instrument_settings'].shape,
                         dtype=grp['instrument_settings'].dtype)
        grp['instrument_settings'].read_direct(instr)
        # for name in grp['instrument_settings'].dtype.names:
        #     instr[name][:] = grp['instrument_settings'][name]

        return instr

    # ---------- class L1Bio::
    def housekeeping_data(self, msm_path):
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group
        """
        if msm_path is None:
            return None

        grp = self.fid[str(Path(msm_path, 'INSTRUMENT'))]
        return np.squeeze(grp['housekeeping_data'])

    # ---------- class L1Bio::
    def msm_dims(self, msm_path, msm_dset):
        """
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
        """
        if msm_path is None:
            return None

        ds_path = Path(msm_path, 'OBSERVATIONS', msm_dset)
        return self.fid[str(ds_path)].shape

    # ---------- class L1Bio::
    def msm_info(self, msm_path):
        """
        Returns sequence number for each unique measurement based on ICID
          and delta_time

        Parameters
        ----------
        msm_path  :  string
           Full path to measurement group

        Returns
        -------
        out  :  array-like
          Numpy rec-array with sequence number, ICID and delta-time
        """
        if msm_path is None:
            self.imsm = None
            return

        grp = self.fid[str(Path(msm_path, 'INSTRUMENT'))]
        icid_list = np.squeeze(grp['instrument_configuration']['ic_id'])
        master_cycle = grp['instrument_settings']['master_cycle_period_us'][0]
        master_cycle /= 1000
        grp = self.fid[str(Path(msm_path, 'OBSERVATIONS'))]
        delta_time = np.squeeze(grp['delta_time'])
        length = delta_time.size
        self.imsm = np.empty((length,), dtype=[('icid', 'u2'),
                                               ('sequence', 'u2'),
                                               ('index', 'u4'),
                                               ('delta_time', 'u4')])
        self.imsm['icid'] = icid_list
        self.imsm['index'] = np.arange(length, dtype=np.uint32)
        self.imsm['delta_time'] = delta_time
        if length == 1:
            self.imsm['sequence'] = [0]
            return

        buff_icid = np.concatenate(([icid_list[0]-10], icid_list,
                                    [icid_list[-1]+10]))
        dt_thres = 10 * master_cycle
        buff_time = np.concatenate(([delta_time[0] - 10 * dt_thres], delta_time,
                                    [delta_time[-1] + 10 * dt_thres]))

        indx = np.where(((buff_time[1:] - buff_time[0:-1]) > dt_thres)
                        | ((buff_icid[1:] - buff_icid[0:-1]) != 0))[0]
        for ii in range(len(indx)-1):
            self.imsm['sequence'][indx[ii]:indx[ii+1]] = ii

    # ---------- class L1Bio::
    def msm_attr(self, msm_path, msm_dset, attr_name):
        """
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
        """
        if msm_path is None:
            return None

        ds_path = str(Path(msm_path, 'OBSERVATIONS', msm_dset))
        if attr_name in self.fid[ds_path].attrs.keys():
            attr = self.fid[ds_path].attrs[attr_name]
            if isinstance(attr, bytes):
                return attr.decode('ascii')

            return attr

        return None

    # ---------- class L1Bio::
    def _get_msm_data(self, msm_path, msm_dset, icid=None, fill_as_nan=False):
        """
        Reads data from dataset "msm_dset" in group "msm_path"

        Parameters
        ----------
        msm_path    :  string
           Full path to measurement group
        msm_dset    :  string
            Name of measurement dataset.
        icid        :  integer
            Select measurement data of measurements with given ICID
        fill_as_nan :  boolean
            Set data values equal (KNMI) FillValue to NaN

        Returns
        -------
        out   :  values read from or written to dataset "msm_dset"

        """
        fillvalue = float.fromhex('0x1.ep+122')

        if msm_path is None:
            return None

        ds_path = str(Path(msm_path, 'OBSERVATIONS', msm_dset))
        dset = self.fid[ds_path]

        if icid is None:
            if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                data = np.squeeze(dset)
                data[(data == fillvalue)] = np.nan
                return data

            return np.squeeze(dset)

        if self.imsm is None:
            return None
        indx = self.imsm['index'][self.imsm['icid'] == icid]
        buff = np.concatenate(([indx[0]-10], indx, [indx[-1]+10]))
        kk = np.where((buff[1:] - buff[0:-1]) != 1)[0]

        res = None
        for ii in range(len(kk)-1):
            ibgn = indx[kk[ii]]
            iend = indx[kk[ii+1]-1]+1
            data = dset[0, ibgn:iend, :, :]
            if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                data[(data == fillvalue)] = np.nan

            if res is None:
                res = data
            else:
                res = np.append(res, data, axis=0)

        return res

    # ---------- class L1Bio::
    def _set_msm_data(self, msm_path, msm_dset, write_data, icid=None):
        """
        Writes data from dataset "msm_dset" in group "msm_path"

        Parameters
        ----------
        msm_path   :  string
           Full path to measurement group
        msm_dset   :  string
            Name of measurement dataset.
        write_data :  array-like
            Data to be written with same dimensions as dataset "msm_dset"
        scan_index :  array-like
            Indices ot the scanlines to be read or written. If scan_index is
            None, then all data is read.
        """
        if msm_path is None:
            return

        # we will overwrite existing data, thus readwrite access is required
        if not self.__rw:
            raise PermissionError('read/write access required')

        ds_path = str(Path(msm_path, 'OBSERVATIONS', msm_dset))
        dset = self.fid[ds_path]

        # overwrite the data
        if icid is None:
            if dset.shape[1:] != write_data.shape:
                print('*** Fatal: patch data has not same shape as original')
                return

            dset[0, ...] = write_data
        else:
            if self.imsm is None:
                return
            indx = self.imsm['index'][self.imsm['icid'] == icid]
            buff = np.concatenate(([indx[0]-10], indx, [indx[-1]+10]))
            kk = np.where((buff[1:] - buff[0:-1]) != 1)[0]

            for ii in range(len(kk)-1):
                ibgn = indx[kk[ii]]
                iend = indx[kk[ii+1]-1]+1
                dset[0, ibgn:iend, :, :] = write_data[kk[ii]:kk[ii+1], :, :]

        # update patch logging
        self.__patched_msm.append(ds_path)


# --------------------------------------------------
class L1BioCAL(L1Bio):
    """
    class with function to access Tropomi offline L1b calibration products

    The L1b calibration products are available for UVN (band 1-6)
    and SWIR (band 7-8).
    """
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__(l1b_product, readwrite=readwrite)

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.bands = ''

    # ---------- class L1BioCAL::
    def select(self, msm_type):
        """
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
        """
        self.bands = ''
        self.imsm = None
        for name in ('CALIBRATION', 'IRRADIANCE', 'RADIANCE'):
            for ii in '12345678':
                grp_path = str(Path('BAND{}_{}'.format(ii, name), msm_type))
                if grp_path in self.fid:
                    if self.__verbose:
                        print('*** INFO: found: ', grp_path)
                    self.bands += ii

            if self.bands:
                grp_path = str(Path('BAND%_{}'.format(name), msm_type))
                break

        if self.bands:
            self.__msm_path = grp_path
            super().msm_info(grp_path.replace('%', self.bands[0]))

        return self.bands

    # ---------- class L1BioCAL::
    def get_ref_time(self, band=None):
        """
        Returns reference start time of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    # ---------- class L1BioCAL::
    def get_delta_time(self, band=None):
        """
        Returns offset from the reference start time of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%', band))

    # ---------- class L1BioCAL::
    def get_instrument_settings(self, band=None):
        """
        Returns instrument settings of measurement

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    # ---------- class L1BioCAL::
    def get_exposure_time(self, band=None):
        """
        Returns pixel exposure time of the measurements, which is calculated
        from the parameters 'int_delay' and 'int_hold' for SWIR.

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        instr_arr = super().instrument_settings(self.__msm_path.replace('%',
                                                                        band))

        # calculate exact exposure time
        if int(band) < 7:
            return [instr['exposure_time'] for instr in instr_arr]

        return [swir_exp_time(instr['int_delay'],
                              instr['int_hold']) for instr in instr_arr]

    # ---------- class L1BioCAL::
    def get_housekeeping_data(self, band=None):
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    # ---------- class L1BioCAL::
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
        if band is None:
            band = self.bands[0]

        nscans = self.fid[self.__msm_path.replace('%', band)]['scanline'].size

        dtype = [('sequence', 'u2')]
        for name in geo_dset.split(','):
            dtype.append((name, 'f4'))
        res = np.empty((nscans,), dtype=dtype)
        res['sequence'] = self.imsm['sequence']

        grp = self.fid[str(Path(self.__msm_path.replace('%', band), 'GEODATA'))]
        for name in geo_dset.split(','):
            res[name][...] = grp[name][0, :]

        return res

    # ---------- class L1BioCAL::
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
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
           value of attribute "attr_name"
        """
        if band is None:
            band = self.bands[0]

        return super().msm_attr(self.__msm_path.replace('%', band),
                                msm_dset, attr_name)

    # ---------- class L1BioCAL::
    def get_msm_data(self, msm_dset, band='78', msm_to_row=None,
                     fill_as_nan=False):
        """
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        msm_to_row : {None, 'padding', 'rebin'}
            Return the measurement data as stored in the product (None), padded
            with NaN's for the spectral band with largest number of rows, or
            rebinned according 'measurement_to_detector_row_table'
            - Default for spectral bands is to return the data as stored.
            - Default for spectral channels is to apply padding

        Returns
        -------
        out   :    array-like
           Data of measurement dataset "msm_dset"
        """
        if not isinstance(band, str):
            raise TypeError('band must be a string')

        if band not in self.bands:
            raise ValueError('band not found in product')

        if len(band) == 2 and msm_to_row is None:
            msm_to_row = 'padding'

        data = ()
        for ii in band:
            data += (super()._get_msm_data(self.__msm_path.replace('%', ii),
                                           msm_dset, fill_as_nan=fill_as_nan),)
        if len(data) == 1:
            return data[0]

        if msm_to_row == 'padding':
            data = pad_rows(data[0], data[1])

        return np.concatenate(data, axis=data[0].ndim-1)

    # ---------- class L1BioCAL::
    def set_msm_data(self, msm_dset, data, band='78'):
        """
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        data       :  array-like
            data to be written with same dimensions as dataset "msm_dset"
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Write data to one spectral band or a channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        """
        if band not in self.bands:
            raise ValueError('band not found in product')

        col = 0
        for ii in band:
            dim = super().msm_dims(self.__msm_path.replace('%', ii), msm_dset)
            super()._set_msm_data(self.__msm_path.replace('%', ii), msm_dset,
                                  data[..., col:col+dim[-1]])
            col += dim[-1]


# --------------------------------------------------
class L1BioIRR(L1Bio):
    """
    class with function to access Tropomi offline L1b irradiance products
    """
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__(l1b_product, readwrite=readwrite)

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.bands = ''

    # ---------- class L1BioIRR::
    def select(self, msm_type='STANDARD_MODE'):
        """
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
        """
        self.bands = ''
        self.imsm = None
        for ii in '12345678':
            grp_path = str(Path('BAND{}_IRRADIANCE'.format(ii), msm_type))
            if grp_path in self.fid:
                if self.__verbose:
                    print('*** INFO: found: ', grp_path)
                self.bands += ii

        if self.bands:
            self.__msm_path = str(Path('BAND%_IRRADIANCE', msm_type))
            super().msm_info(self.__msm_path.replace('%', self.bands[0]))

        return self.bands

    # ---------- class L1BioIRR::
    def get_ref_time(self, band=None):
        """
        Returns reference start time of measurements
        """
        if band is None:
            band = self.bands[0]

        return super().ref_time(self.__msm_path.replace('%', band))

    # ---------- class L1BioIRR::
    def get_delta_time(self, band=None):
        """
        Returns offset from the reference start time of measurement
        """
        if band is None:
            band = self.bands[0]

        return super().delta_time(self.__msm_path.replace('%', band))

    # ---------- class L1BioIRR::
    def get_instrument_settings(self, band=None):
        """
        Returns instrument settings of measurement
        """
        if band is None:
            band = self.bands[0]

        return super().instrument_settings(self.__msm_path.replace('%', band))

    # ---------- class L1BioIRR::
    def get_exposure_time(self, band=None):
        """
        Returns pixel exposure time of the measurements, which is calculated
        from the parameters 'int_delay' and 'int_hold' for SWIR.

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]

        instr_arr = super().instrument_settings(self.__msm_path.replace('%',
                                                                        band))

        # calculate exact exposure time
        if int(band) < 7:
            return [instr['exposure_time'] for instr in instr_arr]

        return [swir_exp_time(instr['int_delay'],
                              instr['int_hold']) for instr in instr_arr]

    # ---------- class L1BioIRR::
    def get_housekeeping_data(self, band=None):
        """
        Returns housekeeping data of measurements
        """
        if band is None:
            band = self.bands[0]

        return super().housekeeping_data(self.__msm_path.replace('%', band))

    # ---------- class L1BioIRR::
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
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
            Value of attribute "attr_name"
        """
        if band is None:
            band = self.bands[0]

        return super().msm_attr(self.__msm_path.replace('%', band),
                                msm_dset, attr_name)

    # ---------- class L1BioIRR::
    def get_msm_data(self, msm_dset, band='78', msm_to_row=None,
                     fill_as_nan=False):
        """
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        msm_to_row : {None, 'padding', 'rebin'}
            Return the measurement data as stored in the product (None), padded
            with NaN's for the spectral band with largest number of rows, or
            rebinned according 'measurement_to_detector_row_table'
            - Default for spectral bands is to return the data as stored.
            - Default for spectral channels is to apply padding.

        Returns
        -------
        out   :    array-like
            Data of measurement dataset "msm_dset"
        """
        if not isinstance(band, str):
            raise TypeError('band must be a string')

        if band not in self.bands:
            raise ValueError('band not found in product')

        if len(band) == 2 and msm_to_row is None:
            msm_to_row = 'padding'

        res = None
        for ii in band:
            data = super()._get_msm_data(self.__msm_path.replace('%', ii),
                                         msm_dset, fill_as_nan=fill_as_nan)
            if res is None:
                res = data
            else:
                if msm_to_row == 'padding':
                    (res, data) = pad_rows(res, data)

                res = np.concatenate((res, data), axis=data.ndim-1)

        return res

    # ---------- class L1BioIRR::
    def set_msm_data(self, msm_dset, data, band='78'):
        """
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset
        data       :  array-like
            data to be written with same dimensions as dataset "msm_dset"
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        """
        if not isinstance(band, str):
            raise TypeError('band must be a string')

        if band not in self.bands:
            raise ValueError('band not found in product')

        col = 0
        for ii in band:
            dim = super().msm_dims(self.__msm_path.replace('%', ii), msm_dset)
            super()._set_msm_data(self.__msm_path.replace('%', ii), msm_dset,
                                  data[..., col:col+dim[-1]])
            col += dim[-1]


# --------------------------------------------------
class L1BioRAD(L1Bio):
    """
    class with function to access Tropomi offline L1b radiance products
    """
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__(l1b_product, readwrite=readwrite)

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.bands = ''

    # ---------- class L1BioRAD::
    def select(self, msm_type='STANDARD_MODE'):
        """
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
        """
        self.bands = ''
        self.imsm = None
        for ii in '12345678':
            grp_path = str(Path('BAND{}_RADIANCE'.format(ii), msm_type))
            if grp_path in self.fid:
                if self.__verbose:
                    print('*** INFO: found: ', grp_path)
                self.bands = ii
                break              # only one band per product

        if self.bands:
            self.__msm_path = grp_path
            super().msm_info(self.__msm_path)

        return self.bands

    # ---------- class L1BioRAD::
    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
        return super().ref_time(self.__msm_path)

    # ---------- class L1BioRAD::
    def get_delta_time(self):
        """
        Returns offset from the reference start time of measurement
        """
        return super().delta_time(self.__msm_path)

    # ---------- class L1BioRAD::
    def get_instrument_settings(self):
        """
        Returns instrument settings of measurement
        """
        return super().instrument_settings(self.__msm_path)

    # ---------- class L1BioRAD::
    def get_exposure_time(self):
        """
        Returns pixel exposure time of the measurements, which is calculated
        from the parameters 'int_delay' and 'int_hold' for SWIR.
        """
        instr_arr = super().instrument_settings(self.__msm_path)

        # calculate exact exposure time
        if int(self.bands) < 7:
            return [instr['exposure_time'] for instr in instr_arr]

        return [swir_exp_time(instr['int_delay'],
                              instr['int_hold']) for instr in instr_arr]

    # ---------- class L1BioRAD::
    def get_housekeeping_data(self, icid=None):
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        icid  :   integer
           select housekeeping data of measurements with given ICID

        Returns
        -------
        out   :   array-like
           Numpy rec-array with housekeeping data
        """
        if icid is None:
            return super().housekeeping_data(self.__msm_path)

        res = super().housekeeping_data(self.__msm_path)
        return res[self.imsm['icid'] == icid]

    # ---------- class L1BioRAD::
    def get_geo_data(self, geo_dset='latitude,longitude', icid=None):
        """
        Returns data of selected datasets from the GEODATA group

        Parameters
        ----------
        geo_dset  :  string
           Name(s) of datasets in the GEODATA group, comma separated
        icid  :   integer
           select geolocation data of measurements with given ICID

        Returns
        -------
        out   :   array-like
           Numpy rec-array with data of selected datasets from the GEODATA group
        """
        nrows = self.fid[self.__msm_path]['ground_pixel'].size

        grp = self.fid[str(Path(self.__msm_path, 'GEODATA'))]

        if icid is None:
            nscans = self.fid[self.__msm_path]['scanline'].size

            dtype = [('sequence', 'u2')]
            for name in geo_dset.split(','):
                dtype.append((name, 'f4'))
            res = np.empty((nscans, nrows), dtype=dtype)
            for ii in range(nscans):
                res['sequence'][ii, :] = self.imsm['sequence'][ii]

            for name in geo_dset.split(','):
                res[name][...] = grp[name][0, :, :]
        else:
            indx = self.imsm['index'][self.imsm['icid'] == icid]
            nscans = len(indx)

            dtype = [('sequence', 'u2')]
            for name in geo_dset.split(','):
                dtype.append((name, 'f4'))
            res = np.empty((nscans, nrows), dtype=dtype)
            res['sequence'][:, :] = np.repeat(
                self.imsm['sequence'][indx],
                nrows, axis=0).reshape(nscans, nrows)

            for name in geo_dset.split(','):
                res[name][:, :] = grp[name][0, indx, :]

        return res

    # ---------- class L1BioRAD::
    def get_msm_attr(self, msm_dset, attr_name):
        """
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

        """
        return super().msm_attr(self.__msm_path, msm_dset, attr_name)

    # ---------- class L1BioRAD::
    def get_msm_data(self, msm_dset, icid=None, msm_to_row=None,
                     fill_as_nan=False):
        """
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           Name of measurement dataset
        icid  :   integer
           Select measurement data of measurements with given ICID
        msm_to_row : {None, 'rebin'}
            Return the measurement data as stored in the product (None), or
            rebinned according 'measurement_to_detector_row_table'
            - Default is to return the data as stored.

        Returns
        -------
        out   :    array-like
           Data of measurement dataset "msm_dset"
        """
        if msm_to_row is None:
            return super()._get_msm_data(self.__msm_path, msm_dset,
                                         icid=icid, fill_as_nan=fill_as_nan)
        return None

    # ---------- class L1BioRAD::
    def set_msm_data(self, msm_dset, data, icid=None):
        """
        writes data to measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset  :  string
           Name of measurement dataset
        data      :  array-like
           Data to be written with same dimensions as dataset "msm_dset"
        icid      :   integer
           ICID of measurement data
        """
        return super()._set_msm_data(self.__msm_path, msm_dset, data, icid=icid)


# --------------------------------------------------
class L1BioENG(L1Bio):
    """
    class with function to access Tropomi offline L1b engineering products

    The L1b engineering products are available for UVN (band 1-6)
    and SWIR (band 7-8).
    """
    def __init__(self, l1b_product, readwrite=False, verbose=False):
        super().__init__(l1b_product, readwrite=readwrite)

        # initialize class-attributes
        self.__verbose = verbose
        self.__msm_path = None
        self.bands = ''

    # ---------- class L1BioENG::
    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
        return self.fid['reference_time'][0].astype(int)

    # ---------- class L1BioENG::
    def get_delta_time(self):
        """
        Returns offset from the reference start time of measurement
        """
        return self.fid['/MSMTSET/msmtset']['delta_time'][:].astype(int)

    # ---------- class L1BioENG::
    def get_msmtset(self):
        """
        Returns L1B_ENG_DB/SATELLITE_INFO/satellite_pos
        """
        return self.fid['/SATELLITE_INFO/satellite_pos'][:]

    # ---------- class L1BioENG::
    def get_msmtset_db(self):
        """
        Returns compressed msmtset from L1B_ENG_DB/MSMTSET/msmtset

        Note this function is used to fill the SQLite product databases
        """
        dtype_msmt_db = np.dtype([('meta_id', np.int32),
                                  ('ic_id', np.uint16),
                                  ('ic_version', np.uint8),
                                  ('class', np.uint8),
                                  ('repeats', np.uint16),
                                  ('exp_per_mcp', np.uint16),
                                  ('exp_time_us', np.uint32),
                                  ('mcp_us', np.uint32),
                                  ('delta_time_start', np.int32),
                                  ('delta_time_end', np.int32)])

        # read full msmtset
        msmtset = self.fid['/MSMTSET/msmtset'][:]

        # get indices to start and end of every measurement (based in ICID)
        icid = msmtset['icid']
        indx = np.where(np.diff(icid) != 0)[0] + 1
        indx = np.insert(indx, 0, 0)
        indx = np.append(indx, -1)

        # compress data from msmtset
        msmt = np.zeros(indx.size-1, dtype=dtype_msmt_db)
        msmt['ic_id'][:] = msmtset['icid'][indx[0:-1]]
        msmt['ic_version'][:] = msmtset['icv'][indx[0:-1]]
        msmt['class'][:] = msmtset['class'][indx[0:-1]]
        msmt['delta_time_start'][:] = msmtset['delta_time'][indx[0:-1]]
        msmt['delta_time_end'][:] = msmtset['delta_time'][indx[1:]]

        # add SWIR timing information
        timing = self.fid['/DETECTOR4/timing'][:]
        msmt['mcp_us'][:] = timing['mcp_us'][indx[1:]-1]
        msmt['exp_time_us'][:] = timing['exp_time_us'][indx[1:]-1]
        msmt['exp_per_mcp'][:] = timing['exp_per_mcp'][indx[1:]-1]
        # duration per ICID execution in micro-seconds
        duration = 1000 * (msmt['delta_time_end'] - msmt['delta_time_start'])
        # duration can be zero
        mask = msmt['mcp_us'] > 0
        # divide duration by measurement period in micro-seconds
        msmt['repeats'][mask] = (duration[mask]
                                 / (msmt['mcp_us'][mask])).astype(np.uint16)

        return msmt

    # ---------- class L1BioENG::
    def get_swir_hk_db(self, stats=None, fill_as_nan=False):
        """
        Returns the most important SWIR house keeping parameters

        Parameters
        ----------
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Note this function is used to fill the SQLite product datbase and
           HDF5 monitoring database
        """
        dtype_hk_db = np.dtype([('detector_temp', np.float32),
                                ('grating_temp', np.float32),
                                ('imager_temp', np.float32),
                                ('obm_temp', np.float32),
                                ('calib_unit_temp', np.float32),
                                ('fee_inner_temp', np.float32),
                                ('fee_board_temp', np.float32),
                                ('fee_ref_volt_temp', np.float32),
                                ('fee_video_amp_temp', np.float32),
                                ('fee_video_adc_temp', np.float32),
                                ('detector_heater', np.float32),
                                ('obm_heater_cycle', np.float32),
                                ('fee_box_heater_cycle', np.float32),
                                ('obm_heater', np.float32),
                                ('fee_box_heater', np.float32)])

        num_eng_pkts = self.fid['nr_of_engdat_pkts'].size
        swir_hk = np.empty(num_eng_pkts, dtype=dtype_hk_db)

        hk_tbl = self.fid['/DETECTOR4/DETECTOR_HK/temperature_info'][:]
        swir_hk['detector_temp'] = hk_tbl['temp_det_ts2']
        swir_hk['fee_inner_temp'] = hk_tbl['temp_d1_box']
        swir_hk['fee_board_temp'] = hk_tbl['temp_d5_cold']
        swir_hk['fee_ref_volt_temp'] = hk_tbl['temp_a3_vref']
        swir_hk['fee_video_amp_temp'] = hk_tbl['temp_d6_vamp']
        swir_hk['fee_video_adc_temp'] = hk_tbl['temp_d4_vadc']

        hk_tbl = self.fid['/NOMINAL_HK/TEMPERATURES/hires_temperatures'][:]
        swir_hk['grating_temp'] = hk_tbl['hires_temp_1']

        hk_tbl = self.fid['/NOMINAL_HK/TEMPERATURES/instr_temperatures'][:]
        swir_hk['imager_temp'] = hk_tbl['instr_temp_29']
        swir_hk['obm_temp'] = hk_tbl['instr_temp_28']
        swir_hk['calib_unit_temp'] = hk_tbl['instr_temp_25']

        hk_tbl = self.fid['/DETECTOR4/DETECTOR_HK/heater_data'][:]
        swir_hk['detector_heater'] = hk_tbl['det_htr_curr']

        hk_tbl = self.fid['/NOMINAL_HK/HEATERS/heater_data'][:]
        swir_hk['obm_heater'] = hk_tbl['meas_cur_val_htr12']
        swir_hk['obm_heater_cycle'] = hk_tbl['last_pwm_val_htr12']
        swir_hk['fee_box_heater'] = hk_tbl['meas_cur_val_htr13']
        swir_hk['fee_box_heater_cycle'] = hk_tbl['last_pwm_val_htr13']

        # Note all elements should be floats!
        if fill_as_nan:
            for key in dtype_hk_db.names:
                swir_hk[key][swir_hk[key] == 999.] = np.nan

        if stats is None:
            return swir_hk

        if stats == 'median':
            hk_median = np.empty(1, dtype=dtype_hk_db)
            for key in dtype_hk_db.names:
                if np.all(np.isnan(swir_hk[key])):
                    hk_median[key][0] = np.nan
                elif np.nanmin(swir_hk[key]) == np.nanmax(swir_hk[key]):
                    hk_median[key][0] = swir_hk[key][0]
                else:
                    hk_median[key][0] = biweight(swir_hk[key])
            return hk_median

        if stats == 'range':
            hk_min = np.empty(1, dtype=dtype_hk_db)
            hk_max = np.empty(1, dtype=dtype_hk_db)
            for key in dtype_hk_db.names:
                if np.all(np.isnan(swir_hk[key])):
                    hk_min[key][0] = np.nan
                    hk_max[key][0] = np.nan
                elif np.nanmin(swir_hk[key]) == np.nanmax(swir_hk[key]):
                    hk_min[key][0] = swir_hk[key][0]
                    hk_max[key][0] = swir_hk[key][0]
                else:
                    hk_min[key][0] = np.nanmin(swir_hk[key])
                    hk_max[key][0] = np.nanmax(swir_hk[key])
            return (hk_min, hk_max)

        return None
