"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The classes L1Bio, L1BioIRR, L1BioRAD and L1BioENG provide read/write access to
offline level 1b products, resp. calibration, irradiance, radiance
and engineering.

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from setuptools_scm import get_version

import h5py
import numpy as np

from moniplot.biweight import biweight

from .swir_texp import swir_exp_time

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
            arr1 = np.full(arr2.shape, np.nan, dtype=arr2.dtype)
            arr1[0:buff.shape[0], :] = buff
        elif arr1.shape[0] > arr2.shape[0]:
            buff = arr2.copy()
            arr2 = np.full(arr1.shape, np.nan, dtype=arr2.dtype)
            arr2[0:buff.shape[0], :] = buff
    else:
        if arr1.shape[1] < arr2.shape[1]:
            buff = arr1.copy()
            arr1 = np.full(arr2.shape, np.nan, dtype=arr2.dtype)
            arr1[:, 0:buff.shape[1], :] = buff
        elif arr1.shape[1] > arr2.shape[1]:
            buff = arr2.copy()
            arr2 = np.full(arr1.shape, np.nan, dtype=arr2.dtype)
            arr2[:, 0:buff.shape[1], :] = buff

    return (arr1, arr2)


# - class definition -------------------------------
class L1Bio:
    """
    class with methods to access Tropomi L1B calibration products

    The L1b calibration products are available for UVN (band 1-6)
    and SWIR (band 7-8).

    Attributes
    ----------
    fid : h5py.File
    filename : string
    bands : string

    Methods
    -------
    close()
       Close recources.
    get_attr(attr_name)
       Obtain value of an HDF5 file attribute.
    get_orbit()
       Returns absolute orbit number.
    get_processor_version()
       Returns version of the L01b processor used to generate this product.
    get_coverage_time()
       Returns start and end of the measurement coverage time.
    get_creation_time()
       Returns datetime when the L1b product was created.
    select(msm_type=None)
       Select a calibration measurement as <processing class>_<ic_id>.
    sequence(band=None)
       Returns sequence number for each unique measurement based on ICID
       and delta_time.
    get_ref_time(band=None)
       Returns reference start time of measurements.
    get_delta_time(band=None)
       Returns offset from the reference start time of measurement.
    get_instrument_settings(band=None)
       Returns instrument settings of measurement.
    get_exposure_time(band=None)
       Returns pixel exposure time of the measurements, which is calculated
       from the parameters 'int_delay' and 'int_hold' for SWIR.
    get_housekeeping_data(band=None)
       Returns housekeeping data of measurements.
    get_geo_data(geo_dset=None, band=None)
       Returns data of selected datasets from the GEODATA group.
    get_msm_attr(msm_dset, attr_name, band=None)
       Returns value attribute of measurement dataset "msm_dset".
    get_msm_data(msm_dset, band=None, fill_as_nan=False, msm_to_row=None)
       Reads data from dataset "msm_dset".
    set_msm_data(msm_dset, new_data)
       Replace data of dataset "msm_dset" with new_data.

    Notes
    -----

    Examples
    --------
    """
    band_groups = ('/BAND%_CALIBRATION', '/BAND%_IRRADIANCE',
                   '/BAND%_RADIANCE')
    geo_dset = 'satellite_latitude,satellite_longitude'
    msm_type = None

    def __init__(self, l1b_product, readwrite=False, verbose=False):
        """
        Initialize access to a Tropomi offline L1b product
        """
        # open L1b product as HDF5 file
        if not Path(l1b_product).is_file():
            raise FileNotFoundError(f'{l1b_product} does not exist')

        # initialize private class-attributes
        self.__rw = readwrite
        self.__verbose = verbose
        self.__msm_path = None
        self.__patched_msm = []
        self.filename = l1b_product
        self.bands = ''

        if readwrite:
            self.fid = h5py.File(l1b_product, "r+")
        else:
            self.fid = h5py.File(l1b_product, "r")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}({self.filename!r}, readwrite={self.__rw!r})'

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

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
        Close resources.

        Notes
        -----
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
            # pylint: disable=no-member
            sgrp = self.fid.require_group("/METADATA/SRON_METADATA")
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = get_version(root='..',
                                                relative_to=__file__)
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

        self.fid.close()
        self.fid = None

    # ---------- PUBLIC FUNCTIONS ----------
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

    def get_orbit(self):
        """
        Returns absolute orbit number
        """
        res = self.get_attr('orbit')
        if res is None:
            return None

        return int(res)

    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        attr = self.get_attr('processor_version')
        if attr is None:
            return None

        # pylint: disable=no-member
        return attr.decode('ascii')

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

        # pylint: disable=no-member
        return (attr_start.decode('ascii'),
                attr_end.decode('ascii'))

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

    def select(self, msm_type=None):
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
        if msm_type is None:
            if self.msm_type is None:
                raise ValueError('parameter msm_type is not defined')
            msm_type = self.msm_type

        self.bands = ''
        for name in self.band_groups:
            for ii in '12345678':
                grp_path = PurePosixPath(name.replace('%', ii), msm_type)
                if str(grp_path) in self.fid:
                    if self.__verbose:
                        print('*** INFO: found: ', grp_path)
                    self.bands += ii

            if self.bands:
                self.__msm_path = str(
                    PurePosixPath(name, msm_type))
                break

        return self.bands

    def sequence(self, band=None):
        """
        Returns sequence number for each unique measurement based on ICID
          and delta_time

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out  :  array-like
          Numpy rec-array with sequence number, ICID and delta-time
        """
        if self.__msm_path is None:
            return None

        if band is None or len(band) > 1:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        grp = self.fid[str(PurePosixPath(msm_path, 'INSTRUMENT'))]

        icid_list = np.squeeze(grp['instrument_configuration']['ic_id'])
        master_cycle = grp['instrument_settings']['master_cycle_period_us'][0]
        master_cycle /= 1000
        grp = self.fid[str(PurePosixPath(msm_path, 'OBSERVATIONS'))]
        delta_time = np.squeeze(grp['delta_time'])

        # define result as numpy array
        length = delta_time.size
        res = np.empty((length,), dtype=[('sequence', 'u2'),
                                         ('icid', 'u2'),
                                         ('delta_time', 'u4'),
                                         ('index', 'u4')])
        res['sequence'] = [0]
        res['icid'] = icid_list
        res['delta_time'] = delta_time
        res['index'] = np.arange(length, dtype=np.uint32)
        if length == 1:
            return res

        # determine sequence number
        buff_icid = np.concatenate(([icid_list[0]-10], icid_list,
                                    [icid_list[-1]+10]))
        dt_thres = 10 * master_cycle
        buff_time = np.concatenate(([delta_time[0] - 10 * dt_thres], delta_time,
                                    [delta_time[-1] + 10 * dt_thres]))

        indx = (((buff_time[1:] - buff_time[0:-1]) > dt_thres)
                | ((buff_icid[1:] - buff_icid[0:-1]) != 0)).nonzero()[0]
        for ii in range(len(indx)-1):
            res['sequence'][indx[ii]:indx[ii+1]] = ii

        return res

    def get_ref_time(self, band=None):
        """
        Returns reference start time of measurements

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

        msm_path = self.__msm_path.replace('%', band)
        grp = self.fid[str(PurePosixPath(msm_path, 'OBSERVATIONS'))]

        return datetime(2010, 1, 1, 0, 0, 0) \
            + timedelta(seconds=int(grp['time'][0]))

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

        msm_path = self.__msm_path.replace('%', band)
        grp = self.fid[str(PurePosixPath(msm_path, 'OBSERVATIONS'))]

        return grp['delta_time'][0, :].astype(int)

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

        msm_path = self.__msm_path.replace('%', band)
        #
        # Due to a bug in python module h5py (v2.6.0), it fails to read
        # the UVN instrument settings directy, with exception:
        #    KeyError: 'Unable to open object (Component not found)'.
        # This is my workaround
        #
        grp = self.fid[str(PurePosixPath(msm_path, 'INSTRUMENT'))]
        instr = np.empty(grp['instrument_settings'].shape,
                         dtype=grp['instrument_settings'].dtype)
        grp['instrument_settings'].read_direct(instr)
        # for name in grp['instrument_settings'].dtype.names:
        #     instr[name][:] = grp['instrument_settings'][name]

        return instr

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

        instr_arr = self.get_instrument_settings(band)

        # calculate exact exposure time
        if int(band) < 7:
            return [instr['exposure_time'] for instr in instr_arr]

        return [swir_exp_time(instr['int_delay'], instr['int_hold'])
                for instr in instr_arr]

    def get_housekeeping_data(self, band=None):
        """
        Returns housekeeping data of measurements

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        grp = self.fid[str(PurePosixPath(msm_path, 'INSTRUMENT'))]

        return np.squeeze(grp['housekeeping_data'])

    def get_geo_data(self, geo_dset=None, band=None):
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
        out   :   dict of numpy
           Compound array with data of selected datasets from the GEODATA group
        """
        if self.__msm_path is None:
            return None

        if geo_dset is None:
            geo_dset = self.geo_dset

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        grp = self.fid[str(PurePosixPath(msm_path, 'GEODATA'))]

        res = {}
        for name in geo_dset.split(','):
            res[name] = grp[name][0, ...]

        return res

    def get_msm_attr(self, msm_dset, attr_name, band=None):
        """
        Returns value attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        attr_name : string
            Name of the attribute
        msm_dset  :  string
            Name of measurement dataset
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band

        Returns
        -------
        out   :   scalar or numpy array
            Value of attribute "attr_name"
        """
        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands[0]

        msm_path = self.__msm_path.replace('%', band)
        ds_path = str(PurePosixPath(msm_path, 'OBSERVATIONS', msm_dset))
        if attr_name in self.fid[ds_path].attrs.keys():
            attr = self.fid[ds_path].attrs[attr_name]
            if isinstance(attr, bytes):
                return attr.decode('ascii')

            return attr

        return None

    def get_msm_data(self, msm_dset, band=None,
                     fill_as_nan=False, msm_to_row=None):
        """
        Reads data from dataset "msm_dset"

        Parameters
        ----------
        msm_dset :  string
            Name of measurement dataset.
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns
                  both bands (Calibration, Irradiance)
                  or one band (Radiance)
        fill_as_nan :  boolean
            Set data values equal (KNMI) FillValue to NaN
        msm_to_row : boolean
            Combine two bands using padding if necessary

        Returns
        -------
        out   :  values read from or written to dataset "msm_dset"
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if self.__msm_path is None:
            return None

        if band is None:
            band = self.bands
        elif not isinstance(band, str):
            raise TypeError('band must be a string')
        elif band not in self.bands:
            raise ValueError('band not found in product')

        if len(band) == 2 and msm_to_row is None:
            msm_to_row = 'padding'

        data = ()
        for ii in band:
            msm_path = self.__msm_path.replace('%', ii)
            ds_path = str(PurePosixPath(msm_path, 'OBSERVATIONS', msm_dset))
            dset = self.fid[ds_path]

            if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                buff = np.squeeze(dset)
                buff[(buff == fillvalue)] = np.nan
                data += (buff,)
            else:
                data += (np.squeeze(dset),)

        if len(band) == 1:
            return data[0]

        if msm_to_row == 'padding':
            data = pad_rows(data[0], data[1])

        return np.concatenate(data, axis=data[0].ndim-1)

    def set_msm_data(self, msm_dset, new_data):
        """
        Replace data of dataset "msm_dset" with new_data

        Parameters
        ----------
        msm_dset   :  string
            Name of measurement dataset.
        new_data :  array-like
            Data to be written with same dimensions as dataset "msm_dset"
        """
        if self.__msm_path is None:
            return

        # we will overwrite existing data, thus readwrite access is required
        if not self.__rw:
            raise PermissionError('read/write access required')

        # overwrite the data
        col = 0
        for ii in self.bands:
            msm_path = self.__msm_path.replace('%', ii)
            ds_path = str(PurePosixPath(msm_path, 'OBSERVATIONS', msm_dset))
            dset = self.fid[ds_path]

            dims = dset.shape
            dset[0, ...] = new_data[..., col:col+dims[-1]]
            col += dims[-1]

            # update patch logging
            self.__patched_msm.append(ds_path)


# --------------------------------------------------
class L1BioIRR(L1Bio):
    """
    class with methods to access Tropomi L1B irradiance products
    """
    band_groups = ('/BAND%_IRRADIANCE',)
    geo_dset = 'earth_sun_distance'
    msm_type = 'STANDARD_MODE'


# --------------------------------------------------
class L1BioRAD(L1Bio):
    """
    class with function to access Tropomi L1B radiance products
    """
    band_groups = ('/BAND%_RADIANCE',)
    geo_dset = 'latitude,longitude'
    msm_type = 'STANDARD_MODE'


# --------------------------------------------------
class L1BioENG:
    """
    class with methods to access Tropomi offline L1b engineering products

    Attributes
    ----------
    fid : HDF5 file object
    filename : string

    Methods
    -------
    close()
       Close recources.
    get_attr(attr_name)
       Obtain value of an HDF5 file attribute.
    get_orbit()
       Returns absolute orbit number.
    get_processor_version()
       Returns version of the L01b processor used to generate this product.
    get_coverage_time()
       Returns start and end of the measurement coverage time.
    get_creation_time()
       Returns datetime when the L1b product was created.
    get_ref_time()
       Returns reference start time of measurements.
    get_delta_time()
       Returns offset from the reference start time of measurement.
    get_msmtset()
       Returns L1B_ENG_DB/SATELLITE_INFO/satellite_pos.
    get_msmtset_db()
       Returns compressed msmtset from L1B_ENG_DB/MSMTSET/msmtset.
    get_swir_hk_db(stats=None, fill_as_nan=False)
       Returns the most important SWIR house keeping parameters.

    Notes
    -----
    The L1b engineering products are available for UVN (band 1-6)
    and SWIR (band 7-8).

    Examples
    --------
    """
    def __init__(self, l1b_product):
        """
        Initialize access to a Tropomi offline L1b product
        """
        # open L1b product as HDF5 file
        if not Path(l1b_product).is_file():
            raise FileNotFoundError(f'{l1b_product} does not exist')

        # initialize private class-attributes
        self.filename = l1b_product
        self.fid = h5py.File(l1b_product, "r")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}({self.filename!r})'

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

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
        close access to product
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    # ---------- PUBLIC FUNCTIONS ----------
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

    def get_orbit(self):
        """
        Returns absolute orbit number
        """
        res = self.get_attr('orbit')
        if res is None:
            return None

        return int(res)

    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        attr = self.get_attr('processor_version')
        if attr is None:
            return None

        # pylint: disable=no-member
        return attr.decode('ascii')

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

        # pylint: disable=no-member
        return (attr_start.decode('ascii'),
                attr_end.decode('ascii'))

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

    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
        return self.fid['reference_time'][0].astype(int)

    def get_delta_time(self):
        """
        Returns offset from the reference start time of measurement
        """
        return self.fid['/MSMTSET/msmtset']['delta_time'][:].astype(int)

    def get_msmtset(self):
        """
        Returns L1B_ENG_DB/SATELLITE_INFO/satellite_pos
        """
        return self.fid['/SATELLITE_INFO/satellite_pos'][:]

    def get_msmtset_db(self):
        """
        Returns compressed msmtset from L1B_ENG_DB/MSMTSET/msmtset

        Notes
        -----
        This function is used to fill the SQLite product databases
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
        indx = (np.diff(icid) != 0).nonzero()[0] + 1
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

    def get_swir_hk_db(self, stats=None, fill_as_nan=False):
        """
        Returns the most important SWIR house keeping parameters

        Parameters
        ----------
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Notes
        -----
        This function is used to fill the SQLite product datbase and
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

        # CHECK: works only when all elements of swir_hk are floats
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
