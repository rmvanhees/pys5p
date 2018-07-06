"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class ICMio provides read access to S5p Tropomi ICM_CA_SIR products

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
from pathlib import Path

import h5py
import numpy as np

from .version import version as __version__

# - global parameters ------------------------------

# - local functions --------------------------------


# - class definition -------------------------------
class ICMio():
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

        assert Path(icm_product).is_file(), \
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
        if self.__patched_msm:
            from datetime import datetime

            sgrp = self.fid.require_group("METADATA/SRON_METADATA")
            sgrp.attrs['dateStamp'] = datetime.utcnow().isoformat()
            sgrp.attrs['git_tag'] = __version__
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()
        return False   # any exception is raised by the with statement.

    # ---------- RETURN VERSION of the S/W ----------
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

    # -------------------------
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
                grp_path = str(Path(msm_path.replace('%', ii), msm_type))
                if grp_path in self.fid:
                    self.bands += ii
        else:
            grp_list = ['ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE']
            for ii in '12345678':
                for name in grp_list:
                    grp_path = str(Path('BAND{}_{}'.format(ii, name), msm_type))
                    if grp_path in self.fid:
                        msm_path = 'BAND{}_{}'.format('%', name)
                        self.bands += ii

        # return in case no data was found
        if self.bands:
            self.__msm_path = Path(msm_path, msm_type)

        return self.bands

    # ---------- Functions that work before MSM selection ----------
    def get_orbit(self):
        """
        Returns reference orbit number
        """
        if 'reference_orbit' in self.fid.attrs:
            return int(self.fid.attrs['reference_orbit'])

        return None

    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        if 'processor_version' not in self.fid.attrs:
            return None

        res = self.fid.attrs['processor_version']
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        if 'time_coverage_start' not in self.fid.attrs \
           or 'time_coverage_end' not in self.fid.attrs:
            return None

        res1 = self.fid.attrs['time_coverage_start']
        if isinstance(res1, bytes):
            res1 = res1.decode('ascii')

        res2 = self.fid.attrs['time_coverage_end']
        if isinstance(res2, bytes):
            res2 = res2.decode('ascii')

        return (res1, res2,)

    def get_creation_time(self):
        """
        Returns version of the L01b processor
        """
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].split(b'=')[1].decode('ascii')

    def get_attr(self, attr_name):
        """
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        """
        if attr_name not in self.fid.attrs:
            return None

        res = self.fid.attrs[attr_name]
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

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

        ref_time = datetime(2010, 1, 1, 0, 0, 0)
        if not self.__msm_path:
            return ref_time

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                ref_time += timedelta(seconds=int(sgrp['time'][0]))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = str(Path(msm_path).parent / 'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                ref_time += timedelta(seconds=int(sgrp['time'][0]))
        else:
            grp = self.fid[msm_path]
            sgrp = grp['OBSERVATIONS']
            ref_time += timedelta(seconds=int(sgrp['time'][0]))
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
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                if res is None:
                    res = sgrp['delta_time'][0, :].astype(int)
                else:
                    res = np.append(res, sgrp['delta_time'][0, :].astype(int))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = str(Path(msm_path).parent / 'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['OBSERVATIONS']
                if res is None:
                    res = sgrp['delta_time'][0, :].astype(int)
                else:
                    res = np.append(res, sgrp['delta_time'][0, :].astype(int))
        else:
            grp = self.fid[msm_path]
            sgrp = grp['OBSERVATIONS']
            res = sgrp['delta_time'][0, :].astype(int)

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
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['instrument_settings'][:]
                else:
                    res = np.append(res, sgrp['instrument_settings'][:])
        elif msm_type == 'DPQF_MAP':
            grp_path = str(Path(msm_path).parent / 'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = sgrp['instrument_settings'][:]
                else:
                    res = np.append(res, sgrp['instrument_settings'][:])
        elif msm_type == 'NOISE':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_msmt_keys']
            icid = dset['icid'][dset.size // 2]
            grp_path = str(Path(
                'BAND{}_CALIBRATION'.format(band),
                'BACKGROUND_RADIANCE_MODE_{:04d}'.format(icid)))
            grp = self.fid[grp_path]
            sgrp = grp['INSTRUMENT']
            res = sgrp['instrument_settings'][:]
        else:
            grp = self.fid[str(Path(msm_path, 'INSTRUMENT'))]
            res = grp['instrument_settings'][:]

        return res

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
        # obtain instrument settings
        instr_arr = self.get_instrument_settings(band)
        if instr_arr is None:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        # calculate exact exposure time
        res = []
        for instr in instr_arr:
            if int(band) > 6:
                res.append(1.25e-6 * (65540
                                      - instr['int_delay'] + instr['int_hold']))
            else:
                res.append(instr['exposure_time'])

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
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = np.squeeze(sgrp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(sgrp['housekeeping_data']))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = str(Path(msm_path).parent / 'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['INSTRUMENT']
                if res is None:
                    res = np.squeeze(sgrp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(sgrp['housekeeping_data']))
        else:
            grp = self.fid[str(Path(msm_path, 'INSTRUMENT'))]
            res = np.squeeze(grp['housekeeping_data'])

        return res

    # -------------------------
    def get_msmt_keys(self, band=None):
        """
        Read msmt_keys from the analysis groups

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band

        Returns
        -------
         [ANALOG_OFFSET_SWIR] analog_offset_swir_group_keys
         [LONG_TERM_SWIR]     long_term_swir_group_keys
         [NOISE]              noise_msmt_keys
         else                 None
        """
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            return np.squeeze(grp[msm_type.lower() + '_group_keys'])

        if msm_type == 'NOISE':
            grp = self.fid[msm_path]
            return np.squeeze(grp[msm_type.lower() + '_msmt_keys'])

        return None

    # -------------------------
    def get_msm_attr(self, msm_dset, attr_name, band=None):
        """
        Returns attribute of measurement dataset "msm_dset"

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
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
            ds_path = str(Path(str(self.__msm_path).replace('%', band),
                               dset_grp, msm_dset))
            if ds_path not in self.fid:
                continue

            if attr_name in self.fid[ds_path].attrs:
                attr = self.fid[ds_path].attrs[attr_name]
                if isinstance(attr, bytes):
                    return attr.decode('ascii')

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
        if not self.__msm_path:
            return None

        if band is None:
            band = str(self.bands[0])
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['GEODATA']
                for key in geo_dset.split(','):
                    if res is None:
                        res = np.squeeze(sgrp[key])
                    else:
                        res = np.append(res, np.squeeze(sgrp[key]))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = str(Path(msm_path).parent / 'ANALOG_OFFSET_SWIR')
            grp = self.fid[grp_path]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = str(Path('BAND{}_CALIBRATION'.format(band),
                                    name.decode('ascii')))
                grp = self.fid[grp_path]
                sgrp = grp['GEODATA']
                for key in geo_dset.split(','):
                    if res is None:
                        res = np.squeeze(sgrp[key])
                    else:
                        res = np.append(res, np.squeeze(sgrp[key]))
        else:
            grp = self.fid[str(Path(msm_path, 'GEODATA'))]
            for key in geo_dset.split(','):
                if res is None:
                    res = np.squeeze(grp[key])
                else:
                    res = np.append(res, np.squeeze(grp[key]))

        return res

    def get_msm_data(self, msm_dset, band='78', columns=None, fill_as_nan=True):
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
        columns    : [i, j]
            Slice data on fastest axis (columns) as from index 'i' to 'j'
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Returns
        -------
        out  :  array
           Data of measurement dataset "msm_dset"
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__msm_path:
            return None

        assert band and len(band) <= 2
        if len(band) == 2:
            assert band in ('12', '34', '56', '78')
        assert self.bands.find(band) >= 0

        # skip row257 from the SWIR detector
        rows = None
        if int(band[0]) > 6:
            rows = [0, -1]

        # list potential names of the dataset dimensions
        time_list = ['time', 'scanline']
        row_list = ['width', 'pixel', 'pixel_window', 'ground_pixel']
        column_list = ['height', 'spectral_channel', 'spectral_channel_window']

        data = []
        column_dim = None   # column dimension is unknown
        for ii in band:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                msm_path = str(self.__msm_path).replace('%', ii)
                ds_path = str(Path(msm_path, dset_grp, msm_dset))
                if ds_path not in self.fid:
                    continue
                dset = self.fid[ds_path]

                skipped = 0
                data_sel = ()
                for xx in range(dset.ndim):
                    if len(dset.dims[xx][0][:]) == 1:
                        skipped += 1

                    if Path(dset.dims[xx][0].name).name in time_list:
                        data_sel += (np.s_[:],)
                    elif Path(dset.dims[xx][0].name).name in row_list:
                        if rows is None:
                            data_sel += (np.s_[:],)
                        else:
                            data_sel += (np.s_[rows[0]:rows[1]],)
                    elif Path(dset.dims[xx][0].name).name in column_list:
                        column_dim = xx - skipped
                        if columns is None:
                            data_sel += (np.s_[:],)
                        else:
                            data_sel += (np.s_[columns[0]:columns[1]],)
                    else:
                        raise ValueError

                if dset.dtype == np.float32:
                    res = np.squeeze(dset[data_sel]).astype(np.float64)
                else:
                    res = np.squeeze(dset[data_sel])
                if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                    res[(res == fillvalue)] = np.nan
                data.append(res)

        # Note the current implementation will not work for channels where
        # the output of its bands can have different spatial dimensions (rows)
        # or different integration times (frames/scanlines)
        #
        # no data found
        if not data:
            return None
        # return selected band
        if len(data) == 1:
            return data[0]
        # return bands stacked
        if column_dim is None:
            return data        # np.stack(data)
        # return band in detector layout
        return np.concatenate(data, axis=column_dim)

    # -------------------------
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

        if not self.__msm_path:
            return

        if band is None:
            band = self.bands[0]
        else:
            assert self.bands.find(band) >= 0

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            pass
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            pass
        else:
            ds_path = str(Path(msm_path, 'INSTRUMENT', 'housekeeping_data'))
            self.fid[ds_path][0, :] = data

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
        if not self.__msm_path:
            return

        assert band and len(band) <= 2
        if len(band) == 2:
            assert band in ('12', '34', '56', '78')
        assert self.bands.find(band) >= 0

        fillvalue = float.fromhex('0x1.ep+122')

        # skip row257 from the SWIR detector
        rows = None
        if int(band[0]) > 6:
            rows = [0, -1]

        # list potential names of the dataset dimensions
        time_list = ['time', 'scanline']
        row_list = ['width', 'pixel', 'ground_pixel']
        column_list = ['height', 'spectral_channel']

        indx = 0
        for ii in band:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                msm_path = str(self.__msm_path).replace('%', ii)
                ds_path = str(Path(msm_path, dset_grp, msm_dset))
                if ds_path not in self.fid:
                    continue
                dset = self.fid[ds_path]

                data_sel = ()
                for xx in range(dset.ndim):
                    if len(dset.dims[xx][0][:]) == 1:
                        data_sel += (np.s_[0],)
                    elif Path(dset.dims[xx][0].name).name in time_list:
                        data_sel += (np.s_[:],)
                    elif Path(dset.dims[xx][0].name).name in row_list:
                        if rows is None:
                            data_sel += (np.s_[:],)
                        else:
                            data_sel += (np.s_[rows[0]:rows[1]],)
                    elif Path(dset.dims[xx][0].name).name in column_list:
                        if len(band) == 2:
                            jj = data.ndim-1
                            data = np.stack(np.split(data, 2, axis=jj))
                        data_sel += (np.s_[:],)
                    else:
                        raise ValueError

                if len(band) == 2:
                    if dset.attrs['_FillValue'] == fillvalue:
                        data[indx, np.isnan(data[indx, ...])] = fillvalue
                    dset[data_sel] = data[indx, ...]
                    indx += 1
                else:
                    if dset.attrs['_FillValue'] == fillvalue:
                        data[np.isnan(data)] = fillvalue
                    dset[data_sel] = data

                self.__patched_msm.append(ds_path)
