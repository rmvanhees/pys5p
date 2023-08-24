#
# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""This module contain class `ICMio` to access in-flight calibration data."""

from __future__ import annotations

__all__ = ['ICMio']

from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from setuptools_scm import get_version

if TYPE_CHECKING:
    from collections.abc import Iterable

# - global parameters ------------------------------

# - local functions --------------------------------


# - class definition -------------------------------
class ICMio:
    """This class should offer all the necessary functionality to read Tropomi
    ICM_CA_SIR products.

    Parameters
    ----------
    icm_product :  Path or str
        full path to in-flight calibration measurement product
    readwrite   :  bool
        open product in read-write mode (default is False)
    """

    def __init__(self, icm_product: Path | str, readwrite: bool = False):
        """Initialize access to an ICM product."""
        icm_product = Path(icm_product)
        if not icm_product.is_file():
            raise FileNotFoundError(f'{icm_product.name} does not exist')

        # initialize class-attributes
        self.__rw = readwrite
        self.__msm_path = None
        self.__patched_msm = []
        self.filename = icm_product
        self.bands = None

        # open ICM product as HDF5 file
        if readwrite:
            self.fid = h5py.File(icm_product, 'r+')
        else:
            self.fid = h5py.File(icm_product, 'r')

    def __iter__(self):
        """Allow iteration."""
        for attr in sorted(self.__dict__):
            if not attr.startswith('__'):
                yield attr

    # def __del__(self):
    #    """
    #    called when the object is destroyed
    #    """
    #    self.close()

    def __enter__(self):
        """Initiate the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self):
        """
        Before closing the product, we make sure that the output product
        describes what has been altered by the S/W. To keep any change
        traceable.

        As attributes of this group, we write:

          - dateStamp ('now')
          - Git-version of S/W
          - list of patched datasets
          - auxiliary datasets used by patch-routines

        """
        if self.fid is None:
            return

        self.bands = None
        if self.__patched_msm:
            # pylint: disable=no-member
            sgrp = self.fid.require_group('METADATA/SRON_METADATA')
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

    # ---------- RETURN VERSION of the S/W ----------
    def find(self, msm_class: str) -> list[str]:
        """Find a measurement as <processing-class name>.

        Parameters
        ----------
        msm_class :  str
          processing-class name without ICID

        Returns
        -------
        list of str
           String with msm_type as used by ICMio.select()
        """
        res = []

        grp_list = ['ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE']
        for ii in '12345678':
            for name in grp_list:
                grp_name = f'BAND{ii}_{name}'
                if grp_name in self.fid:
                    gid = self.fid[grp_name]
                    res += [s for s in gid if s.startswith(msm_class)]

        return list(set(res))

    # -------------------------
    def select(self, msm_type: str, msm_path: str = None) -> str:
        """Select a measurement as <processing class>_<ic_id>.

        Parameters
        ----------
        msm_type :  string
           Name of measurement group
        msm_path : {'BAND%_ANALYSIS', 'BAND%_CALIBRATION',
                    'BAND%_IRRADIANCE', 'BAND%_RADIANCE'}
           Name of path in HDF5 file to measurement group

        Returns
        -------
        string
           String with spectral bands found in product or empty
        """
        self.bands = ''
        self.__msm_path = None

        # if path is given, then only determine available spectral bands
        # else determine path and available spectral bands
        if msm_path is None:
            grp_list = ['ANALYSIS', 'CALIBRATION', 'IRRADIANCE', 'RADIANCE']
            for ii in '12345678':
                for name in grp_list:
                    grp_path = PurePosixPath(f'BAND{ii}_{name}', msm_type)
                    if str(grp_path) in self.fid:
                        msm_path = f'BAND%_{name}'
                        self.bands += ii
        else:
            if not msm_path.startswith('BAND%'):
                raise ValueError('msm_path should start with BAND%')

            for ii in '12345678':
                grp_path = PurePosixPath(msm_path.replace('%', ii), msm_type)
                if str(grp_path) in self.fid:
                    self.bands += ii

        # return in case no data was found
        if self.bands:
            self.__msm_path = PurePosixPath(msm_path, msm_type)

        return self.bands

    # ---------- Functions that work before MSM selection ----------
    @property
    def orbit(self) -> int | None:
        """Return reference orbit number."""
        if 'reference_orbit' in self.fid.attrs:
            return int(self.fid.attrs['reference_orbit'])

        return None

    @property
    def processor_version(self) -> str | None:
        """Return version of the L01b processor."""
        if 'processor_version' not in self.fid.attrs:
            return None

        res = self.fid.attrs['processor_version']
        if isinstance(res, bytes):
            # pylint: disable=no-member
            return res.decode('ascii')

        return res

    @property
    def coverage_time(self) -> tuple[str, str] | None:
        """Return start and end of the measurement coverage time."""
        if 'time_coverage_start' not in self.fid.attrs \
           or 'time_coverage_end' not in self.fid.attrs:
            return None

        res1 = self.fid.attrs['time_coverage_start']
        if isinstance(res1, bytes):
            # pylint: disable=no-member
            res1 = res1.decode('ascii')

        res2 = self.fid.attrs['time_coverage_end']
        if isinstance(res2, bytes):
            # pylint: disable=no-member
            res2 = res2.decode('ascii')

        return res1, res2

    @property
    def creation_time(self) -> str:
        """Return version of the L01b processor."""
        grp = self.fid['/METADATA/ESA_METADATA/earth_explorer_header']
        dset = grp['fixed_header/source']
        return dset.attrs['Creation_Date'].split(b'=')[1].decode('ascii')

    def get_attr(self, attr_name: str) -> str | None:
        """Obtain value of an HDF5 file attribute.

        Parameters
        ----------
        attr_name : str
           name of the attribute
        """
        if attr_name not in self.fid.attrs:
            return None

        res = self.fid.attrs[attr_name]
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time(self, band: str | None = None) -> datetime:
        """Return reference start time of measurements.

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        ref_time = datetime(2010, 1, 1, 0, 0, 0)
        if not self.__msm_path:
            return ref_time

        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'OBSERVATIONS')
                grp = self.fid[str(grp_path)]
                ref_time += timedelta(seconds=int(grp['time'][0]))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = PurePosixPath(msm_path).parent / 'ANALOG_OFFSET_SWIR'
            grp = self.fid[str(grp_path)]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'OBSERVATIONS')
                grp = self.fid[str(grp_path)]
                ref_time += timedelta(seconds=int(grp['time'][0]))
        else:
            grp_path = PurePosixPath(msm_path, 'OBSERVATIONS')
            grp = self.fid[str(grp_path)]
            ref_time += timedelta(seconds=int(grp['time'][0]))

        return ref_time

    def get_delta_time(self, band: str | None = None) -> np.ndarray | None:
        """Return offset from the reference start time of measurement.

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
        elif band not in self.bands:
            raise ValueError('band not found in product')

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'OBSERVATIONS')
                grp = self.fid[str(grp_path)]
                if res is None:
                    res = grp['delta_time'][0, :].astype(int)
                else:
                    res = np.append(res, grp['delta_time'][0, :].astype(int))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = PurePosixPath(msm_path).parent / 'ANALOG_OFFSET_SWIR'
            grp = self.fid[str(grp_path)]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'OBSERVATIONS')
                grp = self.fid[grp_path]
                if res is None:
                    res = grp['delta_time'][0, :].astype(int)
                else:
                    res = np.append(res, grp['delta_time'][0, :].astype(int))
        else:
            grp_path = PurePosixPath(msm_path, 'OBSERVATIONS')
            grp = self.fid[str(grp_path)]
            res = grp['delta_time'][0, :].astype(int)

        return res

    def get_instrument_settings(self,
                                band: str | None = None) -> np.ndarray | None:
        """Return instrument settings of measurement.

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
        elif band not in self.bands:
            raise ValueError('band not found in product')

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'INSTRUMENT')
                grp = self.fid[str(grp_path)]
                if res is None:
                    res = grp['instrument_settings'][:]
                else:
                    res = np.append(res, grp['instrument_settings'][:])
        elif msm_type == 'DPQF_MAP':
            grp_path = PurePosixPath(msm_path).parent / 'ANALOG_OFFSET_SWIR'
            grp = self.fid[str(grp_path)]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'INSTRUMENT')
                grp = self.fid[grp_path]
                if res is None:
                    res = grp['instrument_settings'][:]
                else:
                    res = np.append(res, grp['instrument_settings'][:])
        elif msm_type == 'NOISE':
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_msmt_keys']
            icid = dset['icid'][dset.size // 2]
            grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                     f'BACKGROUND_RADIANCE_MODE_{icid:04d}',
                                     'INSTRUMENT')
            grp = self.fid[str(grp_path)]
            res = grp['instrument_settings'][:]
        else:
            grp_path = PurePosixPath(msm_path, 'INSTRUMENT')
            grp = self.fid[str(grp_path)]
            res = grp['instrument_settings'][:]

        return res

    def get_exposure_time(self, band: str | None = None) -> list | None:
        """Return pixel exposure time of the measurements.

        The exposure time is calculated from the parameters 'int_delay' and
        'int_hold' for SWIR.

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        """
        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

        # obtain instrument settings
        instr_arr = self.get_instrument_settings(band)
        if instr_arr is None:
            return None

        # calculate exact exposure time
        res = []
        for instr in instr_arr:
            if int(band) > 6:
                res.append(1.25e-6 *
                           (65540 - instr['int_delay'] + instr['int_hold']))
            else:
                res.append(instr['exposure_time'])

        return res

    def get_housekeeping_data(self,
                              band: str | None = None) -> np.ndarray | None:
        """Return housekeeping data of measurements.

        Parameters
        ----------
        band :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = None
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'INSTRUMENT')
                grp = self.fid[str(grp_path)]
                if res is None:
                    res = np.squeeze(grp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(grp['housekeeping_data']))
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = PurePosixPath(msm_path).parent / 'ANALOG_OFFSET_SWIR'
            grp = self.fid[str(grp_path)]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath('BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'INSTRUMENT')
                grp = self.fid[str(grp_path)]
                if res is None:
                    res = np.squeeze(grp['housekeeping_data'])
                else:
                    res = np.append(res, np.squeeze(grp['housekeeping_data']))
        else:
            grp_path = PurePosixPath(msm_path, 'INSTRUMENT')
            grp = self.fid[str(grp_path)]
            res = np.squeeze(grp['housekeeping_data'])

        return res

    # -------------------------
    def get_msmt_keys(self, band: str | None = None) -> np.ndarray | None:
        """Read msmt_keys from an analysis groups.

        The msmt_keys are available for the following `analysis`:

           * [ANALOG_OFFSET_SWIR]: analog_offset_swir_group_keys
           * [LONG_TERM_SWIR]: long_term_swir_group_keys
           * [NOISE]: noise_msmt_keys

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band

        Returns
        -------
        numpy.ndarray
            msmt_keys from an analysis groups
        """
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

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
    def get_msm_attr(self, msm_dset: str, attr_name: str,
                     band: str | None = None) -> np.ndarray | None:
        """Return attribute of measurement dataset "msm_dset".

        Parameters
        ----------
        msm_dset  :  str
            Name of measurement dataset
        attr_name :  str
            Name of the attribute
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band

        Returns
        -------
        numpy.ndarray
           value of attribute "attr_name"
        """
        if not self.__msm_path:
            return None

        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

        for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
            ds_path = PurePosixPath(str(self.__msm_path).replace('%', band),
                                    dset_grp, msm_dset)
            if str(ds_path) not in self.fid:
                continue

            if attr_name in self.fid[str(ds_path)].attrs:
                attr = self.fid[str(ds_path)].attrs[attr_name]
                if isinstance(attr, bytes):
                    return attr.decode('ascii')

                return attr

        return None

    def get_geo_data(self, band: str | None = None,
                     geo_dset: str | None = None) -> np.ndarray | None:
        """Return data of selected datasets from the GEODATA group.

        Parameters
        ----------
        band      :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product
            Default is 'None' which returns the first available band
        geo_dset  :  str, optional
            Name(s) of datasets in the GEODATA group, comma separated
            Default is 'satellite_latitude,satellite_longitude'

        Returns
        -------
        dict
           dictionary with data of a selected datasets from the GEODATA group;
           the keys of dictionary have the same names as the datasets.
        """
        if not self.__msm_path:
            return None

        if band is None:
            band = str(self.bands[0])
        elif band not in self.bands:
            raise ValueError('band not found in product')

        if geo_dset is None:
            geo_dset = 'satellite_latitude,satellite_longitude'

        msm_path = str(self.__msm_path).replace('%', band)
        msm_type = self.__msm_path.name

        res = {}
        if msm_type in ['ANALOG_OFFSET_SWIR', 'LONG_TERM_SWIR']:
            grp = self.fid[msm_path]
            dset = grp[msm_type.lower() + '_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'GEODATA')
                grp = self.fid[str(grp_path)]
                for key in geo_dset.split(','):
                    res[key] = np.squeeze(grp[key])
        elif msm_type in ['DPQF_MAP', 'NOISE']:
            grp_path = PurePosixPath(msm_path).parent / 'ANALOG_OFFSET_SWIR'
            grp = self.fid[str(grp_path)]
            dset = grp['analog_offset_swir_group_keys']
            group_keys = dset['group'][:]
            for name in group_keys:
                grp_path = PurePosixPath(f'BAND{band}_CALIBRATION',
                                         name.decode('ascii'),
                                         'GEODATA')
                grp = self.fid[str(grp_path)]
                for key in geo_dset.split(','):
                    res[key] = np.squeeze(grp[key])
        else:
            grp_path = PurePosixPath(msm_path, 'GEODATA')
            grp = self.fid[str(grp_path)]
            for key in geo_dset.split(','):
                res[key] = np.squeeze(grp[key])

        return res

    def get_msm_data(self, msm_dset: str, band: str = '78', *,
                     read_raw: bool = False,
                     columns: list[int, int] | None = None,
                     fill_as_nan: bool = True) -> np.ndarray | None:
        """Read datasets from a measurement selected by class-method "select".

        Parameters
        ----------
        msm_dset :  str
            name of measurement dataset
            if msm_dset is None then show names of available datasets
        band :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        read_raw : bool, default=False
            Perform raw read: without slicing or data conversion,
            and ignore keywords: columns, fill_as_nan.
            Default: False
        columns : [i, j], optional
            Slice data on fastest axis (columns) as from index 'i' to 'j'
        fill_as_nan :  bool, default=True
            Replace (float) FillValues with Nan's, when True

        Examples
        --------
        Access an ICM product and obtain it's orbit-counter
        and read from the ANALYSIS group the dark-offset (band 7/8)::

        >>> flname = Path('/full/name/to/icm_product.nc')
        >>> with ICMio(flname) as icm:
        >>>    orbit = icm.orbit
        >>>    icm.select('LONG_TERM_SWIR')
        >>>    offs = icm.get_msm_data('analog_offset_swir_value',
        >>>                            columns=[1, -1], fill_as_nan=True)

        Returns
        -------
        numpy.ndarray
           data of measurement dataset "msm_dset"
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__msm_path:
            return None

        if not isinstance(band, str):
            raise TypeError('band must be a string')

        if band not in self.bands:
            raise ValueError('band not found in product')

        data = []
        if read_raw:
            for ii in band:
                for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                    msm_path = str(self.__msm_path).replace('%', ii)
                    ds_path = PurePosixPath(msm_path, dset_grp, msm_dset)
                    if str(ds_path) not in self.fid:
                        continue

                    data.append(np.squeeze(self.fid[str(ds_path)]))

            return data

        # skip row257 from the SWIR detector
        rows = None
        if int(band[0]) > 6:
            rows = [0, -1]

        # list potential names of the dataset dimensions
        time_list = ['time', 'scanline']
        row_list = ['width', 'pixel', 'pixel_window', 'ground_pixel']
        column_list = ['height', 'spectral_channel', 'spectral_channel_window']

        column_dim = None   # column dimension is unknown
        for ii in band:
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                msm_path = str(self.__msm_path).replace('%', ii)
                ds_path = PurePosixPath(msm_path, dset_grp, msm_dset)
                if str(ds_path) not in self.fid:
                    continue
                dset = self.fid[str(ds_path)]

                skipped = 0
                data_sel = ()
                for ix in range(dset.ndim):
                    if len(dset.dims[ix][0][:]) == 1:
                        skipped += 1

                    dim_name = PurePosixPath(dset.dims[ix][0].name).name
                    if dim_name in time_list:
                        data_sel += (slice(None),)
                    elif dim_name in row_list:
                        if rows is None:
                            data_sel += (slice(None),)
                        else:
                            data_sel += (slice(*rows),)
                    elif dim_name in column_list:
                        column_dim = ix - skipped
                        if columns is None:
                            data_sel += (slice(None),)
                        else:
                            data_sel += (slice(*columns),)
                    else:
                        raise ValueError

                if dset.dtype == np.float32:
                    res = np.squeeze(dset.astype(float)[data_sel])
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

    def read_direct_msm(self, msm_dset: str,
                        dest_sel: tuple[slice | int] | None = None,
                        dest_dtype: type[Any] | None = None,
                        fill_as_nan: bool = False) -> list[np.ndarray] | None:
        """Read datasets from a measurement selected by class-method `select`
        (fast implementation).

        Parameters
        ----------
        msm_dset    :  str
            Name of measurement dataset
        dest_sel    :  numpy slice
            Selection must be the output of numpy.s_[<args>].
        dest_dtype  :  numpy dtype
            Perform type conversion
        fill_as_nan :  bool
            Replace (float) FillValues with Nan's, when True

        Examples
        --------
        Access an ICM product and obtain it's orbit-counter
        and read from the ANALYSIS group the dark-offset (band 7/8)::

        >>> flname = Path('/full/name/to/icm_product.nc')
        >>> with ICMio(flname) as icm:
        >>>    orbit = icm.orbit
        >>>    icm.select('LONG_TERM_SWIR')
        >>>    offs = np.hstack(
        >>>        icm.read_direct_msm('analog_offset_swir_value',
        >>>                            dest_sel=np.s_[:-1, :],
        >>>                            dest_dtype=float, fill_as_nan=True))

        Returns
        -------
        list
           data of all available bands
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__msm_path:
            return None

        if dest_sel is None:
            dest_sel = np.s_[...]

        data = []
        for ii in self.bands:
            msm_path = str(self.__msm_path).replace('%', ii)
            for dset_grp in ['OBSERVATIONS', 'ANALYSIS', '']:
                ds_path = str(PurePosixPath(msm_path, dset_grp, msm_dset))
                if ds_path not in self.fid:
                    continue

                dset = self.fid[ds_path]
                if dest_dtype is None:
                    buff = dset[dest_sel]
                    if (fill_as_nan
                        and '_FillValue' in dset.attrs
                        and np.issubdtype(buff.dtype, np.floating)):
                        fillvalue = dset.attrs['_FillValue'][0]
                        buff[(buff == fillvalue)] = np.nan
                else:
                    buff = dset.astype(dest_dtype)[dest_sel]
                    if (fill_as_nan
                        and '_FillValue' in dset.attrs
                        and np.issubdtype(buff.dtype, np.floating)):
                        buff[(buff == fillvalue)] = np.nan

                data.append(buff)

        return data

    # -------------------------
    def set_housekeeping_data(self, data: np.ndarray,
                              band: str | None = None) -> None:
        """Return housekeeping data of measurements.

        Parameters
        ----------
        data : np.ndarray
        band :  None or {'1', '2', '3', ..., '8'}
            Select one of the band present in the product.
            Default is 'None' which returns the first available band
        """
        if not self.__rw:
            raise PermissionError('read/write access required')

        if (not self.__msm_path
            or self.__msm_path.name in ['ANALOG_OFFSET_SWIR',
                                        'LONG_TERM_SWIR',
                                        'DPQF_MAP',
                                        'NOISE']):
            return

        if band is None:
            band = self.bands[0]
        elif band not in self.bands:
            raise ValueError('band not found in product')

        msm_path = str(self.__msm_path).replace('%', band)
        ds_path = PurePosixPath(msm_path, 'INSTRUMENT', 'housekeeping_data')
        self.fid[str(ds_path)][0, :] = data
        self.__patched_msm.append(str(ds_path))

    def set_msm_data(self, msm_dset: str, data: np.ndarray | Iterable,
                     band: str = '78') -> None:
        """Alter dataset from a measurement selected using function "select".

        Parameters
        ----------
        msm_dset   :  string
            name of measurement dataset
        band       :  {'1', '2', '3', ..., '8', '12', '34', '56', '78'}
            Select data from one spectral band or channel
            Default is '78' which combines band 7/8 to SWIR detector layout
        data       :  array-like
            data to be written with same dimensions as dataset 'msm_dset'
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__rw:
            raise PermissionError('read/write access required')

        if not self.__msm_path:
            return

        if not isinstance(band, str):
            raise TypeError('band must be a string')

        if band not in self.bands:
            raise ValueError('band not found in product')

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
                ds_path = PurePosixPath(msm_path, dset_grp, msm_dset)
                if str(ds_path) not in self.fid:
                    continue
                dset = self.fid[str(ds_path)]

                data_sel = ()
                for ix in range(dset.ndim):
                    dim_name = PurePosixPath(dset.dims[ix][0].name).name

                    if len(dset.dims[ix][0][:]) == 1:
                        data_sel += (0,)
                    elif dim_name in time_list:
                        data_sel += (slice(None),)
                    elif dim_name in row_list:
                        if rows is None:
                            data_sel += (slice(None),)
                        else:
                            data_sel += (slice(*rows),)
                    elif dim_name in column_list:
                        if len(band) == 2:
                            jj = data.ndim-1
                            data = np.stack(np.split(data, 2, axis=jj))
                        data_sel += (slice(None),)
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
