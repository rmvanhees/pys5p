"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Provides access to the S5P Tropomi CKD (static and dynamic)

ToDo
----
 - access to UVN CKD, still incomplete
 - identify latest Static CKD product, e.g. using the validity period

Copyright (c) 2018-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path, PosixPath

import h5py
import numpy as np
import xarray as xr

from moniplot.image_to_xarray import h5_to_xr


# - local functions ------------------------------
def reject_row257(xarr):
    """
    Remove row 257 from DataArray or Dataset
    """
    return xarr.isel(row=np.s_[0:256])


# - class definition -------------------------------
class CKDio():
    """
    Read Tropomi CKD from the Static CKD product or from dynamic CKD products

    Attributes
    ----------
    ckd_dir : pathlib.Path
    ckd_version : int
    ckd_file : pathlib.Path
    ckd_dyn_file : pathlib.Path
    fid : h5py.File

    Methods
    -------
    close()
       Close resources.
    creation_time()
       Returns datetime when the L1b product was created.
    creator_version()
       Returns version of Tropomi L01B processor used to generate this procuct.
    validity_period()
       Return validity period of CKD product as a tuple of 2 datetime objects.
    get_param(ds_name, band='7')
       Returns value(s) of a CKD parameter from the Static CKD product.
    dn2v_factors()
       Returns digital number to Volt CKD, SWIR only.
    v2c_factors()
       Returns Voltage to Charge CKD, SWIR only.
    absirr(qvd=1, bands='78')
       Returns absolute irradiance responsivity.
    absrad(bands='78')
       Returns absolute radiance responsivity.
    memory()
       Returns memory CKD, SWIR only.
    noise()
       Returns pixel read-noise CKD, SWIR only
    prnu(bands='78')
       Returns Pixel Response Non-Uniformity (PRNU).
    relirr(qvd=1, bands='78')
       Returns relative irradiance correction.
    saa()
       Returns definition of the SAA region.
    wavelength(bands='78')
       Returns wavelength CKD.
    darkflux()
       Returns dark-flux CKD, SWIR only.
    offset()
       Returns offset CKD, SWIR only.
    pixel_quality()
       Returns Detector Pixel Quality Mask (float [0, 1]), SWIR only.
    dpqf(threshold=None)
       Returns Detector Pixel Quality Mask (boolean), SWIR only.
    saturation()
       Returns saturation values (pre-offset), SWIR only

    Notes
    -----
    Not all CKD are defined or derived for all bands.
    You can request a CKD for one band or for a channel (bands: '12', '34',
    '56', '78'). Do not mix bands from different channels

    The option to have dynamic CKD is not used for the Tropomi mission, only
    for S/W version 1 a dynamic CKD product is defined. This product contained
    the OCAL CKD and was not updated automatically. For version 2, all CKD are
    stored in one product, where some CKD have a time-axis to correct any
    in-flight degradation.

    Therefore, the logic to find a CKD is implemented as follows:
    1) ckd_dir, defines the base directory to search for the CKD products
    (see below)
    2) ckd_file, defines the full path to (static) CKD product;
    (version 1) any product with dynamic CKD has to be in the same directory

    Version 1:
    * Static CKD are stored in one file: glob('*_AUX_L1_CKD_*')
    * Dynamic CKD are stored in two files:
      - UVN, use glob('*_ICM_CKDUVN_*')
      - SWIR, use glob('*_ICM_CKDSIR_*')

    Version 2+:
    * All CKD in one file: glob('*_AUX_L1_CKD_*')
    * Dynamic CKD are empty

    Examples
    --------
    """
    def __init__(self, ckd_dir=None, ckd_version=1, ckd_file=None):
        """
        Initialize access to a Tropomi Static CKD product

        Parameters
        ----------
        ckd_dir :  str, optional
           Directory where the CKD files are stored,
           default='/nfs/Tropomi/share/ckd'
        ckd_version :  int, optional
           Version of the CKD, default=1
        ckd_file : str, optional
           Name of the CKD file, default=None then the CKD file is searched
           in the directory ckd_dir with ckd_version in the glob-string
        """
        if ckd_dir is None:
            ckd_dir = '/nfs/Tropomi/share/ckd'
        self.ckd_version = max(1, ckd_version)
        self.ckd_dyn_file = None

        # define path to CKD product
        if ckd_file is None:
            if not Path(ckd_dir).is_dir():
                raise FileNotFoundError(f'Not found CKD directory: {ckd_dir}')
            self.ckd_dir = Path(ckd_dir)
            glob_str = f'*_AUX_L1_CKD_*_*_00000_{self.ckd_version:02d}_*_*.h5'
            if (self.ckd_dir / 'static').is_dir():
                res = sorted((self.ckd_dir / 'static').glob(glob_str))
            else:
                res = sorted(self.ckd_dir.glob(glob_str))
            if not res:
                raise FileNotFoundError('Static CKD product not found')
            self.ckd_file = res[-1]
        else:
            if not Path(ckd_file).is_file():
                raise FileNotFoundError(f'Not found CKD file: {ckd_file}')
            self.ckd_dir = Path(ckd_file).parent
            self.ckd_file = Path(ckd_file)

        # obtain path to dynamic CKD product (version 1, only)
        if self.ckd_version == 1:
            if (self.ckd_dir / 'dynamic').is_dir():
                res = sorted((self.ckd_dir / 'dynamic').glob('*_ICM_CKDSIR_*'))
            else:
                res = sorted(self.ckd_dir.glob('*_ICM_CKDSIR_*'))
            if res:
                self.ckd_dyn_file = res[-1]

        # open access to CKD product
        self.fid = h5py.File(self.ckd_file, "r")

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

    def close(self) -> None:
        """
        Make sure that we close all resources
        """
        if self.fid is not None:
            self.fid.close()

    def creation_time(self) -> str:
        """
        Returns datetime when the L1b product was created
        """
        if self.ckd_version == 2:
            attr = self.fid['METADATA'].attrs['production_datetime']
        else:
            group = PosixPath('METADATA', 'earth_explorer_header',
                              'fixed_header', 'source')
            attr = self.fid[str(group)].attrs['Creator_Date'][0]

        if isinstance(attr, bytes):
            attr = attr.decode('ascii')
        return attr

    def creator_version(self) -> str:
        """
        Returns version of Tropomi L01B processor used to generate this procuct
        """
        group = PosixPath('METADATA', 'earth_explorer_header', 'fixed_header')
        attr = self.fid[str(group)].attrs['File_Version']
        if self.ckd_version == 1:
            attr = attr[0]
        if isinstance(attr, bytes):
            attr = attr.decode('ascii')
        return attr

    @staticmethod
    def __get_spectral_channel(bands: str):
        """
        Check bands is valid: single band or belong to one channel

        Parameters
        ----------
        bands : str
           Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
           default: '78'
        """
        band2channel = ['UNKNOWN', 'UV', 'UV', 'VIS', 'VIS',
                        'NIR', 'NIR', 'SWIR', 'SWIR']

        if 0 < len(bands) > 2:
            raise ValueError('read per band or channel, only')

        if len(bands) == 2:
            if band2channel[int(bands[0])] != band2channel[int(bands[1])]:
                raise ValueError('bands should be of the same channel')

        return band2channel[int(bands[0])]

    def get_param(self, ds_name: str, band='7'):
        """
        Returns value(s) of a CKD parameter from the Static CKD product.

        Parameters
        ----------
        ds_name  :  string
           Name of the HDF5 dataset, default='pixel_full_well'
        band     : string
           Band identifier '1', '2', ..., '8', default='7'

        Returns
        -------
        numpy.ndarray or scalar

        Notes
        -----
        Datasets of size=1 are return as scalar

        Handy function for scalar HDF5 datasets, such as:
         - dc_reference_temp
         - dpqf_threshold
         - pixel_full_well
         - pixel_fw_flag_thresh
        """
        if not 1 <= int(band) <= 8:
            raise ValueError('band must be between and 1 and 8')

        if ds_name not in self.fid[f'/BAND{band}']:
            raise ValueError('dataset not available')

        return self.fid[f'/BAND{band}/{ds_name}'][()]

    # ---------- band or channel CKD's ----------
    def dn2v_factors(self):
        """
        Returns digital number to Volt CKD, SWIR only

        Notes
        -----
        The DN2V factor has no error attached to it.
        """
        return np.concatenate(
            (self.fid['/BAND7/dn2v_factor_swir'][2:],
             self.fid['/BAND8/dn2v_factor_swir'][2:]))

    def v2c_factors(self):
        """
        Returns Voltage to Charge CKD, SWIR only

        Notes
        -----
        The V2C factor has no error attached to it.
        """
        # pylint: disable=no-member
        return np.concatenate(
            (self.fid['/BAND7/v2c_factor_swir'].fields('value')[2:],
             self.fid['/BAND8/v2c_factor_swir'].fields('value')[2:]))

    # ---------- spectral-channel CKD's ----------
    def __rd_dataset(self, dset_name: str, bands: str):
        """
        General function to read non-compound dataset into xarray::Dataset

        Parameters
        ----------
        dset_name: str
           name (including path) of the dataset as '/BAND{}/<name>'
        bands : str
           Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
           default: '78'
        """
        ckd_val = None
        for band in bands:
            # try Static-CKD product
            if dset_name.format(band) in self.fid:
                if ckd_val is None:
                    ckd_val = h5_to_xr(self.fid[dset_name.format(band)])
                else:
                    ckd_val = xr.concat(
                        (ckd_val,
                         h5_to_xr(self.fid[dset_name.format(band)])),
                        dim='column')
            # try Dynamic-CKD product
            else:
                dyn_fid = h5py.File(self.ckd_dyn_file, 'r')
                if dset_name.format(band) in dyn_fid:
                    if ckd_val is None:
                        ckd_val = h5_to_xr(dyn_fid[dset_name.format(band)])
                    else:
                        ckd_val = xr.concat(
                            (ckd_val,
                             h5_to_xr(dyn_fid[dset_name.format(band)])),
                            dim='column')
                dyn_fid.close()

        if ckd_val is None:
            return None

        # Use NaN as FillValue
        ckd_val = ckd_val.where(ckd_val != float.fromhex('0x1.ep+122'),
                                other=np.nan)

        # combine DataArrays to Dataset
        return xr.Dataset({'value': ckd_val}, attrs=ckd_val.attrs)

    def __rd_datapoints(self, dset_name: str, bands: str):
        """
        General function to read datapoint dataset into xarray::Dataset

        Parameters
        ----------
        dset_name: str
           name (including path) of the dataset as '/BAND{}/<name>'
        bands : str
           Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
           default: '78'
        """
        ckd_val = None
        ckd_err = None
        for band in bands:
            # try Static-CKD product
            if dset_name.format(band) in self.fid:
                if ckd_val is None:
                    ckd_val = h5_to_xr(self.fid[dset_name.format(band)],
                                       field='value')
                    ckd_err = h5_to_xr(self.fid[dset_name.format(band)],
                                       field='error')
                else:
                    ckd_val = xr.concat(
                        (ckd_val, h5_to_xr(self.fid[dset_name.format(band)],
                                           field='value')), dim='column')
                    ckd_err = xr.concat(
                        (ckd_err, h5_to_xr(self.fid[dset_name.format(band)],
                                           field='error')), dim='column')
            # try Dynamic-CKD product
            else:
                dyn_fid = h5py.File(self.ckd_dyn_file, 'r')
                if dset_name.format(band) in dyn_fid:
                    if ckd_val is None:
                        ckd_val = h5_to_xr(dyn_fid[dset_name.format(band)],
                                           field='value')
                        ckd_err = h5_to_xr(dyn_fid[dset_name.format(band)],
                                           field='error')
                    else:
                        ckd_val = xr.concat(
                            (ckd_val, h5_to_xr(dyn_fid[dset_name.format(band)],
                                               field='value')), dim='column')
                        ckd_err = xr.concat(
                            (ckd_err, h5_to_xr(dyn_fid[dset_name.format(band)],
                                               field='error')), dim='column')
                dyn_fid.close()

        if ckd_val is None:
            return None

        # Use NaN as FillValue
        ckd_val = ckd_val.where(ckd_val != float.fromhex('0x1.ep+122'),
                                other=np.nan)
        ckd_err = ckd_err.where(ckd_err != float.fromhex('0x1.ep+122'),
                                other=np.nan)

        # combine DataArrays to Dataset
        return xr.Dataset({'value': ckd_val, 'error': ckd_err},
                          attrs=ckd_val.attrs)

    # ---------- static CKD's ----------
    def absirr(self, qvd=1, bands='78'):
        """
        Returns absolute irradiance responsivity

        Parameters
        ----------
        bands : str
          Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
          default: '78'
        qvd : int
          Tropomi QVD identifier. Valid values are 1 or 2, default: 1
        """
        try:
            channel = self.__get_spectral_channel(bands)
        except Exception as exc:
            raise RuntimeError(exc) from exc

        dset_name = '/BAND{}' + f'/abs_irr_conv_factor_qvd{qvd}'
        ckd = self.__rd_datapoints(dset_name, bands)
        if '7' in bands or '8' in bands:
            ckd = reject_row257(ckd)
        ckd.attrs["long_name"] = \
                            f'{channel} absolute irradiance CKD (QVD={qvd})'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def absrad(self, bands='78'):
        """
        Returns absolute radiance responsivity

        Parameters
        ----------
        bands : str
          Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
          default: '78'
        """
        try:
            channel = self.__get_spectral_channel(bands)
        except Exception as exc:
            raise RuntimeError(exc) from exc

        dset_name = '/BAND{}/abs_rad_conv_factor'
        ckd = self.__rd_datapoints(dset_name, bands)
        if '7' in bands or '8' in bands:
            ckd = reject_row257(ckd)
        ckd.attrs["long_name"] = f'{channel} absolute radiance CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def memory(self):
        """
        Returns memory CKD, SWIR only

        Parameters
        ----------
        bands : str
          Tropomi bands [7,8] or channels ['78'], default: '78'
        """
        column = None
        ckd_parms = ['mem_lin_neg_swir', 'mem_lin_pos_swir',
                     'mem_qua_neg_swir', 'mem_qua_pos_swir']

        ckd = xr.Dataset()
        ckd.attrs["long_name"] = 'SWIR memory CKD'
        for key in ckd_parms:
            dset_name = f'/BAND7/{key}'
            ckd_val = h5_to_xr(self.fid[dset_name], field='value')
            ckd_err = h5_to_xr(self.fid[dset_name], field='error')
            dset_name = f'/BAND8/{key}'
            ckd_val = xr.concat(
                (ckd_val, h5_to_xr(self.fid[dset_name], field='value')),
                dim='column')
            if column is None:
                column = np.arange(ckd_val.column.size, dtype='u4')
            ckd_val = ckd_val.assign_coords(column=column)
            ckd_err = xr.concat(
                (ckd_err, h5_to_xr(self.fid[dset_name], field='error')),
                dim='column')
            ckd_err = ckd_err.assign_coords(column=column)
            ckd[key.replace('swir', 'value')] = reject_row257(ckd_val)
            ckd[key.replace('swir', 'error')] = reject_row257(ckd_err)

        return ckd

    def noise(self):
        """
        Returns readout-noise CKD, SWIR only
        """
        dset_name = '/BAND{}/readout_noise_swir'
        ckd = reject_row257(self.__rd_dataset(dset_name, '78'))
        ckd.attrs["long_name"] = 'SWIR readout-noise CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def prnu(self, bands='78'):
        """
        Returns Pixel Response Non-Uniformity (PRNU)

        Parameters
        ----------
        bands : str
          Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
          default: '78'
        """
        try:
            channel = self.__get_spectral_channel(bands)
        except Exception as exc:
            raise RuntimeError(exc) from exc

        ckd = self.__rd_datapoints('/BAND{}/PRNU', bands)
        if '7' in bands or '8' in bands:
            ckd = reject_row257(ckd)
        ckd.attrs["long_name"] = f'{channel} PRNU CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def relirr(self, qvd=1, bands='78'):
        """
        Returns relative irradiance correction

        Parameters
        ----------
        bands : str
          Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
          default: '78'
        qvd : int
          Tropomi QVD identifier. Valid values are 1 or 2, default: 1

        Returns
        -------
        dictionary with keys:
         - band           :    Tropomi spectral band ID
         - mapping_cols   :    coarse irregular mapping of the columns
         - mapping_rows   :    coarse irregular mapping of the rows
         - cheb_coefs     :    chebyshev parameters for elevation and azimuth
                               for pixels on a coarse irregular grid
        """
        try:
            _ = self.__get_spectral_channel(bands)
        except Exception as exc:
            raise RuntimeError(exc) from exc

        res = ()
        for band in bands:
            ckd = {}
            ckd['band'] = int(band)

            dsname = f'/BAND{band}/rel_irr_coarse_mapping_vert'
            ckd['mapping_rows'] = self.fid[dsname][:].astype(int)

            dsname = f'/BAND{band}/rel_irr_coarse_mapping_hor'
            # pylint: disable=no-member
            mapping_hor = self.fid[dsname][:].astype(int)
            mapping_hor[mapping_hor > 1000] -= 2**16
            ckd['mapping_cols'] = mapping_hor

            dsname = f'/BAND{band}/rel_irr_coarse_func_cheb_qvd{qvd}'
            ckd['cheb_coefs'] = self.fid[dsname]['coefs'][:]
            res += (ckd,)

        return res

    def saa(self) -> dict:
        """
        Returns definition of the SAA region
        """
        saa_region = {}
        saa_region['lat'] = self.fid['saa_latitude'][:]
        saa_region['lon'] = self.fid['saa_longitude'][:]

        return saa_region

    def wavelength(self, bands='78'):
        """
        Returns wavelength CKD

        Parameters
        ----------
        bands : str
          Tropomi bands [1..8] or channels ['12', '34', '56', '78'],
          default: '78'

        Notes
        -----
        * The wavelength CKD has no error attached to it.
        """
        try:
            channel = self.__get_spectral_channel(bands)
        except Exception as exc:
            raise RuntimeError(exc) from exc

        dset_name = '/BAND{}/wavelength_map'
        ckd = self.__rd_datapoints(dset_name, bands)
        if '7' in bands or '8' in bands:
            ckd = reject_row257(ckd)
        ckd.attrs["long_name"] = f'{channel} wavelength CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    # ---------- static or dynamic CKD's ----------
    def darkflux(self):
        """
        Returns dark-flux CKD, SWIR only
        """
        dset_name = '/BAND{}/long_term_swir'
        ckd = reject_row257(self.__rd_datapoints(dset_name, '78'))
        ckd.attrs["long_name"] = 'SWIR dark-flux CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def offset(self):
        """
        Returns offset CKD, SWIR only
        """
        dset_name = '/BAND{}/analog_offset_swir'
        ckd = reject_row257(self.__rd_datapoints(dset_name, '78'))
        ckd.attrs["long_name"] = 'SWIR offset CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def pixel_quality(self):
        """
        Returns detector pixel-quality mask (float [0, 1]), SWIR only
        """
        dset_name = '/BAND{}/dpqf_map'
        ckd = reject_row257(self.__rd_dataset(dset_name, '78'))
        ckd.attrs["long_name"] = 'SWIR pixel-quality CKD'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))

    def dpqf(self, threshold=None):
        """
        Returns detector pixel-quality flags (boolean), SWIR only

        Parameters
        ----------
        threshold: float
          Value between [0..1]

        Returns
        -------
        numpy ndarray
        """
        if threshold is None:
            threshold = self.fid['/BAND7/dpqf_threshold'][:]

        # try Static-CKD product
        if '/BAND7/dpqf_map' in self.fid:
            dset = self.fid['/BAND7/dpqf_map']
            dpqf_b7 = dset[:-1, :]
            dset = self.fid['/BAND8/dpqf_map']
            dpqf_b8 = dset[:-1, :]
        else:
            # try Dynamic-CKD product
            with h5py.File(self.ckd_dyn_file, 'r') as fid:
                dset = fid['/BAND7/dpqf_map']
                dpqf_b7 = dset[:-1, :]
                dset = fid['/BAND8/dpqf_map']
                dpqf_b8 = dset[:-1, :]

        return np.hstack((dpqf_b7, dpqf_b8)) < threshold

    def saturation(self):
        """
        Returns pixel-saturation values (pre-offset), SWIR only
        """
        ckd_val = None
        dset_name = '/BAND{}/saturation_preoffset'
        ckd_file = (self.ckd_dir / 'OCAL'
                    / 'ckd.saturation_preoffset.detector4.nc')
        with h5py.File(ckd_file, 'r') as fid:
            ckd_val = xr.concat((h5_to_xr(fid[dset_name.format(7)]),
                                 h5_to_xr(fid[dset_name.format(8)])),
                                dim='column')

        ckd = xr.Dataset({'value': ckd_val}, attrs=ckd_val.attrs)
        ckd = reject_row257(ckd)
        ckd.attrs["long_name"] = 'SWIR pixel-saturation CKD (pre-offset)'

        return ckd.assign_coords(column=np.arange(ckd.column.size, dtype='u4'))
