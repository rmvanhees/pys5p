"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Provides access to the S5P Tropomi CKD (static and dynamic)

ToDo
----
 - access to UVN CKD, still incomplete
 - identify latest Static CKD product, e.g. using the validity period

Copyright (c) 2018-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path, PosixPath

import h5py
import numpy as np

from .s5p_msm import S5Pmsm

# - global parameters ------------------------------


# - class definition -------------------------------
class CKDio():
    """
    Read Tropomi CKD from the Static CKD product or from dynamic CKD products

    Attributes
    ----------
    ckd_dir : pathlib.Path
    ckd_file : pathlib.Path
    ckd_dyn_file : pathlib.Path
    fid : h5py.File

    Methods
    -------
    close()
       Close resources.
    creation_time()
       Returns datetime when the L1b product was created.
    processor_version()
       Returns version of Tropomi L01B processor used to generate this procuct.
    validity_period()
       Return validity period of CKD product as a tuple of 2 datetime objects.
    get_param(ds_name=None, band='7')
       Returns value(s) of a CKD parameter from the Static CKD product.
    memory(bands='78')
       Returns memory CKD, SWIR only.
    dn2v(bands='78')
       Returns digital number to Volt CKD, SWIR only.
    voltage_to_charge(bands='78')
       Returns Voltage to Charge CKD, SWIR only.
    prnu(bands='78')
       Returns Pixel Response Non-Uniformity (PRNU).
    absirr(qvd=1, bands='78')
       Returns absolute irradiance responsivity.
    relirr(qvd=1, bands='78')
       Returns relative irradiance correction.
    absrad(bands='78')
       Returns absolute radiance responsivity.
    wavelength(bands='78')
       Returns wavelength CKD.
    saa()
       Returns definition of the SAA region.
    offset(bands='78')
       Returns offset CKD, SWIR only.
    darkflux(bands='78')
       Returns dark-flux CKD, SWIR only.
    dpqf(threshold=None, bands='78')
       Returns Detector Pixel Quality Mask (boolean), SWIR only.
    pixel_quality(bands='78')
       Returns Detector Pixel Quality Mask (float [0, 1]), SWIR only.
    noise(bands='78')
       Returns pixel noise CKD, SWIR only
    saturation(bands='78')
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

    Examples
    --------
    """
    def __init__(self, ckd_dir='/nfs/Tropomi/share/ckd', ckd_file=None):
        """
        Initialize access to a Tropomi Static CKD product
        """
        # define path to CKD product
        if ckd_file is not None:
            if not Path(ckd_file).is_file():
                raise FileNotFoundError(
                    'Not found CKD file: {}'.format(ckd_file))
            self.ckd_dir = Path(ckd_file).parent
            self.ckd_file = Path(ckd_file)
        else:
            if not Path(ckd_dir).is_dir():
                raise FileNotFoundError(
                    'Not found CKD directory: {}'.format(ckd_dir))
            self.ckd_dir = Path(ckd_dir)
            if (self.ckd_dir / 'static').is_dir():
                res = sorted((self.ckd_dir / 'static').glob('*_AUX_L1_CKD_*'))
            else:
                res = sorted(self.ckd_dir.glob('*_AUX_L1_CKD_*'))
            if not res:
                raise FileNotFoundError('Static CKD product not found')
            self.ckd_file = res[-1]

        # obtain path to dynamic CKD product (version 1, only)
        self.ckd_dyn_file = None
        if (self.ckd_dir / 'dynamic').is_dir():
            res = sorted((self.ckd_dir / 'dynamic').glob('*_ICM_CKDSIR_*'))
        else:
            res = sorted(self.ckd_dir.glob('*_ICM_CKDSIR_*'))
        if res:
            self.ckd_dyn_file = res[-1]

        # initialize class-attributes
        self.__header = PosixPath('/METADATA', 'earth_explorer_header',
                                  'fixed_header')

        self.fid = h5py.File(self.ckd_file, "r")

    # def __del__(self):
    #    """
    #    called when the object is destroyed
    #    """
    #    self.close()

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
        grpname = str(self.__header / 'source')
        attr = self.fid[grpname].attrs['Creator_Date'][0]
        if attr is None:
            return None

        if isinstance(attr, bytes):
            return attr.decode('ascii')
        return attr

    def processor_version(self) -> str:
        """
        Returns version of Tropomi L01B processor used to generate this procuct
        """
        grpname = str(self.__header / 'source')
        attr = self.fid[grpname].attrs['Creator_Version'][0]
        if attr is None:
            return None

        if isinstance(attr, bytes):
            return attr.decode('ascii')
        return attr

    def validity_period(self) -> tuple:
        """
        Return validity period of CKD product as a tuple of 2 datetime objects
        """
        grpname = str(self.__header / 'validity_period')
        attr = self.fid[grpname].attrs['Validity_Start'][0]
        if attr is None:
            return None
        attr_bgn = attr.decode('ascii')

        attr = self.fid[grpname].attrs['Validity_Stop'][0]
        if attr is None:
            return None
        attr_end = attr.decode('ascii')

        return (datetime.strptime(attr_bgn, '%Y%m%dT%H%M%S'),
                datetime.strptime(attr_end, '%Y%m%dT%H%M%S'))

    # ---------- static CKD's ----------
    def get_param(self, ds_name=None, band='7'):
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
        if ds_name is None:
            ds_name = 'pixel_full_well'

        if not 1 <= int(band) <= 8:
            raise ValueError('band must be between and 1 and 8')

        if ds_name not in self.fid['/BAND{}'.format(band)]:
            raise ValueError('dataset not available')

        full_name = '/BAND{}/{}'.format(band, ds_name)
        if self.fid[full_name].size == 1:
            return self.fid[full_name][0]

        return self.fid[full_name][:]

    def memory(self, bands='78') -> dict:
        """
        Returns memory CKD, SWIR only

        Note SWIR row 257 is excluded
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('Memory CKD only available for SWIR')

        long_name = 'SWIR memory CKD ({})'
        ckd_parms = ['mem_lin_neg_swir', 'mem_lin_pos_swir',
                     'mem_qua_neg_swir', 'mem_qua_pos_swir']
        ckd = {}
        for key in ckd_parms:
            for band in bands:
                if key not in ckd:
                    ckd[key] = S5Pmsm(self.fid['/BAND{}/{}'.format(band, key)],
                                      datapoint=True, data_sel=np.s_[:-1, :])
                    ckd[key].set_long_name(long_name.format(key))
                else:
                    buff = S5Pmsm(self.fid['/BAND{}/{}'.format(band, key)],
                                  datapoint=True, data_sel=np.s_[:-1, :])
                    ckd[key].concatenate(buff, axis=1)

            ckd[key].set_fillvalue()
            ckd[key].fill_as_nan()

        return ckd

    def dn2v(self, bands='78'):
        """
        Returns digital number to Volt CKD, SWIR only

        Notes
        -----
        The DN2V factor has no error attached to it.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('DN2V factor is only available for SWIR')

        ckd = S5Pmsm(self.fid['/BAND7/dn2v_factor_swir'])
        ckd8 = S5Pmsm(self.fid['/BAND8/dn2v_factor_swir'])
        ckd.value[2:] = ckd8.value[2:]
        ckd.set_long_name('SWIR DN2V factor')
        return ckd

    # def non_linearity(self, bands='12'):
    #    """
    #    Returns non-linearity CKD, UVN only
    #    """
    #    if len(bands) > 2:
    #        raise ValueError('read per band or channel, only')
    #    if '7' in bands or '8' in bands:
    #        raise ValueError('non-linearity only available for UVN')
    #    else:
    #        raise NotImplementedError('not implemented, yet')

    def voltage_to_charge(self, bands='78'):
        """
        Returns Voltage to Charge CKD, SWIR only

        Notes
        -----
        The V2C CKD has no error attached to it.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('Voltage to Charge only available for SWIR')

        ckd = S5Pmsm(self.fid['/BAND7/v2c_factor_swir'], datapoint=True)
        ckd.set_long_name('SWIR voltage to charge CKD')
        return ckd

    def prnu(self, bands='78'):
        """
        Returns Pixel Response Non-Uniformity (PRNU)

        Notes
        -----
        * SWIR row 257 is excluded.
        * The PRNU-CKD has no error attached to it (always zero).
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')

        data_sel = None
        if '7' in bands or '8' in bands:
            data_sel = np.s_[:-1, :]
            long_name = 'SWIR PRNU CKD'
        elif '5' in bands or '6' in bands:
            long_name = 'NIR PRNU CKD'
        elif '3' in bands or '4' in bands:
            long_name = 'VIS PRNU CKD'
        else:
            long_name = 'UV PRNU CKD'

        ckd = None
        for band in bands:
            if not ckd:
                ckd = S5Pmsm(self.fid['/BAND{}/PRNU'.format(band)],
                             datapoint=True, data_sel=data_sel)
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid['/BAND{}/PRNU'.format(band)],
                              datapoint=True, data_sel=data_sel)
                ckd.concatenate(buff, axis=1)

            ckd.set_fillvalue()
        return ckd

    def absirr(self, qvd=1, bands='78'):
        """
        Returns absolute irradiance responsivity

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')

        data_sel = None
        if '7' in bands or '8' in bands:
            data_sel = np.s_[:-1, :]
            long_name = 'SWIR absolute irradiance CKD (QVD={})'.format(qvd)
        elif '5' in bands or '6' in bands:
            long_name = 'NIR absolute irradiance CKD (QVD={})'.format(qvd)
        elif '3' in bands or '4' in bands:
            long_name = 'VIS absolute irradiance CKD (QVD={})'.format(qvd)
        else:
            long_name = 'UV absolute irradiance CKD (QVD={})'.format(qvd)

        ckd = None
        for band in bands:
            dsname = '/BAND{}/abs_irr_conv_factor_qvd{}'.format(band, qvd)
            if not ckd:
                ckd = S5Pmsm(self.fid[dsname],
                             datapoint=True, data_sel=data_sel)
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname],
                              datapoint=True, data_sel=data_sel)
                ckd.concatenate(buff, axis=1)

            ckd.set_fillvalue()

        return ckd

    def relirr(self, qvd=1, bands='78'):
        """
        Returns relative irradiance correction

        Parameters
        ----------
         - qvd   : integer
                   select data from QVD1 or QVD2.
                   Default QVD1
         - bands : string
                   select CKD for one band or a channel. Channels can be
                   selected by bands equals '12', '34', '56' or '78'.
                   Default SWIR channel

        Returns
        -------
        dictionary with keys:
         - band           :    Tropomi spectral band ID
         - mapping_cols   :    coarse irregular mapping of the columns
         - mapping_rows   :    coarse irregular mapping of the rows
         - cheb_coefs     :    chebyshev parameters for elevation and azimuth
                               for pixels on a coarse irregular grid

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')

        res = ()
        for band in bands:
            ckd = {}
            ckd['band'] = int(band)

            dsname = '/BAND{}/rel_irr_coarse_mapping_vert'.format(band)
            ckd['mapping_rows'] = self.fid[dsname][:].astype(int)

            dsname = '/BAND{}/rel_irr_coarse_mapping_hor'.format(band)
            mapping_hor = self.fid[dsname][:].astype(int)
            mapping_hor[mapping_hor > 1000] -= 2**16
            ckd['mapping_cols'] = mapping_hor

            dsname = '/BAND{}/rel_irr_coarse_func_cheb_qvd{}'.format(band, qvd)
            ckd['cheb_coefs'] = self.fid[dsname]['coefs'][:]
            res += (ckd,)

        return res

    def absrad(self, bands='78'):
        """
        Returns absolute radiance responsivity

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')

        data_sel = None
        if '7' in bands or '8' in bands:
            data_sel = np.s_[:-1, :]
            long_name = 'SWIR absolute radiance CKD'
        elif '5' in bands or '6' in bands:
            long_name = 'NIR absolute radiance CKD'
        elif '3' in bands or '4' in bands:
            long_name = 'VIS absolute radiance CKD'
        else:
            long_name = 'UV absolute radiance CKD'

        ckd = None
        for band in bands:
            dsname = '/BAND{}/abs_rad_conv_factor'.format(band)
            if not ckd:
                ckd = S5Pmsm(self.fid[dsname],
                             datapoint=True, data_sel=data_sel)
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname],
                              datapoint=True, data_sel=data_sel)
                ckd.concatenate(buff, axis=1)

            ckd.set_fillvalue()

        return ckd

    def wavelength(self, bands='78'):
        """
        Returns wavelength CKD

        Notes
        -----
        * SWIR row 257 is excluded.
        * The wavelength CKD has no error attached to it.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')

        data_sel = None
        if '7' in bands or '8' in bands:
            data_sel = np.s_[:-1, :]
            long_name = 'SWIR wavelength CKD'
        elif '5' in bands or '6' in bands:
            long_name = 'NIR wavelength CKD'
        elif '3' in bands or '4' in bands:
            long_name = 'VIS wavelength CKD'
        else:
            long_name = 'UV wavelength CKD'

        ckd = None
        for band in bands:
            dsname = '/BAND{}/wavelength_map'.format(band)
            if not ckd:
                ckd = S5Pmsm(self.fid[dsname],
                             datapoint=True, data_sel=data_sel)
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname],
                              datapoint=True, data_sel=data_sel)
                ckd.concatenate(buff, axis=1)

            ckd.set_fillvalue()

        return ckd

    def saa(self) -> dict:
        """
        Returns definition of the SAA region
        """
        saa_region = {}
        saa_region['lat'] = self.fid['saa_latitude'][:]
        saa_region['lon'] = self.fid['saa_longitude'][:]

        return saa_region

    # ---------- external CKD's ----------
    def offset(self, bands='78'):
        """
        Returns offset CKD, SWIR only

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('Offset CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR offset CKD'

        # try the Static CKD product, first
        for band in bands:
            dsname = '/BAND{}/analog_offset_swir'.format(band)
            if dsname not in self.fid:
                continue

            if not ckd:
                ckd = S5Pmsm(self.fid[dsname],
                             datapoint=True, data_sel=np.s_[:-1, :])
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname],
                              datapoint=True, data_sel=np.s_[:-1, :])
                ckd.concatenate(buff, axis=1)

        if ckd is not None:
            ckd.set_fillvalue()
            return ckd

        # try the dynamic CKD products
        with h5py.File(self.ckd_dyn_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/analog_offset_swir'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname],
                                 datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname],
                                  datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)
        ckd.set_fillvalue()
        return ckd

    def darkflux(self, bands='78'):
        """
        Returns dark-flux CKD, SWIR only

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('Dark-flux CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR dark-flux CKD'

        # try the Static CKD product, first
        for band in bands:
            dsname = '/BAND{}/long_term_swir'.format(band)
            if dsname not in self.fid:
                continue

            if not ckd:
                ckd = S5Pmsm(self.fid[dsname],
                             datapoint=True, data_sel=np.s_[:-1, :])
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname],
                              datapoint=True, data_sel=np.s_[:-1, :])
                ckd.concatenate(buff, axis=1)

        if ckd is not None:
            ckd.set_fillvalue()
            return ckd

        # try the dynamic CKD products
        with h5py.File(self.ckd_dyn_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/long_term_swir'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname],
                                 datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname],
                                  datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)

        ckd.set_fillvalue()
        return ckd

    def dpqf(self, threshold=None, bands='78'):
        """
        Returns Detector Pixel Quality Mask (boolean), SWIR only

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('pixel quality CKD is only available for SWIR')

        if threshold is None:
            threshold = self.fid['/BAND7/dpqf_threshold'][:]

        # try the Static CKD product, first
        if '/BAND7/dpqf_map' in self.fid:
            dset = self.fid['/BAND7/dpqf_map']
            dpqf_b7 = dset[:-1, :]
            dset = self.fid['/BAND8/dpqf_map']
            dpqf_b8 = dset[:-1, :]

            return np.hstack((dpqf_b7, dpqf_b8)) < threshold

        # try the dynamic CKD products
        with h5py.File(self.ckd_dyn_file, 'r') as fid:
            if threshold is None:
                threshold = fid['/BAND7/dpqf_threshold'][:]

            dset = fid['/BAND7/dpqf_map']
            dpqf_b7 = dset[:-1, :]
            dset = fid['/BAND8/dpqf_map']
            dpqf_b8 = dset[:-1, :]

        return np.hstack((dpqf_b7, dpqf_b8)) < threshold

    def pixel_quality(self, bands='78'):
        """
        Returns Detector Pixel Quality Mask (float [0, 1]), SWIR only

        Notes
        -----
        SWIR row 257 is excluded.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('pixel quality CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR pixel-quality CKD'

        # try the Static CKD product, first
        for band in bands:
            dsname = '/BAND{}/dpqf_map'.format(band)
            if dsname not in self.fid:
                continue

            if not ckd:
                ckd = S5Pmsm(self.fid[dsname], data_sel=np.s_[:-1, :])
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname], data_sel=np.s_[:-1, :])
                ckd.concatenate(buff, axis=1)

        if ckd is not None:
            return ckd

        # try the dynamic CKD products
        with h5py.File(self.ckd_dyn_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/dpqf_map'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)

        return ckd

    def noise(self, bands='78'):
        """
        Returns noise CKD, SWIR only

        Notes
        -----
        * SWIR row 257 is excluded.
        * The noise CKD has no error attached to it.
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('noise CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR noise CKD'

        # try the Static CKD product, first
        for band in bands:
            dsname = '/BAND{}/readout_noise_swir'.format(band)
            if dsname not in self.fid:
                continue

            if not ckd:
                ckd = S5Pmsm(self.fid[dsname], data_sel=np.s_[:-1, :])
                ckd.set_long_name(long_name)
            else:
                buff = S5Pmsm(self.fid[dsname], data_sel=np.s_[:-1, :])
                ckd.concatenate(buff, axis=1)

        if ckd is not None:
            ckd.set_fillvalue()
            return ckd

        # try the dynamic CKD products
        ckd_file = self.ckd_dir / 'OCAL' / 'ckd.readnoise.detector4.nc'
        with h5py.File(ckd_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/readout_noise_swir'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname],
                                 datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname],
                                  datapoint=True, data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)

        ckd.set_fillvalue()
        return ckd

    def saturation(self, bands='78'):
        """
        Returns saturation values (pre-offset), SWIR only

        Notes
        -----
        * SWIR row 257 is excluded
        * The saturation CKD has no error attached to it (always zero)
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('saturation CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR saturation(pre-offset) CKD'
        ckd_file = (self.ckd_dir / 'OCAL'
                    / 'ckd.saturation_preoffset.detector4.nc')
        with h5py.File(ckd_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/saturation_preoffset'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)

        ckd.set_fillvalue()
        return ckd
