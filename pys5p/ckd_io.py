"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Provides access to the S5P Static CKD product, type AUX_L1_CKD

ToDo:
 - access to UVN CKD, still incomplete
 - identify latest Static CKD product, e.g. using the validity period

Copyright (c) 2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""
from pathlib import Path

import h5py
import numpy as np

from .s5p_msm import S5Pmsm

# - global parameters ------------------------------

# - local functions --------------------------------

# - class definition -------------------------------
class CKDio():
    """
    Read Tropomi CKD from the Static CKD product or from dynamic CKD products

    Not all CKD are defined or derived for all bands.
    You can request a CKD for one band or for a channel (bands: '12', '34',
    '56', '78'). Do not mix bands from UVN and SWIR!
    """
    def __init__(self, ckd_dir='/nfs/Tropomi/share/ckd/current'):
        """
        Initialize access to a Tropomi Static CKD product
        """
        # initialize private class-attributes
        self.ckd_file = None
        self.fid = None
        self.__header = Path('/METADATA/earth_exploirer_header/fixed_header')

        self.ckd_dir = Path(ckd_dir)
        if not self.ckd_dir.is_dir():
            raise FileNotFoundError('directory {} not found'.format(ckd_dir))

        res = sorted(self.ckd_dir.glob('*_AUX_L1_CKD_*'))
        if not res:
            raise FileNotFoundError('Static CKD product not found')
        self.ckd_file = res[-1]
        self.fid = h5py.File(self.ckd_file, "r")

    def __del__(self):
        """
        Make sure that we close all resources
        """
        if self.fid is not None:
            self.fid.close()

    def creation_time(self):
        """
        Returns datetime when the L1b product was created
        """
        attr = self.fid[self.__header / 'source'].attrs['Creator_Date'][0]
        if attr is None:
            return None

        return attr.decode('ascii')

    def processor_version(self):
        """
        Returns version of the CKD product
        """
        attr = self.fid[self.__header / 'source'].attrs['Creator_Verion'][0]
        if attr is None:
            return None

        return attr.decode('ascii')

    def validity_period(self):
        """
        Return validity period of CKD product as a tuple of 2 datetime objects
        """
        from datetime import datetime

        attr = self.fid[self.__header / 'validity_period'].attrs['Validity_Start']
        if attr is None:
            return None
        attr_bgn = attr.decode('ascii')

        attr = self.fid[self.__header / 'validity_period'].attrs['Validity_Stop']
        if attr is None:
            return None
        attr_end = attr.decode('ascii')

        return (datetime.strptime(attr_bgn, '%Y%m%dT%H%M%S'),
                datetime.strptime(attr_end, '%Y%m%dT%H%M%S'))

    # ---------- static CKD's ----------
    def get_param(self, ds_name, band='7'):
        """
        Returns value(s) of a CKD parameter from the Static CKD product.
        Datasets of size=1 are return as scalar

        Parameters
        ----------
        ds_name  :  string
           Name of the HDF5 dataset
        band     : string
           Band identifier '1', '2', ..., '8'

        Returns
        -------
        numpy.ndarray or scalar

        Handy function for scalar HDF5 datasets, such as:
         - dc_reference_temp
         - dpqf_threshold
         - pixel_full_well
         - pixel_fw_flag_thresh
        """
        if not 1 <= int(band) <= 8:
            raise ValueError('band must be between and 1 and 8')

        if ds_name not in self.fid['/BAND{}'.format(band)]:
            raise ValueError('dataset not available')

        full_name = '/BAND{}/{}'.format(band, ds_name)
        if self.fid[full_name].size == 1:
            return self.fid[full_name][0]

        return self.fid[full_name][:]

    def memory(self, bands='78'):
        """
        Returns memory CKD, SWIR only

        Note SWIR row 257 is excluded
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('Voltage to Charge only available for SWIR')

        long_name = 'SWIR memory CKD'
        ckd_parms = ['mem_lin_neg_swir', 'mem_lin_pos_swir',
                     'mem_qua_neg_swir', 'mem_qua_pos_swir']
        ckd = {}
        for key in ckd_parms:
            for band in bands:
                if key not in ckd:
                    ckd[key] = S5Pmsm(self.fid['/BAND{}/{}'.format(band, key)],
                                      datapoint=True, data_sel=np.s_[:-1, :])
                    ckd[key].set_long_name(long_name)
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

        Note: the DN2V factor has no error attached to it
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('DN2V factor is only available for SWIR')

        ckd = S5Pmsm(self.fid['/BAND7/v2c_factor_swir'])
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

        Note: the V2C CKD has no error attached to it
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

        Note1 SWIR row 257 is excluded
        Note2 the PRNU-CKD has no error attached to it (always zero)
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

        Note SWIR row 257 is excluded
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

    def absrad(self, bands='78'):
        """
        Returns absolute radiance responsivity

        Note SWIR row 257 is excluded
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

        Note1 SWIR row 257 is excluded
        Note2 the wavelength CKD has no error attached to it
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

    # ---------- external CKD's ----------
    def offset(self, bands='78'):
        """
        Returns offset CKD, SWIR only

        Note SWIR row 257 is excluded
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
        ckd_file = self.ckd_dir.parent / 'dynamic' / 'ckd.offset.detector4.nc'
        with h5py.File(ckd_file, 'r') as fid:
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

        Note SWIR row 257 is excluded
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
        ckd_file = self.ckd_dir.parent / 'dynamic' / 'ckd.dark.detector4.nc'
        with h5py.File(ckd_file, 'r') as fid:
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

    def noise(self, bands='78'):
        """
        Returns noise CKD, SWIR only

        Note1 SWIR row 257 is excluded
        Note2 the noise CKD has no error attached to it
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
        ckd_file = self.ckd_dir.parent / 'dynamic' / 'ckd.readnoise.detector4.nc'
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

        Note1 SWIR row 257 is excluded
        Note2 the saturation CKD has no error attached to it (always zero)
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('saturation CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR saturation(pre-offset) CKD'
        ckd_file = self.ckd_dir.parent / 'dynamic'
        ckd_file /= 'ckd.saturation_preoffset.detector4.nc'
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

    def dpqf(self, threshold=None, bands='78'):
        """
        Returns Detector Pixel Quality Mask (boolean), SWIR only

        Note1 SWIR row 257 is excluded
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('pixel quality CKD is only available for SWIR')

        ckd_file = self.ckd_dir.parent / 'dynamic' / 'ckd.dpqf.detector4.nc'
        with h5py.File(ckd_file, 'r') as fid:
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

        Note1 SWIR row 257 is excluded
        """
        if len(bands) > 2:
            raise ValueError('read per band or channel, only')
        if '7' not in bands and '8' not in bands:
            raise ValueError('pixel quality CKD is only available for SWIR')

        ckd = None
        long_name = 'SWIR pixel-quality CKD'
        ckd_file = self.ckd_dir.parent / 'dynamic' / 'ckd.dpqf.detector4.nc'
        with h5py.File(ckd_file, 'r') as fid:
            for band in bands:
                dsname = '/BAND{}/dpqf_map'.format(band)

                if not ckd:
                    ckd = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.set_long_name(long_name)
                else:
                    buff = S5Pmsm(fid[dsname], data_sel=np.s_[:-1, :])
                    ckd.concatenate(buff, axis=1)

        return ckd
