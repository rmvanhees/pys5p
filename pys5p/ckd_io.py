"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Provides acces to the Tropomi SWIR CKD's

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import, print_function

from pathlib import Path

import h5py
import numpy as np

from .s5p_msm import S5Pmsm

# - global parameters ------------------------------


# - class definition -------------------------------
class CKDio():
    """
    This class should offer all the necessary functionality to read Tropomi
    CKD's
    """
    def __init__(self, ckd_dir='/nfs/Tropomi/ocal/ckd'):
        """
        Initialize access to a Tropomi offline L1b product

        Parameters
        ----------
        ckd_dir  :  string
           path to directory with subdirectories per CKD
        """
        # initialize private class-attributes
        self.__ckd_dir = Path(ckd_dir)
        self.pixel_fw_flag_thresh = 0.9  # ToDo read from CKD product

        assert self.__ckd_dir.is_dir(), \
            '*** Fatal, can not find CKD base-directory: {}'.format(ckd_dir)
        assert (self.__ckd_dir / 'ckd_release').is_dir(), \
            '*** Fatal, can not find UVN CKD subdirectory: {}'.format(ckd_dir)
        assert (self.__ckd_dir / 'ckd_release_swir').is_dir(), \
            '*** Fatal, can not find SWIR CKD subdirectory: {}'.format(ckd_dir)

    def get_swir_absirr(self, qvd=1):
        """
        returns absolute irradiance responsivity for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release' / 'abs_irr_uvn'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find absirr directory: {}'.format(ckd_dir)

        file_b7 = ckd_dir / 'irrad_conv_factors.band7.ckd.nc'
        file_b8 = ckd_dir / 'irrad_conv_factors.band8.ckd.nc'
        assert file_b7.is_file() and file_b8.is_file(), \
            '*** Fatal, no ABSIRR CKD found on the system'

        dname = '/BAND7/abs_irr_conv_factor_qvd{}'.format(qvd)
        with h5py.File(str(file_b7), 'r') as fid:
            ckd = S5Pmsm(fid[dname], datapoint=True, data_sel=np.s_[:-1, :])

        dname = '/BAND8/abs_irr_conv_factor_qvd{}'.format(qvd)
        with h5py.File(str(file_b8), 'r') as fid:
            ckd_b8 = S5Pmsm(fid[dname], datapoint=True, data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR absolute irradiance CKD')
        return ckd

    def get_swir_absrad(self):
        """
        returns absolute radiance responsivity for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release' / 'abs_rad_uvn'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find absrad directory: {}'.format(ckd_dir)

        file_b7 = ckd_dir / 'abs_rad.band7.ckd.nc'
        file_b8 = ckd_dir / 'abs_rad.band8.ckd.nc'
        assert file_b7.is_file() and file_b8.is_file(), \
            '*** Fatal, no ABSRAD CKD found on the system'

        dname = '/BAND7/abs_rad_conv_factor'
        with h5py.File(str(file_b7), 'r') as fid:
            ckd = S5Pmsm(fid[dname], datapoint=True, data_sel=np.s_[:-1, :])

        dname = '/BAND8/abs_rad_conv_factor'
        with h5py.File(str(file_b8), 'r') as fid:
            ckd_b8 = S5Pmsm(fid[dname], datapoint=True, data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR absolute radiance CKD')
        return ckd

    def get_swir_darkflux(self):
        """
        returns darkflux CKD for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'darkflux'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find dark-flux directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.dark.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no dark-flux CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/long_term_swir'], datapoint=True,
                         data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/long_term_swir'], datapoint=True,
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR dark-flux CKD')
        return ckd

    def get_swir_dpqf(self, threshold=None):
        """
        returns Detector Pixel Quality Flags for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'dpqf'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find dpqf directory: {}'.format(ckd_dir)

        ckd_file = ckd_dir / 'ckd.dpqf.detector4.nc'
        assert ckd_file.is_file(), \
            '# *** Fatal, no DPQF CKD found on the system'

        with h5py.File(str(ckd_file), 'r') as fid:
            if threshold is None:
                threshold = fid['/BAND7/dpqf_threshold'][:]

            dset = fid['/BAND7/dpqf_map']
            dpqf_b7 = dset[:-1, :]
            dset = fid['/BAND8/dpqf_map']
            dpqf_b8 = dset[:-1, :]

        return np.hstack((dpqf_b7, dpqf_b8)) < threshold

    def get_swir_dpqm(self, ds_name='dpqf_map'):
        """
        returns Detector Pixel Quality Mask for SWIR, except row 257

        Parameters
        ----------
         ds_name  :  string
           name of the pixel-quality map. Default: dpqf_map
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'dpqf'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find dpqf directory: {}'.format(ckd_dir)

        ckd_file = ckd_dir / 'ckd.dpqf.detector4.nc'
        assert ckd_file.is_file(), \
            '# *** Fatal, no DPQM CKD found on the system'

        with h5py.File(str(ckd_file), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/' + ds_name], data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/' + ds_name], data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_long_name('SWIR quality CKD')
        return ckd

    def get_swir_memory(self):
        """
        returns memory CKD for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'memory'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find memory directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.memory.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no memory CKD found on the system'

        ckd_parms = ['mem_lin_neg_swir', 'mem_lin_pos_swir',
                     'mem_qua_neg_swir', 'mem_qua_pos_swir']
        ckd = {}
        with h5py.File(str(file_ch4), 'r') as fid:
            for key in ckd_parms:
                ckd[key] = S5Pmsm(fid['/BAND7/{}'.format(key)],
                                  datapoint=True, data_sel=np.s_[:-1, :])
                ckd_b8 = S5Pmsm(fid['/BAND8/{}'.format(key)],
                                datapoint=True, data_sel=np.s_[:-1, :])

                ckd[key].concatenate(ckd_b8, axis=1)
                ckd[key].set_fillvalue()
                ckd[key].set_long_name(
                    fid['/BAND7/{}'.format(key)].attrs['comment'])
                ckd[key].fill_as_nan()

        return ckd

    def get_swir_noise(self):
        """
        returns noise CKD for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'readnoise_external'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find noise directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.readnoise.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no noise CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/readout_noise_swir'], datapoint=False,
                         data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/readout_noise_swir'], datapoint=False,
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR noise CKD')
        return ckd

    def get_swir_offset(self):
        """
        returns offset CKD for SWIR, except row 257
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'offset'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find offset directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.offset.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no offset CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/analog_offset_swir'], datapoint=True,
                         data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/analog_offset_swir'], datapoint=True,
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR offset CKD')
        return ckd

    def get_swir_prnu(self):
        """
        returns Pixel Response Non-Uniformity (PRNU) for SWIR, except row 257

        Note the PRNU-CKD has no error attached to it (always zero)
        """
        ckd_dir = self.__ckd_dir / 'ckd_release' / 'prnu_uvn'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find prnu directory: {}'.format(ckd_dir)

        file_b7 = ckd_dir / 'stitch.band7.ckd.nc'
        file_b8 = ckd_dir / 'stitch.band8.ckd.nc'
        assert file_b7.is_file() and file_b8.is_file(), \
            '*** Fatal, no PRNU CKD found on the system'

        with h5py.File(str(file_b7), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/PRNU'], datapoint=True,
                         data_sel=np.s_[:-1, :])

        with h5py.File(str(file_b8), 'r') as fid:
            ckd_b8 = S5Pmsm(fid['/BAND8/PRNU'], datapoint=True,
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR PRNU CKD')
        return ckd

    def get_swir_saturation(self):
        """
        returns saturation values (pre-offset) for SWIR, except row 257

        Note the saturation-CKD has no error attached to it (always zero)
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'saturation_preoffset'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find saturation directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.saturation_preoffset.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no saturation CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/saturation_preoffset'],
                         data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/saturation_preoffset'],
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR saturation(pre-offset) CKD')
        return ckd

    def get_swir_saturation_ext(self):
        """
        returns saturation values (external) for SWIR, except row 257

        Note the saturation-CKD has no error attached to it (always zero)
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'saturation_external'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find saturation directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.saturation_external.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no saturation CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(np.full((256, 500),
                                 fid['/BAND7/pixel_full_well'][:]))
            ckd_b8 = S5Pmsm(np.full((256, 500),
                                    fid['/BAND8/pixel_full_well'][:]))

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_fillvalue()
        ckd.set_long_name('SWIR saturation(pre-external) CKD')
        return ckd

    def get_swir_v2c(self):
        """
        returns Voltage to Charge CKD for SWIR

        Note: the V2C CKD has no error attached to it
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'v2c_external'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find v2c directory: {}'.format(ckd_dir)

        ckd_file = ckd_dir / 'ckd.v2c_factor.detector4.nc'
        assert ckd_file.is_file(), \
            '*** Fatal, no V2C CKD found on the system'

        with h5py.File(str(ckd_file), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/v2c_factor_swir'], datapoint=True)

        ckd.set_long_name('SWIR voltage to charge CKD')
        return ckd

    def get_swir_wavelength(self):
        """
        returns wavelength CKD for SWIR, except row 257

        Note: the wavelength CKD has no error attached to it
        """
        ckd_dir = self.__ckd_dir / 'ckd_release_swir' / 'wavelength'
        assert ckd_dir.is_dir(), \
            '*** Fatal, can not find wavelength directory: {}'.format(ckd_dir)

        file_ch4 = ckd_dir / 'ckd.wavelength.detector4.nc'
        assert file_ch4.is_file(), \
            '*** Fatal, no wavelength CKD found on the system'

        with h5py.File(str(file_ch4), 'r') as fid:
            ckd = S5Pmsm(fid['/BAND7/wavelength_map'],
                         data_sel=np.s_[:-1, :])
            ckd_b8 = S5Pmsm(fid['/BAND8/wavelength_map'],
                            data_sel=np.s_[:-1, :])

        ckd.concatenate(ckd_b8, axis=1)
        ckd.set_long_name('SWIR wavelength CKD')
        return ckd
