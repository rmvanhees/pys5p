"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The classes L1BioCAL, L1BioIRR, L1BioRAD provide read access to
offline level 1b products, resp. calibration, irradiance and radiance.

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import os.path

import numpy as np
import h5py

#- global parameters ------------------------------

FILLVALUE = float.fromhex('0x1.ep+122')

#- local functions --------------------------------
class CKDio(object):
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
        self.__ckd_dir = ckd_dir

        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find CKD base-directory: {}'.format(ckd_dir)
        assert os.path.isdir(os.path.join(ckd_dir, 'ckd_release')), \
            '*** Fatal, can not find UVN CKD subdirectory: {}'.format(ckd_dir)
        assert os.path.isdir(os.path.join(ckd_dir, 'ckd_release_swir')), \
            '*** Fatal, can not find SWIR CKD subdirectory: {}'.format(ckd_dir)

    def get_swir_darkflux(self):
        """
        returns darkflux CKD for SWIR, except row 257
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir', 'darkflux')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find darkflux directory: {}'.format(ckd_dir)

        file_ch4 = os.path.join(ckd_dir,'ckd.dark.detector4.nc')
        if not os.path.isfile(file_ch4):
            print( '# *** Warning no darkflux CKD found on the system' )
            return (np.zeros((256, 1000), dtype=np.float64),
                    np.zeros((256, 1000), dtype=np.float64))
        else:
            with h5py.File(file_ch4, 'r') as fid:
                dset = fid['/BAND7/long_term_swir']
                dark_b7 = dset[:-1, :]['value']
                dark_b7[(dark_b7 == FILLVALUE)] = np.nan

                error_b7 = dset[:-1, :]['error']
                error_b7[(error_b7 == FILLVALUE)] = np.nan

                dset = fid['/BAND8/long_term_swir']
                dark_b8 = dset[:-1, :]['value']
                dark_b8[(dark_b8 == FILLVALUE)] = np.nan

                error_b8 = dset[:-1, :]['error']
                error_b8[(error_b8 == FILLVALUE)] = np.nan

            return (np.hstack((dark_b7, dark_b8)),
                    np.hstack((error_b7, error_b8)))

    def get_swir_dpqf(self, threshold=None):
        """
        returns Detector Pixel Quality Flags for SWIR, except row 257
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir', 'dpqf')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find DPQF-CKD directory: {}'.format(ckd_dir)

        ckd_file = os.path.join(ckd_dir, 'ckd.dpqf.detector4.nc')

        if not os.path.isfile(ckd_file):
            print( '# *** Warning no DPQF-CKD found on the system' )
            return np.zeros((256, 1000), dtype=np.bool)

        with h5py.File(ckd_file, 'r' ) as fid:
            if threshold is None:
                threshold = fid['/BAND7/dpqf_threshold'][:]

            dset = fid['/BAND7/dpqf_map']
            dpqf_b7 = dset[:-1, :]
            dset = fid['/BAND8/dpqf_map']
            dpqf_b8 = dset[:-1, :]

        return np.hstack((dpqf_b7, dpqf_b8)) < threshold

    def get_swir_offset(self):
        """
        returns offset CKD for SWIR, except row 257
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir', 'offset')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find offset directory: {}'.format(ckd_dir)

        file_ch4 = os.path.join(ckd_dir,'ckd.offset.detector4.nc')
        if not os.path.isfile(file_ch4):
            print( '# *** Warning no offset CKD found on the system' )
            return (np.zeros((256, 1000), dtype=np.float64),
                    np.zeros((256, 1000), dtype=np.float64))
        else:
            with h5py.File(file_ch4, 'r') as fid:
                dset = fid['/BAND7/analog_offset_swir']
                offs_b7 = dset[:-1, :]['value']
                offs_b7[(offs_b7 == FILLVALUE)] = np.nan

                error_b7 = dset[:-1, :]['error']
                error_b7[(error_b7 == FILLVALUE)] = np.nan

                dset = fid['/BAND8/analog_offset_swir']
                offs_b8 = dset[:-1, :]['value']
                offs_b8[(offs_b8 == FILLVALUE)] = np.nan

                error_b8 = dset[:-1, :]['error']
                error_b8[(error_b8 == FILLVALUE)] = np.nan

            return (np.hstack((offs_b7, offs_b8)),
                    np.hstack((error_b7, error_b8)))

    def get_swir_prnu(self):
        """
        returns Pixel Response Non-Uniformity (PRNU) for SWIR, except row 257

        Note the PRNU-CKD has no error attached to it (always zero)
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release', 'prnu_uvn')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find PRNU-CKD directory: {}'.format(ckd_dir)

        file_b7 = os.path.join(ckd_dir, 'stitch.band7.ckd.nc')
        file_b8 = os.path.join(ckd_dir, 'stitch.band8.ckd.nc')
        if not os.path.isfile(file_b7) or not os.path.isfile(file_b8):
            print( '# *** Warning no PRNU CKD found on the system' )
            return np.ones((256, 1000), dtype=np.float)
        else:
            with h5py.File(file_b7, 'r') as fid:
                dset = fid['/BAND7/PRNU']
                prnu_b7 = dset[:-1, :]['value']

            with h5py.File(file_b8, 'r') as fid:
                dset = fid['/BAND8/PRNU']
                prnu_b8 = dset[:-1, :]['value']

            return np.hstack((prnu_b7, prnu_b8))

    def get_swir_saturation(self):
        """
        returns saturation values (pre-offset) for SWIR, except row 257

        Note the saturation-CKD has no error attached to it (always zero)
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir',
                               'saturation_preoffset')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find saturation directory: {}'.format(ckd_dir)

        file_ch4 = os.path.join(ckd_dir,'ckd.saturation_preoffset.detector4.nc')
        if not os.path.isfile(file_ch4):
            print( '# *** Warning no saturation CKD found on the system' )
            return np.full((256,1000), np.nan)
        else:
            with h5py.File(file_ch4, 'r') as fid:
                dset = fid['/BAND7/saturation_preoffset']
                satur_b7 = dset[:-1, :]
                dset = fid['/BAND8/saturation_preoffset']
                satur_b8 = dset[:-1, :]

            return np.hstack((satur_b7, satur_b8))

    def get_swir_v2c(self):
        """
        returns Voltage to Charge CKD for SWIR

        Note: the V2C CKD has no error attached to it
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir', 'v2c')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find V2C-CKD directory: {}'.format(ckd_dir)

        ckd_file = os.path.join(ckd_dir, 'ckd.v2c_factor.detector4.nc')
        if not os.path.isfile(ckd_file):
            print( '# *** Warning no V2C-CKD found on the system' )
            return np.ones((4,), dtype=np.float32)

        with h5py.File(ckd_file, 'r' ) as fid:
            v2c = fid['/BAND7/v2c_factor_swir']['value']

        return v2c

    def get_swir_wavelength(self):
        """
        returns wavelength CKD for SWIR, except row 257

        Note: the wavelength CKD has no error attached to it
        """
        ckd_dir = os.path.join(self.__ckd_dir, 'ckd_release_swir', 'wavelength')
        assert os.path.isdir(ckd_dir), \
            '*** Fatal, can not find wavelength directory: {}'.format(ckd_dir)

        file_ch4 = os.path.join(ckd_dir,'ckd.wavelength.detector4.nc')
        if not os.path.isfile(file_ch4):
            print( '# *** Warning no wavelength CKD found on the system' )
            return np.zeros((256, 1000), dtype=np.float64)
        else:
            with h5py.File(file_ch4, 'r') as fid:
                dset = fid['/BAND7/wavelength_map']
                wv_b7 = dset[:-1, :]
                dset = fid['/BAND8/wavelength_map']
                wv_b8 = dset[:-1, :]

            return np.hstack((wv_b7, wv_b8))
