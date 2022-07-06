"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path
import shutil

from setuptools_scm import get_version

import h5py
import numpy as np

from .l1b_io import L1BioRAD
from . import swir_region


# - global variables --------------------------------
_MSG_ERR_IO_BAND_ = 'spectral band of input and output products do not match'


# - local functions --------------------------------


# - class definition -------------------------------
class L1Bpatch():
    """
    Definition off class L1Bpatch
    """
    def __init__(self, l1b_product: str, data_dir='/tmp',
                 ckd_dir='/nfs/Tropomi/share/ckd') -> None:
        """
        Initialize access to a Tropomi offline L1b product
        """
        prod_type = Path(l1b_product).name[0:15]
        if prod_type not in ('S5P_OFFL_L1B_RA', 'S5P_RPRO_L1B_RA'):
            raise TypeError(
                'Warning: only implemented for Tropomi L1b radiance products')

        # initialize private class-attributes
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            self.data_dir.mkdir(mode=0o755)
        self.ckd_dir = Path(ckd_dir)
        self.l1b_product = Path(l1b_product)
        self.l1b_patched = \
            self.data_dir / self.l1b_product.name.replace('_01_', '_99_')
        if self.l1b_patched.is_file():
            self.l1b_patched.unlink()
        self.__patched_msm = []

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """
        Before closing the product, we make sure that the output product
        describes what has been altered by the S/W. To keep any change
        traceable.

        In case the L1b product is altered, the attributes listed below are
        added to the group: '/METADATA/SRON_METADATA':
             - dateStamp ('now')
             - Git-version of S/W
             - list of patched datasets
             - auxiliary datasets used by patch-routines
        """
        if not self.l1b_patched.is_file():
            return

        if not self.__patched_msm:
            return

        with h5py.File(self.l1b_patched, 'r+') as fid:
            sgrp = fid.require_group('/METADATA/SRON_METADATA')
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

    # --------------------------------------------------
    def pixel_quality(self, dpqm, threshold=0.8) -> None:
        """
        Patch SWIR pixel_quality.

        Patched dataset: 'quality_level' and 'spectral_channel_quality'

        Requires (naive approach):
        * read original dataset 'spectral_channel_quality'
        * read pixel quality ckd
        * adjust second pixel of each byte of spectral_channel_quality
        * quality_level = int(100 * dpqm)
        * write updated datasets to patched product

        Parameters
        ----------
        dpqm  :   array-like
           SWIR pixel quality as a float value between 0 and 1
        thershold :  float, optional
           threshold for good pixels, default 0.8

        Returns
        -------
        Nothing
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            quality_level = l1b.get_msm_data('quality_level')
            print('quality_level', quality_level.dtype)
            chan_quality = l1b.get_msm_data('spectral_channel_quality')
            print('chan_quality', chan_quality.dtype)

        if band in ('7', '8'):
            l2_dpqm = dpqm[swir_region.coords(mode='level2', band=band)]
        else:
            raise ValueError('only implemented for band 7 or 8')

        # patch dataset 'quality_level'
        quality_level[...] = (100 * l2_dpqm).astype(np.uint8)  # broadcasting

        # patch dataset 'spectral_channel_quality'
        buff = chan_quality & ~2          # set second bit to zero (all good)
        buff[:, l2_dpqm < threshold] += 2                   # flag bad pixels
        chan_quality = buff.astype(np.uint8)

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('quality_level', quality_level)
            l1b.set_msm_data('spectral_channel_quality', chan_quality)

    def offset(self) -> None:
        """
        Patch SWIR offset correction.

        Patched dataset: 'radiance' ('radiance_error' and 'radiance_noise'?)

        Requires (naive approach):
        * reverse applied radiance calibration
        * reverse applied stray-light correction
        * reverse applied PRNU correction
        * reverse applied dark-flux correction
        * reverse applied offset correction
        * apply (alternative) offset correction
        * apply (alternative) dark-flux correction
        * apply (alternative) PRNU correction
        * apply (alternative) stray-light correction
        * apply (alternative) radiance calibration

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            data = l1b.get_msm_data('radiance')

        # read required CKD's

        # patch dataset 'radiance'

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('radiance', data)

    def darkflux(self) -> None:
        """
        Patch SWIR dark-flux correction.

        Patched dataset: 'radiance' ('radiance_error' and 'radiance_noise'?)

        Requires (naive approach):
        * reverse applied radiance calibration
        * reverse applied stray-light correction
        * reverse applied PRNU correction
        * reverse applied dark-flux correction
        * apply (alternative) dark-flux correction
        * apply (alternative) PRNU correction
        * apply (alternative) stray-light correction
        * apply (alternative) radiance calibration

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            data = l1b.get_msm_data('radiance')

        # read required CKD's

        # patch dataset 'radiance'

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('radiance', data)

    def prnu(self) -> None:
        """
        Patch pixel response non-uniformity correction.

        Patched dataset: 'radiance' ('radiance_error' and 'radiance_noise'?)

        Requires (naive approach):
        * reverse applied radiance calibration
        * reverse applied stray-light correction
        * reverse applied PRNU correction
        * apply (alternative) PRNU correction
        * apply (alternative) stray-light correction
        * apply (alternative) radiance calibration

        Alternative: neglect impact stray-light, but apply patch to correct for
        spectral features.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        It is assumed that for the PRNU correction the CKD has to be multiplied
        with the pixel signals.
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            data = l1b.get_msm_data('radiance')

        # read required CKD's

        # patch dataset 'radiance'

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('radiance', data)

    def relrad(self) -> None:
        """
        Patch relative radiance calibration.

        Patched dataset: 'radiance' ('radiance_error' and 'radiance_noise'?)

        Requires:
        * reverse applied radiance calibration
        * apply alternative radiance calibration

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            data = l1b.get_msm_data('radiance')

        # read required CKD's

        # patch dataset 'radiance'

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('radiance', data)

    def absrad(self) -> None:
        """
        Patch absolute radiance calibration.

        Patched dataset: 'radiance' ('radiance_error' and 'radiance_noise'?)

        Requires:
        * reverse applied irradiance calibration
        * apply alternative irradiance calibration

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
        """
        if not self.l1b_patched.is_file():
            shutil.copy(self.l1b_product, self.l1b_patched)

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            data = l1b.get_msm_data('radiance')

        # read required CKD's

        # patch dataset 'radiance'

        # write patched dataset to new product
        with L1BioRAD(self.l1b_patched, readwrite=True) as l1b:
            res = l1b.select('STANDARD_MODE')
            if res != band:
                raise ValueError(_MSG_ERR_IO_BAND_)
            l1b.set_msm_data('radiance', data)

    def check(self) -> None:
        """
        Check patched dataset in L1B product
        """
        if not self.l1b_patched.is_file():
            raise ValueError('patched product not found')

        with h5py.File(self.l1b_patched, 'r+') as fid:
            if 'SRON_METADATA' not in fid['/METADATA']:
                raise ValueError('no SRON metadata defined in L1B product')
            sgrp = fid['/METADATA/SRON_METADATA']
            if 'patched_datasets' not in sgrp:
                raise ValueError('no patched datasets in L1B prduct')
            patched_datasets = sgrp['patched_datasets'][:]

        for ds_name in patched_datasets:
            with L1BioRAD(self.l1b_product) as l1b:
                l1b.select('STANDARD_MODE')
                orig = l1b.get_msm_data(ds_name.split('/')[-1])

            with L1BioRAD(self.l1b_patched) as l1b:
                l1b.select('STANDARD_MODE')
                patch = l1b.get_msm_data(ds_name.split('/')[-1])

            if np.issubdtype(orig.dtype, np.integer):
                if np.array_equiv(orig, patch):
                    print(ds_name.split('/')[-1], ' equal True')
                else:
                    print(f'{ds_name.split("/")[-1]}'
                          f' equal {(orig == patch).sum()}'
                          f' differ {(orig != patch).sum()}')
            else:
                print('test not yet defined')
