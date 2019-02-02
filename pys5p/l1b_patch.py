"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import h5py
import numpy as np

from pys5p.l1b_io import L1BioRAD

# - global variables --------------------------------
_MSG_ERR_IO_BAND_ = 'spectral band of input and output products do not match'


# - local functions --------------------------------
def swir_exp_time(int_delay, int_hold):
    """
    Returns the exact pixel exposure time of the measurements
    """
    return 1.25e-6 * (65540 - int_delay + int_hold)


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
        import shutil

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
        shutil.copy(self.l1b_product, self.l1b_patched)
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
        added to the group: "/METADATA/SRON_METADATA":
             - dateStamp ('now')
             - Git-version of S/W
             - list of patched datasets
             - auxiliary datasets used by patch-routines
        """
        from datetime import datetime
        from .version import version as __version__

        if not self.__patched_msm:
            return

        with h5py.File(self.l1b_patched, 'r+') as fid:
            sgrp = fid.require_group("/METADATA/SRON_METADATA")
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
        from pys5p import swir_region

        # read original data
        with L1BioRAD(self.l1b_product) as l1b:
            band = l1b.select('STANDARD_MODE')
            quality_level = l1b.get_msm_data('quality_level')
            chan_quality = l1b.get_msm_data('spectral_channel_quality')

        if band in ('7', '8'):
            l2_dpqm = dpqm[swir_region.coords(mode='level2', band=band)]
        else:
            raise ValueError('only implemented for band 7 or 8')

        # patch dataset 'quality_level'
        quality_level[...] = int(100 * l2_dpqm)  # broadcasting

        # patch dataset 'spectral_channel_quality'
        chan_quality &= ~2               # set second bit to zero (all bad)
        chan_quality[l2_dpqm >= threshold] &= 2  # flag good pixels

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


# --------------------------------------------------
def _main():
    """
    main function: process command-line parameters and call patch functions
    """
    import argparse

    from pys5p.ckd_io import CKDio

    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description=('Patch Tropomi level 1b product.\n'
                     'This is program is considered alpha, no patches are '
                     'yet available. Foreseen patches for SWIR data are: '
                     'offset, dark flux and combined (ir)radiance factors.'))
    parser.add_argument('l1b_product', default=None,
                        help='name of L1B product (full path)')
    parser.add_argument('--ckd_dir', default=None,
                        help='path to official SWIR CKD')
    # ToDo maybe we want a comma seperated list to perform multiple patches?
    parser.add_argument('--patch', default=None,
                        choices=('pixel_quality', 'offset', 'darkflux',
                                 'prnu', 'relrad', 'absrad'),
                        help='what needs to be patched?')
    parser.add_argument('-o', '--output', default='/tmp',
                        help='directory to store patched product')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        default=True, help='only show error messages')
    args = parser.parse_args()
    if args.verbose:
        print(args)
    if args.l1b_product is None:
        parser.print_usage()
        parser.exit()

    with L1Bpatch(args.l1b_product, data_dir=args.output,
                  ckd_dir=args.ckd_dir) as l1b_patch:
        if args.patch == 'pixel_quality':
            with CKDio(args.ckd_dir) as ckd:
                dpqm = ckd.pixel_quality()

            l1b_patch.pixel_quality(dpqm)
            print('INFO: applied patch pixel_qualtiy')

        if args.patch == 'offset':
            l1b_patch.offset()
            print('WARNING: patch offset not yet implemented')

        if args.patch == 'darkflux':
            l1b_patch.darkflux()
            print('WARNING: patch dark-flux not yet implemented')

        if args.patch == 'prnu':
            l1b_patch.prnu()
            print('WARNING: patch prnu not yet implemented')

        if args.patch == 'relrad':
            l1b_patch.relrad()
            print('WARNING: patch relrad not yet implemented')

        if args.patch == 'absrad':
            l1b_patch.absrad()
            print('WARNING: patch absrad not yet implemented')


# --------------------------------------------------
if __name__ == '__main__':
    _main()
