"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pys5p.l1b_io import L1BioRAD


# --------------------------------------------------
def pixel_quality(l1b_file, l1b_patched):
    """
    Patch SWIR pixel_quality.

    Patched dataset: 'spectral_channel_quality'

    Requires (naive approach):
     * read original dataset 'spectral_channel_quality'
     * read pixel quality ckd
     * adjust third pixel of each byte
     * write updated dataset to patched product

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """
    # read original data
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('spectral_channel_quality')

    # read new pixel-quality CKD

    # patch dataset 'spectral_channel_quality'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('spectral_channel_quality', data)


def offset(l1b_file, l1b_patched):
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

    Alternative: ...

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """
    # read original data
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('radiance')

    # read required CKD's

    # patch dataset 'radiance'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('radiance', data)


def darkflux(l1b_file, l1b_patched):
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

    Alternative: ...

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """
    # read original data
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('radiance')

    # read required CKD's

    # patch dataset 'radiance'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('radiance', data)


def prnu(l1b_file, l1b_patched):
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
    array      :  ndarray
       SWIR detector data, band 7 & 8 combined
    prnu_orig  :  ndarray
       PRNU CKD used by the L01b processor
    prnu_patch :  ndarray
       newly developed PRNU CKD

    Returns
    -------
    out    :  ndarray
       Returns array where the data is corrected with the newly developed PRNU

    Notes
    -----
    It is assumed that for the PRNU correction the CKD has to be multiplied
    with the pixel signals.
    """
    # read original data
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('radiance')

    # read required CKD's

    # patch dataset 'radiance'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('radiance', data)


def relrad(l1b_file, l1b_patched):
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
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('radiance')

    # read required CKD's

    # patch dataset 'radiance'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('radiance', data)


def absrad(l1b_file, l1b_patched):
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
    with L1BioRAD(l1b_file) as l1b:
        l1b.select('STANDARD_MODE')
        data = l1b.get_msm_data('radiance')

    # read required CKD's

    # patch dataset 'radiance'

    # write patched dataset to new product
    with L1BioRAD(l1b_patched, readwrite=True) as l1b:
        l1b.select('STANDARD_MODE')
        l1b.set_msm_data('radiance', data)


# --------------------------------------------------
def _main():
    """
    main function: process command-line parameters and call patch functions
    """
    import argparse
    import shutil
    from pathlib import Path

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
    parser.add_argument('--patched_ckd_dir', default=None,
                        help='path to alternative SWIR CKD')
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

    if not Path(args.output).is_dir():
        Path(args.output).mkdir(mode=0o755)

    l1b_patch = str(Path(args.output,
                         Path(args.l1b_product.replace('_01_', '_02_')).name))
    shutil.copy(args.l1b_product, l1b_patch)

    prod_type = Path(l1b_patch).name[0:15]
    if prod_type not in ('S5P_OFFL_L1B_RA', 'S5P_RPRO_L1B_RA'):
        print('Warning: only implemented for Tropomi offline L1b radiance')
        return

    if args.patch == 'pixel_quality':
        # add dir with patched CKD
        pixel_quality(args.l1b_product, l1b_patch)
        print('INFO: applied patch {}'.format(args.patch))

    if args.patch == 'offset':
        offset(args.l1b_product, l1b_patch)
        print('WARNING: patch {} not yet implemented'.format(args.patch))

    if args.patch == 'darkflux':
        darkflux(args.l1b_product, l1b_patch)
        print('WARNING: patch {} not yet implemented'.format(args.patch))

    if args.patch == 'prnu':
        prnu(args.l1b_product, l1b_patch)
        print('WARNING: patch {} not yet implemented'.format(args.patch))

    if args.patch == 'relrad':
        relrad(args.l1b_product, l1b_patch)
        print('WARNING: patch {} not yet implemented'.format(args.patch))

    if args.patch == 'absrad':
        absrad(args.l1b_product, l1b_patch)
        print('WARNING: patch {} not yet implemented'.format(args.patch))


# --------------------------------------------------
if __name__ == '__main__':
    _main()
