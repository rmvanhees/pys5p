"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""


# --------------------------------------------------
def offset():
    """
    Patch SWIR offset correction.

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
    pass


def darkflux():
    """
    Patch SWIR dark-flux correction.

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
    pass


def prnu(array, prnu_orig, prnu_patch):
    """
    Patch pixel response non-uniformity correction.

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
    return prnu_patch * array / prnu_orig


def radiance():
    """
    Patch radiance calibration.

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
    pass


def irradiance():
    """
    Patch irradiance calibration.

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
    pass


# --------------------------------------------------
def _main():
    """
    Let the user test the software!!!
    """
    import argparse
    import shutil
    from pathlib import Path

    from pys5p.l1b_io import L1BioRAD

    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description=('Patch Tropomi level 1b product.\n'
                     'This is program is considered alpha, no patches are '
                     'yet available. Foreseen patches for SWIR data are: '
                     'offset, dark flux and combined (ir)radiance factors.'))
    parser.add_argument('l1b_product', default=None,
                        help='name of L1B product (full path)')
    parser.add_argument('--ckd_dir', default=None,
                        help='path to SWIR CKD data used by the L01b processor')
    parser.add_argument('--patched_ckd_dir', default=None,
                        help='path to alternative SWIR CKD data')
    parser.add_argument('--msm_type', default=None,
                        help=('define measurement type as: '
                              '<processing class>_<ic_id>'))
    parser.add_argument('--msm_dset', default=None,
                        help='define measurement dataset to be read/patched')
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
    print(args.l1b_product, l1b_patch)
    shutil.copy(args.l1b_product, l1b_patch)

    prod_type = Path(l1b_patch).name[0:15]
    if prod_type == 'S5P_OFFL_L1B_RA':
        msm_type = args.msm_type
        if args.msm_type is None:
            msm_type = 'STANDARD_MODE'
        msm_dset = args.msm_dset
        if args.msm_dset is None:
            msm_dset = 'radiance'
        print('rad: ', msm_type, msm_dset)

        # open L1B product and read dataset
        l1b = L1BioRAD(l1b_patch, readwrite=True)
        l1b.select(msm_type)
        data = l1b.get_msm_data(msm_dset)

        # patch dataset (increase all values by 10%)
        patch = 1.10 * data

        # write patched data to L1B product
        l1b.set_msm_data(msm_dset, patch)

        # update meta-data of product and flush all changes to disk
        del l1b
    else:
        print('Warning: only implemented for Tropomi offline L1b radiance')


# --------------------------------------------------
if __name__ == '__main__':
    _main()
