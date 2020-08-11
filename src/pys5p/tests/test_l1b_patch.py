"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

The class L1Bpatch provides methods to patch Tropomi SWIR measurement data in
offline level 1b products (incl. calibration, irradiance and radiance).

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from ..ckd_io import CKDio
from ..l1b_patch import L1Bpatch


# --------------------------------------------------
def _main():
    """
    main function: process command-line parameters and call patch functions
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description=('Patch Tropomi level 1b product.\n'
                     'This is program is considered alpha, no patches are '
                     'yet available. Foreseen patches for SWIR data are: '
                     'offset, dark flux and combined (ir)radiance factors.'))
    parser.add_argument('l1b_product', default=None,
                        help='name of L1B product (full path)')
    parser.add_argument('--ckd_dir', default='/nfs/Tropomi/share/ckd',
                        help='path to official SWIR CKD')
    # ToDo maybe we want a comma seperated list to perform multiple patches?
    parser.add_argument('--patch', default=None,
                        choices=('pixel_quality', 'offset', 'darkflux',
                                 'prnu', 'relrad', 'absrad'),
                        help='what needs to be patched?')
    parser.add_argument('--check', action='store_true',
                        help='compare patched datasets with original')
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

            l1b_patch.pixel_quality(dpqm.value)
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

        if args.check:
            l1b_patch.check()


# --------------------------------------------------
if __name__ == '__main__':
    _main()
