"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on class CKDio (xarray version)

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

from pys5p.ckd_io import CKDio


def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(
        description=f'{Path(__file__).name}: run units-test on class CKDio')
    parser.add_argument('ckd_dir', nargs=1, type=str, default=None,
                        help=('directory with CKD data with'
                              ' static CKD in a subdirectory static'))
    args = parser.parse_args()

    with CKDio(args.ckd_dir[0], ckd_version=1) as ckd:
        print(ckd.ckd_file)
        for meth in dir(ckd):
            if (meth.startswith('_')
                or meth.startswith('ckd')
                or meth in ('close', 'fid', 'get_param')):
                continue
            print('-------------------------', meth, '[v1]',
                  '-------------------------')
            print(meth, getattr(ckd, meth)())

    with CKDio(args.ckd_dir[0], ckd_version=2) as ckd:
        print(ckd.ckd_file)
        for meth in dir(ckd):
            if (meth.startswith('_')
                or meth.startswith('ckd')
                or meth in ('close', 'fid', 'get_param')):
                continue
            print('-------------------------', meth, '[v2]',
                  '-------------------------')
            print(meth, getattr(ckd, meth)())

# - main code --------------------------------------
if __name__ == '__main__':
    main()
