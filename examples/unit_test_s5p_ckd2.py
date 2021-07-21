"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on class CKDio

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path
from pys5p.ckd_io2 import CKDio

def main():
    """
    """
    if Path('/nfs/Tropomi/share/ckd').is_dir():
        ckd_dir = '/nfs/Tropomi/share/ckd'
    elif Path('/data/richardh/Tropomi/share/ckd').is_dir():
        ckd_dir = '/data/richardh/Tropomi/share/ckd'
    else:
        ckd_dir = '/data2/richardh/Tropomi/share/ckd'

    print(ckd_dir)
    with CKDio(ckd_dir, ckd_version=1) as ckd:
        print(ckd.ckd_file)
        for meth in dir(ckd):
            if (meth.startswith('_')
                or meth.startswith('ckd')
                or meth in ('close', 'fid', 'get_param')):
                continue
            print('-------------------------', meth, '[v1]',
                  '-------------------------')
            print(meth, getattr(ckd, meth)())

    with CKDio(ckd_dir, ckd_version=2) as ckd:
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
