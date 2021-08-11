"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

Copyright (c) 2020-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pys5p import __version__


def get(full=False):
    """
    Returns software version as obtained from git
    """
    if full:
        return __version__

    return __version__.split('+')[0]
