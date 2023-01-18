# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2023 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""Provide access to the software version as obtained from git.
"""
__all__ = ['pys5p_version']

from pys5p import __version__


def pys5p_version(full=False, githash=False):
    """Returns software version as obtained from git.
    """
    if full:
        return __version__

    if githash:
        return __version__.split('+g')[1].split('.')[0]

    return __version__.split('+')[0]
