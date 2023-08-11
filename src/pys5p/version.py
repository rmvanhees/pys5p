# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2023 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""Provide access to the software version as obtained from git."""

__all__ = ['pys5p_version']

from . import __version__


def pys5p_version(full=False, githash=False) -> str:
    """Return the software version as obtained from git.

    Examples
    --------
    Show the software version of the module pys5p::

       > from pys5p.version import pys5p_version
       > pys5p_version()
    '2.1.5'
    """
    if full:
        return __version__

    if githash:
        return __version__.split('+g')[1].split('.')[0]

    return __version__.split('+')[0]
