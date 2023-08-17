#
# This file is part of pys5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2023 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause

"""The pys5p package contains software to read S5p Tropomi L1B products.
And contains plotting routines to display your data beautifully.
"""
import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version(__name__)

