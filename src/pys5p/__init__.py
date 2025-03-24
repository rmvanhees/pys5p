#
# This file is part of pys5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2025 SRON
#   All Rights Reserved
#
# License:  BSD-3-Clause

"""SRON Python package `pys5p`.

It contains software to read Sentinel-5p Tropomi ICM, L1B and L2 products.
"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version(__name__)
