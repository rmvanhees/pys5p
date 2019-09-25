# This file is part of pys5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause

""" The pys5p package contains software to read S5p Tropomi L1B products.
    And contains plotting routines to display your data beautifully."""

from pkg_resources import get_distribution, DistributionNotFound

from .swir_texp import swir_exp_time

from . import biweight
from . import get_data_dir
from . import ckd_io, icm_io, l1b_io, lv2_io, ocm_io
from . import error_propagation
from . import s5p_msm, s5p_plot #, s5p_geoplot
from . import swir_region
from . import tol_colors

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
