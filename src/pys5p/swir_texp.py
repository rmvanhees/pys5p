# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""Calculate the Tropomi SWIR exposure time from detector settings."""


def swir_exp_time(int_delay: int, int_hold: int) -> float:
    """Calculate the correct SWIR exposure time from detector settings.

    Parameters
    ----------
    int_delay :  int
      parameters int_delay from the instrument_settings
    int_hold  :  int
      parameters int_holt from the instrument_settings
    Returns
    -------
    float
        exact (SWIR) pixel exposure time
    """
    return 1.25e-6 * (65540 - int_delay + int_hold)
