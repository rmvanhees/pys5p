"""
This file is part of pys5p

https://github.com/rmvanhees/pys5p.git

Calculates the exact pixel exposure time of the SWIR measurements

Requires parameters int_delay and int_holt from the instrument_settings

Copyright (c) 2019-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""


def swir_exp_time(int_delay: int, int_hold: int) -> float:
    """
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
