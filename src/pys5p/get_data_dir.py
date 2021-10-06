"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implement function to obtain path to test-data

Copyright (c) 2016-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from os import environ
from pathlib import Path


def get_data_dir():
    """
    Obtain directory with test datasets

    Limited to UNIX/Linux/MacOS operating systems

    This module checks if the following directories are available:
     - /data/$USER/pys5p-data
     - /Users/$USER/pys5p-data
     - environment variable PYS5P_DATA_DIR

    It expects the data to be organized in the sub-directories:
     - CKD which should contain the SWIR dpqf CKD
     - OCM which should contain at least one directory of an on-ground
       calibration measurement with one or more OCAL LX products.
     - L1B which should contain at least one offline calibration, irradiance
       and radiance product.
     - ICM which contain at least one inflight calibration product.
    """
    try:
        user = environ['USER']
    except KeyError:
        print('*** Fatal: environment variable USER not set')
        return None

    guesses_data_dir = [f'/data/{user}/pys5p-data',
                        f'/Users/{user}/pys5p-data']

    try:
        _ = environ['PYS5P_DATA_DIR']
    except KeyError:
        pass
    else:
        guesses_data_dir.append(environ['PYS5P_DATA_DIR'])

    for key in guesses_data_dir:
        if Path(key).is_dir():
            return key

    raise FileNotFoundError
