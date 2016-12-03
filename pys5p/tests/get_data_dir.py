from __future__ import absolute_import

import os.path

from os import environ

DATA_DIR_GUESSES = ['/Users/{}/pys5p-data', '/data/{}/pys5p-data']

def get_data_dir():
    """
    Obtain directory with test datasets

    Limited to UNIX/Linux/MacOS operating systems

    This module checks if the following directories are available:
     - /data/$USER/pys5p-data
     - /Users/$USER/pys5p-data

    It expects the data to be organized in the sub-directories:
     - CKD which should contain the SWIR dpqf CKD
     - OCM which should contain at least one directory of an on-ground 
       calibration measurement with one or more OCAL LX products.
     - L1B which should contain at least one offline calibration, irradiance 
       and radiance product.
     - ICM which contain at least one inflight calibration product.
    """
    data_path = None
    
    user = environ['USER']
    for key in DATA_DIR_GUESSES:
        if os.path.isdir(key.format(user)):
            data_path = key.format(user)
            break
    print(data_path)
    return data_path
