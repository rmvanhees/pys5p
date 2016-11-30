from __future__ import absolute_import

import os.path

from os import environ

def get_data_dir():
    """
    Obtain directory with test datasets

    Limited to UNIX/Linux/MacOS operating systems

    This module checks if the following directories are available:
     - /data/$USER/pys5p-data
     - /Users/$USER/pys5p-data

    It expects the data to be organized in the sub-directories:
     - CKD
     - OCM which should contain at least one directory of an on-ground 
       calibration measurement with one or more OCAL LX products
     - L1B which should contain at least one offline calibration, irradiance 
       and radiance product
     - ICM
    """
    user = environ['USER']
    if os.path.isdir(os.path.join('/Users', user)):
        return os.path.join('/Users', user, 'pys5p-data')
    elif os.path.isdir(os.path.join('/data', user)):
        return os.path.join('/data', user, 'pys5p-data')
    else:
        return None
