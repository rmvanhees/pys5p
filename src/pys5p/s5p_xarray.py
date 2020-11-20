"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implements an lite interface with the xarray::DataArray

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import PurePath

import numpy as np
import xarray as xr

def h5_to_xr(h5_dset):
    """
    Create xarray::DataArray from a HDF5 dataset (with dimension scales)

    Parameters
    ----------
    h5_dset :  h5py.Dataset

    Returns
    -------
    xarray.DataArray

    ToDo
    ----
    * Do we need to offer data selection
    * Implement time-axis
    * Direct read of data, specially when original is 'f4' and request is 'f8'
    * Offer to convert missing data to NaN
    """
    # Values for this array
    data = h5_dset[()]

    # Name of this array
    name = PurePath(h5_dset.name).name

    # Attributes to assign to the array
    attrs = {}
    for key in h5_dset.attrs:
        if key == 'DIMENSION_LIST':
            continue
        if isinstance(h5_dset.attrs[key], bytes):
            attrs[key] = h5_dset.attrs[key].decode('ascii')
        else:
            attrs[key] = h5_dset.attrs[key]

    # Coordinates (tick labels) to use for indexing along each dimension.
    # Provided as sequence of tuples [(dims, data), ...]
    coords = []
    if len(h5_dset.dims) == h5_dset.ndim:
        for dim in h5_dset.dims:
            buff = dim[0][()]
            if np.all(buff == 0):
                buff = np.arange(dim[0].size, dtype=dim[0].dtype)

            coords.append((PurePath(dim[0].name).name, buff))

    return xr.DataArray(data, name=name, attrs=attrs, coords=coords)
