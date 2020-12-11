"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Implements a lite interface with the xarray::DataArray

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import PurePath

import numpy as np
import xarray as xr


def __get_attrs(dset, field: str) -> dict:
    """
    Return attributes of the HDF5 dataset

    Parameters
    ----------
    dset :  h5py.Dataset
       h5py dataset from which the attributes are read
    field : str
       Name of field in compound dataset

    Returns
    -------
    dict with numpy arrays
    """
    attrs = {}
    for key in dset.attrs:
        if key == 'DIMENSION_LIST':
            continue

        attr_value = dset.attrs[key]
        if isinstance(attr_value, np.ndarray):
            attr_value = attr_value[0]

        if isinstance(attr_value, bytes):
            attrs[key] = attr_value.decode('ascii')
        elif field is not None and key == '_FillValue' \
           and isinstance(attr_value, np.void):
            attrs[key] = attr_value[0]
        else:
            attrs[key] = attr_value

    return attrs


def __get_coords(dset, dims: list, data_sel) -> list:
    """
    Return coordinates of the HDF5 dataset

    Parameters
    ----------
    dset :  h5py.Dataset
       h5py dataset from which the data is read
    dims : list of strings
       Alternative names for the dataset dimensions if not attached to dataset
    data_sel :  numpy slice
       A numpy slice generated for example numpy.s_
       Default read while array

    Returns
    -------
    A sequence of tuples [(dims, data), ...]
    """
    coords = []
    if len(dset.dims) == dset.ndim:
        try:
            for dim in dset.dims:
                buff = dim[0][()]
                if np.all(buff == 0):
                    buff = np.arange(dim[0].size, dtype=dim[0].dtype)

                coords.append((PurePath(dim[0].name).name, buff))
        except RuntimeError:
            coords = []
        else:
            return coords

    # Only necessary when no dimension scales are defined or they are corrupted
    # set default dimension names
    if dims is None:
        if dset.ndim > 3:
            raise ValueError('not implemented for ndim > 3')

        dims = ['time', 'row', 'column'][-dset.ndim:]

    for ii in range(dset.ndim):
        buff = np.arange(dset.shape[ii], dtype='u4')[data_sel[ii]]
        coords.append((dims[ii], buff))

    return coords


def __get_data(dset, field: str, data_sel):
    """
    Return data of the HDF5 dataset

    Parameters
    ----------
    dset :  h5py.Dataset
       h5py dataset from which the data is read
    field : str
       Name of field in compound dataset or None
    data_sel :  numpy slice
       A numpy slice generated for example numpy.s_
       Default read while array

    Returns
    -------
    Numpy array

    Notes
    -----
    Read floats always as doubles
    """
    if np.issubdtype(dset.dtype, np.floating):
        with dset.astype(float):
            data = dset[data_sel]
        return data

    if field is None:
        return dset[data_sel]

    data = dset.fields(field)[data_sel]
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    return data


def h5_to_xr(h5_dset, field=None, data_sel=None, dims=None):
    """
    Create xarray::DataArray from a HDF5 dataset (with dimension scales)

    Parameters
    ----------
    h5_dset :  h5py.Dataset
       Data, dimensions, coordinates and attributes are read for this dataset
    field : str
       Name of field in compound dataset or None
    data_sel :  numpy slice
       A numpy slice generated for example numpy.s_
    dims :  list of strings
       Alternative names for the dataset dimensions if not attached to dataset

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    ...

    Examples
    --------
    >>> fid = h5py.File(flname, 'r')        # flname is a HDF5/netCDF4 file
    >>> xdata = h5_to_xr(fid['signal'])
    >>> fid.close()

    Combine Tropomi SWIR data of band7 and band8
    >>> fid = h5py.File(s5p_b7_prod, 'r')   # Tropomi band7 product
    >>> xdata7 = h5_to_xr(fid['signal'])
    >>> fid.close()
    >>> fid = h5py.File(s5p_b8_prod, 'r')   # Tropomi band8 product
    >>> xdata8 = h5_to_xr(fid['signal'])
    >>> fid.close()
    >>> xdata = xr.concat((xdata7, xdata8), dim='spectral_channel')

    Optionally, fix the 'column' dimension
    >>> xdata = xdata.assign_coords(
    >>> ... column=np.arange(xdata.column.size, dtype='u4'))
    """
    # ToDo fix cases when data_sel equals int, Ellipsis, ...
    if data_sel is None:
        data_sel = ()

    if field is not None:
        try:
            _field = {'name': field,
                      'oneof': len(h5_dset.dtype.names),
                      'index': h5_dset.dtype.names.index(field)}
        except Exception as exc:
            raise RuntimeError('field {} not found in dataset {}'.format(
                field, h5_dset.name)) from exc

    # Name of this array
    name = PurePath(h5_dset.name).name

    # Values for this array
    data = __get_data(h5_dset, field, data_sel)

    # Attributes to assign to the array
    attrs = __get_attrs(h5_dset, field)

    # Coordinates (tick labels) to use for indexing along each dimension
    coords = __get_coords(h5_dset, data_sel, dims)

    return xr.DataArray(data, name=name, attrs=attrs, coords=coords)
