"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class S5Pmsm read HDF5 measurement data including its attributes and
dimensions.

Limited to 3 dimensional dataset

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import os.path
from collections import namedtuple

import numpy as np
import h5py

#- local functions --------------------------------
def pad_rows(arr1, arr2):
    """
    Pad the array with the least numer of rows with NaN's
    """
    if arr2.ndim == 2:
        if arr1.shape[0] < arr2.shape[0]:
            buff = arr1.copy()
            arr1 = np.full_like(arr2, np.nan)
            arr1[0:buff.shape[0], :] = buff
        elif arr1.shape[0] > arr2.shape[0]:
            buff = arr2.copy()
            arr2 = np.full_like(arr1, np.nan)
            arr2[0:buff.shape[0], :] = buff
    else:
        if arr1.shape[1] < arr2.shape[1]:
            buff = arr1.copy()
            arr1 = np.full_like(arr2, np.nan)
            arr1[:, 0:buff.shape[1], :] = buff
        elif arr1.shape[1] > arr2.shape[1]:
            buff = arr2.copy()
            arr2 = np.full_like(arr1, np.nan)
            arr2[:, 0:buff.shape[1], :] = buff

    return (arr1, arr2)

#- class definition -------------------------------
class S5Pmsm(object):
    """
    Definition of class S5Pmsm
    """
    def __init__(self, h5_dset, index0=None):
        """
        Read measurement data from a Tropomi OCAL, ICM, of L1B product
        """
        assert isinstance(h5_dset, h5py.Dataset)
        assert h5_dset.ndim <= 3

        keys = []
        values = []
        self.coords = None
        self.name = os.path.basename(h5_dset.name)
        self.fillvalue = h5_dset.fillvalue
        print('index0 : ', index0)
        if index0 is None:
            self.value = np.squeeze(h5_dset.value)
            if h5_dset.shape[0] == 1:
                for ii in range(len(h5_dset.dims)-1):
                    keys.append(os.path.basename(h5_dset.dims[ii+1][0].name))
                    buff = h5_dset.dims[ii+1][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    values.append(buff)
            else:
                for ii in range(len(h5_dset.dims)):
                    keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    values.append(buff)
        else:
            self.value = np.squeeze(h5_dset[index0,...])
            if len(list(index0)) == 1:
                for ii in range(len(h5_dset.dims)-1):
                    keys.append(os.path.basename(h5_dset.dims[ii+1][0].name))
                    buff = h5_dset.dims[ii+1][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    values.append(buff)
            else:
                for ii in range(len(h5_dset.dims)):
                    keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    if ii == 0:
                        values.append(buff[index0])
                    else:
                        values.append(buff)
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(values)

        if 'units' in h5_dset.attrs:
            self.units = h5_dset.attrs['units']
        else:
            self.units = None

        if 'long_name' in h5_dset.attrs:
            self.long_name = h5_dset.attrs['long_name']
        else:
            self.long_name = None

    def combine_bands(self, h5_dset, index0=None):
        """
        Combine data of two measurment datasets
        Note:
         - the two datasets have to be from the same detector
         - it is required to first read the first (left band)
        """
        _msm = S5Pmsm(h5_dset, index0=index0)

        xdim = _msm.value.ndim-1
        if self.value.shape[-1] == _msm.value.shape[-1]:
            self.value = np.concatenate((self.value, _msm.value), axis=xdim)
        else:
            data1 = self.value.copy()
            data2 = _msm.value.copy()
            (data1, data2) = pad_rows(data1, data2)
            self.value = np.concatenate((data1, data2), axis=xdim)
        key = self.coords._fields[xdim]
        values = np.concatenate((self.coords[xdim],
                                 len(self.coords[xdim]) + _msm.coords[xdim]))
        self.coords = self.coords._replace(**{key : values})

    def set_units(self, name, force=False):
        """
        Set the units attribute, overwrite when force is true
        """
        if self.units is None or force:
            self.units = name

    def set_long_name(self, name, force=False):
        """
        Set the long_name attribute, overwrite when force is true
        """
        if self.long_name is None or force:
            self.long_name = name

    def fill_as_nan(self):
        """
        Replace fillvalues in data with NaN's
        """
        if self.fillvalue == float.fromhex('0x1.ep+122'):
            self.value[(self.value == self.fillvalue)] = np.nan
