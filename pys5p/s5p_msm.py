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
    def __init__(self, h5_dset, index0=None, datapoint=False):
        """
        Read measurement data from a Tropomi OCAL, ICM, of L1B product

        Parameters
        ----------
        h5_dset   :  h5py.Dataset
        index0    :  integer
        datapoint :  boolean
        
        HDF5 compound datasets of type datapoint can be read as value & error
        """
        assert isinstance(h5_dset, h5py.Dataset)
        assert h5_dset.ndim <= 3

        keys = []
        dims = []
        self.error = None
        self.coords = None
        self.name = os.path.basename(h5_dset.name)
        if datapoint:
            self.fillvalue = h5_dset.fillvalue[0]
        else:
            self.fillvalue = h5_dset.fillvalue
        if index0 is None:
            if datapoint:
                self.value = np.squeeze(h5_dset.value['value'])
                self.error = np.squeeze(h5_dset.value['error'])
            else:
                self.value = np.squeeze(h5_dset.value)
            if h5_dset.shape[0] == 1:
                for ii in range(len(h5_dset.dims)-1):
                    keys.append(os.path.basename(h5_dset.dims[ii+1][0].name))
                    buff = h5_dset.dims[ii+1][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    dims.append(buff)
            else:
                for ii in range(len(h5_dset.dims)):
                    keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    dims.append(buff)
        else:
            if datapoint:
                self.value = np.squeeze(h5_dset[index0,...]['value'])
                self.error = np.squeeze(h5_dset[index0,...]['error'])
            else:
                self.value = np.squeeze(h5_dset[index0,...])
            if len(list(index0)) == 1:
                for ii in range(len(h5_dset.dims)-1):
                    keys.append(os.path.basename(h5_dset.dims[ii+1][0].name))
                    buff = h5_dset.dims[ii+1][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    dims.append(buff)
            else:
                for ii in range(len(h5_dset.dims)):
                    keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)

                    if ii == 0:
                        dims.append(buff[index0])
                    else:
                        dims.append(buff)
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(dims)

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
         - it is required to initialize the class with band[2*chan-1]
        """
        if self.error is not None:
            _msm = S5Pmsm(h5_dset, index0=index0, datapoint=True)
        else:
            _msm = S5Pmsm(h5_dset, index0=index0)

        xdim = _msm.value.ndim-1
        if self.value.shape[-1] == _msm.value.shape[-1]:
            self.value = np.concatenate((self.value, _msm.value), axis=xdim)
            if self.error is not None:
                self.error = np.concatenate((self.error, _msm.error), axis=xdim)
        else:
            self.value = np.concatenate(pad_rows(self.value, _msm.value),
                                        axis=xdim)
            if self.error is not None:
                self.error = np.concatenate(pad_rows(self.error, _msm.error),
                                            axis=xdim)
        key = self.coords._fields[xdim]
        dims = np.concatenate((self.coords[xdim],
                                 len(self.coords[xdim]) + _msm.coords[xdim]))
        self.coords = self.coords._replace(**{key : dims})

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

        Works only on datasets with HDF5 datatype 'float' or 'datapoints'
        """
        if self.fillvalue == float.fromhex('0x1.ep+122'):
            self.value[(self.value == self.fillvalue)] = np.nan

    def remove_row257(self):
        """
        Remove last (not used) row from the data (SWIR, only)
        """
        if ndim == 2:
            self.value = self.value[:-1,:]
            if self.error is not None:
                self.error = self.error[:-1,:]
        elif ndim == 3:
            self.value = self.value[:,:-1,:]
            if self.error is not None:
                self.error = self.error[:,:-1,:]
