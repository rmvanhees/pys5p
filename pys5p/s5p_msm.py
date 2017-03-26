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
    """
    def __init__(self, h5_dset, index0=None):
        """
        """
        assert isinstance(h5_dset, h5py.Dataset)
        assert h5_dset.ndim <= 3

        keys = []
        values = []
        self.coords = None
        self.name = os.path.basename(h5_dset.name)
        self.fillvalue = h5_dset.fillvalue

        if index0 is None:
            self.value = np.squeeze(h5_dset.value)
            if dset.shape[0] == 1:
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
            if isinstance(index0, int):
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
        Coords = namedtuple('Coords', keys)
        self.coords = Coords._make(values)

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
        """
        msm = S5Pmsm(h5_dset, index0=index0)
        
        xdim = msm.value.ndim-1
        print(xdim, self.value.shape[-1], msm.value.shape[-1])
        print(len(self.coords), len(msm.coords))
        if self.value.shape[-1] == msm.value.shape[-1]:
            self.value = np.concatenate((self.value, msm.value), axis=xdim)
        else:
            data1 = self.value.copy()
            data2 = msm.value.copy()
            (data1, data2) = pad_rows(data1, data2)
            self.value = np.concatenate((data1, data2), axis=xdim)
        key = self.coords._fields[xdim]
        values = np.concatenate((self.coords[xdim],
                                 len(self.coords[xdim]) + msm.coords[xdim]))
        self.coords = self.coords._replace(**{key : values})

    def set_units(self, name, force=False):
        """
        """
        if self.units is None or force:
            self.units = name

    def set_long_name(self, name, force=False):
        """
        """
        if self.long_name is None or force:
            self.long_name = name

    def fill_as_nan(self):
        """
        """
        if self.fillvalue == float.fromhex('0x1.ep+122'):
            self.value[(self.value == fillvalue)] = np.nan
#
#--------------------------------------------------
#
if __name__ == '__main__':
    """
    """
    data_dir = '/data/richardh/pys5p-data/ICM'
    flname = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001100_20151002T140000.h5'
    msm_path = 'BAND7_CALIBRATION/BACKGROUND_MODE_1071/OBSERVATIONS'

    fid = h5py.File(os.path.join(data_dir, flname), 'r')
    dset = fid[os.path.join(msm_path, 'signal_avg')]
                    
    msm = S5Pmsm(dset)
    print(msm.name)
    print(msm.long_name)
    print(msm.units)
    print(msm.value.shape)
    print(len(msm.coords[-1]))
    fid.close()

    msm_path = 'BAND8_CALIBRATION/BACKGROUND_MODE_1071/OBSERVATIONS'

    fid = h5py.File(os.path.join(data_dir, flname), 'r')
    dset = fid[os.path.join(msm_path, 'signal_avg')]
                    
    msm.combine_bands(dset)
    print(msm.name)
    print(msm.long_name)
    print(msm.units)
    print(msm.value.shape)
    print(len(msm.coords[-1]))
    print(msm.coords[-1])
    fid.close()

