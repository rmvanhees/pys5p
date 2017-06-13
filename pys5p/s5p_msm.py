"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class S5Pmsm read HDF5 measurement data including its attributes and
dimensions. Initialization:

  S5Pmsm attribute | hdf5 dataset           | Numpy array
  -------------------------------------------------------------------------
  name             | h5_dset.name           | 'value'
  value            | h5_dset.value['value'] | np.squeeze(data)
                   | or h5_dset.value       |
  error            | h5_dset.value['value'] | None
                   | or None                |
  coords           | h5_dset.dims           | [[['time',] 'row',] 'column']
  units            | attrs['units']         | None
  long_name        | attrs['long_name']     | None
  fillvalue        | h5_dset.fillvalue      | None
  coverage         | None                   | None

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
    def __init__(self, dset, data_sel=None, datapoint=False):
        """
        Read measurement data from a Tropomi OCAL, ICM, of L1B product

        Parameters
        ----------
        dset      :  h5py.Dataset or ndarray
           h5py dataset from which the data is read, data is used to
           initalize S5Pmsm object
        data_sel  :  numpy slice
           a numpy slice generated for example numpy.s_
        datapoint :  boolean
           to indicate that the dataset is a compound of type datapoint

        Returns
        -------
        numpy structure with dataset data and attributes, including data,
        fillvalue, coordinates, units, ...

        """
        # initialize object
        self.name = 'value'
        self.value = None
        self.error = None
        self.coords = None
        self.coverage = None
        self.units = None
        self.long_name = ''
        self.fillvalue = None

        if isinstance(dset, h5py.Dataset):
            self.__from_h5_dset(dset, data_sel, datapoint)
        else:
            self.__from_ndarray(dset, data_sel)

    def __from_h5_dset(self, h5_dset, data_sel, datapoint):
        """
        initialize S5Pmsm object from h5py dataset
        """
        self.name = os.path.basename(h5_dset.name)

        # copy dataset values (and error) to object
        if data_sel is None:
            if datapoint:
                self.value = np.array(h5_dset.value['value'])
                self.error = np.array(h5_dset.value['error'])
            else:
                self.value = np.array(h5_dset.value)
        else:
            if datapoint:
                self.value = np.array(h5_dset[data_sel]['value'])
                self.error = np.array(h5_dset[data_sel]['error'])
            else:
                self.value = np.array(h5_dset[data_sel])

        # copy all dimensions with size longer then 1
        # ToDo what happens when no dimensions are assigned to dataset?
        keys = []
        dims = []
        for ii in range(h5_dset.ndim):
            if self.value.shape[ii] == 1:
                continue
            elif self.value.shape[ii] == h5_dset.shape[ii]:
                keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                buff = h5_dset.dims[ii][0][:]
                if np.all(buff == 0):
                    buff = np.arange(buff.size)
                dims.append(buff)
            else:
                keys.append(os.path.basename(h5_dset.dims[ii][0].name))
                buff = h5_dset.dims[ii][0][:]
                if np.all(buff == 0):
                    buff = np.arange(buff.size)
                dims.append(buff[data_sel[ii]])

        # add dimensions as a namedtuple
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(dims)

        # remove all dimensions with size equal 1 from value (and error)
        self.value = np.squeeze(self.value)
        if datapoint:
            self.error = np.squeeze(self.error)

        # copy FillValue (same for value/error in a datapoint)
        if datapoint:
            self.fillvalue = h5_dset.fillvalue[0]
        else:
            self.fillvalue = h5_dset.fillvalue

        # copy its units
        if 'units' in h5_dset.attrs:
            if isinstance(h5_dset.attrs['units'], bytes):
                self.units = h5_dset.attrs['units'].decode('ascii')
            else:
                self.units = h5_dset.attrs['units']

        # copy its long_name
        if 'long_name' in h5_dset.attrs:
            self.long_name = h5_dset.attrs['long_name']

    def __from_ndarray(self, data, data_sel):
        """
        initialize S5Pmsm object from h5py dataset
        """
        # copy dataset values (and error) to object
        if data_sel is None:
            self.value = np.squeeze(data)
        else:
            self.value = np.squeeze(data[data_sel])

        # define coordinates
        if data.ndim == 1:
            keys = ['column']
            dims = [np.arange(self.value.shape[0])]
        elif data.ndim == 2:
            keys = ['row', 'column']
            dims = [np.arange(self.value.shape[0]),
                    np.arange(self.value.shape[1])]
        elif data.ndim == 3:
            keys = ['time', 'row', 'column']
            dims = [np.arange(self.value.shape[0]),
                    np.arange(self.value.shape[1]),
                    np.arange(self.value.shape[2])]
        else:
            raise ValueError('not implemented for ndim > 3')

        # add dimensions as a namedtuple
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(dims)

    def copy(self):
        """
        return a deep copy of the current object
        """
        import copy

        return copy.deepcopy(self)

    def set_coverage(self, coverage, force=False):
        """
        Set the coverage attribute, as (coverageStart, coverageEnd)
        Both elements are expected to be datatime objects.
        Overwrite when force is true
        """
        if self.coverage is None or force:
            self.coverage = coverage

    def set_units(self, units, force=False):
        """
        Set the units attribute, overwrite when force is true
        """
        if self.units is None or force:
            self.units = units

    def set_fillvalue(self):
        """
        Set fillvalue to KNMI undefined
        """
        if self.value.dtype == np.float64 or self.value.dtype == np.float32:
            if self.fillvalue is None or self.fillvalue == 0.:
                self.fillvalue = float.fromhex('0x1.ep+122')

    def set_long_name(self, name, force=False):
        """
        Set the long_name attribute, overwrite when force is true
        """
        if force or self.long_name:
            self.long_name = name

    def fill_as_nan(self):
        """
        Replace fillvalues in data with NaN's

        Works only on datasets with HDF5 datatype 'float' or 'datapoints'
        """
        if self.fillvalue == float.fromhex('0x1.ep+122'):
            self.value[(self.value == self.fillvalue)] = np.nan
            if self.error is not None:
                self.error[(self.error == self.fillvalue)] = np.nan

    def concatenate(self, msm, axis=0):
        """
        Concatenate two measurement datasets, the current with another.

        Parameters
        ----------
        msm   :  pys5p.S5Pmsm
           an S5Pmsm object
        axis  : int, optional
           The axis along which the arrays will be joined. Default is 0.

        Returns
        -------
        The data of the new dataset is concatenated to the existing data along
        an existing axis. The affected coordinate is also extended.

        Note:
         - The arrays must have the same shape, except in the dimension
        corresponding to axis (the first, by default).
        """
        assert self.name == os.path.basename(msm.name), \
            '*** Fatal, only combine the same dataset from different bands'

        if self.error is None and msm.error is None:
            datapoint = False
        elif  self.error is not None and msm.error is not None:
            datapoint = True
        else:
            raise RuntimeError("S5Pmsm: combining non-datapoint and datapoint")

        # all but the last 2 dimensions have to be equal
        assert self.value.shape[:-2] == msm.value.shape[:-2]

        if axis == 0:
            self.value = np.concatenate((self.value, msm.value), axis=axis)
            if datapoint:
                self.error = np.concatenate((self.error, msm.error),
                                            axis=axis)
        elif axis == 1:
            if self.value.shape[0] == msm.value.shape[0]:
                self.value = np.concatenate((self.value, msm.value), axis=axis)
                if datapoint:
                    self.error = np.concatenate((self.error, msm.error),
                                                axis=axis)
            else:
                self.value = np.concatenate(pad_rows(self.value, msm.value),
                                            axis=axis)
                if datapoint:
                    self.error = np.concatenate(pad_rows(self.error, msm.error),
                                                axis=axis)
        elif axis == 2:
            if self.value.shape[1] == msm.value.shape[1]:
                self.value = np.concatenate((self.value, msm.value), axis=axis)
                if datapoint:
                    self.error = np.concatenate((self.error, msm.error),
                                                axis=axis)
            else:
                self.value = np.concatenate(pad_rows(self.value, msm.value),
                                            axis=axis)
                if datapoint:
                    self.error = np.concatenate(pad_rows(self.error, msm.error),
                                                axis=axis)
        else:
            raise ValueError("S5Pmsm: implemented for ndim <= 3")

        # now extent coordinate of the fastest axis
        key = self.coords._fields[axis]
        if msm.coords[axis][0] == 0:
            dims = np.concatenate((self.coords[axis],
                                   len(self.coords[axis]) + msm.coords[axis]))
        else:
            dims = np.concatenate((self.coords[axis], msm.coords[axis]))
        self.coords = self.coords._replace(**{key : dims})

    def nanmedian(self, data_sel=None, *, axis=0, keepdims=False):
        """
        Returns median of the data in the S5Pmsm

        Parameters
        ----------
        data_sel  :  numpy slice, optional
           A numpy slice generated for example numpy.s_. Can be used to skip
           the first and/or last frame
        axis      : int, optional
           Axis or axes along which the medians are computed. Default is 0.
        keepdims  : bool, optional
           If this is set to True, the axes which are reduced are left in the
           result as dimensions with size one. With this option, the result
           will broadcast correctly against the original arr.

        Returns
        -------
        S5Pmsm object with data (value & error) is replaced by their medians
        and the coordinates are adjusted
        """
        if data_sel is None:
            if self.error is not None:
                self.error = np.nanmedian(self.error,
                                          axis=axis, keepdims=keepdims)
            else:
                self.error = np.nanstd(self.value, ddof=1,
                                       axis=axis, keepdims=keepdims)
            self.value = np.nanmedian(self.value, axis=axis, keepdims=keepdims)
        else:
            if self.error is not None:
                self.error = np.nanmedian(self.error[data_sel],
                                          axis=axis, keepdims=keepdims)
            else:
                self.error = np.nanstd(self.value[data_sel], ddof=1,
                                       axis=axis, keepdims=keepdims)
            self.value = np.nanmedian(self.value[data_sel],
                                      axis=axis, keepdims=keepdims)

        # adjust the coordinates
        if keepdims:
            key = self.coords._fields[axis]
            if self.coords[axis][0] == 0:
                dims = [0]
            else:
                dims = np.median(self.coords[axis], keepdims=keepdims)
            self.coords = self.coords._replace(**{key : dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+1):
                if ii != axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

    def nanpercentile(self, vperc, data_sel=None, axis=0, keepdims=False):
        """
        Returns median of the data in the S5Pmsm

        Parameters
        ----------
        vperc       :  list
           range to normalize luminance data between percentiles min and max of
           array data.
        data_sel  :  numpy slice
           A numpy slice generated for example numpy.s_. Can be used to skip
           the first and/or last frame
        axis      : int, optional
           Axis or axes along which the medians are computed. Default is 0.
        keepdims  : bool, optional
           If this is set to True, the axes which are reduced are left in the
           result as dimensions with size one. With this option, the result
           will broadcast correctly against the original arr.

        Returns
        -------
        S5Pmsm object where value is replaced by its median and error by the
        minimum and maximum percentiles. The coordinates are adjusted.

        You should atleast supply one percentile and atmost three.
         vperc is instance 'int' or len(vperc) == 1:
             'value' is replaced by its (nan-)percentile vperc
             'error' is unchanged
         len(vperc) == 2:
             'value' is replaced by its (nan-)median
             'error' is replaced by percentile('value', (min(vperc), max(vperc))
         len(vperc) == 3:
             'value' is replaced by percentile('value', vperc[1])
             'error' is replaced by percentile('value', (vperc[0], vperc[2]))
        """
        if isinstance(axis, int):
            axis = (axis,)

        if isinstance(vperc, int):
            vperc = (vperc,)
        elif len(vperc) == 2:
            vperc += (50,)
        assert len(vperc) == 1 or len(vperc) == 3

        if data_sel is None:
            if self.value.size <= 1 or self.value.ndim <= max(axis):
                return
            perc = np.nanpercentile(self.value, vperc,
                                    axis=axis, keepdims=keepdims)
        else:
            if (self.value[data_sel].size <= 1
                or self.value[data_sel].ndim <= max(axis)):
                return
            perc = np.nanpercentile(self.value[data_sel], vperc,
                                    axis=axis, keepdims=keepdims)
        if len(vperc) == 3:
            vperc = tuple(sorted(vperc))
            self.value = perc[1, ...]
            self.error = [perc[0, ...],
                          perc[2, ...]]
        else:
            self.value = perc[0, ...]

        # adjust the coordinates
        if keepdims:
            key = self.coords._fields[axis]
            if self.coords[axis][0] == 0:
                dims = [0]
            else:
                dims = np.median(self.coords[axis], keepdims=keepdims)
            self.coords = self.coords._replace(**{key : dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+len(axis)):
                if ii not in axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

    def biweight(self, data_sel=None, axis=0, keepdims=False):
        """
        Returns biweight median of the data in the S5Pmsm

        Parameters
        ----------
        data_sel  :  numpy slice
           A numpy slice generated for example numpy.s_. Can be used to skip
           the first and/or last frame
        axis  : int, optional
           Axis or axes along which the medians are computed. Default is 0.
        keepdims  : bool, optional
           If this is set to True, the axes which are reduced are left in the
           result as dimensions with size one. With this option, the result
           will broadcast correctly against the original arr.

        Returns
        -------
        S5Pmsm object with data (value & error) is replaced by their medians
        and the coordinates are adjusted
        """
        from .biweight import biweight

        if data_sel is None:
            if self.error is not None:
                self.value = biweight(self.value, axis=axis)
                self.error = biweight(self.error, axis=axis)
            else:
                (self.value, self.error) = biweight(self.value,
                                                    axis=axis, spread=True)
        else:
            if self.error is not None:
                self.value = biweight(self.value[data_sel], axis=axis)
                self.error = biweight(self.error[data_sel], axis=axis)
            else:
                (self.value, self.error) = biweight(self.value[data_sel],
                                                    axis=axis, spread=True)
        if keepdims:
            self.value = np.expand_dims(self.value, axis=axis)
            self.error = np.expand_dims(self.error, axis=axis)

        # adjust the coordinates
        if keepdims:
            key = self.coords._fields[axis]
            if self.coords[axis][0] == 0:
                dims = [0]
            else:
                dims = np.median(self.coords[axis], keepdims=keepdims)
            self.coords = self.coords._replace(**{key : dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+1):
                if ii != axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

    def transpose(self):
        """
        Transpose data and coordinates of S5Pmsm object
        """
        if self.value.ndim <= 1:
            return

        if self.error is not None:
            self.error = np.transpose(self.error)
        self.value = np.transpose(self.value)

        keys = []
        dims = []
        for ii in range(self.value.ndim):
            keys.append(self.coords._fields[ii])
            dims.append(self.coords[ii][:])
        tmp = keys[1]
        keys[1] = keys[0]
        keys[0] = tmp
        tmp = dims[1]
        dims[1] = dims[0]
        dims[0] = tmp
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(dims)
