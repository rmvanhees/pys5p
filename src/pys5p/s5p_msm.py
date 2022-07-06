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
  error            | h5_dset.value['error'] | None
                   | or None                |
  coords           | h5_dset.dims           | [[['time',] 'row',] 'column']
  units            | attrs['units']         | None
  long_name        | attrs['long_name']     | ''
  fillvalue        | h5_dset.fillvalue      | None
  coverage         | None                   | None

Limited to 3 dimensions

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from collections import namedtuple
from copy import deepcopy
from pathlib import PurePath

from h5py import Dataset
import numpy as np

from moniplot.biweight import biweight


# - local functions --------------------------------
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


# - class definition -------------------------------
class S5Pmsm():
    """
    Definition of class S5Pmsm which contains the data of a HDF5 dataset,
    including its attributes (CF convension)

    Attributes
    ----------
    name : string
       Name of the original HDF5 dataset
    value : ndarray
       Dataset values
    error : ndarray, optional
       Dataset uncertainties or error estimates
    coords : OrderDict
       Dataset coordinates (HDF5/netCDF4 dimensions)
    coverage : string, optional
       Dataset coverage start and end
    units : string
       Data units
    long_name : string
       Dataset long name
    fillvalue : float
       Value of undefined or missings data values

    Methods
    -------
    copy()
       Return a deep copy of the current object.
    set_coords(coords_data, coords_name=None)
       Set coordinates of data.
    set_coverage(coverage, force=False)
       Set the coverage attribute, as (coverageStart, coverageEnd),
    set_units(units, force=False)
       Set the units attribute.
    set_fillvalue()
       Set fillvalue of floating-point data.
    set_long_name(name, force=False)
       Set the long_name attribute.
    fill_as_nan()
       Replace missing floating-point data with NaN's.
    sort(axis=0)
       Sort dataset according a coordinate axis.
    concatenate(msm, axis=0)
       Concatenate the data of a S5Pmsm to the current S5Pmsm object.
    nanpercentile(vperc, data_sel=None, axis=0, keepdims=False)
       Returns percentile(s) of the data in the S5Pmsm object.
    biweight(data_sel=None, axis=0, keepdims=False)
       Returns biweight median of the data in the S5Pmsm object.
    nanmedian(data_sel=None, axis=0, keepdims=False)
       Returns S5Pmsm object containing median & standard deviation of the
        original data.
    nanmean(data_sel=None, axis=0, keepdims=False)
       Returns S5Pmsm object containing mean & standard deviation of the
        original data.
    transpose()
       Transpose data and coordinates of an S5Pmsm object.

    Notes
    -----

    Examples
    --------
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

        if isinstance(dset, Dataset):
            self.__from_h5_dset(dset, data_sel, datapoint)
        else:
            self.__from_ndarray(dset, data_sel)

    def __repr__(self) -> str:
        res = []
        for key, value in self.__dict__.items():
            if key.startswith('__'):
                continue
            if isinstance(value, np.ndarray):
                res.append(f'{key}: {value.shape}')
            else:
                res.append(f'{key}: {value}')

        return '\n'.join(res)

    def __from_h5_dset(self, h5_dset, data_sel, datapoint):
        """
        initialize S5Pmsm object from h5py dataset
        """
        self.name = PurePath(h5_dset.name).name

        # copy dataset values (and error) to object
        if data_sel is None:
            if datapoint:
                self.value = h5_dset['value']
                self.error = h5_dset['error']
            else:
                self.value = h5_dset[...]
        else:
            # we need to keep all dimensions to get the dimensions
            # of the output data right
            if datapoint:
                self.value = h5_dset['value'][data_sel]
                self.error = h5_dset['error'][data_sel]
                if isinstance(data_sel, tuple):
                    for ii, elmnt in enumerate(data_sel):
                        if isinstance(elmnt, (int, np.int64)):
                            self.value = np.expand_dims(self.value, axis=ii)
                            self.error = np.expand_dims(self.error, axis=ii)
            else:
                self.value = h5_dset[data_sel]
                if isinstance(data_sel, tuple):
                    for ii, elmnt in enumerate(data_sel):
                        if isinstance(elmnt, (int, np.int64)):
                            self.value = np.expand_dims(self.value, axis=ii)

        # set default dimension names
        if h5_dset.ndim == 1:
            keys_default = ['column']
        elif h5_dset.ndim == 2:
            keys_default = ['row', 'column']
        elif h5_dset.ndim == 3:
            keys_default = ['time', 'row', 'column']
        else:
            raise ValueError('not implemented for ndim > 3')

        # copy all dimensions with size longer then 1
        keys = []
        dims = []
        for ii in range(h5_dset.ndim):
            if self.value.shape[ii] == 1:
                continue

            if len(h5_dset.dims[ii]) != 1:   # bug in some KMNI HDF5 files
                keys.append(keys_default[ii])
                dims.append(np.arange(self.value.shape[ii]))
            elif self.value.shape[ii] == h5_dset.shape[ii]:
                buff = PurePath(h5_dset.dims[ii][0].name).name
                if len(buff.split()) > 1:
                    buff = buff.split()[0]
                keys.append(buff)
                if h5_dset.dims[ii][0][:].size == h5_dset.shape[ii]:
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)
                else:                          # bug in some KMNI HDF5 files
                    buff = np.arange(h5_dset.shape[ii])
                dims.append(buff)
            else:
                buff = PurePath(h5_dset.dims[ii][0].name).name
                if len(buff.split()) > 1:
                    buff = buff.split()[0]
                keys.append(buff)
                if h5_dset.dims[ii][0][:].size == h5_dset.shape[ii]:
                    buff = h5_dset.dims[ii][0][:]
                    if np.all(buff == 0):
                        buff = np.arange(buff.size)
                else:                          # bug in some KMNI HDF5 files
                    buff = np.arange(h5_dset.shape[ii])

                if isinstance(data_sel, slice):
                    dims.append(buff[data_sel])
                elif len(data_sel) == h5_dset.ndim:
                    dims.append(buff[data_sel[ii]])
                elif not isinstance(data_sel, tuple):
                    dims.append(buff[data_sel])
                elif ii > len(data_sel):
                    dims.append(buff[data_sel[-1]])
                else:
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
            if isinstance(h5_dset.attrs['units'], np.ndarray):
                if h5_dset.attrs['units'].size == 1:
                    self.units = h5_dset.attrs['units'][0]
                    if isinstance(self.units, bytes):
                        self.units = self.units.decode('ascii')
                else:
                    self.units = h5_dset.attrs['units']
                    if isinstance(self.units[0], bytes):
                        self.units = self.units.astype(str)
            else:
                self.units = h5_dset.attrs['units']
                if isinstance(self.units, bytes):
                    self.units = self.units.decode('ascii')

        # copy its long_name
        if 'long_name' in h5_dset.attrs:
            if isinstance(h5_dset.attrs['long_name'], bytes):
                self.long_name = h5_dset.attrs['long_name'].decode('ascii')
            else:
                self.long_name = h5_dset.attrs['long_name']

    def __from_ndarray(self, data, data_sel):
        """
        initialize S5Pmsm object from a ndarray
        """
        # copy dataset values (and error) to object
        if data_sel is None:
            self.value = np.squeeze(data)
        else:
            self.value = np.squeeze(data[data_sel])

        # define coordinates
        dims = [np.arange(sz) for sz in self.value.shape]
        try:
            self.set_coords(dims, coords_name=None)
        except Exception as exc:
            raise RuntimeError('failed to set the coordinates') from exc

    def copy(self):
        """
        return a deep copy of the current object
        """
        return deepcopy(self)

    def set_coords(self, coords_data, coords_name=None):
        """
        Set coordinates of data

        Parameters
        ----------
        coords_data  :  list of ndarrays
          list with coordinates data for each dimension
        coords_name  :  list of strings
          list with the names of each dimension
        """
        if coords_name is None:
            if len(coords_data) == 1:
                keys = ['column']
            elif len(coords_data) == 2:
                keys = ['row', 'column']
            elif len(coords_data) == 3:
                keys = ['time', 'row', 'column']
            else:
                raise ValueError('not implemented for ndim > 3')
        else:
            if isinstance(coords_name, str):
                keys = [coords_name]
            else:
                keys = coords_name

        # add dimensions as a namedtuple
        coords_namedtuple = namedtuple('Coords', keys)
        self.coords = coords_namedtuple._make(coords_data)

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
        if np.issubdtype(self.value.dtype, np.floating):
            if self.fillvalue is None or self.fillvalue == 0.:
                self.fillvalue = float.fromhex('0x1.ep+122')

    def set_long_name(self, name, force=False):
        """
        Set the long_name attribute, overwrite when force is true
        """
        if force or not self.long_name:
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

    def sort(self, axis=0):
        """
        Sort data and its coordinate along a given axis

        Parameters
        ----------
        axis  : int, optional
           axis for which the array will be sorted. Default is 0.
        """
        if not isinstance(axis, int):
            raise TypeError('axis not an integer')
        if not 0 <= axis < self.value.ndim:
            raise ValueError('axis out-of-range')

        indx = np.argsort(self.coords[axis][:])
        self.coords[axis][:] = self.coords[axis][indx]

        if axis == 0:
            self.value = self.value[indx, ...]
            if self.error is not None:
                if isinstance(self.error, list):
                    self.error = (self.error[0][indx, ...],
                                  self.error[1][indx, ...])
                else:
                    self.error = self.error[indx, ...]
        elif axis == 1:
            self.value = self.value[:, indx, ...]
            if self.error is not None:
                if isinstance(self.error, list):
                    self.error = (self.error[0][:, indx, :],
                                  self.error[1][:, indx, :])
                else:
                    self.error = self.error[:, indx, :]
        elif axis == 2:
            self.value = self.value[:, :, indx]
            if self.error is not None:
                if isinstance(self.error, list):
                    self.error = (self.error[0][:, :, indx],
                                  self.error[1][:, :, indx])
                else:
                    self.error = self.error[:, :, indx]
        else:
            raise ValueError("S5Pmsm: implemented for ndim <= 3")

    def concatenate(self, msm, axis=0):
        """
        Concatenate two measurement datasets, the current with another.

        Parameters
        ----------
        msm   :  pys5p.S5Pmsm
           an S5Pmsm object
        axis  : int, optional
           The axis for which the array will be joined. Default is 0.

        Returns
        -------
        The data of the new dataset is concatenated to the existing data along
        an existing axis. The affected coordinate is also extended.

        Note:
         - The arrays must have the same shape, except in the dimension
        corresponding to axis (the first, by default).
        """
        if self.name != PurePath(msm.name).name:
            raise TypeError('combining dataset with different name')

        if self.error is None and msm.error is None:
            datapoint = False
        elif self.error is not None and msm.error is not None:
            datapoint = True
        else:
            raise RuntimeError("S5Pmsm: combining non-datapoint and datapoint")

        # all but the last 2 dimensions have to be equal
        if self.value.shape[:-2] != msm.value.shape[:-2]:
            raise TypeError('all but the last 2 dimensions should be equal')

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
                    self.error = np.concatenate(
                        pad_rows(self.error, msm.error), axis=axis)
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
                    self.error = np.concatenate(
                        pad_rows(self.error, msm.error), axis=axis)
        else:
            raise ValueError("S5Pmsm: implemented for ndim <= 3")

        # now extent coordinate of the fastest axis
        key = self.coords._fields[axis]
        if msm.coords[axis][0] == 0:
            dims = np.concatenate((self.coords[axis],
                                   len(self.coords[axis]) + msm.coords[axis]))
        else:
            dims = np.concatenate((self.coords[axis], msm.coords[axis]))
        self.coords = self.coords._replace(**{key: dims})
        return self

    def nanpercentile(self, vperc, data_sel=None, axis=0, keepdims=False):
        """
        Returns percentile(s) of the data in the S5Pmsm

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
        S5Pmsm object with the original data replaced by the percentiles along
        one of the axis, see below. The coordinates are adjusted, accordingly.

        You should atleast supply one percentile and atmost three.
         vperc is instance 'int' or len(vperc) == 1:
             'value' is replaced by its (nan-)percentile vperc
             'error' is unchanged
         len(vperc) == 2:
             'vperc' is sorted
             'value' is replaced by its (nan-)median
             'error' is replaced by percentile('value', (vperc[0], vperc[1]))
         len(vperc) == 3:
             'vperc' is sorted
             'value' is replaced by percentile('value', vperc[1])
             'error' is replaced by percentile('value', (vperc[0], vperc[2]))
        """
        if isinstance(axis, int):
            axis = (axis,)

        if isinstance(vperc, int):
            vperc = (vperc,)
        else:
            if len(vperc) == 2:
                vperc += (50,)
            # make sure that the values are sorted
            vperc = tuple(sorted(vperc))

        if len(vperc) != 1 and len(vperc) != 3:
            raise TypeError('dimension vperc must be 1 or 3')

        if data_sel is None:
            if self.value.size <= 1 or self.value.ndim <= max(axis):
                return self
            perc = np.nanpercentile(self.value, vperc,
                                    axis=axis, keepdims=keepdims)
        else:
            if self.value[data_sel].size <= 1 \
               or self.value[data_sel].ndim <= max(axis):
                return self
            perc = np.nanpercentile(self.value[data_sel], vperc,
                                    axis=axis, keepdims=keepdims)
        if len(vperc) == 3:
            self.value = perc[1, ...]
            self.error = [perc[0, ...], perc[2, ...]]
        else:
            self.value = perc[0, ...]

        # adjust the coordinates
        if keepdims:
            key = self.coords._fields[axis]
            if self.coords[axis][0] == 0:
                dims = [0]
            else:
                dims = np.median(self.coords[axis], keepdims=keepdims)
            self.coords = self.coords._replace(**{key: dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+len(axis)):
                if ii not in axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

        return self

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
        S5Pmsm object with its data (value & error) replaced by its biweight
        medians along one axis. The coordinates are adjusted, accordingly.
        """
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
            self.coords = self.coords._replace(**{key: dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+1):
                if ii != axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

        return self

    def nanmedian(self, data_sel=None, axis=0, keepdims=False):
        """
        Returns S5Pmsm object containing median & standard deviation of the
        original data

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
        S5Pmsm object with its data (value & error) replaced by its nanmedian
        and standard deviation along one axis.
        The coordinates are adjusted, accordingly.
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
            self.coords = self.coords._replace(**{key: dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+1):
                if ii != axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

        return self

    def nanmean(self, data_sel=None, axis=0, keepdims=False):
        """
        Returns S5Pmsm object containing mean & standard deviation of the
        original data

        Parameters
        ----------
        data_sel  :  numpy slice, optional
           A numpy slice generated for example numpy.s_. Can be used to skip
           the first and/or last frame
        axis      : int, optional
           Axis or axes along which the mean are computed. Default is 0.
        keepdims  : bool, optional
           If this is set to True, the axes which are reduced are left in the
           result as dimensions with size one. With this option, the result
           will broadcast correctly against the original arr.

        Returns
        -------
        S5Pmsm object with its data (value & error) replaced by its nanmean and
        standard deviation along one axis.
        The coordinates are adjusted, accordingly.
        """
        if data_sel is None:
            if self.error is not None:
                self.error = np.nanmean(self.error,
                                        axis=axis, keepdims=keepdims)
            else:
                self.error = np.nanstd(self.value, ddof=1,
                                       axis=axis, keepdims=keepdims)
            self.value = np.nanmean(self.value, axis=axis, keepdims=keepdims)
        else:
            if self.error is not None:
                self.error = np.nanmean(self.error[data_sel],
                                        axis=axis, keepdims=keepdims)
            else:
                self.error = np.nanstd(self.value[data_sel], ddof=1,
                                       axis=axis, keepdims=keepdims)
            self.value = np.nanmean(self.value[data_sel],
                                    axis=axis, keepdims=keepdims)

        # adjust the coordinates
        if keepdims:
            key = self.coords._fields[axis]
            if self.coords[axis][0] == 0:
                dims = [0]
            else:
                dims = np.mean(self.coords[axis], keepdims=keepdims)
            self.coords = self.coords._replace(**{key: dims})
        else:
            keys = []
            dims = []
            for ii in range(self.value.ndim+1):
                if ii != axis:
                    keys.append(self.coords._fields[ii])
                    dims.append(self.coords[ii][:])
            coords_namedtuple = namedtuple('Coords', keys)
            self.coords = coords_namedtuple._make(dims)

        return self

    def transpose(self):
        """
        Transpose data and coordinates of an S5Pmsm object
        """
        if self.value.ndim <= 1:
            return self

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

        return self
