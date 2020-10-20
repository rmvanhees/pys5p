"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

This module contains the helper functions for the class S5Pplot

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from collections import OrderedDict
from datetime import datetime

import matplotlib as mpl
import numpy as np

from pys5p.s5p_msm import S5Pmsm


# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side
    from a prescribed midpoint value)

    e.g. im = ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,
                                                       vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        xx, yy = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, xx, yy), np.isnan(value))


def blank_legend_key():
    """
    Show only text in matplotlib legenda, no key
    """
    return mpl.patches.Rectangle((0, 0), 0, 0, fill=False,
                                 edgecolor='none', visible=False)


def check_data2d(method, data):
    """
    Make sure that the input data is 2-dimensional and contains valid data
    """
    # data is stored in a numpy.ndarray
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('data must be two dimensional')
        if np.all(np.isnan(data)):
            raise ValueError('data must contain valid data')
        return

    # data must be stored in a pys5p.S5Pmsm object
    if not isinstance(data, S5Pmsm):
        raise TypeError('data not an numpy.ndarray or pys5p.S5Pmsm')

    if data.value.ndim != 2:
        raise ValueError('data must be two dimensional')
    if np.all(np.isnan(data.value)):
        raise ValueError('data must contain valid data')
    if method == 'ratio_unc':
        if data.error.ndim != 2:
            raise ValueError('data must be two dimensional')
        if np.all(np.isnan(data.error)):
            raise ValueError('data must contain valid error estimates')


def data_for_trend2d(data_in, coords=None, delta_time=None):
    """
    Prepare data to display the averaged column/row as function of time

    Parameters
    ----------
    data :  numpy.ndarray or pys5p.S5Pmsm
       Object holding measurement data and attributes
    coords :  dictionary, optional
       Dictionary with the names and data of each dimension, where the keys of
       the  dictionary are the names of the dimensions.
       Required when data is a numpy array (otherwise ignored)
    delta_time : int, optional
       Expected resolution of the time-axis.
       Defined in whole seconds or number of orbit revolutions.
    missing_data: 'NaN' or 'previous'
       Default: NaN

    Returns
    -------
    pys5p.S5Pmsm
    """
    try:
        check_data2d('data', data_in)
    except Exception as exc:
        raise RuntimeError('invalid input-data provided') from exc

    if isinstance(data_in, np.ndarray):
        if coords is None:
            raise KeyError('coords needs to be provided when data is a ndarray')
        msm = S5Pmsm(data_in)
        msm.set_coords(list(coords.values()), list(coords.keys()))
    else:
        msm = data_in

    # determine time-axis
    (ylabel, xlabel) = msm.coords._fields
    if ylabel in ('orbit', 'time'):
        time_axis = 0
        time_data = msm.coords[0]
    elif xlabel in ('orbit', 'time'):
        time_axis = 1
        time_data = msm.coords[1]
    else:
        raise RuntimeError('can not find the time-axis')

    if not np.issubdtype(time_data.dtype, np.integer):
        raise ValueError('time-coordinate not of type integer')
    if not np.all(time_data[1:] > time_data[:-1]):
        indx = np.where(time_data[1:] <= time_data[:-1])[0]
        raise ValueError('time-coordinate not increasing at {}'.format(indx))

    # determine delta_time with greatest common divisor (numpy v1.15+)
    if delta_time is None:
        delta_time = np.gcd.reduce(np.diff(time_data))

    time_dim = 1 + (time_data.max() - time_data.min()) // delta_time
    if time_dim == time_data.size:
        return msm

    time_data_full = np.arange(time_data.min(),
                               time_data.max() + delta_time,
                               delta_time)
    mask = np.in1d(time_data_full, time_data)
    if time_axis == 0:
        data_full = np.empty((time_dim, msm.value.shape[1]),
                             dtype=msm.value.dtype)
        data_full[mask, :] = msm.value
        for ii in np.where(~mask)[0]:
            data_full[ii, :] = data_full[ii-1, :]
    else:
        data_full = np.empty((msm.value.shape[0], time_dim),
                             dtype=msm.value.dtype)
        data_full[:, mask] = msm.value
        for ii in np.where(~mask)[0]:
            data_full[ii, :] = data_full[ii-1, :]

    msm.value = data_full
    return msm


def fill_fig_info(fig_info, key: str, value: str):
    """
    Fill ordered dictionary with annotation for the current figure

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    """
    if fig_info is None:
        return OrderedDict({key: value})

    fig_info.update({key: value})
    return fig_info


def get_fig_coords(data_in):
    """
    get labels of the X/Y axis

    Returns
    -------
    dict: {'X': {'label': value, 'data': value},
           'Y': {'label': value, 'data': value}}
    """
    if isinstance(data_in, S5Pmsm):
        ylabel, xlabel = data_in.coords._fields
        ydata = data_in.coords[0]
        xdata = data_in.coords[1]
    else:
        ylabel, xlabel = ['row', 'column']
        ydata = np.arange(data_in.shape[0])
        xdata = np.arange(data_in.shape[1])

    return {'X': {'label': xlabel, 'data': xdata},
            'Y': {'label': ylabel, 'data': ydata}}


def fig_info_as_str(dict_info, count=False):
    """
    Return fig_info as a string
    """
    info_str = ""
    info_lines = 1
    if dict_info is not None:
        for key in dict_info:
            if isinstance(dict_info[key], (float, np.float32)):
                info_str += "{} : {:.5g}".format(key, dict_info[key])
            else:
                info_str += "{} : {}".format(key, dict_info[key])
            info_str += '\n'
            info_lines += 1
    # add timestamp
    info_str += 'created : {}'.format(
        datetime.utcnow().isoformat(timespec='seconds'))

    if count:
        return (info_str, count)

    return info_str


def get_xdata(xdata, use_steps):
    """
    The X-coordinate from the data in object msm is checked for data gaps.
    The data of the X-coordinate is extended to avoid interpolation over
    missing data.

    A list of indices to all data gaps is also returned which can be used to
    update the Y-coordinate.
    """
    # this function only works when the X-coordinate is of type integer and
    # increasing
    if not np.issubdtype(xdata.dtype, np.integer):
        raise ValueError('x-coordinate not of type integer')
    if not np.all(xdata[1:] > xdata[:-1]):
        indx = np.where(xdata[1:] <= xdata[:-1])[0]
        raise ValueError('x-coordinate not increasing at {}'.format(indx))

    xstep = np.gcd.reduce(np.diff(xdata))
    gap_list = 1 + np.where(np.diff(xdata) > xstep)[0]
    for indx in reversed(gap_list):
        xdata = np.insert(xdata, indx, xdata[indx])
        xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)
        xdata = np.insert(xdata, indx, xdata[indx-1] + xstep)

    if use_steps:
        xdata = np.append(xdata, xdata[-1] + xstep)

    return (xdata, gap_list)
