"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The classe OCMio provide read access to Tropomi on-ground calibration products
(Lx)

Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from moniplot.biweight import biweight

# - global parameters ------------------------------


# - local functions --------------------------------
def band2channel(dict_a, dict_b, mode=None):
    """
    Store data from a dictionary as returned by get_msm_data to a ndarray

    Parameters
    ----------
    dict_a      :  dictionary
    dict_b      :  dictionary
    mode        :  list ['combined', 'mean', 'median', 'biweight']
        'combined' will combine data using np.concatenate((data_a, data_b),
                                                          axis=data_a.ndim-1)
        'mean' is calculated using np.nanmean(data, axis=0)
        'median' is calculated using np.nanmedian(data, axis=0)
        'biweight' is calculated using pys5p.biweight(data, axis=0)
        default is None

    Returns
    -------
    out  :  ndarray
        Data from dictionary stored in a numpy array

    Notes
    -----

    Examples
    --------
    >>> data = ocm.band2channel(dict_a, dict_b, mode=['combined', 'median'])
    >>>
    """
    if mode is None:
        mode = []

    data_a = None
    for key in sorted(dict_a):
        buff = dict_a[key][...]

        if data_a is None:
            data_a = buff
        else:
            data_a = np.vstack((data_a, buff))

    if data_a is not None:
        if 'mean' in mode:
            data_a = np.nanmean(data_a, axis=0)
        elif 'median' in mode:
            data_a = np.nanmedian(data_a, axis=0)
        elif 'biweight' in mode:
            data_a = biweight(data_a, axis=0)

    if dict_b is None:
        return data_a

    data_b = None
    for key in sorted(dict_b):
        buff = dict_b[key][...]

        if data_b is None:
            data_b = buff
        else:
            data_b = np.vstack((data_b, buff))

    if data_b is not None:
        if 'mean' in mode:
            data_b = np.nanmean(data_b, axis=0)
        elif 'median' in mode:
            data_b = np.nanmedian(data_b, axis=0)
        elif 'biweight' in mode:
            data_b = biweight(data_b, axis=0)

    if 'combined' in mode:
        return np.concatenate((data_a, data_b), axis=data_a.ndim-1)

    return (data_a, data_b)


# - class definition -------------------------------
class OCMio():
    """
    This class should offer all the necessary functionality to read Tropomi
    on-ground calibration products (Lx)

    Attributes
    ----------
    fid : h5py.File
    filename : string
    band : string

    Methods
    -------
    close()
       Close resources.
    get_processor_version()
       Returns version of the L01b processor used to generate this product.
    get_coverage_time()
       Returns start and end of the measurement coverage time.
    get_attr(attr_name)
       Obtain value of an HDF5 file attribute.
    get_ref_time()
       Returns reference start time of measurements.
    get_delta_time()
       Returns offset from the reference start time of measurement.
    get_instrument_settings()
       Returns instrument settings of measurement.
    get_gse_stimuli()
       Returns GSE stimuli parameters.
    get_exposure_time()
       Returns the exact pixel exposure time of the measurements.
    get_housekeeping_data()
       Returns housekeeping data of measurements.
    select(ic_id=None, *, msm_grp=None)
       Select a measurement as BAND%/ICID_<ic_id>_GROUP_%
    get_msm_attr(msm_dset, attr_name)
       Returns attribute of measurement dataset 'msm_dset'
    get_msm_data(msm_dset, fill_as_nan=True, frames=None, columns=None)
       Returns data of measurement dataset 'msm_dset'
    read_direct_msm(msm_dset, dest_sel=None, dest_dtype=None,
                    fill_as_nan=False)
       The faster implementation of class method 'get_msm_data'.

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, ocm_product):
        """
        Initialize access to an OCAL Lx product

        Parameters
        ----------
        ocm_product :  string
           Full path to on-ground calibration measurement

        """
        if not Path(ocm_product).is_file():
            raise FileNotFoundError(f'{ocm_product} does not exist')

        # initialize class-attributes
        self.__msm_path = None
        self.band = None
        self.filename = ocm_product

        # open OCM product as HDF5 file
        self.fid = h5py.File(ocm_product, "r")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}({self.filename!r})'

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    # def __del__(self):
    #    """
    #    called when the object is destroyed
    #    """
    #    self.close()

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self):
        """
        Close resources
        """
        self.band = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    # ---------- RETURN VERSION of the S/W ----------
    # ---------- Functions that work before MSM selection ----------
    def get_processor_version(self):
        """
        Returns version of the L01b processor
        """
        res = self.fid.attrs['processor_version']
        if isinstance(res, bytes):
            # pylint: disable=no-member
            res = res.decode('ascii')
        return res

    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        t_bgn = self.fid.attrs['time_coverage_start']
        if isinstance(t_bgn, bytes):
            # pylint: disable=no-member
            t_bgn = t_bgn.decode('ascii')

        t_end = self.fid.attrs['time_coverage_end']
        if isinstance(t_end, bytes):
            # pylint: disable=no-member
            t_end = t_end.decode('ascii')
        return (t_bgn, t_end)

    def get_attr(self, attr_name):
        """
        Obtain value of an HDF5 file attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        """
        if attr_name in self.fid.attrs.keys():
            return self.fid.attrs[attr_name]

        return None

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'GEODATA'))]
            res[msm] = datetime(2010, 1, 1, 0, 0, 0)
            res[msm] += timedelta(seconds=int(sgrp['time'][0]))

        return res

    def get_delta_time(self):
        """
        Returns offset from the reference start time of measurement
        """
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'GEODATA'))]
            res[msm] = sgrp['delta_time'][:].astype(int)

        return res

    def get_instrument_settings(self):
        """
        Returns instrument settings of measurement
        """
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['instrument_settings'])

        return res

    def get_gse_stimuli(self):
        """
        Returns GSE stimuli parameters
        """
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['gse_stimuli'])

        return res

    def get_exposure_time(self):
        """
        Returns the exact pixel exposure time of the measurements
        """
        if not self.__msm_path:
            return None

        grp = self.fid[f'BAND{self.band}']
        msm = self.__msm_path[0]  # all measurement sets have the same ICID
        sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
        instr = np.squeeze(sgrp['instrument_settings'])

        if int(self.band) > 6:
            return 1.25e-6 * (65540 - instr['int_delay'] + instr['int_hold'])

        return instr['exposure_time']

    def get_housekeeping_data(self):
        """
        Returns housekeeping data of measurements
        """
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['housekeeping_data'])

        return res

    # -------------------------
    def select(self, ic_id=None, *, msm_grp=None):
        """
        Select a measurement as BAND%/ICID_<ic_id>_GROUP_%

        Parameters
        ----------
        ic_id  :  integer
          used as "BAND%/ICID_{}_GROUP_%".format(ic_id)
        msm_grp : string
          select measurement group with name msm_grp

        All measurements groups are shown when ic_id and msm_grp are None

        Returns
        -------
        out  :  scalar
           Number of measurements found

        Updated object attributes:
          - bands               : available spectral bands
        """
        self.band = ''
        self.__msm_path = []
        for ii in '87654321':
            if f'BAND{ii}' in self.fid:
                self.band = ii
                break

        if self.band:
            gid = self.fid[f'BAND{self.band}']
            if msm_grp is not None and msm_grp in gid:
                self.__msm_path = [msm_grp]
            elif ic_id is None:
                grp_name = 'ICID_'
                for kk in gid:
                    if kk.startswith(grp_name):
                        print(kk)
            else:
                grp_name = f'ICID_{ic_id:05}_GROUP'
                self.__msm_path = [s for s in gid if s.startswith(grp_name)]

        return len(self.__msm_path)

    # -------------------------
    def get_msm_attr(self, msm_dset, attr_name):
        """
        Returns attribute of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset    :  string
            name of measurement dataset
        attr_name : string
            name of the attribute

        Returns
        -------
        out   :   scalar or numpy array
           value of attribute "attr_name"

        """
        if not self.__msm_path:
            return ''

        grp = self.fid[f'BAND{self.band}']
        for msm_path in self.__msm_path:
            ds_path = str(PurePosixPath(msm_path, 'OBSERVATIONS', msm_dset))

            if attr_name in grp[ds_path].attrs.keys():
                attr = grp[ds_path].attrs[attr_name]
                if isinstance(attr, bytes):
                    return attr.decode('ascii')

                return attr

        return None

    # -------------------------
    def get_msm_data(self, msm_dset, fill_as_nan=True,
                     frames=None, columns=None):
        """
        Returns data of measurement dataset "msm_dset"

        Parameters
        ----------
        msm_dset    :  string
            name of measurement dataset
            if msm_dset is None then show names of available datasets

        columns    : [i, j]
            Slice data on fastest axis (columns) as, from index 'i' to 'j'

        frames    : [i, j]
            Slice data on slowest axis (time) as, from index 'i' to 'j'

        fill_as_nan :  boolean
            replace (float) FillValues with Nan's

        Returns
        -------
        out   :   dictionary
           Python dictionary with names of msm_groups as keys
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__msm_path:
            return {}

        # show HDF5 dataset names and return
        grp = self.fid[f'BAND{self.band}']
        if msm_dset is None:
            ds_path = str(PurePosixPath(self.__msm_path[0], 'OBSERVATIONS'))
            for kk in grp[ds_path]:
                print(kk)
            return {}

        # skip row257 from the SWIR detector
        rows = None
        if self.band in ('7', '8'):
            rows = [0, -1]

        # combine data of all measurement groups in dictionary
        res = {}
        for msm_grp in sorted(self.__msm_path):
            dset = grp[str(PurePosixPath(msm_grp, 'OBSERVATIONS', msm_dset))]
            data_sel = ()
            for ii in range(dset.ndim):
                dim_name = PurePosixPath(dset.dims[ii][0].name).name
                if dim_name == 'msmt_time':
                    if frames is None:
                        data_sel += (slice(None),)
                    else:
                        data_sel += (slice(*frames),)
                elif dim_name == 'row':
                    if rows is None:
                        data_sel += (slice(None),)
                    else:
                        data_sel += (slice(*rows),)
                elif dim_name == 'column':
                    if columns is None:
                        data_sel += (slice(None),)
                    else:
                        data_sel += (slice(*columns),)
                else:
                    raise ValueError

            # read data
            if dset.dtype == np.float32:
                data = np.squeeze(dset.astype(float)[data_sel])
            else:
                data = np.squeeze(dset[data_sel])

            if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                data[(data == fillvalue)] = np.nan

            # add data to dictionary
            res[msm_grp] = data

        return res

    # -------------------------
    def read_direct_msm(self, msm_dset, dest_sel=None,
                        dest_dtype=None, fill_as_nan=False):
        """
        The faster implementation of get_msm_data()

        Parameters
        ----------
        msm_dset    :  string
            Name of measurement dataset
        dest_sel    :  numpy slice
            Selection must be the output of numpy.s_[<args>].
        dest_dtype  :  numpy dtype
            Perform type conversion
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Returns
        -------
        out   :   dictionary
           Python dictionary with names of msm_groups as keys
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if not self.__msm_path:
            return None

        if dest_sel is None:
            dest_sel = np.s_[...]

        # combine data of all measurement groups in dictionary
        res = {}
        for msm_grp in sorted(self.__msm_path):
            dset = self.fid[str(PurePosixPath(
                f'BAND{self.band}', msm_grp,
                'OBSERVATIONS', msm_dset))]

            if dest_dtype is None:
                buff = dset[dest_sel]
            else:
                buff = dset.astype(dest_dtype)[dest_sel]

            if fill_as_nan:
                if dset.attrs['_FillValue'] == fillvalue:
                    buff[(buff == fillvalue)] = np.nan

            # add data to dictionary
            res[msm_grp] = buff

        return res
