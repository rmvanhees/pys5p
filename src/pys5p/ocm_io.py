#
# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""This module contain class `OCMio` to access on-ground calibration data."""

from __future__ import annotations

__all__ = ['OCMio']

from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np
from moniplot.biweight import Biweight

# - global parameters ------------------------------


# - local functions --------------------------------
def band2channel(dict_a: dict, dict_b: dict,
                 mode: bool = None) -> np.ndarray | tuple[Any, Any]:
    """Store data from a dictionary as returned by get_msm_data to a ndarray.

    Parameters
    ----------
    dict_a      :  dict
       data of the one spectral band
    dict_b      :  dict
       data of another spectral band
    mode        :  list ['combined', 'mean', 'median', 'biweight']
        'combined'
             will combine data using np.concatenate((data_a, data_b),\
             axis=data_a.ndim-1)

        'mean'
             is calculated using np.nanmean(data, axis=0)

        'median'
             is calculated using np.nanmedian(data, axis=0)

        'biweight'
             is calculated using biweight(data, axis=0)

    Returns
    -------
    numpy.ndarray
        Data from dictionary stored in a numpy array

    Examples
    --------
    > data = ocm.band2channel(dict_a, dict_b, mode=['combined', 'median'])
    """
    if mode is None:
        mode = []

    data_a = None
    for key in sorted(dict_a):
        buff = dict_a[key][...]

        data_a = buff if data_a is None else np.vstack((data_a, buff))

    if data_a is not None:
        if 'mean' in mode:
            data_a = np.nanmean(data_a, axis=0)
        elif 'median' in mode:
            data_a = np.nanmedian(data_a, axis=0)
        elif 'biweight' in mode:
            data_a = Biweight(data_a, axis=0).median

    if dict_b is None:
        return data_a

    data_b = None
    for key in sorted(dict_b):
        buff = dict_b[key][...]

        data_b = buff if data_b is None else np.vstack((data_b, buff))

    if data_b is not None:
        if 'mean' in mode:
            data_b = np.nanmean(data_b, axis=0)
        elif 'median' in mode:
            data_b = np.nanmedian(data_b, axis=0)
        elif 'biweight' in mode:
            data_b = Biweight(data_b, axis=0).median

    if 'combined' in mode:
        return np.concatenate((data_a, data_b), axis=data_a.ndim-1)

    return data_a, data_b


# - class definition -------------------------------
class OCMio:
    """This class should offer all the necessary functionality to read Tropomi
    on-ground calibration products (Lx).

    Parameters
    ----------
    ocm_product :  Path
        Full path to on-ground calibration measurement
    """

    def __init__(self, ocm_product: Path):
        """Initialize access to an OCAL Lx product."""
        if not ocm_product.is_file():
            raise FileNotFoundError(f'{ocm_product.name} does not exist')

        # initialize class-attributes
        self.__msm_path = None
        self.band = None
        self.filename = ocm_product

        # open OCM product as HDF5 file
        self.fid = h5py.File(ocm_product, 'r')

    def __iter__(self):
        """Allow iteration."""
        for attr in sorted(self.__dict__):
            if not attr.startswith('__'):
                yield attr

    # def __del__(self):
    #    """
    #    called when the object is destroyed
    #    """
    #    self.close()

    def __enter__(self):
        """Initiate the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self):
        """Close resources."""
        self.band = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    # ---------- RETURN VERSION of the S/W ----------
    # ---------- Functions that work before MSM selection ----------
    def get_processor_version(self) -> str:
        """Return version of the L01b processor."""
        res = self.fid.attrs['processor_version']
        if isinstance(res, bytes):
            # pylint: disable=no-member
            res = res.decode('ascii')
        return res

    def get_coverage_time(self) -> tuple[str, str]:
        """Return start and end of the measurement coverage time."""
        t_bgn = self.fid.attrs['time_coverage_start']
        if isinstance(t_bgn, bytes):
            # pylint: disable=no-member
            t_bgn = t_bgn.decode('ascii')

        t_end = self.fid.attrs['time_coverage_end']
        if isinstance(t_end, bytes):
            # pylint: disable=no-member
            t_end = t_end.decode('ascii')
        return t_bgn, t_end

    def get_attr(self, attr_name) -> np.ndarray | None:
        """Obtain value of an HDF5 file attribute.

        Parameters
        ----------
        attr_name : string
           name of the attribute
        """
        if attr_name in self.fid.attrs:
            return self.fid.attrs[attr_name]

        return None

    # ---------- Functions that only work after MSM selection ----------
    def get_ref_time(self) -> np.ndarray | None:
        """Return reference start time of measurements."""
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'GEODATA'))]
            res[msm] = datetime(2010, 1, 1, 0, 0, 0)
            res[msm] += timedelta(seconds=int(sgrp['time'][0]))

        return res

    def get_delta_time(self) -> np.ndarray | None:
        """Return offset from the reference start time of measurement."""
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'GEODATA'))]
            res[msm] = sgrp['delta_time'][:].astype(int)

        return res

    def get_instrument_settings(self) -> np.ndarray | None:
        """Return instrument settings of measurement."""
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['instrument_settings'])

        return res

    def get_gse_stimuli(self) -> np.ndarray | None:
        """Return GSE stimuli parameters."""
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['gse_stimuli'])

        return res

    def get_exposure_time(self) -> np.ndarray | None:
        """Return the exact pixel exposure time of the measurements."""
        if not self.__msm_path:
            return None

        grp = self.fid[f'BAND{self.band}']
        msm = self.__msm_path[0]  # all measurement sets have the same ICID
        sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
        instr = np.squeeze(sgrp['instrument_settings'])

        if int(self.band) > 6:
            return 1.25e-6 * (65540 - instr['int_delay'] + instr['int_hold'])

        return instr['exposure_time']

    def get_housekeeping_data(self) -> np.ndarray | None:
        """Return housekeeping data of measurements."""
        if not self.__msm_path:
            return {}

        grp = self.fid[f'BAND{self.band}']
        res = {}
        for msm in sorted(self.__msm_path):
            sgrp = grp[str(PurePosixPath(msm, 'INSTRUMENT'))]
            res[msm] = np.squeeze(sgrp['housekeeping_data'])

        return res

    # -------------------------
    def select(self, ic_id: int = None, *,
               msm_grp: str | None = None) -> int:
        """Select a measurement as BAND%/ICID_<ic_id>_GROUP_%.

        Parameters
        ----------
        ic_id  :  int
          used as "BAND%/ICID_{}_GROUP_%".format(ic_id)
        msm_grp : str
          select measurement group with name msm_grp

        All measurements groups are shown when ic_id and msm_grp are None

        Returns
        -------
        scalar
           Number of measurements found

        Notes
        -----
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
    def get_msm_attr(self, msm_dset: str, attr_name: str) -> str | None:
        """Return attribute of measurement dataset "msm_dset".

        Parameters
        ----------
        msm_dset :  str
            name of measurement dataset
        attr_name : str
            name of the attribute

        Returns
        -------
        scalar or numpy.ndarray
           value of attribute "attr_name"
        """
        if not self.__msm_path:
            return ''

        grp = self.fid[f'BAND{self.band}']
        for msm_path in self.__msm_path:
            ds_path = str(PurePosixPath(msm_path, 'OBSERVATIONS', msm_dset))

            if attr_name in grp[ds_path].attrs:
                attr = grp[ds_path].attrs[attr_name]
                if isinstance(attr, bytes):
                    return attr.decode('ascii')

                return attr

        return None

    # -------------------------
    def get_msm_data(self, msm_dset: str, fill_as_nan: bool = True,
                     frames: list[int, int] | None = None,
                     columns: list[int, int] | None = None) -> dict:
        """Return data of measurement dataset `msm_dset`.

        Parameters
        ----------
        msm_dset    :  str
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
        dict
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
    def read_direct_msm(self, msm_dset: str,
                        dest_sel: tuple[slice | int] | None = None,
                        dest_dtype: type[Any] | None = None,
                        fill_as_nan: bool = False) -> dict | None:
        """Return data of measurement dataset `msm_dset` (fast implementation).

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
        dict
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

            if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
                buff[(buff == fillvalue)] = np.nan

            # add data to dictionary
            res[msm_grp] = buff

        return res
