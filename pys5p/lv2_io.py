"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class LV2io provides read access to S5p Tropomi S5P_OFFL_L2 products

Copyright (c) 2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import h5py
import numpy as np

# - global parameters ------------------------------

# - local functions --------------------------------


# - class definition -------------------------------
class LV2io():
    """
    This class should offer all the necessary functionality to read Tropomi
    S5P_OFFL_L2 products
    """
    def __init__(self, lv2_product):
        """
        Initialize access to an S5P_L2 product

        Parameters
        ----------
        lv2_product :  string
           full path to S5P Tropomi level 2 product
        """
        # initialize class-attributes
        self.filename = lv2_product
        self.fid = None

        if not Path(lv2_product).is_file():
            raise FileNotFoundError('{} does not exist'.format(lv2_product))

        # open LV2 product as HDF5 file
        self.fid = h5py.File(lv2_product, "r")

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r})'.format(class_name, self.filename)

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __del__(self):
        """
        called when the object is destroyed
        """
        self.close()

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
        Close the product.
        """
        if self.fid is not None:
            self.fid.close()

    # -------------------------
    def get_orbit(self):
        """
        Returns reference orbit number
        """
        if 'orbit' in self.fid.attrs:
            return int(self.fid.attrs['orbit'])

        return None

    # -------------------------
    def get_algorithm_version(self):
        """
        Returns version of the level 2 algorithm
        """
        if 'algorithm_version' not in self.fid.attrs:
            return None

        res = self.fid.attrs['algorithm_version']
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    # -------------------------
    def get_processor_version(self):
        """
        Returns version of the L12 processor
        """
        if 'processor_version' not in self.fid.attrs:
            return None

        res = self.fid.attrs['processor_version']
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    # -------------------------
    def get_product_version(self):
        """
        Returns version of the level 2 product
        """
        if 'product_version' not in self.fid.attrs:
            return None

        res = self.fid.attrs['product_version']
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    # -------------------------
    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        if 'time_coverage_start' not in self.fid.attrs \
           or 'time_coverage_end' not in self.fid.attrs:
            return None

        res1 = self.fid.attrs['time_coverage_start']
        if isinstance(res1, bytes):
            res1 = res1.decode('ascii')

        res2 = self.fid.attrs['time_coverage_end']
        if isinstance(res2, bytes):
            res2 = res2.decode('ascii')

        return (res1, res2)

    # -------------------------
    def get_creation_time(self):
        """
        Returns creation date/time of the level 2 product
        """
        if 'date_created' not in self.fid.attrs:
            return None

        res = self.fid.attrs['date_created']
        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    # -------------------------
    def get_attr(self, attr_name, ds_name=None):
        """
        Obtain value of an HDF5 file attribute or dataset attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        ds_name   : string
           name of dataset in group PRODUCT
        """
        if ds_name is not None:
            dset = self.fid['/PRODUCT/{}'.format(ds_name)]
            if attr_name not in dset.attrs.keys():
                return None

            attr = dset.attrs[attr_name]
        else:
            if attr_name not in self.fid.attrs:
                return None

            attr = self.fid.attrs[attr_name]

        if isinstance(attr, bytes):
            return attr.decode('ascii')

        return attr

    # ---------- Functions using data in the PRODUCT group ----------
    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
        from datetime import datetime, timedelta

        ref_time = datetime(2010, 1, 1, 0, 0, 0)
        ref_time += timedelta(seconds=int(self.fid['/PRODUCT/time'][0]))
        return ref_time

    # -------------------------
    def get_delta_time(self):
        """
        Returns offset from the reference start time of measurement
        """
        return self.fid['/PRODUCT/delta_time'][0, :].astype(int)

    # -------------------------
    def get_geo_data(self,
                     geo_dset='satellite_latitude,satellite_longitude'):
        """
        Returns data of selected datasets from the GEOLOCATIONS group

        Parameters
        ----------
        geo_dset  :  string
            Name(s) of datasets in the GEODATA group, comma separated
            Default is 'satellite_latitude,satellite_longitude'

        Returns
        -------
        out   :   array-like
           Compound array with data of selected datasets from the GEODATA group
        """
        grp = self.fid['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS']
        for key in geo_dset.split(','):
            if res is None:
                res = np.squeeze(grp[key])
            else:
                res = np.append(res, np.squeeze(grp[key]))

        return res

    # -------------------------
    def get_geo_bounds(self, extent=None, data_sel=None):
        """
        Returns bounds of latitude/longitude as a mesh for plotting

        Parameters
        ----------
        extent    :  list
           select data to cover a region with geolocation defined by:
             lon_min, lon_max, lat_min, lat_max and return numpy slice
        data_sel  :  numpy slice
           a 3-dimensional numpy slice: time, scan_line, ground_pixel
           Note 'data_sel' will be overwritten when 'extent' is defined

        Returns
        -------
        data_sel  :   numpy slice
           slice of data which covers geolocation defined by extent. Only
           provided if extent is not None.
        out       :   dictionary
           with numpy arrays for latitude and longitude
        """
        if extent is not None:
            if len(extent) != 4:
                raise ValueError('parameter extent must have 4 elements')

            lats = self.fid['/PRODUCT/latitude'][:]
            lons = self.fid['/PRODUCT/longitude'][:]
            indx = np.where((lons >= extent[0]) & (lons <= extent[1])
                            & (lats >= extent[2]) & (lats <= extent[3]))
            data_sel = np.s_[0,
                             indx[0].min():indx[0].max(),
                             indx[1].min():indx[1].max()]

        gid = self.fid['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS']
        if data_sel is None:
            _sz = gid['latitude_bounds'][0, ...].shape
        else:
            _sz = gid['latitude_bounds'][data_sel].shape

        res = {}
        res['latitude'] = np.empty((_sz[0]+1, _sz[1]+1), dtype=float)
        if data_sel is None:
            lat_bounds = gid['latitude_bounds'][0, ...]
        else:
            lat_bounds = gid['latitude_bounds'][data_sel + (slice(None),)]
        res['latitude'][:-1, :-1] = lat_bounds[:, :, 0]
        res['latitude'][-1, :-1] = lat_bounds[-1, :, 1]
        res['latitude'][:-1, -1] = lat_bounds[:, -1, 1]
        res['latitude'][-1, -1] = lat_bounds[-1, -1, 2]

        res['longitude'] = np.empty((_sz[0]+1, _sz[1]+1), dtype=float)
        if data_sel is None:
            lon_bounds = gid['longitude_bounds'][0, ...]
        else:
            lon_bounds = gid['longitude_bounds'][data_sel + (slice(None),)]
        res['longitude'][:-1, :-1] = lon_bounds[:, :, 0]
        res['longitude'][-1, :-1] = lon_bounds[-1, :, 1]
        res['longitude'][:-1, -1] = lon_bounds[:, -1, 1]
        res['longitude'][-1, -1] = lon_bounds[-1, -1, 2]

        if extent is None:
            return res

        return data_sel, res

    # -------------------------
    def get_dataset(self, name, data_sel=None, fill_as_nan=True):
        """
        Read level 2 dataset from PRODUCT group

        Parameters
        ----------
        name   :  string
            name of dataset with level 2 data
        data_sel  :  numpy slice
           a 3-dimensional numpy slice: time, scan_line, ground_pixel
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True

        Returns
        -------
        out  :  array
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if name not in self.fid['/PRODUCT']:
            raise ValueError('dataset {} for found'.format(name))

        dset = self.fid['/PRODUCT/{}'.format(name)]
        if data_sel is None:
            if dset.dtype == np.float32:
                res = dset[0, ...].astype(np.float64)
            else:
                res = dset[0, ...]
        else:
            if dset.dtype == np.float32:
                res = dset[data_sel].astype(np.float64)
            else:
                res = dset[data_sel]
        if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
            res[(res == fillvalue)] = np.nan
        return res

    # -------------------------
    def get_data_as_s5pmsm(self, name, data_sel=None, fill_as_nan=True,
                           mol_m2=False):
        """
        Read level 2 dataset from PRODUCT group as S5Pmsm object

        Parameters
        ----------
        name   :  string
            name of dataset with level 2 data
        data_sel  :  numpy slice
           a 3-dimensional numpy slice: time, scan_line, ground_pixel
        fill_as_nan :  boolean
            Replace (float) FillValues with Nan's, when True
        mol_m2    : boolean
            Leaf units as mol per m^2 or convert units to molecules per cm^2

        Returns
        -------
        out  :  pys5p.S5Pmsm object
        """
        from pys5p.s5p_msm import S5Pmsm

        if name not in self.fid['/PRODUCT']:
            raise ValueError('dataset {} for found'.format(name))

        dset = self.fid['/PRODUCT/{}'.format(name)]
        msm = S5Pmsm(dset, data_sel=data_sel)
        if '{}_precision'.format(name) in self.fid['/PRODUCT']:
            msm.error = self.get_dataset('{}_precision'.format(name),
                                         data_sel=data_sel)

        if not mol_m2:
            factor = dset.attrs[
                'multiplication_factor_to_convert_to_molecules_percm2']
            msm.value[msm.value != msm.fillvalue] *= factor
            msm.units = 'molecules / cm^2'
            if msm.error is not None:
                msm.error[msm.error != msm.fillvalue] *= factor
        else:
            msm.units = 'mol / m^2'

        if fill_as_nan:
            msm.fill_as_nan()

        return msm
