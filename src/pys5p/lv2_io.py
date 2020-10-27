"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

The class LV2io provides read access to S5p Tropomi S5P_OFFL_L2 products

Copyright (c) 2018-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta
from pathlib import Path

import h5py
from netCDF4 import Dataset
import numpy as np

from .s5p_msm import S5Pmsm

# - global parameters ------------------------------


# - local functions --------------------------------


# - class definition -------------------------------
class LV2io():
    """
    This class should offer all the necessary functionality to read Tropomi
    S5P_OFFL_L2 products

    Attributes
    ----------
    fid : h5py.File
    filename : string
    science_product : bool
    ground_pixel : int
    scanline : int

    Methods
    -------
    close()
       Close resources.
    get_attr(attr_name, ds_name=None)
       Obtain value of an HDF5 file attribute or dataset attribute.
    get_orbit()
       Returns reference orbit number
    get_algorithm_version()
       Returns version of the level-2 algorithm.
    get_processor_version()
       Returns version of the L12 processor used to generate this product.
    get_product_version()
       Returns version of the level-2 product
    get_coverage_time()
       Returns start and end of the measurement coverage time.
    get_creation_time()
       Returns creation date/time of the level-2 product.
    get_ref_time()
       Returns reference start time of measurements.
    get_delta_time()
       Returns offset from the reference start time of measurement.
    get_geo_data(geo_dsets=None)
       Returns data of selected datasets from the GEOLOCATIONS group.
    get_geo_bounds(extent=None, data_sel=None)
       Returns bounds of latitude/longitude as a mesh for plotting.
    get_dataset(name, data_sel=None, fill_as_nan=True)
       Read level-2 dataset from PRODUCT group.
    get_data_as_s5pmsm(name, data_sel=None, fill_as_nan=True, mol_m2=False)
       Read dataset from group PRODUCT/target_product group.

    Notes
    -----
    The Python h5py module can read the operational netCDF4 products without
    any problems, however, the SRON science products contain incompatible
    attributes. Thus should be fixed when more up-to-date netCDF software is
    used to generate the products. Currently, the Python netCDF4 module is
    used to read the science products.

    Examples
    --------
    """
    def __init__(self, lv2_product):
        """
        Initialize access to an S5P_L2 product

        Parameters
        ----------
        lv2_product :  string
           full path to S5P Tropomi level 2 product
        """
        if not Path(lv2_product).is_file():
            raise FileNotFoundError('{} does not exist'.format(lv2_product))

        # initialize class-attributes
        self.filename = lv2_product
        self.science_product = False

        # open LV2 product as HDF5 file
        self.fid = h5py.File(lv2_product, "r")
        try:
            science_inst = ['SRON Netherlands Institute for Space Research']

            if self.get_attr('institution') in science_inst:
                self.science_product = True
        except OSError:
            self.science_product = True

        if self.science_product:
            self.fid.close()
            self.fid = Dataset(lv2_product, "r", format="NETCDF4")

            self.ground_pixel = self.fid['/instrument/ground_pixel'][:].max()
            self.ground_pixel += 1
            self.scanline = self.fid['/instrument/scanline'][:].max()
            self.scanline += 1
            # alternative set flag sparse
            if self.fid['/instrument/scanline'].size % self.ground_pixel != 0:
                raise ValueError('not all scanlines are complete')
        else:
            self.ground_pixel = self.fid['/PRODUCT/ground_pixel'].size
            self.scanline = self.fid['/PRODUCT/scanline'].size

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r})'.format(class_name, self.filename)

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

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
    def __h5_attr(self, attr_name, ds_name):
        """
        read attributes from operational products using hdf5
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

    def __nc_attr(self, attr_name, ds_name):
        """
        read attributes from science products using netCDF4
        """
        if ds_name is not None:
            for grp_name in ['target_product', 'side_product', 'instrument']:
                if grp_name not in self.fid.groups:
                    continue

                if ds_name not in self.fid[grp_name].variables:
                    continue

                dset = self.fid['/{}/{}'.format(grp_name, ds_name)]
                if attr_name in dset.ncattrs():
                    return dset.getncattr(attr_name)

            return None

        if attr_name not in self.fid.ncattrs():
            return None

        return self.fid.getncattr(attr_name)

    def get_attr(self, attr_name, ds_name=None):
        """
        Obtain value of an HDF5 file attribute or dataset attribute

        Parameters
        ----------
        attr_name : string
           name of the attribute
        ds_name   : string (optional)
           name of dataset, default is to read the product attributes
        """
        if self.science_product:
            return self.__nc_attr(attr_name, ds_name)

        return self.__h5_attr(attr_name, ds_name)

    # -------------------------
    def get_orbit(self) -> int:
        """
        Returns reference orbit number
        """
        if self.science_product:
            return int(self.__nc_attr('orbit', 'l1b_file'))

        return self.__h5_attr('orbit', None)[0]

    # -------------------------
    def get_algorithm_version(self) -> str:
        """
        Returns version of the level 2 algorithm
        """
        return self.get_attr('algorithm_version')

    # -------------------------
    def get_processor_version(self) -> str:
        """
        Returns version of the L12 processor
        """
        return self.get_attr('processor_version')

    # -------------------------
    def get_product_version(self):
        """
        Returns version of the level 2 product
        """
        return self.get_attr('product_version')

    # -------------------------
    def get_coverage_time(self):
        """
        Returns start and end of the measurement coverage time
        """
        res1 = self.get_attr('time_coverage_start')
        res2 = self.get_attr('time_coverage_end')
        return (res1, res2)

    # -------------------------
    def get_creation_time(self):
        """
        Returns creation date/time of the level 2 product
        """
        return self.get_attr('date_created')

    # ---------- Functions using data in the PRODUCT group ----------
    def get_ref_time(self):
        """
        Returns reference start time of measurements
        """
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
    def __h5_geo_data(self, geo_dsets):
        """
        read gelocation datasets from operational products using HDF5
        """
        res = {}
        if geo_dsets is None:
            geo_dsets = 'latitude,longitude'

        for key in geo_dsets.split(','):
            for grp_name in ['/PRODUCT', '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS']:
                if key in self.fid[grp_name]:
                    res[key] = np.squeeze(
                        self.fid['{}/{}'.format(grp_name, key)])
                    continue

        return res

    def __nc_geo_data(self, geo_dsets):
        """
        read gelocation datasets from science products using netCDF4
        """
        res = {}
        if geo_dsets is None:
            geo_dsets = 'latitude_center,longitude_center'

        for key in geo_dsets.split(','):
            if key in self.fid['/instrument'].variables.keys():
                res[key] = self.fid['/instrument/{}'.format(key)][:]

        return res

    def get_geo_data(self, geo_dsets=None):
        """
        Returns data of selected datasets from the GEOLOCATIONS group

        Parameters
        ----------
        geo_dset  :  string
            Name(s) of datasets, comma separated
            Default:
              * operational: 'latitude,longitude'
              * science: 'latitude_center,longitude_center'

        Returns
        -------
        out   :   dictonary with arrays
           arrays of selected datasets
        """
        if self.science_product:
            return self.__nc_geo_data(geo_dsets)

        return self.__h5_geo_data(geo_dsets)

    # -------------------------
    def __h5_geo_bounds(self, extent, data_sel):
        """
        read bounds of latitude/longitude from operational products using HDF5
        """
        indx = None
        if extent is not None:
            if len(extent) != 4:
                raise ValueError('parameter extent must have 4 elements')

            lats = self.fid['/PRODUCT/latitude'][0, ...]
            lons = self.fid['/PRODUCT/longitude'][0, ...]

            indx = np.where((lons >= extent[0]) & (lons <= extent[1])
                            & (lats >= extent[2]) & (lats <= extent[3]))
            data_sel = np.s_[0,
                             indx[0].min():indx[0].max(),
                             indx[1].min():indx[1].max()]

        gid = self.fid['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS']
        if data_sel is None:
            lat_bounds = gid['latitude_bounds'][0, ...]
            lon_bounds = gid['longitude_bounds'][0, ...]
        else:
            lat_bounds = gid['latitude_bounds'][data_sel + (slice(None),)]
            lon_bounds = gid['longitude_bounds'][data_sel + (slice(None),)]

        return (data_sel, lon_bounds, lat_bounds)

    def __nc_geo_bounds(self, extent, data_sel):
        """
        read bounds of latitude/longitude from science products using netCDF4
        """
        indx = None
        if extent is not None:
            if len(extent) != 4:
                raise ValueError('parameter extent must have 4 elements')

            lats = self.fid['/instrument/latitude_center'][:].reshape(
                self.scanline, self.ground_pixel)
            lons = self.fid['/instrument/longitude_center'][:].reshape(
                self.scanline, self.ground_pixel)

            indx = np.where((lons >= extent[0]) & (lons <= extent[1])
                            & (lats >= extent[2]) & (lats <= extent[3]))
            data_sel = np.s_[indx[0].min():indx[0].max(),
                             indx[1].min():indx[1].max()]

        gid = self.fid['/instrument']
        lat_bounds = gid['latitude_corners'][:].data.reshape(
            self.scanline, self.ground_pixel, 4)
        lon_bounds = gid['longitude_corners'][:].data.reshape(
            self.scanline, self.ground_pixel, 4)
        if data_sel is not None:
            lat_bounds = lat_bounds[data_sel + (slice(None),)]
            lon_bounds = lon_bounds[data_sel + (slice(None),)]

        return (data_sel, lon_bounds, lat_bounds)

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
        if self.science_product:
            res = self.__nc_geo_bounds(extent, data_sel)
        else:
            res = self.__h5_geo_bounds(extent, data_sel)
        data_sel, lon_bounds, lat_bounds = res

        res = {}
        _sz = lon_bounds.shape
        res['longitude'] = np.empty((_sz[0]+1, _sz[1]+1), dtype=np.float)
        res['longitude'][:-1, :-1] = lon_bounds[:, :, 0]
        res['longitude'][-1, :-1] = lon_bounds[-1, :, 1]
        res['longitude'][:-1, -1] = lon_bounds[:, -1, 1]
        res['longitude'][-1, -1] = lon_bounds[-1, -1, 2]

        res['latitude'] = np.empty((_sz[0]+1, _sz[1]+1), dtype=np.float)
        res['latitude'][:-1, :-1] = lat_bounds[:, :, 0]
        res['latitude'][-1, :-1] = lat_bounds[-1, :, 1]
        res['latitude'][:-1, -1] = lat_bounds[:, -1, 1]
        res['latitude'][-1, -1] = lat_bounds[-1, -1, 2]

        if extent is None:
            return res

        return data_sel, res

    # -------------------------
    def __h5_dataset(self, name, data_sel, fill_as_nan):
        """
        read dataset from operational products using HDF5
        """
        fillvalue = float.fromhex('0x1.ep+122')

        if name not in self.fid['/PRODUCT']:
            raise ValueError('dataset {} for found'.format(name))

        dset = self.fid['/PRODUCT/{}'.format(name)]
        if data_sel is None:
            if dset.dtype == np.float32:
                with dset.astype(np.float):
                    res = dset[0, ...]
            else:
                res = dset[0, ...]
        else:
            if dset.dtype == np.float32:
                with dset.astype(np.float):
                    res = dset[data_sel]
            else:
                res = dset[data_sel]

        if fill_as_nan and dset.attrs['_FillValue'] == fillvalue:
            res[(res == fillvalue)] = np.nan

        return res

    def __nc_dataset(self, name, data_sel, fill_as_nan):
        """
        read dataset from science products using netCDF4
        """
        if name not in self.fid['/target_product'].variables.keys():
            raise ValueError('dataset {} for found'.format(name))

        dset = self.fid['/target_product/{}'.format(name)]
        res = dset[:].reshape(self.scanline, self.ground_pixel)
        if data_sel is not None:
            res = res[data_sel]

        if fill_as_nan:
            return res.filled(np.nan)

        return res.data

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
        if self.science_product:
            return self.__nc_dataset(name, data_sel, fill_as_nan)

        return self.__h5_dataset(name, data_sel, fill_as_nan)

    # -------------------------
    def __h5_data_as_s5pmsm(self, name, data_sel, fill_as_nan, mol_m2):
        """
        Read dataset from group target_product using HDF5

        Input: operational product

        Return: S5Pmsm object
        """
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
            msm.units = 'molecules / cm$^2$'
            if msm.error is not None:
                msm.error[msm.error != msm.fillvalue] *= factor
        else:
            msm.units = 'mol / m$^2$'

        if fill_as_nan:
            msm.fill_as_nan()

        return msm

    def __nc_data_as_s5pmsm(self, name, data_sel, fill_as_nan):
        """
        Read dataset from group PRODUCT using netCDF4

        Input: science product

        Return: S5Pmsm object
        """
        msm = S5Pmsm(self.get_dataset(name, data_sel))
        buff = self.get_dataset('{}_precision'.format(name), data_sel)
        if buff is not None:
            msm.error = buff
        msm.set_fillvalue()
        msm.set_units(self.get_attr('units', name))
        msm.set_long_name(self.get_attr('long_name', name))

        if fill_as_nan:
            msm.fill_as_nan()

        return msm

    def get_data_as_s5pmsm(self, name, data_sel=None, fill_as_nan=True,
                           mol_m2=False):
        """
        Read dataset from group PRODUCT/target_product group

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
        if self.science_product:
            return self.__nc_data_as_s5pmsm(name, data_sel, fill_as_nan)

        return self.__h5_data_as_s5pmsm(name, data_sel, fill_as_nan, mol_m2)
