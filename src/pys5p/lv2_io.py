#
# This file is part of pyS5p
#
# https://github.com/rmvanhees/pys5p.git
#
# Copyright (c) 2017-2025 SRON
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""`LV2io`, class to access Tropomi level-2 products."""

from __future__ import annotations

__all__ = ["LV2io"]

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Self

import h5py
import numpy as np
from moniplot.image_to_xarray import data_to_xr, h5_to_xr
from netCDF4 import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

# - global parameters ------------------------------


# - local functions --------------------------------


# - class definition -------------------------------
class LV2io:
    """A class to read Tropomi Level-2 (offline) products.

    Parameters
    ----------
    lv2_product :  Path
        full path to S5P Tropomi level 2 product

    Notes
    -----
    The Python h5py module can read the operational netCDF4 products without
    any problems, however, the SRON science products contain incompatible
    attributes. Thus, should be fixed when more up-to-date netCDF software is
    used to generate the products. Currently, the Python netCDF4 module is
    used to read the science products.

    """

    def __init__(self: LV2io, lv2_product: Path) -> None:
        """Initialize access to an S5P_L2 product."""
        if not lv2_product.is_file():
            raise FileNotFoundError(f"{lv2_product.name} does not exist")

        # initialize class-attributes
        self.filename = lv2_product

        # open LV2 product as HDF5 file
        if self.science_product:
            self.fid = Dataset(lv2_product, "r")
            self.ground_pixel = self.fid["/instrument/ground_pixel"][:].max()
            self.ground_pixel += 1
            self.scanline = self.fid["/instrument/scanline"][:].max()
            self.scanline += 1
            # alternative set flag sparse
            if self.fid["/instrument/scanline"].size % self.ground_pixel != 0:
                raise ValueError("not all scanlines are complete")
        else:
            self.fid = h5py.File(lv2_product, "r")
            self.ground_pixel = self.fid["/PRODUCT/ground_pixel"].size
            self.scanline = self.fid["/PRODUCT/scanline"].size

    def __iter__(self: LV2io) -> None:
        """Allow itertion."""
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self: LV2io) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: LV2io, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: LV2io) -> None:
        """Close the product."""
        if self.fid is not None:
            self.fid.close()

    # ----- Class properties --------------------
    @property
    def science_product(self: LV2io) -> bool:
        """Check if product is a science product."""
        science_inst = b"Space Research Organisation Netherlands"

        res = False
        with h5py.File(self.filename) as fid:
            if "institution" in fid.attrs and fid.attrs["institution"] == science_inst:
                res = True

        return res

    @property
    def orbit(self: LV2io) -> int:
        """Return reference orbit number."""
        if self.science_product:
            return int(self.__nc_attr("orbit", "l1b_file"))

        return int(self.__h5_attr("orbit", None)[0])

    @property
    def algorithm_version(self: LV2io) -> str | None:
        """Return version of the level 2 algorithm."""
        res = self.get_attr("algorithm_version")

        return res if res is not None else self.get_attr("version")

    @property
    def processor_version(self: LV2io) -> str | None:
        """Return version of the level 2 processor."""
        res = self.get_attr("processor_version")

        return res if res is not None else self.get_attr("version")

    @property
    def product_version(self: LV2io) -> str:
        """Return version of the level 2 product."""
        res = self.get_attr("product_version")

        return res if res is not None else self.get_attr("version")

    @property
    def coverage_time(self: LV2io) -> tuple[str, str]:
        """Return start and end of the measurement coverage time."""
        return (
            self.get_attr("time_coverage_start"),
            self.get_attr("time_coverage_end"),
        )

    @property
    def creation_time(self: LV2io) -> str:
        """Return creation date/time of the level 2 product."""
        return self.get_attr("date_created")

    # ----- Attributes --------------------
    def __h5_attr(
        self: LV2io, attr_name: str, ds_name: str | None
    ) -> np.ndarray | None:
        """Read attributes from operational products using hdf5."""
        if ds_name is not None:
            dset = self.fid[f"/PRODUCT/{ds_name}"]
            if attr_name not in dset.attrs:
                return None

            attr = dset.attrs[attr_name]
        else:
            if attr_name not in self.fid.attrs:
                return None

            attr = self.fid.attrs[attr_name]

        if isinstance(attr, bytes):
            return attr.decode("ascii")

        return attr

    def __nc_attr(self: LV2io, attr_name: str, ds_name: str) -> np.ndarray | None:
        """Read attributes from science products using netCDF4."""
        if ds_name is not None:
            for grp_name in ["target_product", "side_product", "instrument"]:
                if grp_name not in self.fid.groups:
                    continue

                if ds_name not in self.fid[grp_name].variables:
                    continue

                dset = self.fid[f"/{grp_name}/{ds_name}"]
                if attr_name in dset.ncattrs():
                    return dset.getncattr(attr_name)

            return None

        if attr_name not in self.fid.ncattrs():
            return None

        return self.fid.getncattr(attr_name)

    def get_attr(
        self: LV2io, attr_name: str, ds_name: str | None = None
    ) -> np.ndarray | None:
        """Obtain value of an HDF5 file attribute or dataset attribute.

        Parameters
        ----------
        attr_name : str
           name of the attribute
        ds_name   : str, optional
           name of dataset, default is to read the product attributes

        """
        if self.science_product:
            return self.__nc_attr(attr_name, ds_name)

        return self.__h5_attr(attr_name, ds_name)

    # ----- Time information ---------------
    @property
    def ref_time(self: LV2io) -> datetime | None:
        """Return reference start time of measurements."""
        if self.science_product:
            return None

        return datetime(2010, 1, 1, 0, 0, 0) + timedelta(
            seconds=int(self.fid["/PRODUCT/time"][0])
        )

    def get_time(self: LV2io) -> np.ndarray | None:
        """Return start time of measurement per scan-line."""
        if self.science_product:
            buff = self.get_dataset("time")[:: self.ground_pixel, :]
            return np.array([datetime(*x) for x in buff])

        buff = self.fid["/PRODUCT/delta_time"][0, :]
        return np.array([self.ref_time + timedelta(seconds=x / 1e3) for x in buff])

    # ----- Geolocation --------------------
    def __h5_geo_data(self: LV2io, geo_dsets: str) -> dict:
        """Read geolocation datasets from operational products using HDF5."""
        res = {}
        if geo_dsets is None:
            geo_dsets = "latitude,longitude"

        for key in geo_dsets.split(","):
            for grp_name in ["/PRODUCT", "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS"]:
                if key in self.fid[grp_name]:
                    res[key] = np.squeeze(self.fid[f"{grp_name}/{key}"])
                    continue

        return res

    def __nc_geo_data(self: LV2io, geo_dsets: str) -> dict:
        """Read geolocation datasets from science products using netCDF4."""
        res = {}
        if geo_dsets is None:
            geo_dsets = "latitude_center,longitude_center"

        for key in geo_dsets.split(","):
            if key in self.fid["/instrument"].variables:
                ds_name = f"/instrument/{key}"
                res[key] = self.fid[ds_name][:].reshape(
                    self.scanline, self.ground_pixel
                )

        return res

    def get_geo_data(self: LV2io, geo_dsets: str | None = None) -> dict:
        """Return data of selected datasets from the GEOLOCATIONS group.

        Parameters
        ----------
        geo_dsets  :  str, optional
            Name(s) of datasets, comma separated. Default:

            * operational: 'latitude,longitude'
            * science: 'latitude_center,longitude_center'

        Returns
        -------
        dict
           dictionary with arrays of selected datasets

        """
        if self.science_product:
            return self.__nc_geo_data(geo_dsets)

        return self.__h5_geo_data(geo_dsets)

    # ----- Footprints --------------------
    def __h5_geo_bounds(
        self: LV2io,
        extent: list[float, float, float, float],
        data_sel: tuple[slice | int],
    ) -> tuple:
        """Read latitude/longitude bounding box [HDF5]."""
        if extent is not None:
            if len(extent) != 4:
                raise ValueError("parameter extent must have 4 elements")

            lats = self.fid["/PRODUCT/latitude"][0, ...]
            lons = self.fid["/PRODUCT/longitude"][0, ...]

            indx = (
                (lons >= extent[0])
                & (lons <= extent[1])
                & (lats >= extent[2])
                & (lats <= extent[3])
            ).nonzero()
            data_sel = np.s_[
                indx[0].min() : indx[0].max(), indx[1].min() : indx[1].max()
            ]

        gid = self.fid["/PRODUCT/SUPPORT_DATA/GEOLOCATIONS"]
        if data_sel is None:
            lat_bounds = gid["latitude_bounds"][0, ...]
            lon_bounds = gid["longitude_bounds"][0, ...]
        else:
            data_sel0 = (0, *data_sel, slice(None))
            lat_bounds = gid["latitude_bounds"][data_sel0]
            lon_bounds = gid["longitude_bounds"][data_sel0]

        return data_sel, lon_bounds, lat_bounds

    def __nc_geo_bounds(
        self: LV2io,
        extent: list[float, float, float, float],
        data_sel: tuple[slice | int],
    ) -> tuple:
        """Read latitude/longitude bounding box [netCDF4]."""
        if extent is not None:
            if len(extent) != 4:
                raise ValueError("parameter extent must have 4 elements")

            lats = self.fid["/instrument/latitude_center"][:].reshape(
                self.scanline, self.ground_pixel
            )
            lons = self.fid["/instrument/longitude_center"][:].reshape(
                self.scanline, self.ground_pixel
            )

            indx = (
                (lons >= extent[0])
                & (lons <= extent[1])
                & (lats >= extent[2])
                & (lats <= extent[3])
            ).nonzero()
            data_sel = np.s_[
                indx[0].min() : indx[0].max(), indx[1].min() : indx[1].max()
            ]

        gid = self.fid["/instrument"]
        lat_bounds = gid["latitude_corners"][:].data.reshape(
            self.scanline, self.ground_pixel, 4
        )
        lon_bounds = gid["longitude_corners"][:].data.reshape(
            self.scanline, self.ground_pixel, 4
        )
        if data_sel is not None:
            lat_bounds = lat_bounds[(*data_sel, slice(None))]
            lon_bounds = lon_bounds[(*data_sel, slice(None))]

        return data_sel, lon_bounds, lat_bounds

    def get_geo_bounds(
        self: LV2io,
        extent: list[float, float, float, float] | None,
        data_sel: tuple[slice | int] | None,
    ) -> np.ndarray | tuple:
        """Return bounds of latitude/longitude as a mesh for plotting.

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
           Select slice of data which covers geolocation defined by extent.
           Only provided if extent is not None.
        out       :   dictionary
           With numpy arrays for latitude and longitude

        """
        if self.science_product:
            res = self.__nc_geo_bounds(extent, data_sel)
        else:
            res = self.__h5_geo_bounds(extent, data_sel)
        data_sel, lon_bounds, lat_bounds = res

        res = {}
        _sz = lon_bounds.shape
        res["longitude"] = np.empty((_sz[0] + 1, _sz[1] + 1), dtype=float)
        res["longitude"][:-1, :-1] = lon_bounds[:, :, 0]
        res["longitude"][-1, :-1] = lon_bounds[-1, :, 1]
        res["longitude"][:-1, -1] = lon_bounds[:, -1, 1]
        res["longitude"][-1, -1] = lon_bounds[-1, -1, 2]

        res["latitude"] = np.empty((_sz[0] + 1, _sz[1] + 1), dtype=float)
        res["latitude"][:-1, :-1] = lat_bounds[:, :, 0]
        res["latitude"][-1, :-1] = lat_bounds[-1, :, 1]
        res["latitude"][:-1, -1] = lat_bounds[:, -1, 1]
        res["latitude"][-1, -1] = lat_bounds[-1, -1, 2]

        if extent is None:
            return res

        return data_sel, res

    # ----- Datasets (numpy) --------------------
    def __h5_dataset(
        self: LV2io, name: str, data_sel: tuple[slice | int], fill_as_nan: bool
    ) -> np.ndarray:
        """Read dataset from operational products using HDF5."""
        fillvalue = float.fromhex("0x1.ep+122")

        if name not in self.fid["/PRODUCT"]:
            raise ValueError(f"dataset {name} for found")

        dset = self.fid[f"/PRODUCT/{name}"]
        if data_sel is None:
            if dset.dtype == np.float32:
                res = dset.astype(float)[0, ...]
            else:
                res = dset[0, ...]
        else:
            if dset.dtype == np.float32:
                res = dset.astype(float)[(0, *data_sel)]
            else:
                res = dset[(0, *data_sel)]

        if fill_as_nan and dset.attrs["_FillValue"] == fillvalue:
            res[(res == fillvalue)] = np.nan

        return res

    def __nc_dataset(
        self: LV2io, name: str, data_sel: tuple[slice | int], fill_as_nan: bool
    ) -> np.ndarray:
        """Read dataset from science products using netCDF4."""
        if name in self.fid["/target_product"].variables:
            group = "/target_product"
        elif name in self.fid["/instrument"].variables:
            group = "/instrument"
        else:
            raise ValueError(f"dataset {name} for found")

        dset = self.fid[f"{group}/{name}"]
        if dset.size == self.scanline * self.ground_pixel:
            res = dset[:].reshape(self.scanline, self.ground_pixel)
        else:
            res = dset[:]
        if data_sel is not None:
            res = res[data_sel]

        if fill_as_nan:
            return res.filled(np.nan)

        return res.data

    def get_dataset(
        self: LV2io,
        name: str,
        data_sel: tuple[slice | int] | slice | None = None,
        fill_as_nan: bool = True,
    ) -> np.ndarray:
        """Read level 2 dataset from PRODUCT group.

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
        numpy.ndarray

        """
        if self.science_product:
            return self.__nc_dataset(name, data_sel, fill_as_nan)

        return self.__h5_dataset(name, data_sel, fill_as_nan)

    # ----- Dataset (xarray) --------------------
    def __h5_data_as_xds(
        self: LV2io, name: str, data_sel: tuple[slice | int]
    ) -> xr.DataArray:
        """Read dataset from group target_product using HDF5.

        Input: operational product

        Return: xarray.Dataset
        """
        if name not in self.fid["/PRODUCT"]:
            raise ValueError(f"dataset {name} for found")
        dset = self.fid[f"/PRODUCT/{name}"]

        # ToDo handle parameter mol_m2
        return h5_to_xr(dset, (0, *data_sel)).squeeze()

    def __nc_data_as_xds(
        self: LV2io, name: str, data_sel: tuple[slice | int]
    ) -> xr.DataArray:
        """Read dataset from group PRODUCT using netCDF4.

        Input: science product

        Return: xarray.DataArray
        """
        if name in self.fid["/target_product"].variables:
            group = "/target_product/"
        elif name in self.fid["/instrument"].variables:
            group = "/instrument/"
        else:
            raise ValueError("dataset {name} for found")

        return data_to_xr(
            self.get_dataset(group + name, data_sel),
            dims=["scanline", "ground_pixel"],
            name=name,
            long_name=self.get_attr("long_name", name),
            units=self.get_attr("units", name),
        )

    def get_data_as_xds(
        self: LV2io, name: str, data_sel: tuple[slice | int] | None = None
    ) -> xr.DataArray:
        """Read dataset from group PRODUCT/target_product group.

        Parameters
        ----------
        name   :  str
            name of dataset with level 2 data
        data_sel  :  numpy slice
           a 3-dimensional numpy slice: time, scan_line, ground_pixel

        Returns
        -------
        xarray.DataArray

        """
        if self.science_product:
            return self.__nc_data_as_xds(name, data_sel)

        return self.__h5_data_as_xds(name, data_sel)
