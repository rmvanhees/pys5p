"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest 

Note
----
Please use the code as tutorial

Copyright (c) 2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from pathlib import Path

import h5py
import numpy as np

from pys5p.s5p_plot import S5Pplot
from pys5p.s5p_geo_plot import S5Pgeoplot


# -------------------------
def read_using_co_h5py(args):
    """
    Perform read using h5py and use S5Pplot to create the figure
    """
    with h5py.File(args.input_file, 'r') as fid:
        print(fid.attrs['title'])
        print(fid.attrs['date_created'])
        print(fid.attrs['algorithm_version'])
        print(fid.attrs['product_version'])
        print(fid.attrs['orbit'])
        print(fid.attrs['time_coverage_start'],
              fid.attrs['time_coverage_end'])
        orbit = fid.attrs['orbit'][0]

        gid = fid['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS']
        sz = gid['latitude_bounds'].shape
        gridlats = np.empty((sz[1]+1, sz[2]+1), dtype=float)
        gridlats[:-1, :-1] = gid['latitude_bounds'][0, :, :, 0]
        gridlats[-1, :-1] = gid['latitude_bounds'][0, -1, :, 1]
        gridlats[:-1, -1] = gid['latitude_bounds'][0, :, -1, 1]
        gridlats[-1, -1] = gid['latitude_bounds'][0, -1, -1, 2]

        gridlons = np.empty((sz[1]+1, sz[2]+1), dtype=float)
        gridlons[:-1, :-1] = gid['longitude_bounds'][0, :, :, 0]
        gridlons[-1, :-1] = gid['longitude_bounds'][0, -1, :, 1]
        gridlons[:-1, -1] = gid['longitude_bounds'][0, :, -1, 1]
        gridlons[-1, -1] = gid['longitude_bounds'][0, -1, -1, 2]

        gridlats = gridlats[2000:2501, :]
        gridlons = gridlons[2000:2501, :]

        dset = fid['PRODUCT/carbonmonoxide_total_column']
        co_data = np.squeeze(dset[0, 2000:2500, :]).astype(float)
        fillvalue = float.fromhex('0x1.ep+122')
        co_data[(co_data == fillvalue)] = np.nan
        print(gridlons.shape, gridlats.shape)
        print(co_data.shape)

    plot = S5Pplot('test_plot_co_h5py.pdf')
    plot.draw_signal(co_data,
                     sub_title='CO orbit={}'.format(orbit))
    plot.close()

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.pcolormesh(gridlons, gridlats, co_data, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.show()


# -------------------------
def read_using_lv2_io(args):
    """
    Perform read using LV2io and use S5Pplot to create the figure
    """
    from pys5p.lv2_io import LV2io

    with LV2io(args.input_file) as lv2:
        print(lv2)
        print(lv2.get_algorithm_version())
        print(lv2.get_product_version())
        print(lv2.get_creation_time())
        print(lv2.get_coverage_time())
        orbit = lv2.get_orbit()

        geo = lv2.get_geo_bounds()
        co_data = lv2.get_msm_data('carbonmonoxide_total_column',
                                   fill_as_nan=True)

    plot = S5Pplot('test_plot_lv2_io.pdf')
    plot.draw_signal(co_data,
                     sub_title='CO orbit={}'.format(orbit))
    plot.close()

    plot = S5Pgeoplot('test_plot_lv2_geo.pdf')
    plot.draw_geo_blocks(geo['longitude'], geo['latitude'], co_data)
    plot.close()

# -------------------------
def read_using_co_msm(args):
    """
    Perform read using h5py and use S5Pplot to create the figure
    """
    from pys5p.s5p_msm import S5Pmsm

    with h5py.File(args.input_file, 'r') as fid:
        print(fid.attrs['title'])
        print(fid.attrs['date_created'])
        print(fid.attrs['algorithm_version'])
        print(fid.attrs['product_version'])
        print(fid.attrs['orbit'])
        print(fid.attrs['time_coverage_start'],
              fid.attrs['time_coverage_end'])
        print(fid['PRODUCT/carbonmonoxide_total_column'].shape)
        orbit = fid.attrs['orbit'][0]
        
        dset = fid['PRODUCT/carbonmonoxide_total_column']
        co_data = S5Pmsm(dset)
        co_data.fill_as_nan()
        print(co_data.coords)
        

    plot = S5Pplot('test_plot_co_msm.pdf')
    plot.draw_signal(co_data,
                     sub_title='CO orbit={}'.format(orbit))
    plot.close()


# - main function --------------------------------------
def main():
    """
    main function when called from the command-line
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='{}: show SWIR data of L1B_RADIANCE product'.format(
            Path(__file__).name))
    parser.add_argument('input_file', nargs='?', type=str, default=None,
                        help='read from INPUT_FILE')
    parser.add_argument('--msm_mode', default=False, action='store_true',
                        help=('use s5p_msm to read the data'))

    args = parser.parse_args()
    if args.input_file is None:
        parser.print_help()
        return

    if args.msm_mode:
        read_using_co_msm(args)
    else:
        read_using_co_h5py(args)
        read_using_lv2_io(args)


# - main code --------------------------------------
if __name__ == '__main__':
    main()
