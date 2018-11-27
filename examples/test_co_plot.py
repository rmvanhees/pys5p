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


# --------------------------------------------------
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
        
        lat = np.squeeze(fid['PRODUCT/latitude'])
        lon = np.squeeze(fid['PRODUCT/longitude'])
        dset = fid['PRODUCT/carbonmonoxide_total_column']
        co_data = np.squeeze(dset).astype(np.float64)
        fillvalue = float.fromhex('0x1.ep+122')
        co_data[(co_data == fillvalue)] = np.nan
        
        

    plot = S5Pplot('test_plot_co_h5py.pdf')
    plot.draw_signal(co_data,
                     sub_title='CO orbit={}'.format(orbit))

    # plot.draw_geolocation(lat, lon)
    plot.close()


# --------------------------------------------------
def read_using_s5p_msm(args):
    """
    Perform read using s5p_msm  and use S5Pplot to create the figure
    """
    pass


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


# - main code --------------------------------------
if __name__ == '__main__':
    main()
