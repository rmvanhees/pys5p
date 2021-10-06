"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on class CKDio (xarray version)

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

import numpy as np

from pys5p.lv2_io import LV2io

def read_lv2(l2_product):
    """
    Read Tropomi level 2 product
    """
    with LV2io(l2_product) as lv2:
        # Class properties
        print('science_product: ', lv2.science_product)
        print('orbit: ', lv2.orbit)
        print('algorithm_version: ', lv2.algorithm_version)
        print('processor_version: ', lv2.processor_version)
        print('product_version: ', lv2.product_version)
        if not lv2.science_product:
            print('coverage_time: ', lv2.coverage_time)
        print('creation_time: ', lv2.creation_time)
        # Attributes
        print('get_attr: ', lv2.get_attr('title'))
        if lv2.science_product:
            ds_name = 'h2o_column'
        else:
            ds_name = 'methane_mixing_ratio'
        print('get_attr: ', lv2.get_attr('long_name', ds_name))

        # Time information
        print('ref_time: ', lv2.ref_time)
        print('get_time: ', lv2.get_time())
        # Geolocation
        geo_var = 'latitude_center' if lv2.science_product else 'latitude'
        print('get_geo_data: ', lv2.get_geo_data()[geo_var].shape)
        # Footprints
        geo_var = 'latitude'
        print('get_geo_bounds: ', lv2.get_geo_bounds()[geo_var].shape)
        print('get_geo_bounds: ', lv2.get_geo_bounds(
            data_sel=np.s_[250:300, 100:110])[geo_var].shape)
        # Datasets (numpy)
        if lv2.science_product:
            ds_name = 'h2o_column'
        else:
            ds_name = 'methane_mixing_ratio'
        print('get_dataset: ', lv2.get_dataset(ds_name).shape)
        print('get_dataset: ', lv2.get_dataset(
            ds_name, data_sel=np.s_[250:300, 100:110]).shape)
        # Datasets (xarray)
        if lv2.science_product:
            ds_name = 'h2o_column'
        else:
            ds_name = 'methane_mixing_ratio'
        print('get_data_as_xds: ', lv2.get_data_as_xds(ds_name))


def main():
    """
    Main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description=f'{Path(__file__).name}: run units-test on class LV2io')
    parser.add_argument('lv2_product', nargs=1, type=str, default=None,
                        help='use this Tropomi level2 product')
    args = parser.parse_args()
    print(args)

    read_lv2(args.lv2_product[0])


# - main code --------------------------------------
if __name__ == '__main__':
    main()
