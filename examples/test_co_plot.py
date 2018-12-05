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

import numpy as np

from pys5p.s5p_plot import S5Pplot
from pys5p.s5p_geoplot import S5Pgeoplot


# -------------------------
def read_using_lv2_io(args):
    """
    Perform read using LV2io and use S5Pplot to create the figure
    """
    from pys5p.lv2_io import LV2io

    with LV2io(args.input_file) as lv2:
        print(lv2)
        print(lv2.science_product)
        print(lv2.get_algorithm_version())
        print(lv2.get_processor_version())
        print(lv2.get_product_version())
        print(lv2.get_creation_time())
        print(lv2.get_coverage_time())
        orbit = lv2.get_orbit()
        if orbit is None:
            orbit = -1

        geo = lv2.get_geo_data()
        for key in geo:
            print(key, geo[key].shape)

        if lv2.science_product:
            data_sel = np.s_[750:-750, :]
        else:
            data_sel = np.s_[0, 750:-750, :]
        geo = lv2.get_geo_bounds(data_sel=data_sel)
        for key in geo:
            print(key, geo[key].shape)

        extent = [-180, 180, -30, 30]
        data_sel, geo = lv2.get_geo_bounds(extent=extent)
        print('data_sel: ', data_sel)
        for key in geo:
            print(key, geo[key].shape)

        if lv2.science_product:
            msm_name = 'co_column'
        else:
            msm_name = 'carbonmonoxide_total_column'
        co_data = lv2.get_dataset(msm_name, data_sel=data_sel,
                                  fill_as_nan=True)
        co_msm = lv2.get_data_as_s5pmsm(msm_name, data_sel=data_sel,
                                        fill_as_nan=True)

    plot = S5Pplot('test_plot_lv2_io.pdf')
    plot.draw_signal(co_data, sub_title='CO orbit={}'.format(orbit))
    plot.draw_signal(co_msm, sub_title='CO orbit={}'.format(orbit))
    plot.close()

    plot = S5Pgeoplot('test_plot_lv2_geo.pdf')
    plot.draw_geo_msm(geo['longitude'], geo['latitude'], co_msm,
                      title='CO orbit={}'.format(orbit), whole_globe=True)
    plot.draw_geo_msm(geo['longitude'], geo['latitude'], co_msm,
                      title='CO orbit={}'.format(orbit))
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

    args = parser.parse_args()
    if args.input_file is None:
        parser.print_help()
        return

    read_using_lv2_io(args)


# - main code --------------------------------------
if __name__ == '__main__':
    main()
