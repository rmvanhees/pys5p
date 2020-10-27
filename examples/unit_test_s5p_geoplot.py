"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Performs unit-tests on S5Pplot methods: draw_signal, draw_quality,
   draw_cmp_images, draw_trend1d and draw_lines

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

import numpy as np

from pys5p.lib.plotlib import FIGinfo
from pys5p.l1b_io import L1BioRAD
from pys5p.lv2_io import LV2io
from pys5p.s5p_geoplot import S5Pgeoplot


# --------------------------------------------------
def run_l1b_tests(plot, l1b_product):
    """
    Run unit tests using an L1B product
    """
    print('Run unit tests on S5Pgeoplot using L1BioRAD')

    with L1BioRAD(l1b_product) as l1b:
        orbit = l1b.get_orbit()
        print('orbit: ', orbit)

        _ = l1b.select()

        seq = l1b.sequence()
        geo_sat = l1b.get_geo_data(
            geo_dset='satellite_latitude,satellite_longitude')

        geo = l1b.get_geo_data()

        l1b_data = l1b.get_msm_data('radiance', fill_as_nan=True)

    mask = (seq['icid'] == 12) | (seq['icid'] == 14)
    for key in geo_sat:
        print(key, geo_sat[key][mask].min(), geo_sat[key][mask].max())
    fig_info_in = FIGinfo()
    fig_info_in.add('ICIDs', [12, 14])
    plot.draw_geo_subsat(geo_sat['satellite_longitude'][mask],
                         geo_sat['satellite_latitude'][mask],
                         title='S5Pgeoplot::draw_geo_subsat',
                         sub_title='orbit={}'.format(orbit),
                         fig_info=fig_info_in.copy())

    plot.draw_geo_tiles(geo['longitude'][mask, :],
                        geo['latitude'][mask, :],
                        title='S5Pgeoplot::draw_geo_tiles',
                        sub_title='orbit={}'.format(orbit),
                        fig_info=fig_info_in.copy())

    plot.draw_geo_tiles(geo['longitude'][mask, :], geo['latitude'][mask, :],
                        sequence=seq['sequence'][mask],
                        title='S5Pgeoplot::draw_geo_tiles',
                        sub_title='orbit={}'.format(orbit),
                        fig_info=fig_info_in.copy())

    plot.draw_geo_msm(geo['longitude'][mask, :], geo['latitude'][mask, :],
                      np.nanmean(l1b_data, axis=2)[mask, :],
                      title='S5Pgeoplot::draw_geo_msm',
                      sub_title='orbit={}'.format(orbit),
                      fig_info=fig_info_in.copy())


def run_lv2_tests(plot, lv2_product):
    """
    Run unit tests using an LV2 product
    """
    print('Run unit tests on S5Pgeoplot using LV2io')

    with LV2io(lv2_product) as lv2:
        orbit = lv2.get_orbit()
        print('orbit: ', orbit)

        extent = [-180, 180, -30, 30]
        data_sel, geo = lv2.get_geo_bounds(extent=extent)
        print('data_sel: ', data_sel)
        for key in geo:
            print(key, geo[key].shape)

        if lv2.science_product:
            msm_name = 'co_column'
        else:
            msm_name = 'carbonmonoxide_total_column'
        lv2_msm = lv2.get_data_as_s5pmsm(msm_name, data_sel=data_sel,
                                         fill_as_nan=True)

    print(lv2_msm.long_name)
    plot.draw_geo_msm(geo['longitude'], geo['latitude'], lv2_msm,
                      title=lv2_msm.long_name,
                      sub_title='orbit={}'.format(orbit))


# --------------------------------------------------
def main():
    """
    main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='{}: run units-test on class S5Pgeoplot'.format(
            Path(__file__).name))
    parser.add_argument('--l1b_file', type=str, default=None,
                        help='Use S5P_L1B_RAD_PRODUCT to run unit-test on ...')
    parser.add_argument('--lv2_file', type=str, default=None,
                        help='Use S5P_LV2_PRODUCT to run unit-test on ...')

    args = parser.parse_args()

    if args.l1b_file is None and args.lv2_file is None:
        parser.print_help()
        return

    plot = S5Pgeoplot('unit_test_s5p_geoplot.pdf')
    if args.l1b_file is not None:
        if not Path(args.l1b_file).is_file():
            raise FileNotFoundError('S5P_L1B_RAD_PRODUCT not found')
        run_l1b_tests(plot, args.l1b_file)

    if args.lv2_file is not None:
        if not Path(args.lv2_file).is_file():
            raise FileNotFoundError('S5P_LV2_PRODUCT not found')
        run_lv2_tests(plot, args.lv2_file)

    # close figure
    plot.close()


# - main code --------------------------------------
if __name__ == '__main__':
    main()
