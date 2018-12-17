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

# -------------------------
def read_using_icm_io(args):
    """
    Perform read of SWIR irradiance data using ICMio
    """
    from pys5p.biweight import biweight
    from pys5p.icm_io import ICMio
    from pys5p.s5p_plot import S5Pplot

    geo_dset='solar_azimuth_angle,solar_elevation_angle'

    plot = S5Pplot('test_plot_icm_irr.pdf')

    result = {}
    with ICMio(args.input_file) as icm:
        print(icm)
        print(icm.get_processor_version())
        print(icm.get_creation_time())
        print(icm.get_coverage_time())
        orbit = icm.get_orbit()
        print('orbit = {}'.format(orbit))
        msm_list = icm.find('SOLAR_IRRADIANCE')
        for msm_type in msm_list:
            print(msm_type)
            bands = icm.select(msm_type)
            print(bands)
            if not bands:
                continue
            res = icm.get_geo_data(geo_dset=geo_dset)
            for key in geo_dset.split(','):
                print(key, res[key].shape)
            irr = icm.get_msm_data('irradiance')
            indx = np.where(np.abs(res['solar_elevation_angle']) <= 1.)[0]
            irr_med = biweight(irr[indx, :, :], axis=0)
            print(irr.shape, irr_med.shape)
            plot.draw_signal(biweight(irr / irr_med, axis=0))
            plot.draw_signal(biweight(irr / irr_med, axis=1))
            plot.draw_signal(biweight(irr / irr_med, axis=2))
            print(res['solar_elevation_angle'][indx])
            result[msm_type] = irr[indx, :, :]

    plot.close()
    return result
        
# - main function --------------------------------------
def main():
    """
    main function when called from the command-line
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='{}: show SWIR irradiance data of L1B_ICM product'.format(
            Path(__file__).name))
    parser.add_argument('input_file', nargs='?', type=str, default=None,
                        help='read from INPUT_FILE')

    args = parser.parse_args()
    if args.input_file is None:
        parser.print_help()
        return

    res = read_using_icm_io(args)
    for key in res:
        print(key, res[key].shape)


# - main code --------------------------------------
if __name__ == '__main__':
    main()
