"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class ICMio

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import argparse

from pathlib import Path

import numpy as np

from pys5p.icm_io import ICMio
from pys5p.s5p_msm import S5Pmsm
from pys5p.s5p_plot import S5Pplot

#--------------------------------------------------
def show_signals(args):
    """
    Perform some simple checks on the ICMio class

    """
    
    icm = ICMio(args.input_file)
    print(icm)
    print(icm.get_processor_version())
    print(icm.get_creation_time())
    print(icm.get_coverage_time())
    orbit = icm.get_orbit()
    
    if icm.select('EARTH_RADIANCE_MODE_{:04}'.format(args.icid)) != '78':
        print('*** NO DATA FOUND')
        return

    res = icm.get_msm_data('radiance_avg')
    print('radiance_avg: ', res.shape)
    print(icm.get_msm_attr('radiance_avg', 'units'))

    plot = S5Pplot('test_plot_icm_io.pdf')
    plot.draw_signal(res,
                     sub_title='orbit={}, ICID={}'.format(orbit, args.icid))

    res = icm.get_msm_data('radiance_avg_col')
    print('radiance_avg_col: ', res.shape)
    plot.draw_trend2d(res[0, ...], time_axis=0, 
                      sub_title='band=7, orbit={}, ICID={}'.format(
                          orbit, args.icid))
    plot.draw_trend2d(res[1, ...], time_axis=0, 
                      sub_title='band=8, orbit={}, ICID={}'.format(
                          orbit, args.icid))

    res = icm.get_msm_data('radiance_avg_row')
    print('radiance_avg_row: ', res.shape)
    plot.draw_trend2d(res, time_axis=1, 
                      sub_title='orbit={}, ICID={}'.format(orbit, args.icid))
    del plot
    del icm
    
#- main function --------------------------------------
def main():
    """
    main function when called from the command-line
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='{}: show SWIR data of ICM product'.format(
            Path(__file__).name))
    parser.add_argument('input_file', nargs='?', type=str, default=None,
                        help='read from INPUT_FILE')
    parser.add_argument( '--icid', default=2, type=int,
                         help=('select ICID'))
    args = parser.parse_args()
    if args.input_file is None:
        parser.print_help()
        return

    show_signals(args)

#- main code --------------------------------------
if __name__ == '__main__':
    main()
