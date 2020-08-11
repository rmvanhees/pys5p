"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class ICMio

Note
----
Please use the code as tutorial

Copyright (c) 2017-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from pathlib import Path

import h5py
import numpy as np

from pys5p.l1b_io import L1BioRAD
from pys5p.biweight import biweight
from pys5p.s5p_msm import S5Pmsm
from pys5p.s5p_plot import S5Pplot


# --------------------------------------------------
def read_using_l1b_io(args):
    """
    Perform read using L1BioRAD and use S5Pplot to create the figure
    """
    l1b = L1BioRAD(args.input_file)
    print(l1b)
    print(l1b.get_processor_version())
    print(l1b.get_creation_time())
    print(l1b.get_coverage_time())
    orbit = l1b.get_orbit()

    l1b.select()
    for key in l1b:
        print('{}: {!r}'.format(key, l1b.__getattribute__(key)))
    geo = l1b.get_geo_data(icid=args.icid, geo_dset='solar_zenith_angle')
    print('geodata: ', geo.dtype.names, geo.shape,
          np.mean(geo['solar_zenith_angle'], axis=1).shape)
    indx = np.where(np.mean(geo['solar_zenith_angle'], axis=1) <= 85)[0]
    res = l1b.get_msm_data('radiance', icid=args.icid, fill_as_nan=True)
    res_b7 = res[indx, ...]
    l1b.close()

    l1b = L1BioRAD(args.input_file.replace('_BD7_', '_BD8_'))
    print(l1b)
    l1b.select()
    res = l1b.get_msm_data('radiance', icid=args.icid, fill_as_nan=True)
    res_b8 = res[indx, ...]
    l1b.close()
    res = np.dstack((res_b7, res_b8))
    print('radiance', res.shape)

    plot = S5Pplot('test_plot_l1b_io.pdf')
    plot.draw_signal(biweight(res, axis=0),
                     sub_title='orbit={}, ICID={}'.format(orbit, args.icid))

    plot.draw_trend2d(biweight(res, axis=2), time_axis=0,
                      sub_title='orbit={}, ICID={}'.format(orbit, args.icid))

    plot.draw_trend2d(biweight(res, axis=1), time_axis=1,
                      sub_title='orbit={}, ICID={}'.format(orbit, args.icid))
    plot.close()


# --------------------------------------------------
def read_using_s5p_msm(args):
    """
    Perform read using s5p_msm  and use S5Pplot to create the figure
    """
    l1b = L1BioRAD(args.input_file)
    orbit = l1b.get_orbit()
    l1b.select()
    geo = l1b.get_geo_data(icid=args.icid, geo_dset='solar_zenith_angle')
    indx = np.where(np.mean(geo['solar_zenith_angle'], axis=1) <= 85)[0]
    l1b.close()

    with h5py.File(args.input_file, 'r') as fid:
        msm_name = '/BAND7_RADIANCE/STANDARD_MODE/OBSERVATIONS/radiance'
        dset = fid[msm_name]
        msm = S5Pmsm(dset, data_sel=np.s_[:, min(indx):max(indx), :, :])
        msm.fill_as_nan()
        print(msm_name, msm.value.shape, msm.coords._fields)

    with h5py.File(args.input_file.replace('_BD7_', '_BD8_'), 'r') as fid:
        msm_name = '/BAND8_RADIANCE/STANDARD_MODE/OBSERVATIONS/radiance'
        dset = fid[msm_name]
        msm_tmp = S5Pmsm(dset, data_sel=np.s_[:, min(indx):max(indx), :, :])
        msm_tmp.fill_as_nan()
        print(msm_name, msm.value.shape, msm.coords._fields)

    msm.concatenate(msm_tmp, axis=2)
    print('msm: ', msm.name, msm.long_name, msm.value.shape)

    plot = S5Pplot('test_plot_s5p_msm.pdf')
    print('step 1')
    msm_tmp = msm.copy()
    print('step 1a', msm_tmp.long_name, msm_tmp.value.shape)
    plot.draw_signal(msm_tmp.biweight(axis=0),
                     sub_title='orbit={}, ICID={}'.format(orbit, args.icid))
    print('step 2')
    msm_tmp = msm.copy()
    print('step 2a', msm_tmp.long_name, msm_tmp.value.shape)
    plot.draw_trend2d(msm_tmp.biweight(axis=2), time_axis=0,
                      sub_title='orbit={}, ICID={}'.format(orbit, args.icid))

    print('step 3')
    msm_tmp = msm.copy()
    print('step 3a', msm_tmp.long_name, msm_tmp.value.shape)
    plot.draw_trend2d(msm_tmp.biweight(axis=1), time_axis=1,
                      sub_title='orbit={}, ICID={}'.format(orbit, args.icid))
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
    parser.add_argument('--icid', default=2, type=int,
                        help=('select ICID'))
    parser.add_argument('--msm_mode', default=False, action='store_true',
                        help=('use s5p_msm to read the data'))

    args = parser.parse_args()
    if args.input_file is None:
        parser.print_help()
        return

    if args.msm_mode:
        read_using_s5p_msm(args)
    else:
        read_using_l1b_io(args)


# - main code --------------------------------------
if __name__ == '__main__':
    main()
