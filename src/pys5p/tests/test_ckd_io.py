"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on ckd_io and s5p_plot

Note
----
Please use the code as tutorial

Copyright (c) 2018 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

import matplotlib as mpl

from pys5p.ckd_io import CKDio
from pys5p.s5p_plot import S5Pplot

mpl.use('Agg')

#-------------------------
def test_ckd_io(ckd_path):
    """
    Read various Tropomi UVN and SWIR CKDs and show their contents
    """
    # open object to read the S5P CKDs
    if ckd_path.is_dir():
        ckd = CKDio(ckd_dir=str(ckd_path))
    else:
        ckd = CKDio(ckd_file=str(ckd_path))
    print(ckd.ckd_file.name)

    # call all exported methods
    for name in dir(ckd):
        if name.startswith("__") or name == 'close':
            continue

        if callable(getattr(ckd, name)):
            print(name, type(getattr(ckd, name)()))

    # open object for report
    plot = S5Pplot('s5p_ckd_report.pdf')

    # read offset CKD and generate figure (SWIR)
    offset = ckd.offset()
    plot.draw_signal(offset, title='Tropomi UVN and SWIR CKD',
                     sub_title=offset.long_name)
    plot.draw_signal(offset.error, title='Tropomi UVN and SWIR CKD',
                     sub_title=offset.long_name + ' (error)')

    # read dark-flux CKD and generate figure (SWIR)
    darkflux = ckd.darkflux()
    plot.draw_signal(darkflux, title='Tropomi UVN and SWIR CKD',
                     sub_title=darkflux.long_name)
    plot.draw_signal(darkflux.error, title='Tropomi UVN and SWIR CKD',
                     sub_title=darkflux.long_name + ' (error)')

    # read background noise CKD and generate figure (SWIR)
    noise = ckd.noise()
    plot.draw_signal(noise, title='Tropomi UVN and SWIR CKD',
                     sub_title=noise.long_name)

    # read saturation CKD and generate figure (SWIR)
    saturation = ckd.saturation()
    plot.draw_signal(saturation, title='Tropomi UVN and SWIR CKD',
                     sub_title=saturation.long_name)

    # read pixel-quality CKD and generate figure (SWIR)
    dpqm = ckd.pixel_quality()
    plot.draw_quality(dpqm, title='Tropomi UVN and SWIR CKD',
                      sub_title=dpqm.long_name)

    # read PRNU CKD and generate figure (UVN & SWIR)
    for bands in ['12', '34', '56', '78']:
        prnu = ckd.prnu(bands=bands)
        plot.draw_signal(prnu, title='Tropomi UVN and SWIR CKD',
                         sub_title=prnu.long_name)

    # read ABSRAD CKD and generate figure (UVN & SWIR)
    for bands in ['12', '34', '56', '78']:
        absrad = ckd.absrad(bands=bands)
        plot.draw_signal(absrad, title='Tropomi UVN and SWIR CKD',
                         sub_title=absrad.long_name)

    # read ABSIRR CKD and generate figure (UVN & SWIR)
    for bands in ['12', '34', '56', '78']:
        for num in [1, 2]:
            absirr = ckd.absirr(qvd=num, bands=bands)
            plot.draw_signal(absirr, title='Tropomi UVN and SWIR CKD',
                             sub_title=absirr.long_name)

    # read wavelength CKD and generate figure (UVN & SWIR)
    for bands in ['12', '34', '56', '78']:
        wavelength = ckd.wavelength(bands=bands)
        plot.draw_signal(wavelength, title='Tropomi UVN and SWIR CKD',
                         sub_title=wavelength.long_name)

    plot.close()
    ckd.close()

def main():
    """
    main function when called from the command-line
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='test of pys5p.ckd_io and pys5p.s5p_plot')
    parser.add_argument('--ckd_dir', default='/nfs/Tropomi/share/ckd')
    parser.add_argument('--ckd_file', default=None)
    args = parser.parse_args()

    if args.ckd_file is None:
        test_ckd_io(Path(args.ckd_dir))
    else:
        test_ckd_io(Path(args.ckd_file))

if __name__ == '__main__':
    main()
