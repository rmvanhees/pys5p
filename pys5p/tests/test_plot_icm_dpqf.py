"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on S5Pplot.draw_quality

Note
----
Please use the code as tutorial

Copyright (c) 2016 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os.path

from glob import glob
#from unittest import TestCase

import matplotlib

matplotlib.use('TkAgg')

#-------------------------
def test_icm_dpqf():
    """
    Check class OCMio and S5Pplot.draw_quality

    """
    from ..get_data_dir import get_data_dir
    from ..icm_io import ICMio
    from ..s5p_plot import S5Pplot

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = glob(os.path.join(data_dir, 'ICM', 'S5P_TEST_ICM_CA_SIR_*.h5'))
    if len(filelist) == 0:
        return

    # open ICM product
    icm = ICMio(filelist[0])
    print(icm, file=sys.stderr)
    if len(icm.select('DPQF_MAP')) > 0:
        dpqm = icm.get_msm_data('dpqf_map', band='78')

        dpqm_dark = icm.get_msm_data('dpqm_dark_flux', band='78')

        dpqm_noise = icm.get_msm_data('dpqm_noise', band='78')

    # generate figure
    plot = S5Pplot('test_icm_dpq.pdf')
    plot.draw_quality(dpqm,
                      title=filelist[0],
                      sub_title='dpqf_map')
    plot.draw_quality(dpqm_dark,
                      title=filelist[0],
                      sub_title='dpqm_dark_flux')
    plot.draw_quality(dpqm_noise,
                      title=filelist[0],
                      sub_title='dpqm_noise')
    del plot
    del icm

if __name__ == '__main__':
    test_icm_dpqf()
