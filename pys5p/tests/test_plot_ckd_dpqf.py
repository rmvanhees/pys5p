from __future__ import absolute_import
from __future__ import print_function

import os.path

from unittest import TestCase

import numpy as np
import h5py

import matplotlib

matplotlib.use('TkAgg')

#-------------------------
def test_ckd_dpqf():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from ..get_data_dir import get_data_dir
    from ..s5p_plot import S5Pplot

    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    dpqm_fl = os.path.join(data_dir, 'CKD', 'dpqf', 'ckd.dpqf.detector4.nc')

    with h5py.File( dpqm_fl, 'r' ) as fid:
        band7 = fid['BAND7/dpqf_map'][:-1,:]
        band8 = fid['BAND8/dpqf_map'][:-1,:]
        dpqm = np.hstack( (band7, band8) )

    # generate figure
    plot = S5Pplot('test_ckd_dpqf.pdf')
    plot.draw_quality(dpqm, title='ckd.dpqf.detector4.nc',
                      sub_title='dpqf_map')
    del plot


class TestCmd(TestCase):
    def test_basic(self):
        test_ckd_dpqf()
