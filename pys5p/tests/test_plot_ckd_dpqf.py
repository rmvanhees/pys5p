from __future__ import absolute_import
from __future__ import print_function

import os.path

import numpy as np
import h5py

import matplotlib

from unittest import TestCase

matplotlib.use('TkAgg')

#-------------------------
def test_ckd_dpqf():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from pys5p.s5p_plot import S5Pplot

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data'
    else:
        data_dir ='/nfs/TROPOMI/ocal/ckd/ckd_release_swir/dpqf'
    dpqm_fl = os.path.join(data_dir, 'ckd.dpqf.detector4.nc')

    with h5py.File( dpqm_fl, 'r' ) as fid:
        band7 = fid['BAND7/dpqf_map'][:-1,:]
        band8 = fid['BAND8/dpqf_map'][:-1,:]
        dpqm = np.hstack( (band7, band8) )

    # generate figure
    figname = 'ckd.dpqf.detector4.pdf'
    plot = S5Pplot( figname )
    plot.draw_quality( dpqm,
                       title='ckd.dpqf.detector4.nc',
                       sub_title='dpqf_map' )
    del plot


class TestCmd(TestCase):
    def test_basic(self):
        test_ckd_dpqf()

#--------------------------------------------------
if __name__ == '__main__':
    print( '*** Info: call function test_ckd_dpqf()')
    test_ckd_dpqf()
