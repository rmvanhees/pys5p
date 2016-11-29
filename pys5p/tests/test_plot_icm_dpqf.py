from __future__ import absolute_import
from __future__ import print_function

import os.path

import matplotlib

from unittest import TestCase

matplotlib.use('TkAgg')

#-------------------------
def test_icm_dpqf():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from pys5p.icm_io import ICMio
    from pys5p.s5p_plot import S5Pplot

    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/S5P_ICM_CA_SIR/001000/2012/09/18'
    elif os.path.isdir('/nfs/TROPOMI/ical/'):
        data_dir = '/nfs/TROPOMI/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    else:
        data_dir = '/data/richardh/Tropomi/ical/S5P_ICM_CA_SIR/001100/2012/09/18'
    fl_name = 'S5P_TEST_ICM_CA_SIR_20120918T131651_20120918T145629_01890_01_001100_20151002T140000.h5'
    icm_product = os.path.join(data_dir, fl_name)

    # open ICM product
    icm = ICMio( icm_product )
    if len(icm.select('DPQF_MAP')) > 0:
        dpqm = icm.get_msm_data( 'dpqf_map', band='78' )

        dpqm_dark = icm.get_msm_data( 'dpqm_dark_flux', band='78' )

        dpqm_noise = icm.get_msm_data( 'dpqm_noise', band='78' )

    # generate figure
    figname = fl_name + '.pdf'
    plot = S5Pplot( figname )
    plot.draw_quality( dpqm,
                       title=fl_name,
                       sub_title='dpqf_map' )
    plot.draw_quality( dpqm_dark,
                       title=fl_name,
                       sub_title='dpqm_dark_flux' )
    plot.draw_quality( dpqm_noise,
                       title=fl_name,
                       sub_title='dpqm_noise' )
    del plot
    del icm
    

class TestCmd(TestCase):
    def test_basic(self):
        test_icm_dpqf()

#--------------------------------------------------
if __name__ == '__main__':
    print( '*** Info: call function test_icm_dpqf()')
    test_icm_dpqf()
