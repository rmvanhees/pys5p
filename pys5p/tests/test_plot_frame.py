from __future__ import absolute_import
from __future__ import print_function

import os.path

from glob import glob
from unittest import TestCase

import matplotlib

matplotlib.use('TkAgg')

#-------------------------
def test_frame():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from .get_data_dir import get_data_dir
    from ..ocm_io import OCMio
    from ..s5p_plot import S5Pplot
    
    # obtain path to directory pys5p-data
    data_dir = get_data_dir()
    msmlist = glob(os.path.join(data_dir, 'OCM', '*'))
    sdirlist = glob(os.path.join(msmlist[0], '*'))

    # read background measurements
    icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = os.path.join( sdirlist[0], product_b7 )
    
    # open OCAL Lx poduct
    ocm7 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm7.select( icid ) > 0:
        dict_b7 = ocm7.get_msm_data( 'signal' )
        dict_b7_std = ocm7.get_msm_data( 'signal_error_vals' )

    # Read BAND8 product
    product_b8 = 'trl1brb8g.lx.nc'
    ocm_product = os.path.join( sdirlist[0], product_b8 )
    
    # open OCAL Lx poduct
    ocm8 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm8.select( icid ) > 0:
        dict_b8 = ocm8.get_msm_data( 'signal' )
        dict_b8_std = ocm8.get_msm_data( 'signal_error_vals' )

    # Combine band 7 & 8 data
    for key in dict_b7:
        print( key, dict_b7[key].shape )
    data = ocm7.band2channel( dict_b7, dict_b8,
                              mode=['combine', 'median'],
                              skip_first=True, skip_last=True )
    error = ocm7.band2channel( dict_b7_std, dict_b8_std,
                               mode=['combine', 'median'],
                               skip_first=True, skip_last=True )

    # Generate figure
    plot = S5Pplot('test_plot_frame.pdf')
    plot.draw_signal( data,
                      data_label='signal',
                      data_unit=ocm7.get_msm_attr('signal', 'units'),
                      title=ocm7.get_attr('title'),
                      sub_title='ICID={}'.format(icid),
                      fig_info=None )
    plot.draw_signal( error,
                      data_label='signal_error_vals',
                      data_unit=ocm7.get_msm_attr('signal_error_vals', 'units'),
                      title=ocm7.get_attr('title'),
                      sub_title='ICID={}'.format(icid),
                      fig_info=None )
    plot.draw_hist( data, error,
                    data_label='signal',
                    data_unit=ocm7.get_msm_attr('signal', 'units'),
                    error_label='signal_error_vals',
                    error_unit=ocm7.get_msm_attr('signal_error_vals', 'units'),
                    title=ocm7.get_attr('title'),
                    sub_title='ICID={}'.format(icid),
                    fig_info=None )
    del plot
    del ocm7
    del ocm8

class TestCmd(TestCase):
    def test_basic(self):
        test_frame()
