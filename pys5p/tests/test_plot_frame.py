from __future__ import absolute_import
from __future__ import print_function

import os.path

import matplotlib

from unittest import TestCase

matplotlib.use('TkAgg')

#-------------------------
def test_frame():
    """
    Let the user test the software!!!

    Please use the code as tutorial
    """
    from pys5p.ocm_io import OCMio
    from pys5p.s5p_plot import S5Pplot
    
    if os.path.isdir('/Users/richardh'):
        data_dir = '/Users/richardh/Data/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'
    else:
        data_dir ='/nfs/TROPOMI/ocal/proc_knmi/2015_02_23T01_36_51_svn4709_CellEarth_CH4'

    icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = os.path.join( data_dir, 'before_strayl_l1b_val_SWIR_2',
                                product_b7 )
    # open OCAL Lx poduct
    ocm7 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm7.select( icid ) > 0:
        dict_b7 = ocm7.get_msm_data( 'signal' )
        dict_b7_std = ocm7.get_msm_data( 'signal_error_vals' )

    # Read BAND8 product
    product_b8 = 'trl1brb8g.lx.nc'
    ocm_product = os.path.join( data_dir, 'before_strayl_l1b_val_SWIR_2',
                                product_b8 )
    # open OCAL Lx poduct
    ocm8 = OCMio( ocm_product )

    # select data of measurement(s) with given ICID
    if ocm8.select( icid ) > 0:
        dict_b8 = ocm8.get_msm_data( 'signal' )
        dict_b8_std = ocm8.get_msm_data( 'signal_error_vals' )

    # Combine band 7 & 8 data
    # del dict_b7['ICID_31524_GROUP_00000']
    # del dict_b8['ICID_31524_GROUP_00000']
    for key in dict_b7:
        print( key, dict_b7[key].shape )
    data = ocm7.band2channel( dict_b7, dict_b8,
                              mode=['combine', 'median'],
                              skip_first=True, skip_last=True )
    error = ocm7.band2channel( dict_b7_std, dict_b8_std,
                               mode=['combine', 'median'],
                               skip_first=True, skip_last=True )

    # Generate figure
    figname = os.path.basename(data_dir) + '.pdf'
    plot = S5Pplot( figname )
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

#--------------------------------------------------
if __name__ == '__main__':
    print( '*** Info: call function test_frame()')
    test_frame()
