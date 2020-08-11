"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on S5Pplot.draw_signal

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import numpy as np

from pys5p.get_data_dir import get_data_dir
from pys5p.ocm_io import OCMio
from pys5p.s5p_msm import S5Pmsm
from pys5p.s5p_plot import S5Pplot


#-------------------------
def test_frame():
    """
    Check class OCMio and S5Pplot.draw_signal

    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    if not Path(data_dir, 'OCM').is_dir():
        return
    msmlist = list(Path(data_dir, 'OCM').glob('*'))
    sdirlist = list(Path(msmlist[0]).glob('*'))

    # read background measurements
    icid = 31523

    # Read BAND7 product
    product_b7 = 'trl1brb7g.lx.nc'
    ocm_product = str(Path(sdirlist[0], product_b7))
    print(ocm_product)

    # open OCAL Lx poduct
    ocm7 = OCMio(ocm_product)

    # select data of measurement(s) with given ICID
    assert ocm7.select(icid) > 0
    dict_b7 = ocm7.get_msm_data('signal')
    dict_b7_std = ocm7.get_msm_data('signal_noise_vals')

    # Read BAND8 product
    product_b8 = 'trl1brb8g.lx.nc'
    ocm_product = str(Path(sdirlist[0], product_b8))

    # open OCAL Lx poduct
    ocm8 = OCMio(ocm_product)

    # select data of measurement(s) with given ICID
    assert  ocm8.select(icid) > 0
    dict_b8 = ocm8.get_msm_data('signal')
    dict_b8_std = ocm8.get_msm_data('signal_noise_vals')

    # Combine band 7 & 8 data
    msm = None
    msm_err = None
    for key in dict_b7:
        print(key)
        if msm is None:
            msm = S5Pmsm(dict_b7[key])
            msm_err = S5Pmsm(dict_b7_std[key])
        else:
            msm.concatenate(S5Pmsm(dict_b7[key]), axis=0)
            msm_err.concatenate(S5Pmsm(dict_b7_std[key]), axis=0)
        print(key, msm.value.shape, msm.coords._fields)
    msm.nanmedian(data_sel=np.s_[1:-1, ...], axis=0)
    msm_err.nanmedian(data_sel=np.s_[1:-1, ...], axis=0)
    print('msm_b7: ', msm.name, msm.value.shape, msm.coords._fields)

    msm_b8 = None
    msm_b8_err = None
    for key in dict_b8:
        if msm_b8 is None:
            msm_b8 = S5Pmsm(dict_b8[key])
            msm_b8_err = S5Pmsm(dict_b8_std[key])
        else:
            msm_b8.concatenate(S5Pmsm(dict_b8[key]), axis=0)
            msm_b8_err.concatenate(S5Pmsm(dict_b8_std[key]), axis=0)
        print(key, msm_b8.value.shape, msm_b8.coords._fields)
    msm_b8.nanmedian(data_sel=np.s_[1:-1, ...], axis=0)
    msm_b8_err.nanmedian(data_sel=np.s_[1:-1, ...], axis=0)
    print('msm_b8: ', msm_b8.name, msm_b8.value.shape, msm_b8.coords._fields)

    # combine both bands
    msm.concatenate(msm_b8, axis=1)
    msm_err.concatenate(msm_b8_err, axis=1)
    print('msm: ', msm.name, msm.value.shape)

    # Generate figure
    plot = S5Pplot('test_plot_frame.pdf')
    plot.draw_signal(msm,
                     title=ocm7.get_attr('title'),
                     sub_title='signal ICID={}'.format(icid),
                     fig_info=None)
    plot.draw_signal(msm_err,
                     title=ocm7.get_attr('title'),
                     sub_title='signal_error_vals ICID={}'.format(icid),
                     fig_info=None)
    plot.draw_hist(msm, msm_err,
                   title=ocm7.get_attr('title'), fig_info=None)
    plot.close()
    ocm7.close()
    ocm8.close()

if __name__ == '__main__':
    test_frame()
