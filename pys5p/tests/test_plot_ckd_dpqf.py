"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on S5Pplot.draw_quality

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  Standard 3-clause BSD
"""


#-------------------------
def test_ckd_dpqf():
    """
    Check S5Pplot.draw_quality

    """
    from ..ckd_io   import CKDio
    from ..s5p_plot import S5Pplot

    # read pixel-quality CKD
    ckd = CKDio()
    dpqm = ckd.get_swir_dpqm()

    # generate figure
    plot = S5Pplot('test_plot_ckd_dpqm.pdf')
    plot.draw_quality(dpqm, title='ckd.dpqf.detector4.nc',
                      sub_title='dpqf_map')
    del plot

if __name__ == '__main__':
    test_ckd_dpqf()
