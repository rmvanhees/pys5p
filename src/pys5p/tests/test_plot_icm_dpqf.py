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

License:  BSD-3-Clause
"""
from pathlib import Path
import sys

from pys5p.get_data_dir import get_data_dir
from pys5p.icm_io import ICMio
from pys5p.s5p_plot import S5Pplot


#-------------------------
def test_icm_dpqf():
    """
    Check class OCMio and S5Pplot.draw_quality
    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'ICM').glob('S5P_OPER_ICM_CA_SIR_*.H5'))
    if not filelist:
        return

    # open ICM product
    for flname in filelist:
        icm = ICMio(flname)
        print(icm, file=sys.stderr)
        if icm.select('DPQF_MAP'):
            break

    assert icm.select('DPQF_MAP')
    dpqm = icm.get_msm_data('dpqf_map', band='78')
    dpqm_dark = icm.get_msm_data('dpqm_dark_flux', band='78')
    dpqm_noise = icm.get_msm_data('dpqm_noise', band='78')

    # generate figure
    plot = S5Pplot('test_plot_icm_dpqm.pdf')
    plot.draw_quality(dpqm,
                      title=Path(icm.filename).name,
                      sub_title='dpqf_map')
    plot.draw_quality(dpqm_dark,
                      title=Path(icm.filename).name,
                      sub_title='dpqm_dark_flux')
    plot.draw_quality(dpqm_noise,
                      title=Path(icm.filename).name,
                      sub_title='dpqm_noise')
    plot.draw_quality(dpqm_noise, ref_data=dpqm_dark,
                      title=Path(icm.filename).name,
                      sub_title='dpqm_noise')
    plot.draw_quality(dpqm_noise, add_medians=False,
                      title=Path(icm.filename).name,
                      sub_title='dpqm_noise')
    plot.close()
    icm.close()

if __name__ == '__main__':
    test_icm_dpqf()
