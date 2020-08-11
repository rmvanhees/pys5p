"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform unittest on S5Pplot.draw_geo

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
from pys5p.l1b_io import L1Bio, L1BioRAD
from pys5p.s5p_geoplot import S5Pgeoplot


#-------------------------
def test_geo():
    """
    Check classes L1BioCAL, L1BioRAD and S5Pplot.draw_geo

    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_RA_*.nc'))
    if not filelist:
        return

    # test footprint mode
    with L1BioRAD(filelist[-1]) as l1b:
        print(l1b, file=sys.stderr)
        l1b.select()
        seq = l1b.sequence()
        geo = l1b.get_geo_data()

    # select measurements with ICID=14
    indx = seq['index'][seq['icid'] == 14]
    plot = S5Pgeoplot('test_plot_geo.pdf')
    plot.draw_geo_tiles(geo['longitude'][indx, :],
                        geo['latitude'][indx, :],
                        sequence=seq['sequence'][indx, None])

    # test subsatellite mode
    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_CA_*.nc'))
    with L1Bio(filelist[-1]) as l1b:
        print(l1b, file=sys.stderr)
        l1b.select('BACKGROUND_RADIANCE_MODE_0005')
        geo = l1b.get_geo_data()

    plot.draw_geo_subsat(geo['satellite_longitude'],
                         geo['satellite_latitude'])
    plot.close()

if __name__ == '__main__':
    test_geo()
