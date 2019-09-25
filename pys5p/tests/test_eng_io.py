"""
This file is part of pyS5p

https://github.com/rmvanhees/pys5p.git

Purpose
-------
Perform (quick) unittest on the class L1BioENG

Note
----
Please use the code as tutorial

Copyright (c) 2017 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

from pys5p.get_data_dir import get_data_dir
from pys5p.l1b_io import L1BioENG

#--------------------------------------------------
def test_rd_eng():
    """
    Perform some simple checks on the L1BioENG class

    """
    # obtain path to directory pys5p-data
    try:
        data_dir = get_data_dir()
    except FileNotFoundError:
        return

    filelist = list(Path(data_dir, 'L1B').glob('S5P_OFFL_L1B_ENG_DB_*.nc'))
    if not filelist:
        return

    for flname in sorted(filelist):
        eng = L1BioENG(flname)
        print(eng)
        print(eng.get_processor_version())
        print(eng.get_creation_time())
        print(eng.get_coverage_time())
        print(eng.get_ref_time())
        print(eng.get_delta_time())
        print(eng.get_msmtset_db())
        print(eng.get_swir_hk_db())
        print(eng.get_swir_hk_db(stats='median'))
        print(eng.get_swir_hk_db(stats='range'))

if __name__ == '__main__':
    test_rd_eng()
