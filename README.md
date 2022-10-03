# pyS5p
[![Package Release](https://img.shields.io/pypi/v/pys5p.svg?label=version)](https://pypi.org/project/pys5p/)
[![Package Status](https://img.shields.io/pypi/status/pys5p.svg?label=status)](https://pypi.org/project/pys5p/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pys5p.svg?label=PyPI%20downloads)](https://github.com/rmvanhees/pys5p/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5665827.svg)](https://doi.org/10.5281/zenodo.5665827)

pyS5p provides a Python interface to S5p Tropomi Level-1B (and 2) products.

For more information on the Sentinel 5 precursor mission visit:

* https://earth.esa.int/web/guest/missions/esa-future-missions/sentinel-5P
* http://www.tropomi.eu

For more information on the Tropomi Level-1B products visit:

* http://www.tropomi.eu/documents/level-0-1b-products

## Documentation
Online documentation is available from [Read the Docs](https://pys5p.readthedocs.io).

## Installation
The module pys5p requires Python3.8+ and Python modules: h5py, netCDF4, numpy and xarray.

Installation instructions are provided on [Read the Docs](https://pys5p.readthedocs.io/en/latest/build.html) or in the INSTALL file.

## Note
Most of the plotting related S/W has been moved from pyS5p (v2.1+) to [moniplot](https://pypi.org/project/moniplot).
Removed are th following modules:
* module biweight.py - contains a Python implementation of the Tukey's biweight algorithm.
* module tol_colors.py - definition of colour schemes for lines and maps that also work for colour-blind
people by [Paul Tol](https://personal.sron.nl/~pault/).
* module s5p_plot.py - the class S5Pplot is rewritten and now available as MONplot in the module mon_plot.py.
