Installing pyS5p
================

Python Distributions
--------------------
If you use a Python Distribution, the installation of pyS5p can be done on
the command line via:

>  `conda install pys5p`

for [Anaconda](https://www.anaconda.com/)/[MiniConda](http://conda.pydata.org/miniconda.html).


Wheels
------
I you have an existing Python installation, pyS5p can be installed via pip
from PyPI:

>  `pip install pys5p`


Source installation
--------------------

The latest release of pyS5p is available from
[gitHub](https://github.com/rmvanhees/pys5p).

Once you have satisfied the requirements detailed below, simply run:

>  `pip install .`

or

>  'pip install . --user`

### Requirements

These external packages are required to install pyS5p or gain access to
significant pyS5p functionality.

It is adviced to use pip to install the Python packages and to use the latest
versions of the software libraries. However, many of these packages are also
available in Linux package managers such as aptitude and yum.

**python** 3.7 or later (https://www.python.org/)
    pyS5p requires Python 3.7 or later.

**Cython** 0.25 or later (http://cython.org/)
    The Cython compiler for writing C extensions for the Python language.

**numpy** 1.17 or later (http://www.numpy.org/)
    Python package for scientific computing including a powerful N-dimensional array object.

**h5py** 2.10  or later (https://www.h5py.org/)
    HDF5 for Python

**matplotlib** 3.1 or later (http://matplotlib.org/)
    Python package for 2D plotting.  This package is required for any graphical capability.

**PROJ.4** 7.0 or later (https://trac.osgeo.org/proj/)
    Cartographic Projections library.

**GEOS** 3.7 or later (https://trac.osgeo.org/geos/)
    GEOS is an API of spatial predicates and functions for processing geometry written in C++.

**shapely** 1.7 0r later (https://github.com/Toblerity/Shapely)
    Shapely is a BSD-licensed Python package for manipulation and analysis of planar geometric objects.

**cartopy** 0.17 or later (http://scitools.org.uk/cartopy/)
    Cartopy is a Python package designed to make drawing maps for data analysis and visualisation as easy as possible.
    
**setuptools-scm** 3.1 or later (https://github.com/pypa/setuptools_scm/)
    The blessed package to manage your versions by scm tags
