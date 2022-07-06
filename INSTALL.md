Installing pys5p
================

Wheels
------
I you have an existing Python (v3.8+) installation, pys5p can be installed
via pip from PyPI:

    pip install pys5p [--user]


Python Distributions
--------------------
If you use a Python Distribution, the installation of pyS5p can be done on
the command line via:

    conda install pys5p

for [Anaconda](https://www.anaconda.com/)/[MiniConda](http://conda.pydata.org/miniconda.html).


Install from source
-------------------

The latest release of pys5p is available from
[gitHub](https://github.com/rmvanhees/pys5p).
Where you can download the source code as a tar-file or zipped archive.
Or you can use git do download the repository:

    git clone https://github.com/rmvanhees/pys5p.git

Before you can install pys5p, you need:

* Python version 3.8+ with development headers
* HDF5, installed with development headers
* netCDF4, installed with development headers

And have the following Python modules available:

* numpy v1.19+
* h5py v3.5+
* netCDF4 v1.5+
* xarray v0.20+

The software is known to work using:

* HDF5 v1.8.21, netCDF4 v4.7.3 and python-netCDF4 v1.5+
* HDF5 v1.10+, netCDF4 v4.7.3 or v4.8+ and python-netCDF4 v1.5+
* HDF5 v1.12+, netCDF4 v4.8+ and python-netCDF4 v1.5+

You can install pys5p once you have satisfied the requirements listed above.
Run at the top of the source tree:

    python3 -m build
    pip3 install dist/pys5p-<version>.whl [--user]

The Python scripts can be found under `/usr/local/bin` or `$USER/.local/bin`.


Known Issues
------------

* You may need to use the environment variable SETUPTOOLS\_SCM\_PRETEND\_VERSION
if your source tree is not a git clone.
