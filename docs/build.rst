.. _install:

Installation
============

Wheels
------

It is highly recommended that you use a pre-built wheel of `pys5p` from PyPI.

If you have an existing Python (3.8+) installation (e.g. a python.org download,
or one that comes with your OS), then on Windows, MacOS/OSX, and Linux on
Intel computers, pre-built `pys5p` wheels can be installed via pip
from PyPI::

  pip install [--user] pys5p

OS-Specific remarks
-------------------

On a Debian Bullseye or Ubuntu 22.04 installation,
we have successfully installed `pys5p` as follows::

  sudo apt install python3-numpy python3-scipy
  sudo apt install python3-h5py python3-netCDF4
  pip install --user pys5p

This will also install a working version of the package xarray.

.. important::
   The version of xarray which comes with the Debian package
   `python3-xarray` is too old, and will not work with `pys5p`.

Building from source
--------------------

The latest release of `pys5p` is available from
`gitHub <https://github.com/rmvanhees/pys5p>`_.
You can obtain the source code using::

  git clone https://github.com/rmvanhees/pys5p.git

We develop the code using `Python <https://www.python.org/>`_ 3.10 using the
latest stable release of the libraries
`HDF5 <https://hdfgroup.org/solutions/hdf5>`_ and
`netCDF4 <https://www.unidata.ucar.edu/software/netcdf/>`_,
and Python packages:
`numpy <https://numpy.org>`_, `h5py <https://www.h5py.org>`_,
`netCDF4-python <https://github.com/Unidata/netcdf4-python>`_
and `xarray <https://xarray.dev/>`_.

To compile the code you need the Python packages: setuptools, setuptools-scm
and wheels. Then you can install `pys5p` as follows::

  python3 -m build
  pip3 install dist/pys5p-<version>.whl [--user]

