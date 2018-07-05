from pathlib import Path

from setuptools import setup
from setuptools_scm import get_version

version_py = str(Path('.', 'pys5p', 'version.py').resolve())
__version__ = get_version(root='.', relative_to=__file__, write_to=version_py)

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pys5p',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='Software package to access S5p Tropomi data products',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License (standard 3-clause)',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
      ],
      url='https://github.com/rmvanhees/pys5p',
      author='Richard van Hees',
      author_email='r.m.van.hees@sron.nl',
      maintainer='Richard van Hees',
      maintainer_email='r.m.van.hees@sron.nl',
      license='BSD',
      packages=['pys5p'],
      install_requires=[
          'setuptools-scm>=2.1',
          'numpy>=1.14',
          'scipy>=1.0',
          'h5py>=2.8',
          'matplotlib>=2.0',
          'Cartopy>=0.15',
          'Shapely>=1.5',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
