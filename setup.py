import os.path
from setuptools import setup

from setuptools_scm import get_version

__version__ = get_version(root='.', relative_to=__file__)

version_py = os.path.join(os.path.dirname(__file__), 'pys5p', 'version.py')
version_msg = "# Do not edit this file, pipeline versioning is governed by git tags"
with open(version_py, 'w') as fh:
    fh.write(version_msg + os.linesep
             + "__version__=" + repr(__version__) + os.linesep)

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pys5p',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='Software to access S5p Tropomi L1B (offline) products',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License (standard 3-clause)',
          'Programming Language :: Python :: 3.5',
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
          'setuptools-scm>=1.1',
          'h5py>=2.6',
          'numpy>=1.11',
          'matplotlib>=1.5',
          'Cartopy>=0.14',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
