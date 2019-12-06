from setuptools import setup

def readme():
    with open('README.rst') as fp:
        return fp.read()

setup(name='pys5p',
      use_scm_version={"root": ".",
                       "relative_to": __file__,
                       "fallback_version": "1.0.1"},
      setup_requires=['setuptools_scm'],
      description='Software package to access S5p Tropomi data products',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License (standard 3-clause)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
      ],
      url='https://github.com/rmvanhees/pys5p',
      author='Richard van Hees',
      author_email='r.m.van.hees@sron.nl',
      maintainer='Richard van Hees',
      maintainer_email='r.m.van.hees@sron.nl',
      license='BSD-3-Clause',
      packages=['pys5p'],
      install_requires=[
          'numpy>=1.17',
          'scipy>=1.3',
          'h5py>=2.9',
          'matplotlib>=3.1',
          'Cartopy'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
