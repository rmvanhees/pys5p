from setuptools import setup

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
      license='BSD',
      packages=['pys5p'],
      install_requires=[
          'setuptools-scm>=1.1',
          'h5py>=2.6',
          'numpy>=1.11',
      ],
      zip_safe=False)
