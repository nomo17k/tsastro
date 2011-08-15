#!/usr/bin/env python2.6
"""
Astro computing utilities by Taro Sato.

This is a package to collect miscellaneous Python astrophysical
computing utility modules and scripts used by Taro Sato.  It is for
organizing modules and scripts that are not substantial enough to be
made into stand-alone packages.
"""
import glob
import os
import sys

# BEFORE importing distutils, remove MANIFEST. distutils doesn't
# properly update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
from distutils.core import setup, Extension


__version__ = '20110101'
__author__ = 'Taro Sato'
__author_email__ = 'ubutsu@gmail.com'


# Check installed Python version.
python_ver = (2, 6, 1, 'final', 0)
if not hasattr(sys, 'version_info') or sys.version_info < python_ver:
    raise SystemExit('Python %s or later required.'
                     % '.'.join([str(c) for c in python_ver[:3]]))


def make_descriptions(docstr=__doc__):
    """
    Make __doc__ into short and long package descriptions.
    """
    docstrs = docstr.strip().split("\n")
    description = docstrs[0].strip()
    long_description = "\n".join(docstrs[2:])
    return description, long_description

def get_scripts():
    """
    Return paths of scripts to install
    """
    paths = ['bin/addartobj',
             'bin/convgauss',
             'bin/ds9reg2fits',
             'bin/fitshead',
             'bin/radprof',
             'bin/sexcat2fits',
             'bin/simbgim',
             'bin/stackmasks',
             'bin/xyonds9']
    #paths.extend(glob.glob('bin/*.py')]
    return paths


def main():
    descr_short, descr_long = make_descriptions()

    setup(name='tsastro',
          version=__version__,
          author=__author__,
          author_email=__author_email__,
          maintainer=__author__,
          maintainer_email=__author_email__,
          url='',
          description=descr_short,
          long_description=descr_long,
          download_url='',
          platforms=['Linux'],
          license='GPL',
          packages=['tsastro'],
          package_dir={'tsastro': 'tsastro'},
          scripts=get_scripts(),
          ext_modules=[],
          #requires=['pyraf', 'pysao']
          )


if __name__ == "__main__":
    main()
