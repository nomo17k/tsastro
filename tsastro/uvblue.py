#!/usr/bin/env python2.6
"""
The uvblue package contains the utility class 'UVBlue' and the
function 'broaden_spectrum' which are functionally equivalent to the
IDL programs readUVBLUE.pro and broadenUVBLUE.pro in the official
UVBLUE distribution.  The class and function can be imported when
uvblue is used as a package.

When this script is invoked from a command line, the given spectrum is
either plotted using Matplotlib (-p) or dumped to standard output
(-t).

The library of stellar spectra to be used with this module can be
downloaded from the following website:

  http://www.inaoep.mx/~modelos/uvblue/uvblue.html

"""
from __future__ import division
from __future__ import print_function
import gzip
import logging
from optparse import OptionParser
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


__version__ = '20120628'
__author__ = 'Taro Sato'
__author_email__ = 'ubutsu@gmail.com'


# logging configuration
logger = logging.getLogger('uvblue')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
info = logger.info


def _uncompress(filename):
    return gzip.GzipFile(filename=filename, mode='rb')


def _fwhm2kern(fwhm, lamb, center, nsig):
    idx = np.searchsorted(lamb, center)
    if not (0 < idx < len(lamb)):
        raise RuntimeError("center must be in (%d, %d)." % (lamb[0], lamb[-1]))
    dlamb = lamb[idx] - lamb[idx - 1]
    sigma = fwhm / 2.35482
    nk = 2. * nsig * sigma / dlamb
    if nk % 2 == 0:
        nk += 1
    lamk = (np.arange(0, nk) - float(nk // 2)) * dlamb
    kern = np.exp(-0.5 * (lamk / sigma)**2)
    return kern / kern.sum()


def _res2kern(res, _res, _reslamb, nsig):
    res2 = np.sqrt((res * _res)**2 / (_res**2 - res**2))
    imax = int(nsig / 2.35482 * (_reslamb / res2))
    ik = np.arange(0, 2 * imax + 1) - imax
    kern = np.exp(-0.5 * (2.35482 * ik * (res2 / _reslamb))**2)
    return kern / kern.sum()


def _vel2kern(vel, _res, _reslamb, nsig):
    c = 2.99792458e5
    res = c / vel
    return _res2kern(res, _res, _reslamb, nsig)


def broaden_spectrum(lamb, flam, fwhm=None, res=None, vel=None, center=None, nsig=5.,
                     _res=10000., _reslamb=40000.):
    """
    Given the spectrum, degrade its resolution with a Gaussian kernel
    of given size.

    Inputs:

      lamb -- wavelengths in angstroms
      flam -- fluxes
      fwhm -- output FWHM in angstrom (computed at wavelength
              specified by the center keyword argument
      center -- wavelength at which fwhm is computed
      res -- output resolving power [R = lambda / (delta lambda)]
      vel -- output velocity dispersion in km/s
      nsig -- maximum Gaussian kernel width in sigma

    Output:

      Broadened spectrum (just flam).
    """
    if (fwhm is not None and center is not None) and (res is vel is None):
        kern = _fwhm2kern(fwhm, lamb, center, nsig)
    elif res is not None and (fwhm is vel is None):
        kern = _res2kern(res, _res, _reslamb, nsig)
    elif vel is not None and (fwhm is res is None):
        kern = _vel2kern(vel, _res, _reslamb, nsig)
    else:
        return flam
    info('Broadening the spectrum...')
    flam = convolve(flam, kern, mode='same')
    return flam


class UVBlue(object):
    """
    This object stores a high-resolution UV spectrum from UVBLUE library.

    After reading a spectrum from a file, the following attributes are
    available:

      lamb -- wavelengths in angstrom
      flam -- emergent fluxes in erg s^-1 cm^-2 A^-1
      fclam -- continuum fluxes in erg s^-1 cm^-2 A^-1
      fnlam -- normalized fluxes (bolometric integral integrates to one)
      residual -- residual fluxes (set to zero where fclam is zero)
      teff -- effective temperature (K)
      logg -- surface gravity log(g)
      metallicity -- metallicity [M/H]
      resolution -- spectral resolution (R)
      ratio -- ratio between wavelengths in adjacent pixels
    """

    def __init__(self, filename):
        """
        Input:

          filename -- compressed spectrum file (.gz)
        """
        self.read(filename)

    def read(self, filename):
        """
        Read a compressed UVBLUE spectrum file.

        Input:

          filename -- compressed spectrum file (.gz)
        """

        f = _uncompress(filename)

        s = f.next().split()
        teff, grav, metal = float(s[0]), float(s[1]), float(s[2])
        s = f.next().split()
        nwl, wlbeg, wlend = float(s[0]), float(s[1]), float(s[2])
        s = f.next().split()
        resolu, ratio = float(s[0]), float(s[1])

        info('Reading spectrum... Teff: %d  Log g: %.1f  [M/H]: %+.1f'
             % (teff, grav, metal))

        lamb = wlbeg * ratio ** np.arange(nwl, dtype=float)
        res = np.zeros(nwl)
        flam = []
        fclam = []
        for o in f:
            o = o.split()
            flam.append(float(o[0]))
            fclam.append(float(o[1]))
        f.close()

        flam = np.asarray(flam)
        fclam = np.asarray(fclam)

        # Stefan-Boltzmann const (used by Kurucz in his codes):
        sigma = 5.6697e-5

        fnlam = flam / (sigma * teff**4)

        ind1 = (fclam > 0.)
        ind0 = np.logical_not(fclam)
        if ind0.sum() > 0:
            res[ind0] = 0.
        res[ind1] = flam[ind1] / fclam[ind1]

        info('%d wavelength points read.' % nwl)

        self.teff = teff
        self.logg = grav
        self.metallicity = metal
        self.resolution = resolu
        self.ratio = ratio
        self.lamb = lamb
        self.flam = flam
        self.fclam = fclam
        self.fnlam = fnlam
        self.residual = res


def main(filename, plot, text, fwhm, res, vel, center):
    o = UVBlue(filename)
    xs, ys = o.lamb, o.flam
    ys = broaden_spectrum(xs, ys,
                          fwhm=fwhm, res=res, vel=vel, center=center)
    if text:
        for i in range(len(xs)):
            print('%.5e %.5e' % (xs[i], ys[i]))
    if plot:
        plt.plot(xs, ys, 'b.-')
        plt.xlabel('wavelength [angstrom]')
        plt.ylabel('flam [erg s^-1 cm^-2 A^-1]')
        plt.show()


if __name__ == '__main__':
    usage = ('usage: %prog [OPTIONS] FILENAME')
    p = OptionParser(usage=usage,
                     description=__doc__.strip(),
                     version='%prog '+__version__)
    p.add_option('-p', '--plot', action='store_true', default=False,
                 help='set for plotting the spectrum')
    p.add_option('-t', '--text', action='store_true', default=False,
                 help='set for generating wavelength vs flam table')
    p.add_option('--resolution', action='store', type='float', default=-1,
                 help='spectral resolving power (R)')
    p.add_option('--fwhm', action='store', type='float', default=-1,
                 help='FWHM in angstrom')
    p.add_option('--center', action='store', type='float', default=-1,
                 help='central wavelength in angstrom for computing FWHM')
    p.add_option('--velocity', action='store', type='float', default=-1,
                 help='velocity in km/s')
    p.add_option('-v', '--verbose', action='store_true', default=False,
                 help='set for verbose output')
    opts, args = p.parse_args()

    if len(args) != 1:
        p.error('Must specify the file.')
    filename = args[0]
    if not os.path.isfile(filename):
        p.error('File %s is not a valid filename.' % filename)

    fwhm = float(opts.fwhm) if opts.fwhm > 0 else None
    res = float(opts.resolution) if opts.resolution > 0 else None
    vel = float(opts.velocity) if opts.velocity > 0 else None
    center = opts.center

    logger.setLevel({True: logging.INFO,
                     False: logging.ERROR}[opts.verbose])

    main(filename, opts.plot, opts.text, fwhm, res, vel, center)
