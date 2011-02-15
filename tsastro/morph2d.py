#!/usr/bin/env python2.6
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals
import numpy as np
from numpy import arctan, cos, exp, pi, sin, sqrt
from scipy.special import gamma


__version__ = '20101228'


class Morph2D(object):
    """
    2D morphology class for imaging

    The parameters are in data units (DN, pixels, etc.) for the
    convenience of data representation.  For more intuitive interface,
    subclass this and have users input physical parameters and convert
    them into data units.
    """

    def __init__(self, **kwargs):
        # x and y coordinates in which the model is centered within
        # the (0, 0) pixel; "within" since x and y can be fractions
        self.xs, self.ys = None, None

        # pixel indices of the region boundary for updating
        self.ixmin, self.ixmax = None, None
        self.iymin, self.iymax = None, None

        # maximum half-width of the region to be updated.
        self.xhw, self.yhw = np.ceil(1), np.ceil(1)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def parameters(self):
        return []

    def _prepare_coords(self, data, x, y):
        """
        Compute the boundary indices defining a square region to be
        updated and the x and y coordinates centered at the object
        located at (x, y).

        Returns True if the object is outside the image; False
        otherwise.

        Input:

          data -- a 2D image array to which the object is added
          x, y -- center coordinates of the object to be generated
        """
        ysize, xsize = data.shape
        
        # find the region in data that needs updating.
        maxrad = max(self.xhw, self.yhw)
        ixmin, ixmax = int(round(x - maxrad)), int(round(x + maxrad))
        iymin, iymax = int(round(y - maxrad)), int(round(y + maxrad))

        if ixmin >= xsize or ixmax < 0 or iymin >= ysize or iymax < 0:
            # model is entirely outside the data so nothing to be done.
            self.ixmin = self.ixmax = self.iymin = self.iymax = None
            self.xs = self.ys = None
            return True

        # only update where parent data exist.
        ixmin = 0 if ixmin < 0 else ixmin
        ixmax = xsize - 1 if ixmax >= xsize else ixmax
        iymin = 0 if iymin < 0 else iymin
        iymax = ysize - 1 if iymax >= ysize else iymax

        dx, dy = ixmax - ixmin + 1, iymax - iymin + 1
        ys = np.repeat(np.arange(iymin, iymax + 1), dx,
                       axis=0).reshape((dy, dx))
        xs = (np.repeat(np.arange(ixmin, ixmax + 1), dy,
                       axis=0).reshape((dx, dy))).transpose()
        
        self.xs, self.ys = xs - x, ys - y
        self.ixmin, self.ixmax = ixmin, ixmax + 1
        self.iymin, self.iymax = iymin, iymax + 1
        return False

    def generate(self, x, y, data):
        """
        Generate a 2D profile centered at (x, y) on a 2D image.
        """
        return data


class Gaussian(Morph2D):
    """
    Gaussian approximation to a point source.
    """

    def __init__(self, ftot=1., gsig=2., hwsigma=5., **kwargs):
        """
        Input:

          ftot -- Total flux
          gsig -- Gaussian sigma in pixels
          hwsigma -- half-width size in gsig of stamp enclosing the model
        """
        super(Gaussian, self).__init__(**kwargs)
        self.ftot = ftot
        self.gsig = gsig
        self.hwsigma = hwsigma
        self.xhw = np.ceil(hwsigma * gsig)
        self.yhw = np.ceil(hwsigma * gsig)

    @property
    def parameters(self):
        return [self.ftot, self.gsig]

    def generate(self, x, y, data):
        if self._prepare_coords(data, x, y):
            # model is entirely outside the data so nothing to be done
            return data
        rs = sqrt(self.xs**2 + self.ys**2)  # in polar coords
        dref = data[self.iymin:self.iymax, self.ixmin:self.ixmax]
        dref += self.ftot * (exp(-rs**2 / 2 / self.gsig**2)
                             / (sqrt(2 * pi * self.gsig**2))**2)
        return data


class GIM2D(Morph2D):
    """
    GIM2D-parametrized morphology

    The model follows that of GIM2D and Simard et al. ApJ (2002) 142,
    1 should be referred for detail.
    """

    def __init__(self, ftot=1., bf=.5, n=4., re=6., el=.5, pab=0., rd=10.,
                 inc=0., pad=0., hwsigma=5., **kwargs):
        """
        Input:

          ftot -- total flux
          bf -- bulge fraction (0 <= B/T <= 1)
          n -- Sersic index
          re -- photobulge semimajor axis effective radius in pixels
          el -- photobulge ellipticity
          pab -- photobulge position angle in degrees
          rd -- photodisk semimajor axis exponential scale length in pixels
          inc -- photodisk inclination in degrees
          pad -- photodisk position angle in degrees
          hwsigma -- half-width size in gsig of stamp enclosing the model
        """
        super(GIM2D, self).__init__(**kwargs)
        self.ftot = ftot
        self.bf = bf
        self.n = n
        self.re = re
        self.el = el
        self.pab = pab * pi / 180.
        self.rd = rd
        self.inc = inc * pi / 180.
        self.pad = pad * pi / 180.
        self.hwsigma = hwsigma
        self.xhw = np.ceil(hwsigma * max(self.re, self.rd))
        self.yhw = np.ceil(hwsigma * max(self.re, self.rd))

    @property
    def parameters(self):
        return [self.ftot, self.bf, self.re, self.el, self.pab, self.rd,
                self.inc, self.pad, self.n]

    def generate(self, x, y, data):
        if self._prepare_coords(data, x, y):
            # model is entirely outside the data so nothing to be done
            return data
        # in polar coordinates
        rs = sqrt(self.xs**2 + self.ys**2)
        phi = arctan(self.ys / self.xs)
        phi[np.isnan(phi)] = 0.    # fix the origin cuz it always blows up

        dref = data[self.iymin:self.iymax, self.ixmin:self.ixmax]

        ftot = self.ftot
        bf = self.bf
        re = self.re
        el = self.el
        pab = self.pab
        rd = self.rd
        inc = self.inc
        pad = self.pad
        n = self.n

        k = 1.9992 * n - 0.3271

        # compute bulge surface brightness at re and disk surface
        # brightness at center
        #sbb = ftot * bf / (2*pi*n * exp(k) * k**(-2*n) * re**2 * gamma(2*n))
        #sbd = ftot * (1 - bf) / (2 * pi * rd**2)

        # obtain buldge effective length.
        a = re
        b = a * (1. - el)
        phi_pab = phi - pab
        res = a * b / sqrt((b * cos(phi_pab))**2 + (a * sin(phi_pab))**2)
        bulge = exp(-k * ((rs / res)**(1. / n) - 1.))
        bulge *= ftot * bf / bulge.sum()

        # obtain disk scale length, which in general is an ellipse and
        # rotated by a position angle.
        a = rd    # semimajor axis
        eld = 1. - cos(inc)**2    # disk ellipticity
        b = a * (1. - eld)    # semiminor axis
        phi_pad = phi - pad
        rds = a * b / sqrt((b * cos(phi_pad))**2 + (a * sin(phi_pad))**2)
        disk = exp(-rs / rds)
        disk *= ftot * (1. - bf) / disk.sum()

        dref += bulge + disk
        return data
