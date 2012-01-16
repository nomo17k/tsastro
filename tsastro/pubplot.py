#!/usr/bin/env python2.6
from matplotlib import rcParams


# for publication-quality plotting
FONTSIZE = 18
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Computer Modern Roman'
rcParams['font.size'] = FONTSIZE
rcParams['text.usetex'] = True
rcParams['xtick.labelsize'] = FONTSIZE
rcParams['ytick.labelsize'] = FONTSIZE
rcParams['axes.labelsize'] = FONTSIZE
rcParams['legend.fontsize'] = FONTSIZE
rcParams['figure.figsize'] = (6, 6)
