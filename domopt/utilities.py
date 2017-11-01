from __future__ import division

import copy
import pdb
import numpy as np
import math
import scipy
import scipy.special as scp
import os
import subprocess
import time

from scipy.stats import beta

import pickle
import json

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

# TO USE THESE FUNCTIONS, ADD THE FOLLOWING TO YOUR PYTHON FILE:
# import sys
# home_dir = os.getenv("HOME")
# sys.path.insert(0, home_dir + '/phd-code/utils/')
# import utilities as utils

#blue = [50./255., 100./255., 180./255.]
#red = [180./255., 34./255., 34./255.]
#green = [34./255., 139./255., 34./255.]

grey = [150./255., 150./255., 150./255.]
purple = [128./255., 0./255., 130./255.]
blue = [136/255, 178/255, 225/255]
orange = [253/255, 174/255, 97/255]
red = [140/255, 20/255, 32/255]
green = [20./255, 177/255, 50./255]

def test_const(x, u):
    return 0.0


def test_constr(x, u):
    return 0.0


def TP1x1u(x, u):

    try:
        iter(u)
        u = u[0]
    except:
        pass

    try:
        iter(x)
        x = x[0]
    except:
        pass

    c = math.atan(x + 0.3)
    return 1. + 8.*c + (1./c)*(np.exp(1.5*u) - 1)

def TP3Dopt(x, u, delta=1):

    try:
        iter(u)
        u = u[0]
    except:
        pass

    return x[0]**2 + 2*u*x[1] + x[2]*u**2 + 5
#    return x[0]**2 + 2*u*x[1] + 1*x[2]*abs(u)**2 + 5 + 0.1*x[2]


def TP2xNu(x, u):
    y = x[0]/2
    z = x[1]/2
    qrosen = ((2-y)**2 + 2*(z - y**2)**2)/200.

    qu = 0.0
    for ii, ui in enumerate(u):
        if ii % 2 == 0:
            scale = (1 + 0.1*z**2)
        else:
            scale = (1 + 0.1*y**2)

    if True:  # Exponential Test Function
        for ii, ui in enumerate(u):
            qu += (1./(ii+1.)) * scale*ui
        qu = np.exp(-qu)
    else:  # Reciprocal Test Function
        for ii, ui in enumerate(u):
            qu += scale*(ui**2)
        qu = 1./(1. + qu)

    return 1 + (qrosen + qu)


def TP4x2u(x, u, delta=1):
    '''x should be 4D, u should be 2D'''

    return x[0]**2 + 2*u[0]*(x[1] + x[3]) + (x[2] + x[3])*u[1]**2 + 5


def TP2Dopt(x, u, delta=1):
    try:
        iter(u)
        u = u[0]
    except:
        u = u

    return (x[0] + 0.5*x[1])**2*u + x[1]*u**2 + 5


def TP2x2uOld(x, u):
    '''Both x and u should be 2D'''
    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5


def TP2D(x, u):
    '''Both x and u should be 2D'''
    y = x[0]/5
    z = x[1]/5
    return 1 + ((-1-y)**2 + 10*(z - y**2)**2)/2 + \
        z*math.sinh(u[0])*math.sinh(u[1]) - y*math.cosh(u[1])

def TP2x2u(x, u):
    '''Both x and u should be 2D'''

    p1 = 2*np.linalg.norm(np.array(x) - np.array([0, 0.7]))**2 - 0.4
    p2 = 5*np.linalg.norm(np.array(x) - np.array([0, -0.7]))**2

    q1 = 5*u[0]*u[1]*x[0]
    q2 = x[1]*u[1]**2
    q3 = x[1]*u[1]**3

    return 10 + 2*(q1 + q2 + q3 + minsmooth(p1, p2))


def TP2x2uBackup(x, u):
    '''Both x and u should be 2D'''

#    return (y0)**2 + (y1)*(v1**2)**.9 + y2*(v2**2)**.7 + 10
#    return y0 + (y1)*5*u[0]**2*v2 + y2*u[1]**2 + 10
#    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5
#    return 10 + 0.33*(y1**2 + y2**2)**0.5 + \
#        (1 + (y1))*2*u[0]*(u[1]**2) + \
#        (y2/(y1**2+1))*0.5*u[1]

    p1 = 2*np.linalg.norm(np.array(x) - np.array([0, 0.7]))**2
    p2 = 5*np.linalg.norm(np.array(x) - np.array([0, -0.7]))**2

    q1 = 5*u[0]*u[1]*x[0]
    q2 = x[1]*u[1]**2
    q3 = x[1]*u[1]**3

    return 10 + 2*(q1 + q2 + q3 + minsmooth(p1, p2))

def TP10x1u(x, u):
    '''Both x and u should be 2D'''
    try:
        iter(u)
        u = u[0]
    except:
        u = u

    qrosen = ((2-u)**2 + 2*(u - u**2)**2)/5

    qx = 0.0

    for ii, xi in enumerate(x):
        scale = (1 + 0.2*u**2)
        qx += (1./(ii+1.)) * scale*xi

    qx = np.exp(-qx)

    return 1 + (qrosen + qx)


def TP2x2uGrad(x, u):
    '''Both x and u should be 2D'''

#    return (y0)**2 + (y1)*(v1**2)**.9 + y2*(v2**2)**.7 + 10
#    return y0 + (y1)*5*u[0]**2*v2 + y2*u[1]**2 + 10
#    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5
#    return 10 + 0.33*(y1**2 + y2**2)**0.5 + \
#        (1 + (y1))*2*u[0]*(u[1]**2) + \
#        (y2/(y1**2+1))*0.5*u[1]
    y = x[0]/2.
    z = x[1]/2. + 12
    q =  0.25*((y**2+z**2)/40. + 5*u[0]*u[1]*y - z*u[1]**2) \
        + 0.2*(z*u[1]**3) + 10

    dqdx1 = (1./8.)*( (2*y)/40. + 5*u[0]*u[1])
    dqdx2 = (1./8.)*( (2*z)/40. - u[1]**2) + 0.1*u[1]**3

    return q, [dqdx1, dqdx2]


def finite_diff(fobj, dv, f0=None, dvi=None, eps=10**-6):

    try:
        iter(dv)
    except:
        dv = [dv]

    if f0 is None: f0 = fobj(dv)
    if dvi is None:
        grad = []
        for ii in range(len(dv)):
            fbase = copy.copy(f0)
            x = copy.copy(dv)
            x[ii] += eps
            fnew = fobj(x)
            grad.append(float((fnew - fbase)/eps))
        if len(grad) == 1:
            return float(grad[0])
        else:
            return grad
    else:
        x = copy.copy(dv)
        x[dvi] += eps
        return (fobj(x) - f0) / eps


def float_gen(iterable):
    for obj in iterable:
        try:
            float(obj)
            yield float(obj)
        except ValueError:
            pass


def mpl2tex(figsize=[2.8, 2.2]):
        '''Makes matplotlib's parameters give a plot formatted
        for latex documents
        -figsize: the figures dimension's in inches.
        Default is small enough for single column'''

        # Note: 1.0/72.27 inches per pt
        # [2.8,2.2] fits on the smallest of these (CMAME) and is a good ratio
        # CMAME template has a 390.0 pt wide textwidth - 5.396 inches
        # Thesis: 437.46 - 6.05 inches

        mpl.rcParams.update({"figure.figsize": figsize,
                             "font.family": "serif",
                             "text.usetex": True,
                             "text.latex.preamble": r"\usepackage{amsmath}",
                             "font.size": 8,
                             "font.weight": "light",
                             'axes.labelsize': 9,
                             'axes.titlesize': 8,
                             'legend.fontsize': 8,
                             'xtick.labelsize': 8,
                             'ytick.labelsize': 8,
                             'lines.linewidth': 0.6,
                             'axes.linewidth': 0.75,
                             'patch.linewidth': 0.75,
                             'legend.fontsize': 'medium',
                             'legend.scatterpoints': 1
                             })

def savefig(name='saved_fig', bSaveBase=False,
            base='/phd-thesis/Figs/', bSaveData=False, formatstr='pdf'):
    '''Function that saves the plot as well as the
    underlying data of the currently open figure:
    -name: string that the figure is saved as'''

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subprocess.call(["mkdir", "-p", "./figs/"])
#    plt.savefig('./output/' + str(name) + '.pdf', format='pdf')
    plt.savefig('./figs/' +  str(name) + '_' + str(date.day) +
                months[date.month-1] + '.' + formatstr, format=formatstr)
