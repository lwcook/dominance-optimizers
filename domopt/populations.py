import time
import itertools
import math
import copy
import pdb
import json

import matplotlib.pyplot as plt

import utilities as utils

from random import Random
import numpy as np



class Stochastic(object):
    """Represents an design's performance through stochastic samples
    """

    def __init__(self, design=None, samples=None, stype='MV'):
        self.design = design
        self.samples = None
        self.mean = None
        self.std = None
        self.CDF = None
        self.supCDF = None
        self.stype = stype # SD for stochastic dominance or MV for mean var

        self.samples = samples
        self.compute_stats(samples)
        self.compute_CDF(samples)


    def __setattr__(self, name, val):
        if name == 'design':
            self.__dict__[name] = val
            self.stochastic = None
        else:
            self.__dict__[name] = val

    def __str__(self):
        return 'Stochastic with stats ['+str(self.mean)+', '+str(self.std)+']'

    def __repr__(self):
        return 'Stochastic with stats ['+str(self.mean)+', '+str(self.std)+']'

    def __lt__(self, other):
        if isinstance(self.stype, basestring):
            stypes = [self.stype.lower()]
        else:
            stypes = [s.lower() for s in self.stype]

        # If dominates under any criterion, return dominating
        for s in stypes:
            if s == 'zsd':
                if self.ZSDcompare(self, other) == 1:
                    return True
            if s == 'sd' or s == 'fsd':
                if self.FSDcompare(self, other) == 1:
                    return True
            if s == 'ssd':
                if self.SSDcompare(self, other) == 1:
                    return True
            elif s == 'mv':
                if self.MVcompare(self, other) == 1:
                    return True

        # If not dominating by any criterion, return not dominating
        return False

    def __le__(self, other):
        return self < other or not other < self

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return other < self or not self < other

    def __eq__(self, other):
        tol = 1e-7
        if (np.linalg.norm(np.array(self.design) - np.array(other.design))
                < tol):
            return True
        else:
            return False

    def __ne__(self, other):
        return self.design != other.design


    def compute_stats(self, samps=None):
        """Evaluate mean and variance from statistics"""
        if samps is None:
            samps = self.samples
        self.mean = np.mean(np.array(samps))
        self.std = np.sqrt(np.var(np.array(samps)))

    def compute_CDF(self, samps=None):
        """Order samples to give the ECDF in the form of a list of tuples of
        (q_value, CDF_value)"""
        if samps is None:
            samps = self.samples
        self.CDF = []
        sorted_samples = np.sort(samps)
        M = float(len(samps))
        for ii, samp in enumerate(sorted_samples):
            tupl = (samp, float(ii)/M + 0.5/M)
            self.CDF.append(tupl)

    def compute_supCDF(self):
        if self.CDF is None:
            self.compute_CDF()
        M = float(len(self.samples))

        self.supCDF = []
        for ii, (quantile, h) in enumerate(self.CDF):
            qsum = sum([tup[0] for tup in self.CDF[ii:]])
            supquant = qsum/(M-ii)
            self.supCDF.append((supquant, h))

    def ZSDcompare(self, design, other):

        if design.CDF is None or other.CDF is None:
            if max(design.samples) < min(other.samples):
                return 1
            else:
                return 0

        else:
            if design.CDF[-1][0] < other.CDF[0][0]:
                return 1
            else:
                return 0

    def FSDcompare(self, design, other):
        """Compares two designs for stochastic dominance, uses the
        empirical CDF for comparison based off samples.

        - 1 for 1st argument dominating, 0 otherwise
        """

        if len(design.samples) != len(other.samples):
            raise Exception('Number of samples of two designs must be equal')
        else:
            M = len(design.samples)

        # For now assume a fixed number of samples
        if design.CDF is None:
            design.compute_CDF()

        if other.CDF is None:
            other.compute_CDF()

        ## design cannot cominate other if mean or worst case are inferior
        if other.mean < design.mean:
            return 0
        if other.CDF[-1][0] < design.CDF[-1][0]:
            return 0

        hlims = [0.0, 1.0]
        b1dominating, b2dominating = True, True
        for ii in range(M):
            if design.CDF[ii][1] >= hlims[0] and design.CDF[ii][1] <= hlims[1]:
                if design.CDF[ii][0] < other.CDF[ii][0]:
                    b2dominating = False
                if other.CDF[ii][0] < design.CDF[ii][0]:
                    b1dominating = False

        if b1dominating and not b2dominating:
            return 1
        else:
            return 0

    def SSDcompare(self, design, other):
        """Compares two designs for 2nd order stochastic dominance, uses the
        empirical CDF for comparison based off samples.

        returns 1 for 1st argument dominating, 2 for 2nd argument, 0 for draw.
        """

        if len(design.samples) != len(other.samples):
            raise Exception('Number of samples of two designs must be equal')
        else:
            M = len(design.samples)

        # For now assume a fixed number of samples
        if design.CDF is None:
            design.compute_CDF()

        if other.CDF is None:
            other.compute_CDF()

        if design.supCDF is None:
            design.compute_supCDF()

        if other.supCDF is None:
            other.compute_supCDF()

        ## design cannot cominate other if mean or worst case are inferior
        if other.mean < design.mean:
            return 0
        if other.CDF[-1][0] < design.CDF[-1][0]:
            return 0


        hlims = [0.0, 1.0]
        b1dominating, b2dominating = True, True
        for ii in range(M):
            if design.supCDF[ii][1] >= hlims[0] and design.supCDF[ii][1] <= hlims[1]:
                if design.supCDF[ii][0] < other.supCDF[ii][0]:
                    b2dominating = False
                if other.supCDF[ii][0] < design.supCDF[ii][0]:
                    b1dominating = False

        if b1dominating and not b2dominating:
            return 1
        else:
            return 0


    def MVcompare(self, design, other):
        """Compares designs using Pareto dominance of mean and std
        """

        if design.mean is None or design.std is None:
            design.compute_stats()
        if other.mean is None or other.std is None:
            other.compute_stats()

        if (design.mean < other.mean and design.std <= other.std) or \
                (design.mean <= other.mean and design.std < other.std):
            return 1
        elif (other.mean < design.mean and other.std <= design.std) or \
                (other.mean <= design.mean and other.std < design.std):
            return 2
        else:
            return 0


def measureAvgDistance(front, filename):

    if len(front) > 0:

        with open(filename, 'r') as f:
            dlist = json.loads(f.readlines()[0])

        points = np.array([d['objectives'] for d in dlist])
        inds = np.argsort(points[:, 0])
        true_points = points[inds]

        apoints = np.array([[p.stochastic.mean, p.stochastic.std]
                                for p in front])
        inds = np.argsort(apoints[:, 0])
        arch_points = apoints[inds]

        distances = np.zeros(len(true_points))
        for it, t in enumerate(true_points):
            dists = np.zeros(len(arch_points))
            for ip, p in enumerate(arch_points):
                dists[ip] = np.linalg.norm(p - t)
            min_dist = np.min(dists)
            distances[it] = min_dist

        avg_distance = np.mean(distances)
        return avg_distance

    else:
        return 0

def saveArchive(front, name):
    l = []
    for p in front:
        d = {'design': p.design,
            'mean': p.mean,
            'std': p.std,
            'CDF': p.CDF}
        l.append(d)

    with open('output/' + str(name) + '.txt', 'w') as f:
        f.write(json.dumps(l))

def readArchive(name):
    arch = []
    with open(name, 'r') as f:
        jdicts = json.loads(f.readlines()[0])
        for d in jdicts:
#            objs = d['objectives']
#            dv = d['design']
#            samps = [tup[0] for tup in d['CDF']]
#            assert(abs(objs[0] - np.mean(samps)) < 1e-12)
#            assert(abs(objs[1] - np.std(samps)) < 1e-12)
#            ind = Stochastic(dv, samps)
#            ind.mean = objs[0]
#            ind.std = objs[1]
#            ind.CDF = d['CDF']
            arch.append(d)

    for ip, ind in enumerate(arch):
        if abs(ind['mean']) > 1e2 or abs(ind['std']) > 1e2:
            arch[ip]['mean'] = 1e2
            arch[ip]['std'] = 1e2
            arch[ip]['CDF'] = [[1e2, 0], [1e2, 1]]

    return arch
