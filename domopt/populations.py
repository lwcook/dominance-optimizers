import time
import itertools
import math
import copy
import pdb
import json

import matplotlib.pyplot as plt

import utilities_SDGA as utils

from random import Random 
import numpy as np

opt_lb = 0.
opt_ub = 10.

class Optimization(object):

    def __init__(self, evaluator, bounds):

        self.bounds = bounds
        self.evaluator = evaluator
        self.LTM = []

    def bounder(self, candidate, *args):
        for ii, c in enumerate(candidate):
            candidate[ii] = max(min(c, opt_ub), opt_lb)
        return candidate

    def scaleCandidateToDV(self, candidate):
        return [(c/10)*(self.bounds[i][1] - self.bounds[i][0]) +
                self.bounds[i][0] for i, c in enumerate(candidate)]

    def scaleDVToCandidate(self, design_variables):
        return [10.*(x - self.bounds[i][0])/(self.bounds[i][1] -
                self.bounds[i][0]) for i, x in enumerate(design_variables)]

    def evalCandidate(self, candidate):
        visited_cands = [d.candidate for d in self.LTM]
        if candidate in visited_cands:
            other = copy.copy(self.LTM[visited_cands.index(candidate)])
            ind = Individual(candidate=candidate)
            samples = other.stochastic.samples
            ind.stochastic = Stochastic(samples=samples, stype=self.stype)
        else:
            ind = Individual(candidate=candidate)
            samples = self.evaluator(self.scaleCandidateToDV(candidate))
            ind.stochastic = Stochastic(samples=samples, stype=self.stype)
            self.LTM.append(ind)

        return ind

    def removeDominatedPoints(self, A, memory):
        out_memory = copy.copy(memory)
        for B in memory:
            if A < B:
                out_memory.remove(B)
        return out_memory

    def addIfNotDominated(self, A, memory):

        if len(memory) == 0:
            memory.append(A)
            return True

        for B in memory:
            if B < A:
                return False
            elif B == A:
                return False

        memory.append(A)
        return True


class Individual(object):
    """Represents an individual in an evolutionary computation.

    An individual is defined by its candidate solution and the
    fitness (or value) of that candidate solution.

    Public Attributes:

    - *candidate* -- the candidate solution
    - *stochastic* -- the stochastic behaviour of the candidate, including
      which measures are of interest and the appropriate comparison. The
      required data is samples of the quantity of interest at the design
      specified by the candidate property.
    - *birthdate* -- the system time at which the individual was created

    """
    def __init__(self, candidate=None):
        self.candidate = candidate
        self.stochastic = None
        self.birthdate = time.time()
        self.sigma = 0.1
        self.alpha = 0.0

    def __setattr__(self, name, val):
        if name == 'candidate':
            self.__dict__[name] = val
            self.stochastic = None
        else:
            self.__dict__[name] = val

    def __str__(self):
        return '%s' % str(self.candidate)

    def __repr__(self):
        return 'candidate = %s, stats = (%s), birthdate = %s' %  \
                (str(self.candidate), str(self.stochastic), str(self.birthdate))

    def __lt__(self, other):
        if self.stochastic is not None and other.stochastic is not None:
            return self.stochastic < other.stochastic
        else:
            raise Exception('stochastic samples of candidate is not defined')

    def __le__(self, other):
        return self < other or not other < self

    def __gt__(self, other):
        if self.stochastic is not None and other.stochastic is not None:
            return other < self
        else:
            raise Exception('stochastic is not defined')

    def __ge__(self, other):
        return other < self or not self < other

    def __eq__(self, other):
        tol = 1e-7
        if (np.linalg.norm(np.array(self.candidate) - np.array(other.candidate))
                < tol):
            return True
        else:
            return False

    def __ne__(self, other):
        return self.candidate != other.candidate


class Stochastic(object):
    """Represents a solution which can be compared for types of dominance"""

    def __init__(self, samples=[], stype='MV'):
        self.samples = samples
        self.mean = None
        self.std = None
        self.CDF = None
        self.supCDF = None
        self.stype = stype # SD for stochastic dominance or MV for mean var

        self.compute_stats()  # Automatically compute mean and std
        self.compute_CDF()  # Don't automatically sort for CDF to save time

    def compute_stats(self):
        """Evaluate mean and variance from statistics"""
        self.mean = np.mean(np.array(self.samples))
        self.std = np.sqrt(np.var(np.array(self.samples)))

    def compute_CDF(self):
        """Order samples to give the ECDF in the form of a list of tuples of
        (q_value, CDF_value)"""
        self.CDF = []
        sorted_samples = np.sort(self.samples)
        M = float(len(self.samples))
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

    def __str__(self):
        return 'Mean: ' + str(self.mean) + '  Std: ' + str(self.std)

    def __repr__(self):
        return 'Mean: ' + str(self.mean) + '  Std: ' + str(self.std)

    def ZSDcompare(self, cand, other):

        if cand.CDF is None or other.CDF is None:
            if max(cand.samples) < min(other.samples):
                return 1
            else:
                return 0

        else:
            if cand.CDF[-1][0] < other.CDF[0][0]:
                return 1
            else:
                return 0

    def FSDcompare(self, cand, other):
        """Compares two candidates for stochastic dominance, uses the
        empirical CDF for comparison based off samples.

        - 1 for 1st argument dominating, 0 otherwise
        """

        if len(cand.samples) != len(other.samples):
            raise Exception('Number of samples of two candidates must be equal')
        else:
            M = len(cand.samples)

        # For now assume a fixed number of samples
        if cand.CDF is None:
            cand.compute_CDF()

        if other.CDF is None:
            other.compute_CDF()

        ## cand cannot cominate other if mean or worst case are inferior
        if other.mean < cand.mean:
            return 0
        if other.CDF[-1][0] < cand.CDF[-1][0]:
            return 0

        hlims = [0.0, 1.0]
        b1dominating, b2dominating = True, True
        for ii in range(M):
            if cand.CDF[ii][1] >= hlims[0] and cand.CDF[ii][1] <= hlims[1]:
                if cand.CDF[ii][0] < other.CDF[ii][0]:
                    b2dominating = False
                if other.CDF[ii][0] < cand.CDF[ii][0]:
                    b1dominating = False

        if b1dominating and not b2dominating:
            return 1
        else:
            return 0

    def SSDcompare(self, cand, other):
        """Compares two candidates for 2nd order stochastic dominance, uses the
        empirical CDF for comparison based off samples.

        returns 1 for 1st argument dominating, 2 for 2nd argument, 0 for draw.
        """

        if len(cand.samples) != len(other.samples):
            raise Exception('Number of samples of two candidates must be equal')
        else:
            M = len(cand.samples)

        # For now assume a fixed number of samples
        if cand.CDF is None:
            cand.compute_CDF()

        if other.CDF is None:
            other.compute_CDF()

        if cand.supCDF is None:
            cand.compute_supCDF()

        if other.supCDF is None:
            other.compute_supCDF()

        ## cand cannot cominate other if mean or worst case are inferior
        if other.mean < cand.mean:
            return 0
        if other.CDF[-1][0] < cand.CDF[-1][0]:
            return 0


        hlims = [0.0, 1.0]
        b1dominating, b2dominating = True, True
        for ii in range(M):
            if cand.supCDF[ii][1] >= hlims[0] and cand.supCDF[ii][1] <= hlims[1]:
                if cand.supCDF[ii][0] < other.supCDF[ii][0]:
                    b2dominating = False
                if other.supCDF[ii][0] < cand.supCDF[ii][0]:
                    b1dominating = False

#        plt.figure()
#        plt.plot(np.array(cand.CDF)[:, 0], np.array(cand.CDF)[:, 1], 'b')
#        plt.plot(np.array(cand.supCDF)[:, 0], np.array(cand.supCDF)[:, 1], 'b:')
#        plt.plot(np.array(other.CDF)[:, 0], np.array(other.CDF)[:, 1], 'r')
#        plt.plot(np.array(other.supCDF)[:, 0], np.array(other.supCDF)[:, 1], 'r:')
#        plt.show()

#        pdb.set_trace()

        if b1dominating and not b2dominating:
            return 1
        else:
            return 0


    def MVcompare(self, cand, other):
        """Compares candidates using Pareto dominance of mean and std
        """

        if cand.mean is None or cand.std is None:
            cand.compute_stats()
        if other.mean is None or other.std is None:
            other.compute_stats()

        if (cand.mean < other.mean and cand.std <= other.std) or \
                (cand.mean <= other.mean and cand.std < other.std):
            return 1
        elif (other.mean < cand.mean and other.std <= cand.std) or \
                (other.mean <= cand.mean and other.std < cand.std):
            return 2
        else:
            return 0


class Bounder(object):
    """Defines a basic bounding function for numeric lists.

    This callable class acts as a function that bounds a 
    numeric list between the lower and upper bounds specified.
    These bounds can be single values or lists of values. For
    instance, if the candidate is composed of five values, each
    of which should be bounded between 0 and 1, you can say
    ``Bounder([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])`` or just
    ``Bounder(0, 1)``. If either the ``lower_bound`` or 
    ``upper_bound`` argument is ``None``, the Bounder leaves 
    the candidate unchanged (which is the default behavior).

    Public Attributes:

    - *lower_bound* -- the lower bound for a candidate
    - *upper_bound* -- the upper bound for a candidate

    """
    def __init__(self, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if self.lower_bound is not None and self.upper_bound is not None:
            try:
                iter(self.lower_bound)
            except TypeError:
                self.lower_bound = itertools.repeat(self.lower_bound)
            try:
                iter(self.upper_bound)
            except TypeError:
                self.upper_bound = itertools.repeat(self.upper_bound)

    def __call__(self, candidate, args):
        # The default would be to leave the candidate alone
        # unless both bounds are specified.
        if self.lower_bound is None or self.upper_bound is None:
            return candidate
        else:
            try:
                iter(self.lower_bound)
            except TypeError:
                self.lower_bound = [self.lower_bound] * len(candidate)
            try:
                iter(self.upper_bound)
            except TypeError:
                self.upper_bound = [self.upper_bound] * len(candidate)
            bounded_candidate = copy.copy(candidate)
            for i, (c, lo, hi) in enumerate(
                    zip(candidate, self.lower_bound, self.upper_bound)):
                bounded_candidate[i] = max(min(c, hi), lo)
            return bounded_candidate

def measureHyperVolume(front, ref):
    if len(front) > 0:

        pfront = [[i.stochastic.mean, i.stochastic.std] for i in front]
        sinds = np.argsort(np.array(pfront)[:, 0])
        sfront = [pfront[ind] for ind in sinds]

        p = sfront[0]
        prev_y = ref[1]
        now_y = min(p[1], ref[1])
        now_x = min(p[0], ref[0])
        hv = abs(ref[0] - now_x)*abs(prev_y - now_y)

        for ip, p in enumerate(sfront[1:]):
            prev_y = min(sfront[ip][1], ref[1])
            now_y = min(p[1], ref[1])
            now_x = min(p[0], ref[0])

            hv += abs(ref[0] - now_x)*abs(prev_y - now_y)

        return hv
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

def measureBestPoint(front):

    if len(front) > 0:

        min_dist = np.infty

        for p in front:
            distance = np.linalg.norm([p.stochastic.mean, p.stochastic.std])
            if distance < min_dist:
                min_dist = 1.*distance

        return min_dist
    else:
        return 0
