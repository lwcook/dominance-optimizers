import pdb
import random
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

from optimizers import Optimization, Point

class TabuSearch(Optimization):

    def __init__(self, evaluator, bounds,
            max_points=200, verbose=False,
            search_method='exhaustive'):

        super(TabuSearch, self).__init__(evaluator, bounds)

        self.STM = []

        self.IM = []
        self.MTM = []
        self.LTM = []

        self.STMsize = 10
        self.intensify_count = 10
        self.diversify_count = 15
        self.reduce_count = 22
        self.reduce_ratio = 0.6
        self.step_size = 1

        self.current_step_size = self.step_size
        self.orig_step_size = self.step_size

        self.max_points = max_points

        self.verbose = verbose
        self.search_method = search_method
        self.intensify_method = 'mixture'

    def shouldTerminate(self):
        return len(self.LTM) > self.max_points

    def optimize(self, dv0, verbose=None, restart_data=None):

        if verbose is None:
            verbose = self.verbose

        x0 = self.scaleDVtoX(dv0)
        base = self.pointFromX(x0)
        counter = 0
        move = 'hookejeeves'
        added_to_MTM = False
        add_to_STM = True

        while True:

            if add_to_STM:
                if verbose:
                    print(len(self.LTM))

                self.STM.insert(0, base)
                if len(self.STM) > self.STMsize:
                    self.STM.pop()

            add_to_STM = True
            visited_points = []

            if move == 'hookejeeves':
                if verbose:
                    print 'HookeJeeves'
                non_dom_points, visited_points = self.doLocalSearch(base)
                if len(non_dom_points) > 0:
                    next_point = random.choice(non_dom_points)
                    non_dom_points.remove(next_point)
                    for point in non_dom_points:
                        added_to_IM = self.addIfNotDominated(point, self.IM)
                        self.IM = self.removeDominatedPoints(point, self.IM)

                    if next_point < base:
                        move = 'pattern'
                    else:
                        move = 'hookejeeves'
                else:
                    x = self.doIntensification()
                    next_point = self.pointFromX(x)
                base = next_point
                counter += 1

            elif move == 'pattern':
                if verbose:
                    print 'Pattern'
                pat_point = self.STM[1]
                base, success = self.doPatternMove(base, pat_point)
                if success:
                    move = 'pattern'
                    counter += 1
                else:
                    move = 'hookejeeves'
                    add_to_STM = False

            elif move == 'intensify':
                if verbose:
                    print 'Intensification'
                x = self.doIntensification()
                base = self.pointFromX(x)
                move = 'hookejeeves'
                counter += 1

            elif move == 'diversify':
                if verbose:
                    print 'Diversification'
                x = self.doDiversification()
                base = self.pointFromX(x)
                move = 'hookejeeves'
                counter += 1

            elif move == 'reduction':
                if verbose:
                    print 'Reduction'
                x = self.doStepSizeReduction()
                base = self.pointFromX(x)
                move = 'hookejeeves'
                counter = 0

            added_to_MTM = self.addIfNotDominated(base, self.MTM)
            self.MTM = self.removeDominatedPoints(base, self.MTM)
            if added_to_MTM:
                counter = 0

            visited_points.append(base)
            for point in visited_points:
                self.addIfNotDominated(point, self.MTM)
                self.MTM = self.removeDominatedPoints(point, self.MTM)

            if counter >= self.reduce_count:
                move = 'reduction'

            if counter == self.diversify_count:
                move = 'diversify'

            if counter == self.intensify_count:
                move = 'intensify'

            if self.shouldTerminate():
                break

        return self.MTM

    def doLocalSearch(self, base_point, search_method=None):
        base_x, base_f = base_point[0], base_point[1]
        x0 = np.array(base_x).flatten()
        xdim = x0.size
        if search_method is None:
            search_method = self.search_method

        best_points = []
        visited_points = []
        xLocal = []
        for ii in np.arange(xdim):
            x1 = copy.copy(x0)+np.eye(xdim)[ii, :]*self.step_size
            x2 = copy.copy(x0)-np.eye(xdim)[ii, :]*self.step_size
            xLocal += [x1, x2]

        if search_method.lower() == 'exhaustive':

            for x in xLocal:
                xb = self.bounder(x)
                if not self.checkIfTabu(xb, self.STM):

                    pnt = self.pointFromX(xb)

                    self.addIfNotDominated(pnt, best_points)
                    best_points = self.removeDominatedPoints(pnt, best_points)
                    visited_points.append(pnt)

        elif search_method.lower() == 'random':

            xRandomized = np.random.permutation(xLocal)
            for x in xRandomized:
                xb = self.bounder(x)
                if not self.checkIfTabu(xb, self.STM):

                    pnt = self.pointFromX(xb)

                    self.addIfNotDominated(pnt, best_points)
                    best_points = self.removeDominatedPoints(pnt, best_points)
                    visited_points.append(pnt)

                    if pnt < base_point:
                        break

        else:
            raise ValueError('Unsupported local search method')

        return best_points, visited_points

    def doPatternMove(self, current_point, last_point):
        current_x, current_f = current_point[0], current_point[1]
        last_x, last_f = last_point[0], last_point[1]

        x_pat = self.bounder((np.array(current_x) -
                np.array(last_x) + np.array(current_x)))

        if not self.checkIfTabu(x_pat, self.STM):
            pattern_point = self.pointFromX(x_pat.tolist())

            if pattern_point < current_point:
                return pattern_point, True
            else:
                return current_point, False
        else:
            return current_point, False

    def doIntensification(self, method=None):
        self.step_size = self.current_step_size

        if method is None:
            case = self.intensify_method
        else:
            case = method

        if case == 'multiple':
            newM = copy.copy(self.MTM)
            random.shuffle(newM)
            x_array = np.array([x for (x, f) in newM[0:min(4, len(newM))]])
            new_x = np.mean(x_array, axis=0)

        elif case == 'random':
            if len(self.IM) > 0:
                new_x, new_f = random.choice(self.IM)
            else:
                new_x, new_f = random.choice(self.MTM)

        elif case == 'density':
            distances = []
            for point in self.MTM:
                distance = 0
                for other in self.MTM:
                    distance += np.linalg.norm(np.array(point[0]) -
                            np.array(other[0]))
                distances.append(distance)
            imax = np.argsort(distances)[-1]
            new_x, new_f = self.MTM[imax]

        elif case == 'mixture':
            if len(self.MTM) > 0:
                if len(self.IM) > 0:
                    new_x, f = random.choice(self.IM)
                else:
                    x1, f1 = random.choice(self.MTM)
                    distances = []
                    for point in self.MTM:
                        distances.append(np.linalg.norm(np.array(point[0]) -
                            np.array(x1)))
                    imin = np.argsort(distances)[0]
                    x2, f2 = self.MTM[imin]
                    new_x = 0.5*(np.array(x1) + np.array(x2))
            else:
                new_x = self.doDiversification()

        return self.bounder([xi for xi in new_x])

    def doDiversification(self, tol=3):
        self.step_size = self.orig_step_size/2.
        dim = len(self.LTM[0][0])

        new_x = [random.uniform(0, 10) for x in np.arange(dim)]
        D = 100
        for point in self.LTM:
            D = min(np.linalg.norm(np.array(new_x) -
                    np.array(point[0])), D)

        if D < tol:
            new_x = self.doDiversification(tol=tol*0.8)

        if not self.checkIfTabu(new_x, self.STM):
            return self.bounder(new_x)
        else:
            return self.doDiversification()

    def doStepSizeReduction(self):
        self.current_step_size = self.current_step_size*self.reduce_ratio
        self.step_size = self.current_step_size
        return self.doIntensification()

    def checkIfTabu(self, x, memory):
        tol = 1e-7
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        for ii, xi in enumerate(x):
            if xi > 10. or xi < 0.:
                return True

        for (x_stored, f_stored) in self.STM:
            if np.linalg.norm(np.array(x) - np.array(x_stored)) < tol:
                return True

        return False
