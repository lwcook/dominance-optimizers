import pdb
import random
import copy
import time

import numpy as np
import matplotlib.pyplot as plt

import utilities_SDGA as utils
from populations import Individual, Stochastic, Optimization

class TabuSearch(Optimization):

    def __init__(self, evaluator, stype='MV', bounds=None, max_points=200,
            bplot=False, step_size=1, observer=None, verbose=False,
            search_method='exhaustive', restart_data=None):

        super(TabuSearch, self).__init__(evaluator, bounds)

        self.stype = stype
        self.STM = []
        self.STMsize = 10

        self.IM = []
        if restart_data is not None:
            self.MTM = self.recreateMemory(restart_data['MTM'])
            self.LTM = self.recreateMemory(restart_data['LTM'])
        else:
            self.MTM = []
            self.LTM = []

        self.intensify_count = 10
        self.diversify_count = 15
        self.reduce_count = 22
        self.reduce_ratio = 0.6

        self.step_size = step_size
        self.current_step_size = step_size
        self.orig_step_size = step_size

        self.max_points = max_points
        self.bplot = bplot
        self.observer = observer
        self.verbose = verbose
        self.search_method = search_method
        self.intensify_method = 'mixture'
        self.when_to_observe = 0

    def recreateMemory(self, Memory):
        points = []
        for d in Memory:
            pdb.set_trace()
            cand = d['candidate']
            objs = d['objectives']
            ind = Individual(cand)
            samps = [tup[0] for tup in d['CDF']]
            ind.stochastic = Stochastic(samples=samps)
            ind.stochastic.mean = objs[0]
            ind.stochastic.std = objs[1]
            pdb.set_trace()
            assert(abs(objs[0] -   np.mean(samps)) < 1e-12)
            assert(abs(objs[1] - np.std(samps)) < 1e-12)
            pdb.set_trace()
            ind.stochastic.CDF = d['CDF']
            points.append(ind)

        return points

    def shouldTerminate(self):
        return len(self.LTM) > self.max_points

    def optimize(self, x0, verbose=None, restart_data=None):

        if verbose is None:
            verbose = self.verbose

        if self.bplot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
            self.ax1 = ax1
            self.ax2 = ax2
            self.ax3 = ax3
            ax1.set_xlim([0, 10])
            ax1.set_ylim([0, 10])
            plt.ion()
            plt.show()

        base = self.evalCandidate(self.scaleDVToCandidate(x0))
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

            if self.bplot:
                self.plot(added_to_MTM)

            add_to_STM = True
            visited_designs = []
#            print('COUNTER: ', counter, 'MOVE: ', move)
            if move == 'hookejeeves':
                non_dom_designs, visited_designs = self.doLocalSearch(base)
                if len(non_dom_designs) > 0:
                    next_design = random.choice(non_dom_designs)
                    non_dom_designs.remove(next_design)
                    for des in non_dom_designs:
                        added_to_IM = self.addIfNotDominated(des, self.IM)
                        self.IM = self.removeDominatedPoints(des, self.IM)

                    if next_design < base:
                        move = 'pattern'
                    else:
                        move = 'hookejeeves'
                else:
                    candidate = self.doIntensification()
                    next_design = self.evalCandidate(candidate)
                base = next_design
                counter += 1

            elif move == 'pattern':
                base, success = self.doPatternMove(base, self.STM[1])
                if success:
                    move = 'pattern'
                    counter += 1
                else:
                    move = 'hookejeeves'
                    add_to_STM = False

            elif move == 'intensify':
                if verbose:
                    print 'Intensification'
                candidate = self.doIntensification()
                base = self.evalCandidate(candidate)
                move = 'hookejeeves'
                counter += 1

            elif move == 'diversify':
                if verbose:
                    print 'Diversification'
                candidate = self.doDiversification()
                base = self.evalCandidate(candidate)
                move = 'hookejeeves'
                counter += 1

            elif move == 'reduction':
                if verbose:
                    print 'Reduction'
                candidate = self.doStepSizeReduction()
                base = self.evalCandidate(candidate)
                move = 'hookejeeves'
                counter = 0

            added_to_MTM = self.addIfNotDominated(base, self.MTM)
            self.MTM = self.removeDominatedPoints(base, self.MTM)
            if added_to_MTM:
                counter = 0

            visited_designs.append(base)
            for des in visited_designs:
                self.addIfNotDominated(des, self.MTM)
                self.MTM = self.removeDominatedPoints(des, self.MTM)

            if counter >= self.reduce_count:
                move = 'reduction'

            if counter == self.diversify_count:
                move = 'diversify'

            if counter == self.intensify_count:
                move = 'intensify'

            if self.observer is not None:
                if len(self.LTM) > self.when_to_observe + 5:
                    self.observer(len(self.LTM), self.MTM, self.LTM)
                    self.when_to_observe = len(self.LTM)

            if self.shouldTerminate():
                break


        return self.MTM

    def doLocalSearch(self, base_design, search_method=None):
        x0 = np.array(base_design.candidate).flatten()
        xdim = x0.size
        if search_method is None:
            search_method = self.search_method

        best_designs = []
        visited_designs = []
        xLocal = []
        for ii in np.arange(xdim):
            x1 = copy.copy(x0)+np.eye(xdim)[ii, :]*self.step_size
            x2 = copy.copy(x0)-np.eye(xdim)[ii, :]*self.step_size
            xLocal += [x1, x2]

        if search_method.lower() == 'exhaustive':

            for x in xLocal:
                if not self.checkIfTabu(self.bounder(x), self.STM):
                    design = self.evalCandidate(x.tolist())
                    self.addIfNotDominated(design, best_designs)
                    best_designs = self.removeDominatedPoints(
                            design, best_designs)
                    visited_designs.append(design)

        elif search_method.lower() == 'random':

            xRandomized = np.random.permutation(xLocal)
            for x in xRandomized:
                if not self.checkIfTabu(self.bounder(x), self.STM):
                    design = self.evalCandidate(x.tolist())
                    self.addIfNotDominated(design, best_designs)
                    best_designs = self.removeDominatedPoints(
                            design, best_designs)
                    visited_designs.append(design)

                    if design < base_design:
                        break

        else:
            raise ValueError('Unsupported local search method')

        return best_designs, visited_designs

    def doPatternMove(self, current, last):

        x_pattern = (np.array(current.candidate) -
                np.array(last.candidate) + np.array(current.candidate))

        if not self.checkIfTabu(self.bounder(x_pattern), self.STM):
            pattern = self.evalCandidate(x_pattern.tolist())

            if pattern < current:
                return pattern, True
            else:
                return current, False
        else:
            return current, False

    def doIntensification(self, method=None):
        self.step_size = self.current_step_size

        if method is None:
            case = self.intensify_method
        else:
            case = method

        if case == 'multiple':
            newM = copy.copy(self.MTM)
            random.shuffle(newM)
            cand_array = np.array([d.candidate for d in
                newM[0:min(4, len(newM))]])
            new_cand = np.mean(cand_array, axis=0)

        elif case == 'random':
            if len(self.IM) > 0:
                new_cand = random.choice(self.IM).candidate
            else:
                new_cand = random.choice(self.MTM).candidate

        elif case == 'density':
            distances = []
            for des in self.MTM:
                distance = 0
                for other in self.MTM:
                    distance += np.linalg.norm(np.array(des.candidate) -
                            np.array(other.candidate))
                distances.append(distance)
            imax = np.argsort(distances)[-1]
            new_cand = self.MTM[imax].candidate

        elif case == 'mixture':
            if len(self.IM) > 0:
                cand1 = random.choice(self.IM).candidate
            else:
                cand1 = random.choice(self.MTM).candidate
            distances = []
            for des in self.MTM:
                distances.append(np.linalg.norm(np.array(des.candidate) -
                    np.array(cand1)))
            imin = np.argsort(distances)[0]
            cand2 = self.MTM[imin].candidate
            new_cand = 0.5*(np.array(cand1) + np.array(cand2))

        elif case == 'crowding_distance':
            front = np.array([[d.stochastic.mean, d.stochastic.std, i] for
                                i, d in enumerate(self.MTM)])
            mean_inds = np.argsort(front[:, 0])
            std_inds = np.argsort(front[:, 1])
            sfront = front[mean_inds]
            distances = np.zeros(len(front))
#            distances[0] = np.inf
#            distances[-1] = np.inf
            mrange = max(front[:, 0]) - min(front[:, 0])
            srange = max(front[:, 1]) - min(front[:, 1])
            for ii, _ in enumerate(front[1:-1]):
                distances[ii+1] += abs(front[ii+1][0] - front[ii-1][0])/mrange
                distances[ii+1] += abs(front[ii+1][1] - front[ii-1][1])/srange

            ichoice = np.argsort(distances)[-1]
            if np.random.uniform(1) < 0.5:
                cand1 = self.MTM[int(sfront[imax, 2])].candidate
                if np.random.uniform(1) < 0.5:
                    cand2 = self.MTM[int(sfront[imax+1, 2])].candidate
                else:
                    cand2 = self.MTM[int(sfront[imax-1, 2])].candidate
            else:
                if np.random.uniform(1) < 0.5:
                    cand1 = self.MTM[int(sfront[0, 2])].candidate
                    cand2 = self.MTM[int(sfront[1, 2])].candidate
                else:
                    cand1 = self.MTM[int(sfront[-1, 2])].candidate
                    cand2 = self.MTM[int(sfront[-2, 2])].candidate
            new_cand = 0.5*(np.array(cand1) + np.array(cand2))

        return [c for c in new_cand]

    def doDiversification(self, tol=3):
        self.step_size = self.orig_step_size/2.
        xdim = len(self.LTM[0].candidate)
#        regionCount = np.array(xdim*10)
        new_cand = [random.uniform(0, 10) for x in np.arange(xdim)]
        D = 100
        for des in self.LTM:
            D = min(np.linalg.norm(np.array(new_cand) -
                    np.array(des.candidate)), D)

        if D < tol:
            new_cand = self.doDiversification(tol=tol*0.8)

        if not self.checkIfTabu(new_cand, self.STM):
            return new_cand
        else:
            return self.doDiversification()

    def doStepSizeReduction(self):
        self.current_step_size = self.current_step_size*self.reduce_ratio
        self.step_size = self.current_step_size
        return self.doIntensification()

    def checkIfTabu(self, cand, memory):
        tol = 1e-7
        if not isinstance(cand, np.ndarray):
            cand = np.array(cand)

        for ii, ci in enumerate(cand):
            if ci > 10. or ci < 0.:
                return True

        for des in self.STM:
            if np.linalg.norm(np.array(des.candidate) - cand) < tol:
                return True

        return False
