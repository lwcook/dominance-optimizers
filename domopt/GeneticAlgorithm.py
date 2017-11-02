import time
import itertools
import math
import copy
import pdb

from random import Random
import numpy as np
import matplotlib.pyplot as plt

import utilities as utils

from optimizers import Optimization

class GeneticAlgorithm(Optimization):
    """Evolutionary computation representing the nondominated
    sorting genetic algorithm under uncertainty.

    :param function evaluator: Function that takes a vector of design variables
        and returns an object that can be compared via < and == operators.

    :param list bounds: List of bounds on design variables in the form
        [(l0, u0), (l1, u1), ..., (ln, un)] where lb, uk are the lower and upper
        bounds on the kth design variable respectively.

    """

    def __init__(self, evaluator, bounds,
            population_size=100, max_generations=50,
            verbose=False):

        super(GeneticAlgorithm, self).__init__(evaluator, bounds)

        self._random = Random()
        self._random.seed(time.time())

        self.population_size = population_size
        self.max_generations = max_generations
        self.archive = None
        self.population = None
        self.num_evaluations = 0
        self.num_generations = 0

        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.stdv = 0.1
        self.mean = 0.0

        self.verbose = verbose


    def optimize(self, seeds=[]):
        """Performs the genetic algorithm optimization.

        :param list seeds: List of initial design vectors to use as seeds in
            the optimization. Only the first population_size will be used.


        :return: Archive of non-dominated points obtained by the
                optimization.
        :rtype: List of Point objects
        """

        self.population = []

        self.archive = []
        self.LTM = []

        evaluator = self.evaluator

        # Create the initial population.
        try:
            iter(seeds)
        except TypeError:
            seeds = [seeds]

        initial_dvs = list(seeds)
        initial_xs = self.bounder([self.scaleDVtoX(dv) for dv in initial_dvs])

        num_generated = max(self.population_size - len(seeds), 0)
        i = 0
        while i < num_generated:

            dvs = [self._random.uniform(l, u) for l, u in self.bounds]
            xs = self.scaleDVtoX(dvs)
            if xs not in initial_xs:
                initial_xs.append(xs)
                i += 1

        self.population = [self.pointFromX(xs) for xs in
                           initial_xs[0:self.population_size]]

        self.num_evaluations = len(self.population)
        self.num_generations = 0

        for point in self.population:
            added_to_archive = self.addIfNotDominated(point, self.archive)
            self.archive = self.removeDominatedPoints(point, self.archive)

        ##############################################################
        # MAIN LOOP
        ##############################################################
        while self.num_generations < self.max_generations:
#
            self.num_generations += 1
            if self.verbose:
                print 'Generation: ', self.num_generations

            # Temperature decreasing encouraging less exploring as time goes on.
            T = 1 - float(self.num_generations) /  \
                (self.max_generations+1)
            self.crossover_rate = 0.9
            self.mutation_rate = 0.2 - 0.05*T
            self.stdev = 0.6 + 0.2*T

            parents = self.doTournamentSelection(self.population)

            child_xs = [copy.deepcopy(point[0]) for point in parents]
            child_xs = self.doBlendCrossover(child_xs)
            child_xs = self.doGaussianMutation(child_xs)

            children = [self.pointFromX(x) for x in child_xs]
            self.num_evaluations += len(children)

            self.population = self.doReplacement(self.population, children)

            # Archive individuals.
            for point in self.population:
                added_to_archive = self.addIfNotDominated(point, self.archive)
                self.archive = self.removeDominatedPoints(point, self.archive)

        return self.archive


    def doTournamentSelection(self, population):
        tsize = 2
        selected = [min(self._random.sample(population, tsize))
                        for _ in population]
        return selected

    def doBlendCrossover(self, designs):

        xs = list(designs)
        if len(xs) % 2 == 1:
            xs = xs[:-1]
        self._random.shuffle(xs)
        moms = xs[::2]
        dads = xs[1::2]
        new_xs = []

        for mom, dad in zip(moms, dads):
            if self._random.random() < self.crossover_rate:
                bro = []
                sis = []
                for (xi, yi) in zip(mom, dad):
                    smallest = min(xi, yi)
                    largest = max(xi, yi)
                    delta = 0.1 * (largest - smallest)
                    bro_val = smallest - delta + self._random.random() * \
                            (largest - smallest + 2 * delta)
                    sis_val = smallest - delta + self._random.random() * \
                            (largest - smallest + 2 * delta)
                    bro.append(bro_val)
                    sis.append(sis_val)
                new_xs.append(self.bounder(bro))
                new_xs.append(self.bounder(sis))
            else:
                new_xs.append(mom)
                new_xs.append(dad)
        return new_xs

    def doGaussianMutation(self, designs):

        new_xs = list(designs)
        for j, design in enumerate(designs):
            for i, _ in enumerate(design):
                if self._random.random() < self.mutation_rate:
                    design[i] += self._random.gauss(self.mean, self.stdev)
            new_xs[j] = self.bounder(design)
        return new_xs

    def doReplacement(self, population, offspring):

        new_points = []
        unsorted_points = [p for p in population] + [p for p in offspring]

        while len(new_points) < len(population):
            ## Get current non-dominated front
            front = []
            for point in unsorted_points:
                self.addIfNotDominated(point, front)

            ## Add whole front if space permits
            if len(new_points) + len(front) < len(population):
                for p in front:
                    new_points.append(p)

            ## Otherwise sort by crowding distance and add those
            else:
                crowd = sortByCrowdingDistance(front)

                for point in crowd[0:len(population) - len(new_points)]:
                    new_points.append(point)

            ## Break if got full population
            if len(new_points) > len(population):
                new_points = new_points[0:len(population)]
            elif len(new_points) == len(population):
                break
            ## Otherwise remove the front from candidate points
            else:
                for point in front:
                    unsorted_points.remove(point)

        return new_points

def sortByCrowdingDistance(front):
    """Find crowding distance from each point to other points in a front.
    The front is a list of Point objects."""

    num_points = len(front)
    distances = [0 for _ in range(num_points)]
    pdicts = [{'point': p, 'index': i} for i, p in enumerate(front)]

    ## Go through front and find distance from each point to closest points in
    ## mean and standard deviation by sorting a list of dictionaries that keeps
    ## track of the indices in the original front
    pdicts.sort(key=lambda p: p['point'][1].std)
    distances[pdicts[0]['index']] = float('inf')
    distances[pdicts[-1]['index']] = float('inf')
    for i in range(1, num_points-1):
        distances[pdicts[i]['index']] += \
        (pdicts[i+1]['point'][1].std - pdicts[i-1]['point'][1].std)

    pdicts.sort(key=lambda p: p['point'][1].mean)
    for i in range(1, num_points-1):
        distances[pdicts[i]['index']] += \
        (pdicts[i+1]['point'][1].mean - pdicts[i-1]['point'][1].mean)

    indices = range(num_points)
    indices.sort(key=lambda i: distances[i], reverse=True)
    crowd = [front[i] for i in indices]

    return crowd

