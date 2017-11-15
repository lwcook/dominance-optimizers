import time
import math
import pdb
from random import Random

import numpy as np

from optimizers import Optimization

class GeneticAlgorithm(Optimization):
    """Evolutionary computation representing the nondominated
    sorting genetic algorithm under uncertainty.
    """

    def __init__(self, evaluator, bounds, observer=None,
            population_size=100, max_generations=50,
            verbose=False):

        super(GeneticAlgorithm, self).__init__(evaluator, bounds, observer)

        self._random = Random()
        self._random.seed(time.time())

        self.population_size = population_size
        self.max_generations = max_generations

        self.mutation_rate = 0.15
        self.crossover_rate = 0.9
        self.stdev = 0.6
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
        self.history = []
        self.archive = []

        self.num_generations = 0
        self.num_evaluations = 0  # is incremented by self.PointFromX()

        # Create the initial population.
        initial_xs = [self.bounder(self.scaleDVtoX(dv)) for dv in seeds]
        while len(initial_xs) < self.population_size:
            initial_xs.append([self._random.uniform(self.opt_lb, self.opt_ub)
                                for _ in self.bounds])

        self.population = [self.pointFromX(x) for x in
                           initial_xs[0:self.population_size]]

        ##############################################################
        # MAIN LOOP
        ##############################################################
        while True:

            if self.num_generations >= self.max_generations:
                break
            self.num_generations += 1

            if self.verbose:
                print 'Generation: ', self.num_generations

            parents = self.doTournamentSelection(self.population)
            child_xs = [point[0] for point in parents]  # get x from Point

            child_xs = self.doBlendCrossover(child_xs)
            child_xs = self.doGaussianMutation(child_xs)

            children = [self.pointFromX(x, bArchive=True) for x in child_xs]

            self.population = self.doReplacement(self.population, children)


        return self.archive


    def doTournamentSelection(self, population):
        tsize = 2
        return [min(self._random.sample(population, tsize))
                        for _ in population]

    def doBlendCrossover(self, xs):

        self._random.shuffle(xs)
        new_xs = []

        for ii in range(math.trunc(0.5*len(xs))):
            p1, p2 = xs[2*ii], xs[2*ii+1]
            if self._random.random() < self.crossover_rate:
                c1, c2 = [], []
                for (xi, yi) in zip(p1, p2):
                    a, b, delt = min(xi, yi), max(xi, yi), 0.1*abs(xi-yi)
                    c1.append(self._random.uniform(a-delt, b+delt))
                    c2.append(self._random.uniform(a-delt, b+delt))

                new_xs.append(self.bounder(c1))
                new_xs.append(self.bounder(c2))
            else:
                new_xs.append(p1)
                new_xs.append(p2)

        return new_xs

    def doGaussianMutation(self, xs):
        new_xs = []
        for x in xs:
            new_x = []
            for ii, xi in enumerate(x):
                if self._random.random() < self.mutation_rate:
                    new_x.append(xi + self._random.gauss(self.mean,
                        self.stdev))
                else:
                    new_x.append(xi)
            new_xs.append(self.bounder(new_x))
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
    distances[pdicts[0]['index']] = np.inf
    distances[pdicts[-1]['index']] = np.inf
    for i in range(1, num_points-1):
        distances[pdicts[i]['index']] += \
        (pdicts[i+1]['point'][1].std - pdicts[i-1]['point'][1].std)

    pdicts.sort(key=lambda p: p['point'][1].mean)
    for i in range(1, num_points-1):
        distances[pdicts[i]['index']] += \
        (pdicts[i+1]['point'][1].mean - pdicts[i-1]['point'][1].mean)

    indices = range(num_points)
    indices.sort(key=lambda i: distances[i], reverse=True)
    sorted_dists = [front[i] for i in indices]

    return sorted_dists

