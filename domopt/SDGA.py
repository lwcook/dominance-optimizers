import time
import itertools
import math
import copy
import pdb

import matplotlib.pyplot as plt

import utilities_SDGA as utils

from populations import Individual, Stochastic, Optimization

from random import Random
import numpy as np

def stochastic_replacement(random, population, parents, offspring, args):
    """Replaces population using the non-dominated sorting technique.
    """
    survivors = []
    combined = population[:]
    combined.extend(offspring[:])

    # Perform the non-dominated sorting to determine the fronts.
    fronts = []
    pop = set(range(len(combined)))
    while len(pop) > 0:
        front = []
        for p in pop:
            dominated = False
            for q in pop:
                if combined[q] < combined[p]:
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        fronts.append([dict(individual=combined[f], index=f) for f in front])
        pop = pop - set(front)

    # Go through each front and add all the elements until doing so
    # would put you above the population limit. At that point, fall
    # back to the crowding distance to determine who to put into the
    # next population. Individuals with higher crowding distances
    # (i.e., more distance between neighbors) are preferred.
    for i, front in enumerate(fronts):
        if len(survivors) + len(front) > len(population):
            # Determine the crowding distance.
            crowd = PF_crowding_distance(front, combined)

            crowd.sort(key=lambda x: x['dist'], reverse=True)
            last_rank = [combined[c['index']] for c in crowd]
            r = 0
            num_added = 0
            num_left_to_add = len(population) - len(survivors)
            while r < len(last_rank) and num_added < num_left_to_add:
                if last_rank[r] not in survivors:
                    survivors.append(last_rank[r])
                    num_added += 1
                r += 1
            # If we've filled out our survivor list, then stop.
            # Otherwise, process the next front in the list.
            if len(survivors) == len(population):
                break

        else:
            for f in front:
                if f['individual'] not in survivors:
                    survivors.append(f['individual'])

    return survivors


def PF_crowding_distance(front, combined):
    """Find crowding distance for each point in a front where crowding distance
    is to every other point in a combined population"""

    distance = [0 for _ in range(len(combined))]
    individuals = front[:]
    num_individuals = len(individuals)

    individuals.sort(key=lambda x: x['individual'].stochastic.mean)
    distance[individuals[0]['index']] = float('inf')
    distance[individuals[-1]['index']] = float('inf')
    for i in range(1, num_individuals-1):
        distance[individuals[i]['index']] = \
        (distance[individuals[i]['index']] +  \
        (individuals[i+1]['individual'].stochastic.mean - \
            individuals[i-1]['individual'].stochastic.mean))

    individuals.sort(key=lambda x: x['individual'].stochastic.std)
    distance[individuals[0]['index']] = float('inf')
    distance[individuals[-1]['index']] = float('inf')
    for i in range(1, num_individuals-1):
        distance[individuals[i]['index']] = \
        (distance[individuals[i]['index']] +  \
        (individuals[i+1]['individual'].stochastic.std - \
            individuals[i-1]['individual'].stochastic.std))

    crowd = [dict(dist=distance[f['index']], index=f['index'])
            for f in front]

    return crowd


def tournament_selection(random, population, args):
    """Return a tournament sampling of individuals from the population."""

    tourn_size = args.setdefault('tourn_size', 2)
    num_selected = len(population)
    selected = []
    for _ in range(num_selected):
        tourn = random.sample(population, tourn_size)
        selected.append(min(tourn))
    return selected


def blend_crossover(random, candidates, args):
    """Return the offspring of blend crossover on the candidates.

    This function assumes that the candidate solutions are iterable
    and composed of values on which arithmetic operations are defined.
    It performs blend crossover, which is similar to a generalized 
    averaging of the candidate elements.

    .. Arguments:
       random -- the random number generator object
       candidates -- the candidate solutions
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:

    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *blx_alpha* -- the blending rate (default 0.1)
    - *lower_bound* -- the lower bounds of the chromosome elements (default 0)
    - *upper_bound* -- the upper bounds of the chromosome elements (default 1)

    The lower and upper bounds can either be single values, which will
    be applied to all elements of a chromosome, or lists of values of 
    the same length as the chromosome.
    """
    blx_alpha = args.setdefault('blx_alpha', 0.1)
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    bounder = args['_ec'].bounder

    cand = list(candidates)
    if len(cand) % 2 == 1:
        cand = cand[:-1]
    random.shuffle(cand)
    moms = cand[::2]
    dads = cand[1::2]
    children = []
    for mom, dad in zip(moms, dads):
        if random.random() < crossover_rate:
            bro = []
            sis = []
            for index, (m, d) in enumerate(zip(mom, dad)):
                smallest = min(m, d)
                largest = max(m, d)
                delta = blx_alpha * (largest - smallest)
                bro_val = smallest - delta + random.random() * \
                        (largest - smallest + 2 * delta)
                sis_val = smallest - delta + random.random() * \
                        (largest - smallest + 2 * delta)
                bro.append(bro_val)
                sis.append(sis_val)
            bro = bounder(bro, args)
            sis = bounder(sis, args) 
            children.append(bro)
            children.append(sis)
        else:
            children.append(mom)
            children.append(dad)
    return children

def gaussian_mutation(random, candidates, args):
    """Return the mutants created by Gaussian mutation on the candidates.

    Optional keyword arguments in args:

    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    - *mean* -- the mean used in the Gaussian function (default 0)
    - *stdev* -- the standard deviation used in the Gaussian function
      (default 1.0)
    """

    mut_rate = args.setdefault('mutation_rate', 0.1)
    mean = args.setdefault('mean', 0.0)
    stdev = args.setdefault('stdev', 1.0)
    bounder = args['_ec'].bounder

    mutants = list(candidates)
    for j, candidate in enumerate(candidates):
        for i, c in enumerate(candidate):
            if random.random() < mut_rate:
                candidate[i] += random.gauss(mean, stdev)
        mutants[j] = bounder(candidate, args)
    return mutants

class SDGA(Optimization):
    """Evolutionary computation representing the nondominated
    sorting genetic algorithm under uncertainty.  """

    def __init__(self, evaluator, bounds, generator, log_file='GA_log',
            bplot=False, observer=None, stype='MV', verbose=False):

        super(SDGA, self).__init__(evaluator, bounds)

        random = Random()
        random.seed(time.time())
        self.verbose = verbose

        self.observer = observer
        self.generator = generator
        self.stype = stype
        self.archive = None

        self.population = None
        self.num_evaluations = 0
        self.num_generations = 0
        self._random = random
        self._kwargs = dict()

    def evolve(self, pop_size=100, seeds=[], **args):
        """Perform the evolution."""

        self._kwargs = args
        self._kwargs['_ec'] = self
        self._kwargs['num_selected'] = pop_size
        self.population = []

        self.archive = []
        self.LTM = []

        generator = self.generator
        evaluator = self.evaluator

        # Create the initial population.
        try:
            iter(seeds)
        except TypeError:
            seeds = [seeds]

        initial_dvs = list(seeds)
        initial_cs = [self.scaleDVToCandidate(dvs) for dvs in initial_dvs]

        num_generated = max(pop_size - len(seeds), 0)
        i = 0
        while i < num_generated:
            dvs = generator(random=self._random, args=self._kwargs)
            cs = self.scaleDVToCandidate(dvs)
            if cs not in initial_cs:
                initial_cs.append(cs)
                i += 1

        self.population = [self.evalCandidate(cs) for cs in
                           initial_cs[0:pop_size]]

        self.num_evaluations = len(self.population)
        self.num_generations = 0

        for ind in self.population:
            added_to_archive = self.addIfNotDominated(ind, self.archive)
            self.archive = self.removeDominatedPoints(ind, self.archive)

        ##############################################################
        # MAIN LOOP
        ##############################################################
        while self.num_generations < self._kwargs['max_generations']:

            if self.observer is not None:
                self.observer(self.num_evaluations, self.archive, self.LTM)

            self.num_generations += 1
            print 'Generation: ', self.num_generations

            # Select individuals to be parents based on tournament selection
            # (compare two random individuals and pick best)
            parents = tournament_selection(random=self._random,
                    population=list(self.population), args=self._kwargs)
            parent_cs = [copy.deepcopy(i.candidate) for i in parents]
            offspring_cs = parent_cs

            # Temperature decreasing encouraging less exploring as time goes on.
            T = 1 - float(self.num_generations) /  \
                (self._kwargs['max_generations']+1)

            self._kwargs['crossover_rate'] = 0.9
            offspring_cs = blend_crossover(random=self._random,
                    candidates=offspring_cs, args=self._kwargs)

            self._kwargs['mutation_rate'] = 0.2 - 0.05*T
            self._kwargs['stdev'] = 0.6 + 0.2*T
            offspring_cs = gaussian_mutation(random=self._random,
                    candidates=offspring_cs, args=self._kwargs)

            # Evaluate offspring.
            offspring = [self.evalCandidate(cs) for cs in offspring_cs]
            self.num_evaluations += len(offspring_cs)

            # Replace individuals (using nsga-II non-dominated sorting)
            self.population = stochastic_replacement(random=self._random,
                    population=list(self.population),
                    parents=parents, offspring=offspring, args=self._kwargs)

            # Archive individuals.
            for ind in self.population:
                added_to_archive = self.addIfNotDominated(ind, self.archive)
                self.archive = self.removeDominatedPoints(ind, self.archive)

        return self.archive
