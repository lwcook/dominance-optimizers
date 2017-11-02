#!/usr/bin/python
import numpy as np
import pdb
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../domopt/')))

import GeneticAlgorithm, TabuSearch
from populations import Stochastic, saveArchive, readArchive

lb = [-1, -1]
ub = [1, 1]

def qfunc(x, u):

    p1 = 2*np.linalg.norm(np.array(x) - np.array([0, 0.7]))**2 - 0.4
    p2 = 5*np.linalg.norm(np.array(x) - np.array([0, -0.7]))**2

    q1 = 5*u[0]*u[1]*x[0]
    q2 = x[1]*u[1]**2
    q3 = x[1]*u[1]**3

    return 10 + 2*(q1 + q2 + q3 + min(p1, p2))


def run(stype='mv', bplot=False):

    # Here we use the same set of samples for each optimization for fair comparison
    with open('algebraic_test_usamples.txt', 'r') as f:
        us = json.loads(f.readlines()[0])

    # The evaluator method returns an object that measures the performance
    # of the design at dv - the design variable vector.
    # Here it is a Stochastic object, for which the < comparison operator
    # is defined by 'stype' the variable
    def evaluator(dv):
        samples = []
        for u in us:
            samples.append(qfunc(dv, u))
        return Stochastic(dv, samples, stype)

    # Generate a random initial design point
    dv0 = [np.random.uniform()*(ub[i] - lb[i]) + lb[i] for i in np.arange(len(lb))]

    # Run the genetic algorithm
    GA = GeneticAlgorithm.GeneticAlgorithm(evaluator=evaluator,
            bounds=[(lb[0], ub[0]), (lb[1], ub[1])],
            population_size=25, max_generations=10, verbose=True)
    ga_front = GA.optimize(seeds=[dv0])

    # Run the tabu search algorithm
    TS = TabuSearch.TabuSearch(evaluator=evaluator,
            bounds=[(lb[0], ub[0]), (lb[1], ub[1])],
            max_points=250, verbose=True)
    ts_front = TS.optimize(dv0)

    # Save the results
    # Use the python script "plot_example1.py" to plot the results
    if isinstance(stype, basestring):
        name = stype
    else:
        name = ''.join(stype)
    saveArchive([p[1] for p in ga_front], 'GA_' + name + '_front')
    saveArchive([p[1] for p in GA.LTM], 'GA_' + name + '_points')
    saveArchive([p[1] for p in ts_front], 'TS_' + name + '_front')
    saveArchive([p[1] for p in TS.LTM], 'TS_' + name + '_points')


def main():
    run('fsd')
    run('mv')
    run(['mv', 'fsd'])
    run('ssd')

if __name__ == "__main__":
    main()
