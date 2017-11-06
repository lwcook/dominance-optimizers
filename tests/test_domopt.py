import unittest
import sys
import os
import pdb

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../domopt/')))

from optimizers import Optimization, Point
from populations import Stochastic
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm


class DomOptTest(unittest.TestCase):

    def testStochastic(self):

        samples = np.random.random(10)
        perf = Stochastic(design=[0, 0], samples=samples)
        mu = perf.mean
        std = perf.std
        CDF = perf.CDF
        supCDF = perf.supCDF

        samples1 = [0, 1, 2, 3, 4, 5]
        samples2 = [1, 2, 3, 4, 5, 6]

        s1 = Stochastic([0, 0], samples1, criteria='zsd')
        s2 = Stochastic([0, 0], samples2, criteria='zsd')
        self.assertFalse(s1 < s2)

        s1 = Stochastic([0, 0], samples1, criteria='fsd')
        s2 = Stochastic([0, 0], samples2, criteria='fsd')
        self.assertTrue(s1 < s2)

        s1 = Stochastic([0, 0], samples1, criteria='ssd')
        s2 = Stochastic([0, 0], samples2, criteria='ssd')
        self.assertTrue(s1 < s2)

    def testPoints(self):
        p1 = Point(0, 1)
        p2 = Point(0, 2)

        x, f = p1[0], p1[1]

        self.assertTrue(p1 < p2)

        samples1 = [0, 1, 2, 3, 4, 5]
        p1 = Point([0, 0], Stochastic(design=[0, 0], samples=samples1))

        samples2 = [1, 2, 3, 4, 5, 6]
        p2 = Point([0, 0], Stochastic(design=[0, 0], samples=samples2))

        self.assertEqual(p1, p2)
        self.assertTrue(p1 < p2)

        p1 = Point([0, 0], Stochastic([0, 0], samples1, criteria='zsd'))
        p2 = Point([0, 0], Stochastic([0, 0], samples2, criteria='zsd'))
        self.assertFalse(p1 < p2)

        p1 = Point([0, 0], Stochastic([0, 0], samples1, criteria='fsd'))
        p2 = Point([0, 0], Stochastic([0, 0], samples2, criteria='fsd'))
        self.assertTrue(p1 < p2)

        p1 = Point([0, 0], Stochastic([0, 0], samples1, criteria='ssd'))
        p2 = Point([0, 0], Stochastic([0, 0], samples2, criteria='ssd'))
        self.assertTrue(p1 < p2)

    def testOptimization(self):

        def evaluator(design):
            samples = [x*abs(design[1]) + design[0] for x in [1, 2, 3, 4, 5]]
            f = Stochastic(design, samples, criteria='mv')
            return f

        bounds = [(-5, -4), (-5, -4)]

        theOpt = Optimization(evaluator, bounds)

        dv0 = [-4.5, -4.5]
        xv = theOpt.scaleDVtoX(dv0)
        dv = theOpt.scaleXtoDV(xv)
        self.assertEqual(dv0, dv)

        p1 = theOpt.pointFromX(xv)
        p2 = theOpt.pointFromX(theOpt.scaleDVtoX([-4.2, -4.8]))

        memory = [p2]
        theOpt.addIfNotDominated(p1, memory)
        memory = theOpt.removeDominatedPoints(p1, memory)
        self.assertTrue(p1 in memory)
        self.assertFalse(p2 in memory)

    def testAlgorithms(self):

        def evaluator(design):
            samples = [x*abs(design[1]) + design[0] for x in [1, 2, 3, 4, 5]]
            f = Stochastic(design, samples, criteria='mv')
            return f

        bounds = [(-5, -4), (-5, -4)]

        TS = TabuSearch(evaluator, bounds, max_points=10, verbose=False)
        TS.optimize()

        GA = GeneticAlgorithm(evaluator, bounds, population_size=2,
                max_generations=2, verbose=False)
        GA.optimize()


if __name__ == "__main__":
    unittest.main()
