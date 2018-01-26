import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../domopt/')))

from surrogates import PolySurrogate

class TestSurrogate(unittest.TestCase):

    def testPolySurrogate(self):

        def fun(x):
            return x**3 + x**2

        poly1 = PolySurrogate(dimensions=1, order=3, poly_type='hermite')
        u = poly1.getQuadraturePoints()
        poly1.train([fun(ui) for ui in u])

        self.assertAlmostEqual(poly1.predict(1), fun(1))

        poly2 = PolySurrogate(dimensions=1, order=3, poly_type=['legendre'])
        poly1.train(fun)

        self.assertAlmostEqual(poly1.predict(2), fun(2))
