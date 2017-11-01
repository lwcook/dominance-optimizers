import copy

class Point(object):
    """Object for storing designs in optimization.
    It can be called as a tuple, corresponding to (x, f)

    :param list x: Object representing the design that the optimization
        manipulates
    :param object f: Object that measures performance provided to the
        optimization, it must have the < and == operators defined.

    """
    def __init__(self, x, f):
        self.x = x
        self.f = f
        self.tuple = (x, f)

    def __repr__(self):
        return '('+repr(self.x)+', '+repr(self.f)+')'

    def __str__(self):
        return '('+str(self.x)+', '+str(self.f)+')'

    def __getitem__(self, key):
        return self.tuple[key]

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.f == other.f

class Optimization(object):
    """Optimization for dealing with populations of designs.

    Designs are stored as Point objects - the first entry is what the optimizer
    controls (i.e. scaled design variables), the second entry is an object
    provided obtained by the evaluator method that measures performance,
    it must have a comparison operator.

    This generalization allows any comparison operator to be defined and thus
    general non-dominance criteria to be used.

    The optimizer scales the design variables to range over [0,10] so that
    an algorithm's methods operate over an appropriate scaled space.

    :param function evaluator: function that takes a list of design variables
        (that are contained within the bounds) and returns an object that
        must have the comparison operators < and ==
    :param list bounds: list of bounds on the design variables in the form
        [(l0, u0), (l1, u1), ..., (ln, un)], where li and ui are the lower and
        upper bounds on the ith design variable.
    """

    def __init__(self, evaluator, bounds):

        self.bounds = bounds
        self.evaluator = evaluator
        self.LTM = []

        self.opt_ub = 10.0
        self.opt_lb = 0.0

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, lbubs):
        self._bounds = lbubs
        lbs, ubs = [], []
        for l, u in lbubs:
            lbs.append(l)
            ubs.append(u)
        self._lbs = lbs
        self._ubs = ubs

#    def bounder(self, x):
#        bounded_x = copy.copy(x)
#        for i, (c, l, u) in enumerate(zip(x, self._lbs, self._ubs)):
#            bounded_x[i] = max(min(c, u), l)
#        return bounded_x

    def bounder(self, x):
        bounded_x = copy.copy(x)
        for i, c in enumerate(x):
            bounded_x[i] = max(min(c, self.opt_ub), self.opt_lb)
        return bounded_x

    def scaleXtoDV(self, x):
        return [((xi-self.opt_lb)/(self.opt_ub-self.opt_lb) *
                (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0])
                for i, xi in enumerate(x)]

    def scaleDVtoX(self, design_variables):
        return [self.opt_lb + ((self.opt_ub - self.opt_lb) *
                (dvi - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0]))
                for i, dvi in enumerate(design_variables)]

    def evalX(self, x):
        return self.evaluator(self.scaleXtoDV(x))

    def evalDV(self, x):
        return self.evaluator(DV)

    def pointFromX(self, x):
        x = [xi for xi in x]
        visited_xs = [y for (y, fy) in self.LTM]
        if x in visited_xs:
            other = copy.copy(self.LTM[visited_xs.index(x)])
            point = copy.copy(other)
        else:
            f = self.evalX(x)
            point = Point(x, f)
            self.LTM.append(point)

        return point

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

