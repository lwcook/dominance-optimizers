import copy

class Point(object):
    """Object for storing designs in optimization.
    It can be called as a tuple, corresponding to (x, f)

    :param list x: Object representing the design that the optimization
        manipulates, i.e. scaled design variables
    :param object f: Object that measures performance,
        it must have the < and == operators defined.

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
    """Base class for optimziations operating on Point objects.

    Designs are stored as Point objects - the first entry is what the optimizer
    controls (i.e. scaled design variables), the second entry is an object
    provided obtained by the evaluator method that measures performance,
    it must have a comparison operator.

    This generalization allows any comparison operator to be defined and thus
    general non-dominance criteria to be used.

    The optimizer scales the design variables to range over [0,10] so that
    an algorithm's methods operate over an appropriate scaled space.

    :param function evaluator: Function that takes a vector of design variables
        and returns an object that can be compared via < and == operators.

    :param list bounds: List of bounds on design variables in the form
        [(l0, u0), (l1, u1), ..., (ln, un)] where lb, uk are the lower and upper
        bounds on the kth design variable respectively.

    :param function observer: User defined function that is called every time
        a Point is evaluated, it is passed the current Point, the current
        non-dominated archive, and the list of all points visited.
        It should be able to be called by myObserver(point, archive, history)

    """

    def __init__(self, evaluator, bounds, observer=None):

        self.bounds = bounds
        self.evaluator = evaluator
        if observer is None:
            self.observer = lambda point, archive, visited: None
        else:
            self.observer = observer

        self.history = [] # Log of all Points visited
        self.archive = [] # Archive of non-dominated Points

        self.opt_ub = 10.0 # Upper bound for scaling of design variables
        self.opt_lb = 0.0  # Lower bound for scaling of design variables

        self.num_evaluations = 0

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

    def evalDV(self, DV):
        self.num_evaluations += 1
        return self.evaluator(DV)

    def evalX(self, x):
        return self.evalDV(self.scaleXtoDV(x))

    def pointFromX(self, x, bArchive=True):
        x = [xi for xi in x]
        visited_xs = [y for (y, fy) in self.history]
        if x in visited_xs:
            other = copy.copy(self.history[visited_xs.index(x)])
            point = copy.copy(other)
        else:
            f = self.evalX(x)
            point = Point(x, f)
            self.history.append(point)

        if bArchive:
            self.addIfNotDominated(point, self.archive)
            self.archive = self.removeDominatedPoints(point, self.archive)

        self.observer(point, self.archive, self.history)

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
