'''
Statistical and mathematical routines. Built on top of 'numpy'.

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import random, bisect, json, copy, numbers, math, numpy as np
import numpy.linalg as linalg
from numpy import sum, mean, zeros, sqrt, pi, exp, isnan, isinf, arctan
from collections import OrderedDict, namedtuple


if __name__ != "__main__":
    from .util import isnumber, isstring, isdict, isfunction, getattrs
else:
    from nifty.util import isnumber, isstring, isdict, isfunction, getattrs


########################################################################################################################
###
###   UTILITIES. Numpy extensions
###

ipi = 1./pi         # inverted PI

def isarray(x):    return isinstance(x, np.ndarray)


def np_find(condition):
    "Return the indices where ravel(condition) is true. Copied from matplotlib/mlab.py"
    res, = np.nonzero(np.ravel(condition))
    return res
np.find = np_find

def np_dumps(V, format_spec = '%.5g'):
    "Return compact string (JSON) representation of numpy vector, with commas between items instead of spaces!"
    return '[' + ','.join([format_spec % x for x in V]) + ']'

def np_loads(s):
    "Load numpy vector from a string in JSON format: [x1,x2,...,xn]"
    return np.array(json.loads(s))

# shorthand; norm() calculates norm of a matrix or vector
norm = linalg.norm


def ceildiv(num, divisor):
    "Ceil division that uses pure integer arithmetics. Always correct, unlike floating-point ceil() + conversion to int."
    return -(-num // divisor)

def round_up(num, divisor):
    "Round `num` up to the nearest multiple of `divisor`."
    divisor = abs(divisor)
    return -(-num // divisor) * divisor

def round_down(num, divisor):
    "Round `num` down to the nearest multiple of `divisor`."
    return num - (num % divisor)


########################################################################################################################
###
###   RANDOM NUMBERS and PROBABILITY DISTRIBUTIONS
###

# see http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
def weighted_random(weights, rnd = random):
    """Random value chosen from a discrete set of values 0,1,... with weights. 
    Weights are (unscaled) probabilities of values, possibly different for each one.
    You can pass your own Random object in 'rnd' to provide appropriate seeding."""
    totals = np.cumsum(weights)
    throw = rnd.random() * totals[-1]
    return np.searchsorted(totals, throw)

class WeightedRandom(object):
    """Generator of random values (can be non-numeric) from a discrete set with weights. 
    Weights are (unscaled) probabilities of values, possibly different for each one."""
    def __init__(self, weights, vals = None, seed = None):
        self.totals = []
        self.total = 0
        for w in weights:
            self.total += w
            self.totals.append(self.total)        
        
        if vals is None: vals = range(len(weights))
        self.vals = vals
        self.rnd = random.Random(seed)
    
    def random(self):
        rnd = self.rnd.random() * self.total
        i = bisect.bisect_right(self.totals, rnd)
        return self.vals[i]


#####################################################################################################################################################

class Distribution(object):
    "Base class for probability distributions."
    
    
    rand = None                         # Random generator to use in get_random() if `rand` argument is None
    rand_default = random.Random()      # fallback Random instance to use in random() if both `rand` argument and self.rand are None
    
    def __init__(self, **common):
        self.set_rand(**common)

    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        if seed is not None:
            rand = random.Random(seed)
        if rand is not None:
            self.rand = rand
        else:
            self.rand_default = random.Random()
        # print self, rand

    def _set_rand_recursive(self, items, overwrite):
        "To be used by subclasses that contain nested Distribution objects."
        if not (overwrite or self.rand): return
        for item in items:
            if isinstance(item, Distribution) and (overwrite or item.rand is None):
                item.set_rand(self.rand, recursive = True, overwrite = overwrite)
        
    def _fix_rand(self, rand = None):
        fixed_rand = rand or self.rand or self.rand_default
        assert fixed_rand is not None
        return fixed_rand

    def _get_random_value(self, rand):
        """
        Returns a single item from the probability distribution represented by self.
        Override in subclasses to implement selection from a custom probability distribution.
        :param rand: instance of standard python <random.Random> class that should be used as a source of randomness
        :return: value drawn from the probability distribution represented by self
        """
        return rand.random()

    def get_random(self, rand = None):
        """
        Returns a single item from the probability distribution represented by self.
        In subclasses, always ovveride _get_random_value() instead of this method.
        """
        rand = self._fix_rand(rand)
        return self._get_random_value(rand)
        # return self._get_rand(rand).random()
    
    def generate_random(self, rand = None):
        "Generate an infinite stream of random items from the distribution represented by self."
        rand = self._fix_rand(rand)
        while True:
            yield self._get_random_value(rand)
    
    # Distribution instances are callable, which provides a shorthand for get_random():
    #   distribution() is equiv. to distribution.get_random()
    #
    def __call__(self, *args, **kwargs):
        return self.get_random(*args, **kwargs)
    
    def copy(self):
        "Deep copy of the random distribution represented by self."
        return copy.deepcopy(self)
    
    
class Fixed(Distribution):
    "Fixed value (deterministic, no randomness)."
    
    def __init__(self, value, **common):
        super(Fixed, self).__init__(**common)
        self.value = value

    def _get_random_value(self, rand):
        return self.value

    
class Interval(Distribution):
    """Uniform distribution over [start,stop) or [start,stop] interval, depending on rounding, like in random.uniform().
       Or just the `start` value if stop=None.
       If `cast` is not-None, values are passed through cast() before being returned.
       Typically, cast is a type (e.g., int), or a rounding function that should be applied to the selected value.
    """
    
    def __init__(self, start = 0.0, stop = None, cast = None, **common):
        super(Interval, self).__init__(**common)
        self.start = start
        self.stop = stop
        self.cast = cast

    def _get_random_value(self, rand):
        if self.stop in (None, self.start): return self.start
        val = rand.uniform(self.start, self.stop)
        if self.cast:
            return self.cast(val)
        return val
    
    
class Range(Distribution):
    """Uniform distribution over integral numbers in [start,stop] range, including both endpoints, like in random.randint().
       Or just the `start` value if stop=None.
    """
    
    def __init__(self, start = 0.0, stop = None, **common):
        super(Range, self).__init__(**common)
        self.start = start
        self.stop = stop

    def _get_random_value(self, rand):
        if self.stop in (None, self.start): return self.start
        return rand.randint(self.start, self.stop)
    
    
class Choice(Distribution):
    """Discrete probability distribution over a fixed set of possible outcomes (choices).
       In random(), if a chosen value is an instance of Distribution, a subsequent choice from this distribution is performed,
       which allows nesting of distributions and building composite ones.
    """
    
    choices = None      # list of possible outcomes (values) to choose from
    is_dist = None      # list of flags: is_dist[i]==True iff choices[i] is a probability distribution itself (an instance of Distribution)
    
    def __init__(self, *choices, **common):
        """`choices` is either a sequence (then uniform distribution is assumed), or a dictionary of {value: probability} pairs.
           The sum of `probability` values must be positive, but not necessarily 1.0 - probabilities are normalized by default to unit sum.
        """
        super(Choice, self).__init__(**common)
        if not choices: raise Exception('Choice.__init__: the list/dict of choices must be non-empty')
        if len(choices) == 1: choices = choices[0]
        
        self.choices = self._list_outcomes(choices)
        self.is_dist = [isinstance(v, Distribution) for v in self.choices]      # this is pre-computed to speed up execution of random()
        
        if isdict(choices):
            self.probs = np.array([choices[v] for v in self.choices], dtype = float)
            self.probs /= float(self.probs.sum())                               # normalize probabilities to unit sum
        else:
            self.probs = None
        
    def _list_outcomes(self, choices):
        """
        Convert a collection (list, dict, iterable) of `choices` to a list of outcomes: ordered
        and with probabilities dropped for now. Make sure the outcomes are unique and their ordering is deterministic,
        to guarantee repeatable outcomes under the same random seed.
        Deterministic ordering can be either predefined by a caller, if `choices` are provided originally
        as a list or OrderedDict - but not as a standard (unordered) dict in Python 2 (!) -
        or by explicit sorting of choice values, the latter being reliable for standard types only, though.
        """
        outcomes = list(choices)
        
        if len(outcomes) != len(set(outcomes)): raise Exception('The list of choices contain duplicates: %s' % outcomes)
        if isinstance(choices, (list, OrderedDict)): return outcomes
        
        # `choices` collection has no predefined ordering? must sort `outcomes` explicitly,
        # but first ensure all values are reliably (deterministically) sortable...
        for v in outcomes:
            if not isinstance(v, (basestring, list, tuple, numbers.Number)):
                raise Exception('Unsortable choice value, must be a number/string/list/tuple: %s' % v)
        
        return sorted(outcomes)
    
        
    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        
        super(Choice, self).set_rand(rand, seed, recursive, overwrite)
        if recursive:
            self._set_rand_recursive(self.choices, overwrite)
        
    def _get_random_value(self, rand):
        
        seed = rand.randrange(4294967296)                               # 4294967296 == 2**32 == 1 + the maximum seed for RandomState
        np_rand = np.random.RandomState(seed)                           # use numpy's random to be able to choose items with non-uniform distribution
        
        choice = np_rand.choice(len(self.choices), p = self.probs)      # choose an index into self.choices[]
        val = self.choices[choice]
        
        # print self, 'random() choices:', self.choices, '  is_dist:', self.is_dist
        # if isinstance(val, Distribution):
        if self.is_dist[choice]:
            val = val.get_random(rand)
        
        return val
            
    
class Switch(Choice):
    """Binary True/False switch that generates True with a given probability. Behaves like Choice with 2 outcomes.
       To assign custom names to outcomes use Choice directly instead of Switch.
    """
    def __init__(self, p_true = 0.5, **common):
        
        assert 0 <= p_true <= 1
        p_false = 1 - p_true
        choices = {True: p_true, False: p_false}
        super(Switch, self).__init__(choices, **common)

    
class Intervals(Choice):
    "Combination of uniform distributions defined on intervals."
    
    def __init__(self, intervals, cast = None, **common):
        "`intervals` is either a list of (start,stop) pairs, or a dict of {(start,stop): probability} tuples."
        
        outcomes = self._list_outcomes(intervals)

        # wrap up interval pairs in Interval class, so that the Choice class knows these pairs should be treated as sub-distributions
        if isdict(intervals):
            choices = OrderedDict([(Interval(start, stop, cast = cast), intervals[(start,stop)]) for (start,stop) in outcomes])
        else:
            choices = [Interval(start, stop, cast = cast) for (start,stop) in outcomes]
            
        super(Intervals, self).__init__(choices, **common)
        
    
class RandomInstance(Distribution):
    """Probability distribution over a space of instances of a given class.
       Attribute values of a generated instance are chosen at random from probability distributions defined in the subclass,
       separately for each attribute.
    """
    
    class_type = None           # the class whose instances are going to be created and returned in random()
    attributes = None           # list of attributes that shall be initialized when a random <class_type> object is created;
                                # in the subclass, these attributes should contain instances of Distribution that define probability distr. to choose values from
    
    def __init__(self, **common):
        
        # initialize the list of attributes
        if self.attributes is None:
            self.attributes = [attr for attr in dir(self.__class__) if not attr.startswith('__')]
            
        # copy nested sub-Distributions from class-level attributes to instance attributes, to allow their customization
        # (e.g., setting a random seed) without affecting the shared class-level object
        for attr in self.attributes:
            if attr in self.__dict__: continue              # already has a value at instance level? skip
            item = getattr(self, attr, None)
            if isinstance(item, Distribution):
                setattr(self, attr, item.copy())

        super(RandomInstance, self).__init__(**common)
        
    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        
        super(RandomInstance, self).set_rand(rand, seed, recursive, overwrite)
        
        # walk through attributes of `self` and for every instance of Distribution initialize its `rand` if missing, or if `overwrite`=True
        if recursive:
            items = getattrs(self, self.attributes).values()
            self._set_rand_recursive(items, overwrite)

        
    def _get_random_value(self, rand):
        """Walk through attributes of `self` and for each one being an instance of Distribution (probability distribution)
           draw a random value and assign to this attribute.
        """
        # print 'RandomInstance class & attributes:', self.class_type, self.class_attr
        assert self.class_type is not None
        obj = self.class_type()
        
        for attr in self.attributes:
            distr = getattr(self, attr, None)
            if isinstance(distr, Distribution):
                val = distr.get_random(rand)
                setattr(obj, attr, val)
        
        self.postprocess(obj, self._fix_rand(rand))
        return obj
    
    def postprocess(self, obj, rand):
        """Override in sublasses to perform additional post-processing of an object generated by random().
           `rand` is a Random generator that is guaranteed to be not-None and should be used instead of self.rand.
        """
        pass
    

#####################################################################################################################################################

class Randomized(object):
    """
    Base class for classes that can generate their own instances randomly with the following syntax:
    
        ClassName.get_random() or
        ClassName.generate_random(),
    
    where get_random() and generate_random() are class-level methods and are implemented by this base class.
    """

    _rand = None                        # Random generator to use in get_random() if `rand` argument is None
    _rand_default = random.Random()     # fallback Random instance to use in random() if both `rand` argument and self.rand are None
    

    @classmethod
    def get_random(cls, randomness = "__random__", rand = None, **params):
        """
        Returns a single item from the probability distribution represented by self.
        In subclasses, always overide _get_random_instance() instead of this method.
        """
        if isstring(randomness): randomness = getattr(cls, randomness)
        if isinstance(randomness, RandomInstance):
            return randomness.get_random(rand)
        
        attributes = cls._get_attributes(randomness, params)
        return cls._get_random_instance(attributes, rand)
    
    
    @classmethod
    def generate_random(cls, randomness = "__random__", rand = None):
        "Generate an infinite stream of random items from the distribution represented by self."
        
        if isstring(randomness): randomness = getattr(cls, randomness)
        if isinstance(randomness, RandomInstance):
            def get_next():
                return randomness.get_random(rand)
        else:
            attributes = cls._get_attributes(randomness)
            def get_next():
                return cls._get_random_instance(attributes, rand)

        while True:
            yield get_next()
    
    
    @classmethod
    def _get_attributes(cls, randomness, params):
        "Check the type of 'randomness' object and convert it appropriately to a dictionary of attribute values/distributions."
        
        if isdict(randomness):          return randomness
        elif isfunction(randomness):    return randomness(**params)
        else:                           return getattrs(randomness)

    
    @classmethod
    def _get_random_instance(cls, attributes, rand):

        obj = cls()
        for attr, val in attributes.items():
            if isinstance(val, Distribution):
                val = val.get_random(rand)
            setattr(obj, attr, val)
        
        obj.postprocess(cls._fix_rand(rand))
        return obj
    

    @classmethod
    def _fix_rand(cls, rand = None):
        fixed_rand = rand or cls._rand or cls._rand_default
        assert fixed_rand is not None
        return fixed_rand


    # def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
    #
    #     super(RandomInstance, self).set_rand(rand, seed, recursive, overwrite)
    #
    #     # walk through attributes of `self` and for every instance of Distribution initialize its `rand` if missing, or if `overwrite`=True
    #     if recursive:
    #         items = getattrs(self, self.class_attr).values()
    #         self._set_rand_recursive(items, overwrite)

    def postprocess(self, obj, rand):
        """
        Override in sublasses to perform additional post-processing of a newly generated random object.
        `rand` is a Random generator that is guaranteed to be not-None and should be used instead of self.rand.
        """
        pass
    
    
    
########################################################################################################################
###
###   SCALAR functions & point-wise transformations
###

def minmax(x):
    "Calculate minimum and maximum in one step."
    return (np.min(x), np.max(x))
    
def heat(X, magnitude, random = np.random.RandomState()):
    "Add random heat to the values"
    shape = X.shape if isarray(X) else X
    return (random.random_sample(shape)-0.5) * (magnitude*2)

def logx(x):
    "Natural logarithm shifted by 1 so that 0-->0, and extended to entire R range, symmetrically respective to (0,0); logx(-1,0,1) == [-log(2), 0, log(2)]."
    return np.log(abs(x) + 1) * np.sign(x)

def mexican(x, std = 1.0, mean = 0.0):
    '''Values of mexican hat function, calculated in point x (can be an ndarray).
       See: http://en.wikipedia.org/wiki/Mexican_hat_wavelet
    '''
    if mean != 0.0 or std != 1.0:
        x = (x - mean) / std
    x2 = x**2
    f1 = 2.0 / ((3.0*std)**0.5 * pi**0.25)
    f2 = 1.0 - x2
    f3 = exp(-x2 / 2)
    return f1 * f2 * f3
    

########################################################################################################################
###
###   SIGMOIDAL functions, for predictive models and data processing
###

def logistic(x, center = None, slope = None, deriv = False):
    "Logistic function: f(x) = 1/(1+e^(-x)). Derivative: f'(x) = f(x)*(1-f(x))"
    if center is not None: x = x - center
    if slope is not None: x = x * slope
    y = 1. / (1. + exp(-x))
    if not deriv: return y
    d = y * (1-y)
    if slope is not None: d *= slope
    return y, d

def cauchy(x, center = None, slope = None, deriv = False):
    """CDF of a Cauchy distribution: f(x) = arctan(x)/pi + 0.5. Derivative: f'(x) = 1/(1+x^2) * 1/pi.
    Has similar shape as logistic function but doesn't saturate so fast, 
    so is safer to use when saturation is undesirable.
    Good for modeling probabilities that will be used in multiplications, like log-likelihood estimates,
    and should stay away from boundary values of 0 and 1.
    In the range [-1.5,1.5], cauchy(x) differs from logistic(x) by no more than 8%,
    with intersections at x=0.0 and near x=1.4. Only after |x|=1.5 the two functions differ substantially.
    """
    if center is not None: x = x - center
    if slope is not None: x = x * slope
    y = arctan(x)/pi + 0.5
    if not deriv: return y
    d = ipi / (1. + x**2)
    if slope is not None: d *= slope
    return y, d

def sigmoid_sqrt(x, center = None, slope = None):
    """
    Smooth sigmoidal function based on 'sqrt'. Symmetrical. All values in (-1,1) range.
    The prototype function is y = x/sqrt(1+x^2).
    f(2.0)  =  0.89
    f(-2.0) = -0.89
    """
    if center is not None: x = x - center
    if slope is not None: x = x * slope
    return x / np.sqrt(1 + x**2)

def sigmoid_lin(x, p0, p1):
    "Piece-wise linear sigmoidal function, with values in [0,1], 0/1 glue points in p0/p1 respectively"
    if x is None: return 0.5
    y = (x - p0) / (p1 - p0)
    return y.clip(0, 1)
    

def binarize(x, x01 = 0.1, x09 = 0.9, funsigm = sigmoid_sqrt, delta = 2.0):
    """
    Softly binarize number X or all values in array X, by non-linear mapping of [0,1] range onto itself, 
    through a sigmoidal function 'funsigm' (f), which has values in [-1,1].
    f is shifted and scaled linearly to map 0 to 0 and 1 to 1. Center of [x01,x09] range is mapped to 0.5 = (1+f(0))/2.
    Slope is defined in such a way that x01 is mapped onto (1+f(-delta))/2 ~= 0.1 
    and x09 is mapped onto (1+f(+delta))/2 ~= 0.9
    (but then value range is slightly stretched to fully fill out [0,1] range).
    'delta' is an approximate length of the range on which 'funsigm' attains intermediate values (far from +/-1), 
    ideally f(delta) >= 0.9. Values in (-inf,x01] map to "almost 0"; [x09,+inf) map to "almost 1".
    All output values are truncated to [0,1] range.
    Usage:
       plot(t,binarize(t,0.3,0.7))
    """
    if isinstance(x, np.ndarray):
        X = x
    elif isinstance(x, list):
        X = np.array(x)
    else:
        X = np.array([x])
    slope = 2 * delta / (x09 - x01)
    center = (x01 + x09) / 2.0
    
    Y = funsigm(X, center, slope)
    v0 = funsigm(0, center, slope)
    v1 = funsigm(1, center, slope)
    Y = (Y - v0) * (1.0/(v1-v0))
    
    Y.clip(0, 1)
    if isnumber(x):
        Y = Y.flatten()[0]
        
    return Y


########################################################################################################################
###
###   VECTOR-to-SCALAR transformations
###

def normv2(x, axis = -1):
    "Squared euclidean norm of vectors contained in matrix 'x', along 'axis'. Last axis by default."
    return np.sum(x*x, axis)
def normv(x, axis = -1):
    "Euclidean norm of vectors contained in matrix 'x', along 'axis'. Last axis by default."
    return np.sqrt(normv2(x,axis))

def softmax(scores, slope = None, eps = 1e-10):
    "softmax function: turns a vector of real-valued scores into unit-sum probabilities by applying exp() and normalization."
    scores = scores - np.max(scores)            # shift values to avoid overflow in exp()
    if slope is not None: scores *= slope
    E = exp(scores) #+ EPS                      # +EPS to avoid 0.0 probabilities
    Z = np.sum(E)
    #print "", Z, list(exps.flat)
    assert not isnan(Z) and not isinf(Z)
    return E / (Z + eps)                     # 1-d vector


########################################################################################################################
###
###   VECTOR-to-VECTOR transformations
###

def zeroSum(X):
    "Shift values of (each) vector by a constant to obtain zero sum. In 2D array, vectors are rows."
    if X.ndim == 1:
        return X - mean(X)
    if X.ndim == 2:
        return X - mean(X,1)[:,np.newaxis]

def unitSum(X):
    "Scale 1D vector X, or all rows of 2D array X, to unit sum. All sums must be originally non-zero."
    if X.ndim == 1:
        return X / np.sum(X)
    if X.ndim == 2:
        scale = 1. / np.sum(X,1)
        scale = scale[:,np.newaxis]
        return X * scale                # numpy "broadcasting" activates here, it automatically copies 'scale' to all columns
    
def unitNorm(X, p = 2):
    "Scale vector(s) to unit norm."


########################################################################################################################
###
###   AGGREGATIONS of vectors/series to scalars
###

def likelihood(probs, log = np.log, exp = False):
    """Average log-likelihood of observed events, given a sequence of their 'a priori' probabilities.
    If exp=True, returns exp() of this: a geometric average of likelihoods of observed events.
    'log' is the logarithm function to use (log/log2/log10)."""
    loglike = mean(log(probs))
    return np.exp(loglike) if exp else loglike
    

########################################################################################################################
###
###   ACCUMULATORS of streams of values or arrays
###

class MinMax(object):
    """
    Monitors a stream of values and (optionally) their arguments, and upon request returns
    min(), max(), argmin(), argmax(), idxmin() or idxmax() of values/arguments/indices seen so far.
    Values can be not-None objects of any type that supports comparison.
    Arguments can be not-None objects of any type; None is treated as a missing argument: index in the stream is used instead.
    """
    
    curr_min = (None, None, None)           # (idx, argument, value) of the 1st minimum value seen so far
    curr_max = (None, None, None)           # (idx, argument, value) of the 1st maximum value seen so far
    
    count = 0                               # no. of objects seen so far
    
    def __init__(self, values = None):
        if values is not None:
            for v in values: self.add(v)
    
    def reset(self):
        self.curr_min = self.curr_max = (None, None, None)
        self.count = 0
    
    def add(self, value, arg = None):
        cmin = self.curr_min[2]
        if cmin is None or value < cmin:
            self.curr_min = (self.count, arg, value)

        cmax = self.curr_max[2]
        if cmax is None or value > cmax:
            self.curr_max = (self.count, arg, value)
    
        self.count += 1
        
    def min(self): return self.curr_min[2]
    def max(self): return self.curr_max[2]
        
    def argmin(self):
        "If corresponding argument was missing (None), index of the minimum value is returned instead, like in idxmin()."
        return self.curr_min[1] if self.curr_min[1] is not None else self.curr_min[0]
    
    def argmax(self):
        "If corresponding argument was missing (None), index of the maximum value is returned instead, like in idxmax()."
        return self.curr_max[1] if self.curr_max[1] is not None else self.curr_max[0]
    
    def idxmin(self):
        "0-based index in the stream of the 1st minimum value seen so far."
        return self.curr_min[0]
    
    def idxmax(self):
        "0-based index in the stream of the 1st maximum value seen so far."
        return self.curr_max[0]
    

class Accumulator(object):
    """Weighted sequence of values (scalars or numpy arrays) where new items are added incrementally
       and weighted mean() of all the items added so far can be computed at any point.
       Only the current sum of items is remembered; actual items are not.
    """
    total  = 0              # total values*weights
    weight = 0              # total weights
    #count = 0              # total no. of items, excluding the initial value
    dtype  = float
        
    def __init__(self, init_value = None, init_weight = None, dtype = None):
        if dtype: self.dtype = dtype
        if init_value is not None and init_weight is not None:
            self.add(init_value, init_weight)
        # self.count = 0
        
    def add(self, value, weight = 1):
        self.total += value * weight
        self.weight += weight
        # self.count += repeat
    
    def mean(self):
        "Weighted mean of all the items added so far."
        return self.total / self.dtype(self.weight)
    

class Accumulator2D(object):
    """A 2D+ numpy array, typically a large one, built incrementally
    from multiple (smaller) 2D patches, possibly overlapping.
    Each patch comes with a non-negative weight or an array of weights (of the same shape as the patch),
    which can be interpreted as confidence in corresponding patch values.
    The resulting array is computed as a weighted average of all accumulated patches.
    """
    
    def __init__(self, X, weight = 0.001, dtype = None):
        """
        'weight' of the initial fullsize patch X should be strictly positive
        to avoid "division by zero" errors for the elements where no patch was provided.
        """
        assert X.ndim >= 2
        self.total = X.copy().astype(dtype) * weight
        # self.count = np.ones_like(self.total) * weight
        self.count = np.zeros_like(self.total)
        self.count[:] = weight
    
    def add(self, x, y, patch, weight = 1.0):
        "'weight' can be a scalar or an array of weights of the same shape as 'patch'"
        h, w = patch.shape[:2]
        self.total[y : y + h, x : x + w, ...] += patch * weight
        self.count[y : y + h, x : x + w, ...] += weight
    
    def mean(self):
        return self.total / self.count
    
    get = mean
    
    def __getitem__(self, key):
        "Efficient sliced read access to the current value of the accumulated array."
        return self.total[key] / self.count[key]
    

class Stack(object):
    """
    An automatically growing numpy vector or array.
    An accumulator that collects a sequence of scalar values or numpy arrays of a predefined shape, passed one be one through add() method;
    stacks them up along the new 1st axis of a "greater" resizable array; and upon get() returns the accumulated
    greater array as a regular numpy vector or array. All data values are cast onto a predefined `dtype` (float by default).
    
    >>> stack = Stack()
    >>> stack.append(-5)
    >>> stack.append(3.1)
    >>> stack.get()
    array([-5. ,  3.1])
    
    >>> stack = Stack((3,), maxsize = 200)
    >>> stack.append([1,2,3])
    >>> stack.append([5,6,7])
    >>> stack.get()
    array([[1., 2., 3.],
           [5., 6., 7.]])

    >>> for _ in range(100): stack.append([10,10,10])
    >>> stack.get().sum()
    3024.0
    """
    
    GROWTH_RATE = 1.5       # `data` array can be at most 50% larger than the actual data stored in it
    
    data = None             # the preallocated greater array; new items are added along the 1st dimension
    size = None             # the current no. of items in `data`
    maxsize = None          # maximum `size` allowed in this Stack object

    def __init__(self, shape = (), dtype = float, like = None, init = None, size = None, maxsize = None):
        if init is not None:
            self.data = init.copy()
            self.size = init.shape[0]
            return
            
        if like is not None:
            shape = like.shape
            dtype = like.dtype
        
        initsize = size or 10
        if maxsize: initsize = min(initsize, maxsize)

        assert shape is not None
        if isnumber(shape): shape = (shape,)
        
        self.data = np.zeros((initsize,) + shape, dtype)
        self.size = 0
        self.maxsize = maxsize
        
    def append(self, item):
        
        # do we have empty space in `data` where to insert another item? if not, perform resizing
        if self.size >= self.data.shape[0]:
            assert self.size == self.data.shape[0]
            self._resize()
        
        self.data[self.size, ...] = item
        self.size += 1

    def append_all(self, items):

        extend = len(items)
        if not extend: return
        if self.size + extend > self.data.shape[0]:
            self._resize(extend)

        self.data[self.size : self.size + extend, ...] = items
        self.size += extend
        
    def _resize(self, extend = 1):
        
        data, size = self.data, self.size
        requested  = size + extend
        new_size   = max(requested, int(math.ceil(size * self.GROWTH_RATE)))
        
        if self.maxsize:
            new_size = min(new_size, self.maxsize)
            if new_size < requested: raise Exception("Can't resize the Stack beyond its maximum size (%s)" % self.maxsize)
        assert new_size >= requested >= data.shape[0]
        
        extended = (new_size,) + data.shape[1:]
        newdata  = self.data.resize(extended, refcheck = False)     # resize() may work in place (with a standard np.array) or return a new array (with a derived array type)
        if newdata is not None:
            self.data = newdata
        # self.data = np.resize(self.data, extended)

        # new_data = np.zeros(extended, data.dtype)
        # new_data[:size,...] = data
        # self.data = new_data
        
    def get(self):
        
        return self.data[:self.size,...]
    
    def __getitem__(self, pos):
        
        return self.data[:self.size,...][pos]
        # if pos > self.size: raise IndexError("Index %s is out of bounds of the Stack (size %s)" % (pos, self.size))
        # return self.data[pos,...]
        
    def __setitem__(self, pos, value):
    
        self.data[:self.size,...][pos] = value
        # if pos > self.size: raise IndexError("Index %s is out of bounds of the Stack (size %s)" % (pos, self.size))
        # self.data[pos,...] = value
        
    def __len__(self):
        
        return self.size
    

#####################################################################################################################################################
#####
#####  NAMEDARRAY
#####

class namedarray(np.ndarray):
    """
    namedarray is a Numpy's ndarray that keeps its column names internally, similar to Pandas,
    but works around performance issues of Pandas being even 20x slower than Numpy in mathematical operations.
    namedarray class provides fast computation without compromising readability of the code.
    
    namedarray performs approx. 7x faster than Pandas' DataFrame, thanks to the robust underlying
    implementation of matrices and vectors as provided by Numpy.
    As such, namedarray can be used as a replacement for Numpy's 2D arrays in the applications
    where column names must be stored and tracked across mathematical operations, so as to guarantee
    high code readability and maintainability.
    namedarray is to Numpy's ndarray like collections.namedtuple is to tuple.
    
    Being based on Numpy's ndarray, namedarray is restricted to a single data type (dtype)
    for all columns of the array, unlike Pandas' DataFrames where this restriction is not present.
    If your application requires the use of different dtypes for columns, or some other advanced functionality
    of Pandas (e.g., the groupby() method), Pandas' DataFrame may still be a better choice than namedarray.
    
    A namedarray has either 1 or 2 dimensions. A 1-dimensional namedarray is interpreted as a row vector.
    namedarray can be used with all Numpy's operators and methods, similar to a standard array (ndarray),
    but additionally it keeps and transfers to result arrays the list of column names,
    and provides some syntactic extensions to be more suitable as a replacement for Pandas' DataFrames.
    Technically, a namedarray is always a *view* on an underlying standard Numpy's ndarray (self.base);
    a namedarray never keeps array data by itself.
    
    New syntax provided by namedarray:
    - arr = namedarray(x)
    - arr[:,'COL1'] or arr['COL1']         -- columns can be accessed by name
    - arr.COL1                             -- columns can be accessed like a property
    New methods for (partial) compatibility with Pandas:
    - arr.empty (property)
    - arr.iloc (property)        -- returns self, so the effect is the same as using "arr" alone
    - arr.median()
    - arr.itertuples(), requires index=False

    TESTS:
    
    >>> A = namedarray([[1,2,3],[-5.0,0.1,-10]], names = ['x','y','z'])
    >>> A
    namedarray([[  1. ,   2. ,   3. ],
                [ -5. ,   0.1, -10. ]])
    >>> A.x
    namedarray([ 1., -5.])
    >>> A[:,'x'], A['x']
    (namedarray([ 1., -5.]), namedarray([ 1., -5.]))
    >>> A[1:].x, A[1].x
    (namedarray([-5.]), -5.0)
    >>> A.x[:]
    namedarray([ 1., -5.])
    >>> A.y[0], A.y[:-1], A.z[-1]
    (2.0, namedarray([2.]), -10.0)
    >>> A.x[1:][:1]
    namedarray([-5.])
    >>> A[1,:2].y, A[1,'y']
    (0.1, 0.1)
    
    >>> A + A               # TODO: checking if names are compatible in both arrays (?)
    namedarray([[  2. ,   4. ,   6. ],
                [-10. ,   0.2, -20. ]])
    >>> (A+A).names.tolist()
    ['x', 'y', 'z']
    >>> B = A**2 / A * 2
    >>> B
    namedarray([[  2. ,   4. ,   6. ],
                [-10. ,   0.2, -20. ]])
    >>> B.names.tolist()
    ['x', 'y', 'z']
    >>> abs(A)
    namedarray([[ 1. ,  2. ,  3. ],
                [ 5. ,  0.1, 10. ]])
    >>> abs(A).names.tolist()
    ['x', 'y', 'z']
    >>> A.min(), A.max(), A.x.min(), A.y.max(), A.z.std()
    (-10.0, 3.0, -5.0, 2.0, 6.5)
    >>> A.median(), A.x.median()
    (0.55, -2.0)
    >>> A.extended_with('a').a
    namedarray([0., 0.])
    >>> A.extended_with(a = [8,9], b = A.x * 2)
    namedarray([[  1. ,   2. ,   3. ,   8. ,   2. ],
                [ -5. ,   0.1, -10. ,   9. , -10. ]])
    >>> A.z = A.z * 3
    >>> A.z
    namedarray([  9., -30.])
    """
    pandas_compatible = True
    
    names   = None         # column names, as a numpy array of strings to allow indexing by lists
    columns = None         # dict of names and their column indices in the underlying numpy array: {name: column}
    
    def __new__(cls, input_array, names = None, pandas_compatible = True):
        if input_array is NotImplemented: raise Exception("input_array is NotImplemented")
        obj = np.asarray(input_array).view(cls)
        assert not isinstance(obj.base, namedarray)
        # if isinstance(input_array, namedarray):
        #     obj.init_like(input_array)
        # else:
        obj.pandas_compatible = pandas_compatible
        if names is not None:
            if not isinstance(names, list): raise Exception("'names' must be a list")
            if not all(isinstance(n, str) and n for n in names): raise Exception("all names must be non-empty strings")
            if not obj.ndim in (1, 2): raise Exception("namedarray must have 1 or 2 dimensions")
            obj._set_names(names)
        return obj
    
    def init_like(self, other, only_params = False):
        self.pandas_compatible = other.pandas_compatible
        if not only_params:
            self._set_names(other.names, other.columns)
        return self
    
    def _set_names(self, names, columns = None):
        if names is None: return
        self.names = np.array(names)
        self.columns = {name: column for column, name in enumerate(names)} if columns is None else columns
        if len(names) != len(self.columns): raise Exception("names of columns are not unique")
        if len(names) != self.shape[-1]: raise Exception("the no. of names is different than the no. of columns")
        
    @staticmethod
    def from_pandas(frame):
        return namedarray(frame.values, names = frame.columns.to_list())
        
    def asarray(self):
        assert not isinstance(self.base, namedarray)
        return self.base
    
    def copy(self, order = 'K'):
        # dup = np.array(self, order = order, dtype = self.dtype)
        assert not isinstance(self.base, namedarray)
        dup = self.base.copy()
        return namedarray(dup, names = list(self.names), pandas_compatible = self.pandas_compatible)
        
    def resize(self, new_shape, refcheck = True):
        """
        This returns a COPY of the array, unlike the base class resize()
        which works in place and returns None. `refcheck` is ignored.
        Calling a base class resize() on a namedarray instance raises a ValueError:
        "cannot resize this array: it does not own its data".
        """
        new = np.resize(self, new_shape)        # this does NOT preserve the array type (namedarray)
        return namedarray(new, names = list(self.names), pandas_compatible = self.pandas_compatible)

    def __array_finalize__(self, src_array):
        """
        For details on implementing __array_finalize__() see:
        https://numpy.org/doc/stable/user/basics.subclassing.html
        """
        if not isinstance(src_array, namedarray): return
        # if hasattr(self, 'names'): return
        # if getattr(self, 'names', None) is not None: return
        
        # copy config parameters
        self.pandas_compatible = src_array.pandas_compatible
        
        # only propagate names/columns when the no. of columns hasn't changed
        # (warning: this does NOT mean that the meaning of columns hasn't changed either)
        if self.shape[-1] == src_array.shape[-1]:
            names = getattr(src_array, 'names', None)
            columns = getattr(src_array, 'columns', None)
            self._set_names(names, columns)

    def __array_ufunc__(self, ufunc, method, *inputs, out = None, **kwargs):
        # print(f'in __array_ufunc__{ufunc, method, *inputs, kwargs}')
        
        # convert input/output namedarrays to nd.array
        args = tuple(x.asarray() if isinstance(x, namedarray) else x for x in inputs)
        if out:
            out = tuple(x.asarray() if isinstance(x, namedarray) else x for x in out)
            kwargs['out'] = out
            
        results = super(namedarray, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        # print('results:', type(results), results)
        
        if results is NotImplemented: return results
        
        # if ufunc.nout == 1:
        #     results = (results,)
        # if isinstance(results[0], np.ndarray):
        
        first = results[0] if ufunc.nout > 1 else results
        
        # convert the result back to namedarray; this should work for multi-output results, as well,
        # in this case the 1st array is converted back
        if isinstance(first, np.ndarray):
        
            if not isinstance(first, namedarray):
                first = namedarray(first)
            first.init_like(self)
            
            results = first if ufunc.nout == 1 else (first,) + results[1:]

        return results
    

    def __getattr__(self, attr):
        if self.columns and attr in self.columns:
            column = self.columns[attr]
            return self[column] if self.ndim <= 1 else self[...,column]
        raise AttributeError(attr)
    
    def __getitem__(self, key, _full_slice = slice(None)):
        # print('key:', keys, type(keys))
        
        key, column = self._decode_key(key)
        ret = self.base.__getitem__(key) #if self.base is not None else super(namedarray, self).__getitem__(key)
        # ret = super(namedarray, self).__getitem__(key)
        
        if not isinstance(ret, np.ndarray):
            return ret
        
        if not isinstance(ret, namedarray):
            ret = namedarray(ret).init_like(self, only_params = True)
            
        # column dimension gets reduced, so the result is a vertical vector? don't assign names anymore
        if isinstance(column, int):
            return ret
            # return np.array(ret)
        
        if self.names is not None:                          # self.names are None for a vertical vector
            if isinstance(column, slice) and column == _full_slice:
                # full slice of the column dimension? names stay the same, no need for recalculation
                ret._set_names(self.names, self.columns)
            else:
                ret._set_names(self.names[column])
                
        return ret

    def __setitem__(self, key, value):
        """"""
        key, column = self._decode_key(key)
        super(namedarray, self).__setitem__(key, value)

    def _decode_key(self, key, _full_slice = slice(None)):
        """
        Normalization and decoding of index key for __getitem__ and __setitem__, with conversion
        of column names (strings) to numeric indexes.
        # :param create: if True and a given column name is missing in `self`, it gets created
        #                and is filled out with zeros through numpy's zeros_like();
        #                WARNING: this operation involves full array copy!
        """
        ndim = self.ndim
        if self.pandas_compatible and isinstance(key, str):
            # if create and key not in self.columns: self._add_column(key)
            key = (_full_slice,) * (ndim - 1) + (self.columns[key],)
            
        if not isinstance(key, tuple): key = (key,)
        if len(key) < ndim:
            key = key + (_full_slice,) * (ndim - len(key))
            
        # column index is either of: an integer, slice, ellipsis (...), newaxis, array, string, sequence of strings
        col_dim = ndim - 1
        column  = key[col_dim]

        # convert column name(s) in `key` to numeric index(es)
        if isinstance(column, str):
            column = self.columns[column]
            key = key[:col_dim] + (column,) + key[col_dim+1:]
            
        return key, column
    
    def assign(self, **columns):
        """
        Assign new and/or existing columns to a namedarray, similar to DataFrame.assign() in Pandas.
        Creates and returns a NEW namedarray. The original array remains unchanged.
        """
        create = [col for col in columns if col not in self.columns]
        new = self.extended_with(create)
        for name, values in columns.items():
            new[...,name] = values
        return new
    
    def extended_with(self, *names, **columns):
        """
        Add new columns and return as a NEW namedarray. The current array (self) remains unchanged.
        New columns as given in `names` are filled with zeros.
        For `columns`, the columns are assigned the values of corresponding `columns` arguments.
        """
        if len(names) == 1: names = names[0]
        if isinstance(names, str): names = names.split()

        names = list(self.names) + list(names) + list(columns.keys())
        cols  = self.shape[-1]
        shape = self.shape[:-1] + (len(names), )
        new   = namedarray(np.zeros(shape, dtype = self.dtype), names = names, pandas_compatible = self.pandas_compatible)
        new[...,:cols] = self
        for name, values in columns.items():
            new[...,name] = values
        return new
        
        # if self.names is None: raise Exception(f"cannot add columns to a namedarray whose names=None")
        # cols  = self.shape[-1]
        # shape = self.shape[:-1] + (cols + 1, )
        # new   = np.zeros(shape, dtype = self.dtype)
        # new[...,:cols] = np.array(self)
        # self.base = new
        # self._set_names(list(self.names) + [name])
    
    ###  Extra properties and methods, for partial compatibility with Pandas  ###
    
    @property
    def empty(self):
        """For compatibility with Pandas."""
        return self.size == 0
        
    @property
    def iloc(self):
        """For compatibility with Pandas."""
        return self
        
    def median(self, *a, **kw):
        return np.median(self.asarray(), *a, **kw)
        
    def itertuples(self, index = True, name = 'Pandas', strict = False):
        assert index == False
        assert self.ndim == 2
        if strict:
            assert self.names is not None
            t = namedtuple(name, self.names)
            for row in self:
                yield t(*row)
        else:
            for row in self:
                yield row

    
#####################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

