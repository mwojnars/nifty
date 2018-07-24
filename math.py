'''
Statistical and mathematical routines. Built on top of 'numpy'.

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import random, bisect, json, copy
import numpy as np
import numpy.linalg as linalg
from numpy import sum, mean, zeros, sqrt, pi, exp, isnan, isinf, arctan

from .util import isnumber, isdict, getattrs


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
    
    
    rand = None                         # Random generator to use in random() if `rand` argument is None
    rand_default = random.Random()      # fallback Random instance to use in random() if both `rand` argument and self.rand are None
    
    def __init__(self, **common):
        self.set_rand(**common)

    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        if seed is not None:
            rand = random.Random(seed)
        if rand is not None:
            self.rand = rand
        # print self, rand

    def _set_rand_recursive(self, items, overwrite):
        "To be used by subclasses that contain nested Distribution objects."
        if not (overwrite or self.rand): return
        for item in items:
            if isinstance(item, Distribution) and (overwrite or item.rand is None):
                item.set_rand(self.rand, recursive = True, overwrite = overwrite)
        
    def get_rand(self, rand = None):
        return rand or self.rand or self.rand_default

    def random(self, rand = None):
        "Override in subclasses to implement random selection from a probability distribution implemented by a subclass."
        return self.get_rand(rand).random()
    
    # Distribution instances are callable, which provides a shorthand for random():
    #   rand_distribution() is eq. to rand_distribution.random()
    #
    def __call__(self, *args, **kwargs):
        return self.random(*args, **kwargs)
    
    def copy(self):
        "Deep copy of the random distribution represented by self."
        return copy.deepcopy(self)
    
    
class Fixed(Distribution):
    "Fixed value (deterministic, no randomness)."
    
    def __init__(self, value, **common):
        super(Fixed, self).__init__(**common)
        self.value = value

    def random(self, rand = None):
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

    def random(self, rand = None):
        rand = self.get_rand(rand)
        if self.stop in (None, self.start): return self.start
        val = rand.uniform(self.start, self.stop)
        if self.cast:
            return self.cast(val)
        return val
    
class Choice(Distribution):
    """Discrete probability distribution over a fixed set of possible outcomes (choices).
       In random(), if a chosen value is an instance of Distribution, a subsequent choice from this distribution is performed,
       which allows nesting of distributions and building composite ones.
    """
    
    choices = None      # list of possible outcomes (values) to choose from
    is_dist = None      # list of flags: is_dist[i]==True iff choices[i] is a probability distribution itself (an instance of Distribution)
    
    def __init__(self, choices, **common):
        """`choices` is either a sequence (then uniform distribution is assumed), or a dictionary of {value: probability} pairs.
           The sum of `probability` values must be positive, but not necessarily 1.0 - probabilities are normalized by default to unit sum.
        """
        super(Choice, self).__init__(**common)
        if not choices: raise Exception('Choice.__init__: the list/dict of choices must be non-empty')
        
        self.choices = sorted(list(choices))                                # sorting is necessary to ensure repeatable outcomes under the same random seed, in case `choices` was a standard (unordered) dict
        self.is_dist = [isinstance(v, Distribution) for v in self.choices]  # this is pre-computed to speed up execution of random()
        
        if isdict(choices):
            self.probs = np.array([choices[v] for v in self.choices], dtype = float)
            self.probs /= float(self.probs.sum())                           # normalize probabilities to unit sum
        else:
            self.probs = None
        
    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        
        super(Choice, self).set_rand(rand, seed, recursive, overwrite)
        if recursive:
            self._set_rand_recursive(self.choices, overwrite)
        
    def random(self, rand = None):
        rand = self.get_rand(rand)
        seed = rand.randrange(4294967296)                               # 4294967296 == 2**32 == 1 + the maximum seed for RandomState
        np_rand = np.random.RandomState(seed)                           # use numpy's random to be able to choose items with non-uniform distribution
        
        choice = np_rand.choice(len(self.choices), p = self.probs)      # choose an index into self.choices[]
        val = self.choices[choice]
        
        # print self, 'random() choices:', self.choices, '  is_dist:', self.is_dist
        # if isinstance(val, Distribution):
        if self.is_dist[choice]:
            val = val.random(rand)
        
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
        
        # wrap up interval pairs in Interval class, so that the Choice class knows these pairs should be treated as sub-distributions
        if isdict(intervals):
            choices = {Interval(start, stop, cast = cast): prob for (start,stop), prob in intervals.iteritems()}
        else:
            choices = [Interval(start, stop, cast = cast) for (start,stop) in intervals]
            
        super(Intervals, self).__init__(choices, **common)
        
    
class RandomInstance(Distribution):
    """Probability distribution over a space of instances of a given class.
       Attribute values of a generated instance are chosen at random from probability distributions defined in the subclass,
       separately for each attribute.
    """
    
    class_type = None           # the class whose instances are going to be created and returned in random()
    class_attr = None           # list of attributes that shall be initialized when a random <class_type> object is created;
                                # in the subclass, these attributes should contain instances of Distribution that define probability distr. to choose values from
    
    def __init__(self, **common):
        
        # initialize `class_attr` list of attributes
        if self.class_attr is None:
            self.class_attr = [attr for attr in dir(self.__class__) if not attr.startswith('__')]
            
        # copy nested sub-Distributions from class-level attributes to instance attributes, to allow their customization
        # (e.g., setting a random seed) without affecting the shared class-level object
        for attr in dir(self.__class__):
            if attr in self.__dict__: continue              # already has a value at instance level? skip
            item = getattr(self, attr, None)
            if isinstance(item, Distribution):
                setattr(self, attr, item.copy())

        super(RandomInstance, self).__init__(**common)
        
    def set_rand(self, rand = None, seed = None, recursive = True, overwrite = False):
        
        super(RandomInstance, self).set_rand(rand, seed, recursive, overwrite)
        
        # walk through attributes of `self` and for every instance of Distribution initialize its `rand` if missing, or if `overwrite`=True
        if recursive:
            items = getattrs(self, self.class_attr).values()
            self._set_rand_recursive(items, overwrite)

        
    def random(self, rand = None):
        """Walk through attributes of `self` and for each one being an instance of Distribution (probability distribution)
           draw a random value and assign to this attribute.
        """
        # print 'RandomInstance class & attributes:', self.class_type, self.class_attr
        assert self.class_type is not None
        obj = self.class_type()
        
        for attr in self.class_attr:
            distr = getattr(self, attr, None)
            if isinstance(distr, Distribution):
                val = distr.random(rand)
                setattr(obj, attr, val)
        
        self.postprocess(obj, self.get_rand(rand))
        return obj
    
    def postprocess(self, obj, rand):
        """Override in sublasses to perform additional post-processing of an object generated by random().
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
    count  = 0              # total no. of items, excluding the initial value
    dtype  = float
        
    def __init__(self, init_value = None, init_weight = None, dtype = None):
        if dtype: self.dtype = dtype
        if init_value is not None and init_weight is not None:
            self.add(init_value, init_weight)
        self.count = 0
        
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
    

