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
import numpy as np
from numpy import sum, mean, zeros, sqrt, pi, exp, isnan, isinf, arctan
import random, bisect, json

from nifty.util import isnumber


########################################################################################################################
###
###   UTILITIES. Numpy extensions
###

ipi = 1./pi         # inverted PI

def isarray(x):    return isinstance(x, numpy.ndarray)


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


########################################################################################################################
###
###   RANDOM NUMBERS
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


########################################################################################################################
###
###   SCALAR & point-wise mathematical transformations
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


### Sigmoidal functions for use in predictive models and data processing

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
    y = float(x - p0) / (p1 - p0)
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
    
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    if isnumber(x):
        Y = Y.flatten()[0]
        
    return Y

def mexican(x, std = 1.0, mean = 0.0):
    '''
    Values of mexican hat function, calculated in point x (can be an ndarray).
    See: http://en.wikipedia.org/wiki/Mexican_hat_wavelet
    '''
    if mean != 0.0 or std != 1.0:
        x = (x - mean) / std
    x2 = x**2
    f1 = 2.0 / ((3.0*std)**0.5 * pi**0.25)
    f2 = 1.0 - x2
    f3 = exp(-x2 / 2)
    return f1 * f2 * f3
    
def softmax(scores, slope = None, eps = 1e-10):
    "softmax function: turns a vector of real-valued scores into unit-sum probabilities by applying exp() and normalization."
    scores = scores - np.max(scores)            # shift values to avoid overflow in exp()
    if slope is not None: scores *= slope
    exps = exp(scores) #+ EPS                   # +EPS to avoid 0.0 probabilities
    Z = np.sum(exps)
    #print "", Z, list(exps.flat)
    assert not isnan(Z) and not isinf(Z)
    return exps / (Z + eps)                     # 1-d vector


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
###   VECTOR-wise operations
###

def normv2(x, axis = -1):
    "Squared euclidean norm of vectors contained in matrix 'x', along 'axis'. Last axis by default."
    return np.sum(x*x, axis)
def normv(x, axis = -1):
    "Euclidean norm of vectors contained in matrix 'x', along 'axis'. Last axis by default."
    return np.sqrt(normv2(x,axis))

