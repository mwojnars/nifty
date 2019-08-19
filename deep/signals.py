'''
High-level representation of output/target signals for Deep Learning models.

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import

import numpy as np
from collections import OrderedDict


# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages;
# only when executed as a standalone file, for unit tests, do an absolute import
if __name__ != "__main__":
    from ..util import list2dict
    from ..math import isarray
else:
    from nifty.util import list2dict
    from nifty.math import isarray


#####################################################################################################################################################
#####
#####  GENERAL UTILITIES
#####

def f_softmax(x):
    "Non-tensor softmax working along the last dimension of x, with x being an array of any dimensions."
    e = np.exp(x - np.max(x, axis = -1)[...,None])
    return e / e.sum(axis = -1)[...,None]


#####################################################################################################################################################
#####
#####  KERAS UTILITIES
#####

try:
    from keras import backend as K
    from keras.layers import Activation, Conv2D, Concatenate
    from keras.activations import softmax
    from keras.utils.generic_utils import get_custom_objects
    _use_keras = True
except:
    _use_keras = False


if _use_keras:
    
    def keras_log1x(x):
        """Keras activation function: ln(|x|+1)*sgn(x)
           This function resembles a sigmoid: it is symmetric with respect to (0,0), negative for x<0 and positive for x>0,
           with a derivative at the origin f'(0)=1 (approximates linear function),
           but unlike `sigmoid` or `relu` is unbounded on both sides and thus can approximate functions
           with values in <-inf,+inf>, not just in <-1,1>, <0,1> or <0,+inf>.
        """
        return K.log(K.abs(x) + 1) * K.sign(x)
    
    def keras_exp1x(x):
        """Keras activation function: exp(|x|)-1)*sgn(x) -- the inverse of log1x()
           This function is symmetric with respect to (0,0), negative for x<0 and positive for x>0,
           but unlike sigmoidal functions it gets steeper when further away from 0.
        """
        return (K.exp(K.abs(x)) - 1) * K.sign(x)
    
    def keras_multi_softmax(x, group_lengths = None):
        """
        Softmax applied multiple times, within disjoint subranges of output indices corresponding to separate subgroups of signals.
        
        >>> x = np.random.rand(10, 20, 50)
        >>> y = np.dstack([f_softmax(x[...,0:15]), f_softmax(x[...,15:50])])
        >>> ky = K.eval(keras_multi_softmax(K.variable(x), [15, 35]))
        >>> np.abs(y - ky).max() < 0.00001
        True
        """
        if not group_lengths: return softmax(x)
        
        # ngroups = len(group_lengths)
        # shape   = tuple(x.shape[:-1] + (ngroups,))
        # y = K.zeros(shape, dtype = dtype, name = 'multi_softmax')

        depth = x.shape[-1]
        total = sum(group_lengths)
        
        if depth != total: raise Exception("multi_softmax: depth of `x` (%s) differs from the declared total length of subgroups (%s)" % (depth, total))
        
        out = []        # list of output tensors, one for each group
        
        start = 0
        for glen in group_lengths:
            stop = start + glen
            y = softmax(x[..., start:stop])
            out.append(y)
            start = stop
            
        return K.concatenate(out)

            
    activ_log1x = Activation(keras_log1x)
    activ_exp1x = Activation(keras_exp1x)
    
    get_custom_objects().update({'log1x': activ_log1x})
    get_custom_objects().update({'exp1x': activ_exp1x})


    def MSE_weighted(weights, normalize = True):
        "MSE with class weighing along the last axis (channels). Can handle multi-dimensional tensors unlike standard Keras MSE."
        
        assert all(w >= 0 for w in weights)
        weights = np.array(weights)[np.newaxis, :]
        if normalize: weights /= weights.sum()
        weights = K.constant(weights)                   # convert to a tensor
    
        def _mse(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true) * weights, axis = -1)
        
        return _mse


#####################################################################################################################################################
#####
#####  SIGNAL base class
#####

class Signal(object):
    
    name = None                 # name of the signal, for debugging and reporting purposes


class Signal2D(Signal):
    
    # all dimensions and coordinates are stored in geometrical (x,y) order, not (y,x) as in numpy images!
    imagesize = None            # (width, height) of the main part of the signal array
    bleedsize = None            # (dx, dy) of a bleed added to `imagesize` when creating the working signal array; bleed is truncated when returning the final array in get()
    totalsize = None
    
    depth     = None            # 3rd dimension of `value`; if None, `value` has 2 dimensions
    channels  = None            # optional list of names/labels of all channels
    
    value = None                # array of activation values of this signal, stored in (x,y) order (!); if merge_type='avg', `value` stores accumulated (weight*value) sums
    dtype = np.float16          # dtype to use when initializing `signal` array
    
    weight       = None         # if merge_type='avg', an array of total weights of values accumulated (summed up) in `value`
    weight_init  = 0.001        # weight of the initial 0.0 value
    
    merge_type   = None         # 'min', 'max', 'avg', 'sum'
    
    keras_activation = 'linear' # default Keras activation to be used in get_keras_layer() when creating a layer instance
    keras_loss       = 'mse'    # default Keras loss funcion to be returned by get_keras_loss()
    
    def __init__(self, imagesize = None, bleedsize = (0, 0), name = None, dtype = None, value = None, default = None):
        """
        If initial `value` is given, it must be in (y,x) order (!) and is transposed before assigning to self.value.
        """
        
        self.name = name or self.name or self.__class__.__name__
        self.imagesize = np.array(value.shape[:2][::-1] if value is not None else imagesize)
        self.bleedsize = np.array(bleedsize)
        self.totalsize = self.imagesize + 2 * self.bleedsize
        self.dtype = dtype or self.dtype
        
        self.create_array(value, default)
        
    def create_array(self, value = None, default = None, value_weight = 1):
        """Create and initialize this signal's numpy array, where all marks will be stored. Override in subclasses.
           By default, an array for storing scalar activation values is created. Override in subclasses to change this.
        """
        if not self.depth and value is not None and len(value.shape) >= 3:
            self.depth = value.shape[2]

        totalshape = tuple(self.totalsize) if self.depth is None else tuple(self.totalsize) + (self.depth,)
        self.value = np.zeros(totalshape, dtype = self.dtype) + (default or 0)
        self.weight = np.zeros(self.totalsize) + self.weight_init
        
        # initial `value` provided? assign it to self.value at a proper position (with or without the bleed)
        if value is not None:
            assert len(value.shape) in (2, 3)
            assert (value.shape[2:] or (None,)) == (self.depth,)
            
            value = self._transpose(value)
            value_shape = value.shape[:2]
            
            if value_shape == totalshape:
                self.value = value
                self.weight += value_weight
            else:
                assert value_shape == tuple(self.imagesize)
                x, y   = self.bleedsize
                dx, dy = self.imagesize
                self.value[x:x+dx, y:y+dy] = value
                self.weight[x:x+dx, y:y+dy] += value_weight

        
    def get(self, dtype = None, raw = False, subsignals = False):
        "Retrieve an array of activation values stored in this signal."
        
        if self.value is None: return None
        value = self.value

        # compute the average when needed
        if self.merge_type == 'avg':
            # if not self.depth: value = value / self.weight
            # else: value = value / self.weight[:,:,None]
            value = value / (self.weight if not self.depth else self.weight[:,:,None])
        
        if raw: return value
        
        x, y   = self.bleedsize
        dx, dy = self.imagesize
        # print 'x, y, dx, dy:', x, y, dx, dy

        # drop the bleed
        value = value[x:x+dx, y:y+dy]

        # reshape from (x,y) to (y,x) ordering of dimensions
        value = self._transpose(value)
        
        # axes = range(value.ndim)
        # axes[:2] = [1,0]                            # (0,1,2,3...) changed to (1,0,2,3...), to indicate permutation of the first 2 axes only
        # value = value.transpose(axes)
        
        if dtype is not None:
            value = value.astype(dtype)
        
        # if subsignals:
        #     return self.split_subsignals(value)
        
        return value
    
    @staticmethod
    def _transpose(X):
        "Reshape X array from (x,y) to (y,x) ordering of dimensions, or the other way round. X can have any number of dimensions >= 2."
        axes = range(X.ndim)
        axes[:2] = [1,0]                            # (0,1,2,3...) changed to (1,0,2,3...) to indicate permutation of the first 2 dimensions only
        return X.transpose(axes)

    def mark(self, center, shape, value = None):
        """Write a value to the signal array on a given position. The value can be transformed through an
           encoding function before being written to the array.
        """
        p1, p2 = self._bounding_rect(center, shape)
        activ  = self.encode(value, p1, p2)
        self._merge_rect(p1, p2, activ)
        
    def encode(self, value, p1, p2):
        raise NotImplementedError

    def merge(self, signal, pos):
        "Merge another signal of the same type onto self at position `pos` (global coordinates); include the bleed during merge."
        
        # translate `pos` so as to include the bleed of `signal`
        p1 = np.array(pos) - signal.bleedsize
        p2 = p1 + signal.totalsize
        
        # convert p1, p2, so as to include the bleed of `self`
        p1 = self._signal_coordinates(p1)
        p2 = self._signal_coordinates(p2)
        
        # print "signal.get()...",
        raw_value = signal.get(raw = True)
        # print "done"
        
        # print "Signal2D.merge()...",
        self._merge_rect(p1, p2, raw_value, signal.weight)
        # print "done"
    
    def _signal_coordinates(self, point):
        "Convert a point in data space to coordinates of self.value: with bleed length added; and to numpy array."
        return np.array(point) + self.bleedsize
    
    def _signal_dimensions(self, vector):
        "Convert a relative dimension vector from data space to self.value space; for now this means just converting to numpy array."
        return np.array(vector)
    
    def _bounding_rect(self, center, shape):
        "Top-left and bottom-right corners of a bounding rectangle, converted to self.value space."
        
        center = self._signal_coordinates(center)
        shape = self._signal_dimensions(shape).astype(int)
        topleft = (center - shape/2).astype(int)
        
        # top-left and bottom-right corners of the bounding rectangle
        p1 = topleft
        p2 = topleft + shape
        # print 'p1, p2, shape:', p1, p2, shape

        return p1, p2
        
    def _clip(self, point):
        "If `point` lies outside self.array, adjust its coordinates and return the difference between the new point and original."
        
        p = point
        p = np.maximum(p, (0,0))
        p = np.minimum(p, self.totalsize - 1)
        diff = p - point
        
        return diff
        
    def _merge_rect(self, p1, p2, value, weight = 1.0):
        "`value` is either a scalar, or an array of the same 2D shape as <p1,p2> rectangle."
        
        # top-left and bottom-right corners of the bounding rectangle
        x1, y1 = p1
        x2, y2 = p2
        shape = p2 - p1
        
        # truncate the rectangle if it exceeds signal array boundaries
        dx1, dy1 = clip1 = self._clip(p1)
        dx2, dy2 = clip2 = self._clip(p2)
        cx2, cy2 = clip2 + shape
        # cshape = ((x2 + dx2) - (x1 + dx1), (y2 + dy2) - (y1 + dy1))
        
        slice_outer = (slice(x1 + dx1, x2 + dx2), slice(y1 + dy1, y2 + dy2), Ellipsis)
        slice_inner = (slice(dx1, cx2), slice(dy1, cy2))
        
        if isarray(value) and value.ndim == self.value.ndim:        # `value` is a patch of values, not a singleton? truncate its 2D dimensions when necessary
            assert value.shape[:2] == tuple(shape)
            value = value[slice_inner + (Ellipsis,)]
            # else:                                           # `value` is a singleton, but having a form of an array? must manually broadcast dimensions
            #     value = np.tile(value, cshape + (1,) * value.ndim)
        
        if isarray(weight):
            weight_w = weight[slice_inner]
            weight_v = weight[slice_inner + (None,) * (value.ndim-2)]
        else:
            weight_w = weight_v = weight
        
        if   self.merge_type == 'min':  merge_fun = np.minimum
        elif self.merge_type == 'max':  merge_fun = np.maximum
        elif self.merge_type == 'sum':  merge_fun = np.add
        elif self.merge_type == 'avg':
            def merge_fun(prev, new):
                self.weight[slice_outer] += weight_w
                val = prev + new * weight_v
                # print 'weight:', self.weight[slice_rect]
                # print 'val:', val.shape, '\n', val
                return val
        else:
            raise Exception("Unknown merge_type in %s: %s" % (self.name, self.merge_type))
        
        self.value[slice_outer] = merge_fun(self.value[slice_outer], value)

    # def split_subsignals(self, value):
    #     raise NotImplementedError
    
    def get_keras_layer(self, input_layer, *args, **kwargs):
        """Create a Keras layer, or an OrderedDict of named Keras layers (for a multi-output signal),
           that would produce this signal from activations of an `input_layer`.
           A layer is a tuple: (layer_object, loss_function)
        """
        assert _use_keras
        depth = self.depth or 1
        kwargs.setdefault('activation', self.keras_activation)
        kwargs.setdefault('name', self.__class__.__name__)
        layer = Conv2D(depth, *args, **kwargs)(input_layer)
        return layer
        
    def get_keras_loss(self):
        return self.keras_loss
    

#####################################################################################################################################################
#####
#####  SIGNAL subclasses
#####

class EmptySignal(Signal):
    "Mockup signal."

    # def __init__(self, name = None):
    #     self.name = name or self.name or self.__class__.__name__

    def mark(self, center, shape, value = None):
        pass    # do nothing

    def __nonzero__(self):
        return False
    

#####################################################################################################################################################

class PositionSignal(Signal2D):
    "Localizing function, typically in <0.0,1.0> or <-1.0,1.0>, whose extremums mark the position of a given object in 2D data space."
    
    activ_shape      = 'triangle'           # 'triangle', 'ellipse', 'sawtooth_x'
    keras_activation = 'linear'
    
    def encode(self, value, p1, p2):
        assert value is None
        shape = p2 - p1
        
        # position vectors along X and Y axes, with values growing from 0.0 to 1.0 (normalized)
        X = np.linspace(0, 1, shape[0])
        Y = np.linspace(0, 1, shape[1])
        
        return self.activ_fun(X[:,None], Y[None,:])

    def activ_fun(self, x, y):
        fun = self.FUNC.get(self.activ_shape)
        if fun is None: raise Exception("Unknown activ_type in %s: %s" % (self.name, self.activ_shape))
        return fun(x, y)
        # if self.activ_shape == 'triangle':  return self.fun_triangle(x, y)
        # if self.activ_shape == 'ellipse':   return self.fun_ellipse(x, y)
        
    @staticmethod
    def fun_triangle(x, y):
        "triangle function: f(x,y) = 1.0 in the center, drops linearly to 0.0 at the edges of a unit square"
        return 1 - 2 * np.maximum(np.abs(x - 0.5), np.abs(y - 0.5))

    @staticmethod
    def fun_ellipse(x, y):
        return np.maximum(0, 1 - 2 * np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2))
    
    @staticmethod
    def fun_sawtooth_x(x, y):
        "reversed sawtooth function along X: f(x,y) = 1-2*x; constant along Y dimension"
        return 1 - 2*x + y*0            # y*0 informs numpy of what Y dimension to use
    
    
    FUNC = {'triangle':     fun_triangle.__func__,
            'ellipse':      fun_ellipse.__func__,
            'sawtooth_x':   fun_sawtooth_x.__func__,
            }
    

#####################################################################################################################################################

class PropertySignal(Signal2D):
    """A function whose value encodes locally the value of a specific property of an object
       present in a given region of the data space. Often a locally constant function.
       Optionally, original values of the property can be encoded when written to the signal array,
       to obtain a representation more suitable for learning.
    """
    encoding = None         # None: no encoding; 'log': signed log(1+|x|);
                            # if activations are vectors, the default encoding functions are applied separately to each element of the vector
    
    log_shift = 0.0         # non-negative constant added to abs(value) before scaling and log() to separate non-zero values from 0.0 after encoding and boost sensitivity of the model to these values
    log_scale = 1.0         # scaling factor that divides value before log(), for better differentiation of values over the entire value range

    keras_activation = 'linear'     # in general, the set of property values is <-inf,+inf>, thus no standard non-linear activation can be used as a default
    # keras_activation = 'log1x'
    

    def get_decoded(self, validate = False):
        "Retrieve a decoded array of activations, converted back to property values."
        activ = self.get()
        return self.decode(activ, validate)
        
    def encode(self, value, p1, p2):
        if self.encoding is None:  return value
        if self.encoding == 'log': return self._logx(value)
        raise Exception("Incorrect value of encoding: %s" % repr(self.encoding))
        
    def decode(self, activ, validate = False):
        if self.encoding is None:  return activ
        if self.encoding == 'log': return self._logx_inv(activ, validate)
        raise Exception("Incorrect value of encoding: %s" % repr(self.encoding))
        
    def _logx(self, value):
        assert self.log_shift >= 0 and self.log_scale > 0
        return np.log(1 + (np.abs(value) + self.log_shift) / self.log_scale) * np.sign(value)
    
    def _logx_inv(self, activ, validate):
        """If validate=True, a pair of arrays is returned: the array of (possibly corrected after decoding) property values,
           and the array of probabilities that the property really existed (is non-zero) at a given pixel.
        """
        assert self.log_shift >= 0 and self.log_scale > 0
        
        # the decoded property values; some of them can possibly be invalid
        property = ((np.exp(np.abs(activ)) - 1) * self.log_scale - self.log_shift) * np.sign(activ)

        if not validate: return property

        # some `activ` values around 0.0 can be disallowed, if shift > 0
        min_activ = np.log(1 + self.log_shift / self.log_scale)             # the first lowest valid value above 0.0
        if min_activ == 0.0:
            property_exists = np.ones_like(activ)
        else:
            # <-min_activ,+min_activ> is the disallowed range of activation values
            property_exists = np.abs(activ / min_activ).clip(0.0, 1.0)      # probability of non-zero is calculated in addition to the decoded property value
            property[(0 < activ) & (activ < +min_activ)] = +min_activ       # fix invalid values by replacing with +/- min_activ
            property[(0 > activ) & (activ > -min_activ)] = -min_activ
        
        return property, property_exists


#####################################################################################################################################################

class VectorSignal(PropertySignal):
    ""
    offset_position = None      # if True, 2D vectors are offset by their pixel position (distance vector) relative to the center of the rectangle,
                                # like if the original vector was a link from the center to a destination point in the data space:
                                #       v = dest - center
                                # and the adjusted vector `w` was a link from another point `p` within the rectangle, to dest:
                                #       w = dest - p = (dest - center) + (center - p) = v + (center - p) = v + offset(p)
    
    def create_array(self, *args, **kwargs):
        assert self.depth
        super(VectorSignal, self).create_array(*args, **kwargs)
        
    def mark(self, center, shape, vector):
        assert np.array(vector).shape == (self.depth,)
        super(VectorSignal, self).mark(center, shape, vector)
        
    def encode(self, vector, p1, p2):
        if self.offset_position is True:
            vector = self._make_offset(vector, p1, p2)
        elif self.offset_position is not False:
            raise Exception("Unknown offset_position in %s: %s" % (self.name, self.offset_position))
            
        return super(VectorSignal, self).encode(vector, p1, p2)
    
    def _make_offset(self, vector, p1, p2):

        # convert `vector` to a 2D matrix of vectors = 3D tensor of depth `self.depth`
        shape  = p2 - p1
        tshape = tuple(shape)
        tensor = np.tile(vector, tshape + (1,))
        assert tensor.shape == tshape + (self.depth,)
        
        # subtract distance-to-center vector from each pixel
        w, h = shape
        X = np.arange(w)
        Y = np.arange(h)
        (cx, cy) = center = shape / 2               # center = shape/2 like in _bounding_rect(), hence the adjustment at the original center is (0,0)
        tensor[:,:,0] += (cx - X[:,None])
        tensor[:,:,1] += (cy - Y[None,:])
        
        return tensor

    
#####################################################################################################################################################

class OneHotSignal(Signal2D):
    """Signal2D whose pixel values have a form of class indicators encoded as (smooth) one-hot vectors, i.e.,
       vectors of <0.0,1.0> probabilities of particular classes.
    """
    merge_type = 'avg'      # averaging is most often the only reasonable merge function for vectors of probabilities, to maintain sum(probs)==1.0

    labels  = None          # list of labels for each class ID:        labels[class_id] == label
    classes = None          # dictionary of class IDs for each label:  classes[label] == class_id   -- derived from `labels`
    
    default = None          # label to fill out the signal array during initialization
    unknown = None          # label that replaces an original one when the latter is not in `labels`
    unknown_id = None       # ID of `unknown`, if unknown<>None  -- derived

    keras_activation = 'softmax'
    keras_loss       = 'mse'    #'categorical_crossentropy'
    
    def __init__(self, *args, **kwargs):
        self.labels = kwargs.pop('labels', None) or self.labels
        assert self.labels
        self.labels = list(self.labels)
        self.channels = list(self.labels)
        self.classes = list2dict(self.labels, invert = True)
        self.unknown_id = self.classes[self.unknown] if self.unknown is not None else None
        self.depth = len(self.labels)
        super(OneHotSignal, self).__init__(*args, **kwargs)
        
    def create_array(self, *args, **kwargs):
        super(OneHotSignal, self).create_array(*args, **kwargs)

        # fill out the signal array with a vector representing `default` label
        assert self.default is not None
        # onehot = self.encode(self.default)
        # self.value[:,:,None] = onehot[None,None,:]
        default = self.class_of(self.default)
        value = 1.0
        self.value[:,:,default] = (value * self.weight_init) if self.merge_type == 'avg' else value

    def encode(self, label, p1 = None, p2 = None):
        class_id = self.class_of(label, self.unknown_id)
        assert 0 <= class_id < self.depth
        onehot = np.zeros(self.depth, self.dtype)
        onehot[class_id] = 1
        return onehot
    
    def class_of(self, label, unknown_id = None):
        "Convert a label to its corresponding class ID."
        cls = self.classes.get(label, unknown_id)
        if cls is None:
            raise Exception(u"OneHotSignal.class_of(): unknown label (%s)" % repr(label))
        return cls
        
    
#####################################################################################################################################################

class MultiHotSignal(Signal2D):
    """Like OneHotSignal, but the vector is a concatenation of a fixed number of one-hot subvectors, each one encoding
       a different property having its own set of labels (classes).
       Subclasses should override value_to_labels() to convert a raw external value to a multi-label (tuple of labels)
       that would be encoded subsequently by the base class implementation.
    """
    merge_type = 'avg'      # averaging is most often the only reasonable merge function for vectors of probabilities, to maintain sum(probs)==1.0
    
    groups  = None          # list of specifications of groups to be one-hot encoded; each specification as a tuple: (group name, training weight, labels); labels must be unique within and between groups
    default = None          # value to fill out the signal array during initialization
    unknown = None          # value that replaces an original one during encoding when the latter is not recognized
    
    ngroups       = None    # [derived] no. of groups
    labels        = None    # [derived] flattened list of all labels
    positions     = None    # [derived] mapping: label -> position_of_label in multi-hot vector
    group_lengths = None    # [derived] list of lengths of consecutive groups
    
    # if group_weights or label_weights are provided, weighted MSE not `loss` is used as a loss function
    label_weights = None    # optional dict of name->weight pairs determining a relative weight of a label *within its subgroup* (1.0 if not specified for a particular label)
    group_weights = None    # [derived] list of weights determining relative importance of subgroups in the calculation of loss function

    keras_activation = 'softmax'    # this activation is applied to each subgroup separately
    # keras_activation = None       # keras_multi_softmax() is used as activation: separate softmax for each subgroup
    keras_loss       = 'mse'        #'categorical_crossentropy'


    def __init__(self, *args, **kwargs):
        self.groups = kwargs.pop('groups', None) or self.groups
        self.group_lengths = [len(labels) for _, _, labels in self.groups]
        self.group_weights = [weight if weight != None else 1 for _, weight, _ in self.groups]
        if set(self.group_weights) == {1.0}: self.group_weights = None

        self.ngroups  = len(self.groups)
        self.labels   = sum([labels for _, _, labels in self.groups], [])
        self.depth    = sum(self.group_lengths)
        self.channels = list(self.labels)
        
        assert len(set(self.labels)) == self.depth, "Labels must be unique within and between groups"

        # initialize `positions` mapping
        self.positions = {}
        for label in self.labels:
            self.positions[label] = len(self.positions)

        super(MultiHotSignal, self).__init__(*args, **kwargs)
        
        # # set `keras_activation`
        # def multi_softmax(x): return keras_multi_softmax(x, self.group_lengths)
        # self.keras_activation = multi_softmax
        
    def create_array(self, *args, **kwargs):
        super(MultiHotSignal, self).create_array(*args, **kwargs)
        
        # fill out the signal array with a vector representing `default` value
        assert self.default is not None
        
        value = 1.0
        labels = self.value_to_labels(self.default)
        assert len(labels) == len(set(labels)) == self.ngroups
        
        # in every group, turn ON the corresponding label as present in self.default
        for label in labels:
            pos = self.positions[label]
            self.value[:,:,pos] = (value * self.weight_init) if self.merge_type == 'avg' else value
        
        # prob = self.value / self.weight[:,:,None]
        # sumprob = prob.sum(2)
        # print self.name, 'create_array() init sumprob: %.5f-%.5f' % (sumprob.min(), sumprob.max())
        # print self.name, 'create_array() self.weight:  %.5f-%.5f' % (self.weight.min(), self.weight.max())
        
    def encode(self, value, p1 = None, p2 = None):
        "Convert a raw value to its vector representation."
        labels = self.value_to_labels(value)
        positions = self.labels_to_positions(labels)
        assert 0 <= min(positions) and max(positions) < self.depth
        hot = np.zeros(self.depth, self.dtype)
        hot[positions] = 1
        assert abs(hot.sum() - self.ngroups) < .0001
        return hot

    def labels_to_positions(self, labels):
        assert len(labels) == self.ngroups
        return [self.positions[label] for label in labels]
    
    def value_to_labels(self, value):
        "Decompose a raw value to a multi-label (tuple of labels, one for each group)."
        raise NotImplementedError
    
    def labels_to_value(self, labels):
        "Convert a multi-label back to a raw value."
        raise NotImplementedError
    
    
    def get_keras_layer(self, input_layer, *args, **kwargs):
        
        assert _use_keras
        depth = self.depth or 1
        name = kwargs.pop('name') or self.__name__
        kwargs.setdefault('activation', self.keras_activation)
        
        sublayers = OrderedDict()
        for group_name, _, labels in self.groups:
            depth = len(labels)
            gname = '%s_%s' % (name, group_name)
            sublayers[gname] = Conv2D(depth, *args, name = gname, **kwargs)(input_layer)
            
        return Concatenate(name = name)(sublayers.values())


    def get_keras_loss(self):
    
        if not (self.group_weights or self.label_weights):
            return self.keras_loss
    
        weights = np.ones(self.depth)
        
        # adapt weights of entire groups
        start = stop = 0
        for length, group_weight in zip(self.group_lengths, self.group_weights or []):
            stop += length
            weights[start:stop] *= group_weight
            start = stop
        
        # adapt weights of individual labels
        for label, label_weight in (self.label_weights or {}).iteritems():
            pos = self.positions[label]
            weights[pos] *= label_weight
        
        print 'MultiHotSignal.get_keras_loss/weights:'
        for w, label in zip(weights, self.labels):
            print '%7.3f  %s' % (w, label)
        
        return MSE_weighted(weights, normalize = False)
        
    

#####################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()


    # x = np.random.rand(2, 3, 5)
    #
    # print 'x:'
    # print x[0,0,:]
    #
    # y = np.dstack([f_softmax(x[...,0:2]), f_softmax(x[...,2:5])])
    # print 'softmax:'
    # print y[0,0,:]
    #
    # ky = K.eval(keras_multi_softmax(K.variable(x), [2, 3]))
    # print 'K-multi_softmax:'
    # print ky[0,0,:]
    # print np.abs(y - ky).max()
    