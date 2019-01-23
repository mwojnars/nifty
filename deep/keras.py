'''
Elements for building Deep Neural Networks with Keras.

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
import keras.layers
from keras import backend as K
from keras.layers import Layer, Conv2D, Activation, Lambda, concatenate, add as layers_add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.utils.generic_utils import get_custom_objects
from keras.utils.conv_utils import normalize_tuple

# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages;
# only when executed as a standalone file, for unit tests, do an absolute import
if __name__ != "__main__":
    from ..util import isstring, istuple, islist
else:
    from nifty.util import isstring, istuple, islist


#####################################################################################################################################################
#####
#####  LAYERS & ACTIVATIONS
#####

def Relu(x): return Activation('relu')(x)
def LeakyReLU(x): return keras.layers.LeakyReLU()(x)
def Softmax(x): return Activation('softmax')(x)

get_custom_objects().update({'Relu': Relu})
# get_custom_objects().update({'Leaky_relu': LeakyReLU})


def relu_BN(y):
    "Relu activation preceeded by BatchNormalization."
    y = BatchNormalization()(y)
    y = Relu(y)
    return y

def leaky_BN(y):
    "LeakyReLU activation preceeded by BatchNormalization."
    y = BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)
    return y

def conv2D_BN(y, *args, **kwargs):
    """Extra arguments:
    - add: (optional) tensor or a list of tensors (typically a shortcut connection) to be added to the output
           right after BatchNormalization, but before activation
    """
    activation = kwargs.pop('activation', None)
    if isstring(activation): activation = Activation(activation)

    add = kwargs.pop('add', None)
    if add and not islist(add): add = [add]
    
    y = Conv2D(*args, **kwargs)(y)
    y = BatchNormalization()(y)
    if add: y = layers_add([y] + add)
    if activation: y = activation(y)
    
    return y
    

#####################################################################################################################################################
#####
#####  ADVANCED LAYERS
#####

class LocalNormalization(Layer):
    """
    Shifts and rescales input activations by means and standard deviations and magnitudes of activation
    of a given channel around a given pixel. Scaling parameters of normalization are trainable.
    Depending on the target and role of a given channel, the network can use LocalNormalization to either
    locally normalize the channel (enhance contrast between neighboring activations), or de-normalize it.
    The latter happens, for instance, when the channel's output must exhibit positive local correlation
    (i.e., high activations should co-occur on neighboring spatial positions) - in such case,
    LocalNormalization will learn negative weights, so as to reinforce local correlation through negative normalization.
    On the other hand, positive weights and positive normalization are learnt when the channel should expose
    negative correlation between neighboring locations (e.g., when performing edge detection).
    """

    def __init__(self, kernel_size = (7, 7), init_normal = 1.0, **kwargs):
        """
        init_normal: initial value of (normal_dev+normal_mag) weight +/- uniform random shift of max. 0.05
        """
        super(LocalNormalization, self).__init__(**kwargs)
        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        self.init_normal = init_normal          # initial value of (normal_dev+normal_mag), +/- random shift of max. 0.05
        self.seed = None

    def build(self, input_shape):
        
        self.channel_axis = -1          # channel_axis = 1 if self.data_format == 'channels_first' else -1
        
        depth = input_shape[self.channel_axis]
        if depth is None: raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        # shift = 1.0: activations are translated such that their local mean == 0.0
        # shift = 0.0: activiations are not translated at all
        # shift outside of <0.0,1.0>: excessive translation (towards mean or below 0.0)
        self.shift  = self.add_weight(name = 'shift',  shape = (depth,), initializer = 'uniform', trainable = True)

        # normal_dev & normal_mag initialized with ~self.init_normal/2 each (~0.5 by default)
        mid = self.init_normal / 2
        uniform_05 = RandomUniform(mid - .05, mid + .05, seed = self.seed)

        # normal = 1.0: activations are normalized to local std.deviation == 1.0
        # normal = 0.0: activations are not normalized at all
        self.normal_dev = self.add_weight(name = 'normal_dev', shape = (depth,), initializer = uniform_05, trainable = True) #constraint = Clip(0.0, 1.0)
        self.normal_mag = self.add_weight(name = 'normal_mag', shape = (depth,), initializer = uniform_05, trainable = True) #constraint = Clip(0.0, 1.0)
        
        # scale <> 0.0: activations are rescaled by the factor of e^scale after normalization
        # scale = 0.0: activations are not rescaled
        self.scale  = self.add_weight(name = 'scale',  shape = (depth,), initializer = 'uniform', trainable = True) #constraint = Clip(-1.0, 1.0)
        
        # self.offset = self.add_weight(name = 'offset', shape = (depth,), initializer = 'uniform', trainable = True)
        
        super(LocalNormalization, self).build(input_shape)          # Be sure to call this at the end

    def call(self, x):
        
        # print 'LocalNormalization.kernel_size:', self.kernel_size
        
        def mean2d(y):
            
            y = K.pool2d(y, (self.kernel_size[0], 1), pool_mode = 'avg', padding = 'same')
            y = K.pool2d(y, (1, self.kernel_size[1]), pool_mode = 'avg', padding = 'same')
            return y

            # return K.pool2d(y, self.kernel_size, pool_mode = 'avg', padding = 'same')
            
            # (dy, dx) = self.kernel_size
            # top  = dy/2 + 1                             # if even `dy`, averaging window is shifted to the top
            # left = dx/2 + 1                             # if even `dx`, averaging window is shifted to the left
            #
            # padding = ((top, dy-top), (left, dx-left))
            #
            # z  = K.spatial_2d_padding(y, padding)       # `y` padded with zeros
            # s1 = K.cumsum(z,  axis = -3)                # cumulative sums along Y axis only
            # s  = K.cumsum(s1, axis = -2)                # cumulative sums along (Y,X) axes
            #
            # t = s[...,dy:,dx:,:] + s[...,:-dy,:-dx,:] - s[...,dy:,:-dx,:] - s[...,:-dy,dx:,:]
            #
            # # t[0,0] = s[dy,dx] + s[0,0] - ... = cumsum(y)[0,0] + cumsum(y)[dy,dx] - ... = z[0,0] + (z[0,0]+...+z[dy,dx]) - ...
            # #        = area_sum(z, (1,1)...(dy,dx)) = area_sum(y, (0,0)...(dy-top,dx-left)) = area_sum(y, (0,0)...(dy-(dy/2+1), dx-(dx/2+1))) =
            #
            # return t / float(dx*dy)
            
            
        # mean of `x` and x^2 in local area around given pixel
        M   = mean2d(x)
        M2  = mean2d(x**2)
        V   = mean2d((x-M)**2)
        eps = 0.001  #K.epsilon()
        
        scale = K.exp(self.scale) / K.pow(M2 + eps, self.normal_mag/2) / K.pow(V + eps, self.normal_dev/2)  #(V + eps) #K.exp(K.log(D + eps) * self.normal[None,None,:])
        return (x - self.shift * M) * scale
        
        # D = K.pool2d(x, self.kernel_size, pool_mode = 'avg', padding = 'same', data_format = 'channels_last') #self.data_format
        # return x / K.exp(D * self.scale - self.offset)

    def get_config(self):
        config = {'kernel_size': self.kernel_size}
        base_config = super(LocalNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class FeaturesNormalization(Layer):
    """
    Normalize input values along channels dimension, independently on every spatial position.
    Meta-parameters of normalization are learnt during training.
    
    Normalization that works along channels dimension. Does 2 things:
    1) normalizes channel activations so that their sum on every spatial position is (roughly) equal to a predefined (but trainable) value
    2) when total activation is small, adds random noise to small activations to stimulate training
    """
    def __init__(self, **kwargs):
        super(FeaturesNormalization, self).__init__(**kwargs)
        self.seed = None

    def build(self, input_shape):

        self.channel_axis = -1          # channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.depth = input_shape[self.channel_axis] or 100

        uniform_05 = RandomUniform(0.45, 0.55, seed = self.seed)        # norm_dev & norm_mag initialized with ~0.5 each

        # norm_dev = 1.0: features are normalized to std.deviation == 1.0
        # norm_mag = 1.0: features are normalized to quadratic average (magnitude) == 1.0
        # norm_abs = 1.0: features are normalized to mean absolute value == 1.0
        self.norm_dev = self.add_weight(name = 'norm_dev', shape = (), initializer = uniform_05, trainable = True)
        self.norm_mag = self.add_weight(name = 'norm_mag', shape = (), initializer = uniform_05, trainable = True)
        self.norm_abs = self.add_weight(name = 'norm_abs', shape = (), initializer = 'uniform',  trainable = True)
        
        super(FeaturesNormalization, self).build(input_shape)       # Be sure to call this at the end

    def call(self, x):
        
        # statistics computed along features dimension, on every spatial position of the input tensor
        A   = K.mean(K.abs(x), axis = self.channel_axis)            # mean absolute value
        M1  = K.mean(x, axis = self.channel_axis)                   # mean value
        M2  = K.mean(x**2, axis = self.channel_axis)                # squared quadratic average
        V   = M2 - M1**2                                            # variance: V[X] = E[X^2] - E[X]^2
        eps = 0.001 #K.epsilon()
        
        norm = K.pow(V + eps, self.norm_dev/2) * K.pow(M2 + eps, self.norm_mag/2) * K.pow(A + eps, self.norm_abs)
        return x / norm[...,None]

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class SmartNoise(Layer):
    """
    When total or maximum absolute activation of input is small on a particular spatial position,
    SmartNoise selectively adds uniform noise to individual activations to stimulate training.
    
    If a particular activiation is negative, adding noise is done by further decreasing its value
    (i.e., the noise added has the same sign as the original value).
    """
    def __init__(self, **kwargs):
        super(SmartNoise, self).__init__(**kwargs)
        self.seed = None
        # self.supports_masking = True

    def build(self, input_shape):

        self.channel_axis = -1          # channel_axis = 1 if self.data_format == 'channels_first' else -1
        
        uniform_0   = RandomUniform(-.05, +.05, seed = self.seed)
        uniform_1   = RandomUniform(0.95, 1.05, seed = self.seed)
        # uniform_3   = RandomUniform(2.95, 3.05, seed = self.seed)
        # uniform_01  = RandomUniform(0.10, 0.15, seed = self.seed)
        # uniform_001 = RandomUniform(0.01, 0.02, seed = self.seed)
        
        self.scale       = self.add_weight(name = 'scale',       shape = (), initializer = uniform_1, trainable = True)     # scale of added noise
        self.sensitivity = self.add_weight(name = 'sensitivity', shape = (), initializer = uniform_0, trainable = True)
        # self.reduction   = self.add_weight(name = 'reduction',   shape = (), initializer = uniform_3, trainable = True)

        super(SmartNoise, self).build(input_shape)       # Be sure to call this at the end

    def call(self, x, training = None):
        
        eps = 0.01
        ax  = K.abs(x)
        M   = K.mean((ax+eps) ** 4, axis = self.channel_axis) ** (1./4)     # Minkowsky's average to focus more on the (few) large values than on (many) smaller ones
        
        noise = K.random_uniform(shape = K.shape(x), minval = -1.0, maxval = 1.0, seed = self.seed)
        
        # xr  = ax * K.exp(self.reduction)
        # red = xr / (1 + xr**2)
        red = 1 / (1 + ax)                                          # individual noise reduction for each element of input
        mag = K.exp(-M / K.exp(self.sensitivity)) * self.scale      # global magnitude:  if M = 0.0 -> large magnitude (1.0) ... if M >> 0.0 -> low magnitude (~0.0)
        
        noisy = x + noise * red * mag[...,None]
        
        return noisy
        # return K.in_train_phase(noisy, x, training = training)
        

    def compute_output_shape(self, input_shape):
        return input_shape
    

#####################################################################################################################################################
#####
#####  BLOCKS
#####

def grouped_convolution(y, channels, groups, strides = 1, dilation_rate = 1):
    """Grouped convolution with `groups` number of groups, between layers of depth: channels[0] to channels[1].
       When `groups`=1 this is just a standard convolution.
       If `channels` is a single number, the same depth on input and output is assumed.
    """
    
    if not istuple(channels): channels = (channels, channels)
    if not istuple(groups):   groups   = (0, 0, groups)

    (channels_in, channels_out) = channels
    groups_H, groups_V, groups_C = groups
    groups_total = sum(groups)
    
    # if not groups:
    #     assert channels_out % groupsize == 0
    #     groups = channels_out // groupsize
    #
    # if not groupsize:
    #     assert channels_out % groups == 0
    #     groupsize = channels_out // groups
        
    # if not channels:
    #     channels = groups * groupsize
    # else:
    #     assert channels == groups * groupsize
        
    # when groups==1 this is just a standard convolution
    if groups == (0, 0, 1):
        return Conv2D(channels_out, (3, 3), strides = strides, dilation_rate = dilation_rate, padding = 'same')(y)
    
    depth_in = y.shape[-1]
    
    assert channels_in == depth_in, "grouped_convolution(): declared no. of input channels (%s) differs from the actual depth of input layer (%s)" % (channels_in, depth_in)
    assert channels_in  % groups_total == 0, "grouped_convolution(): no. of input channels (%s) must be a multiplicity of the no. of groups (%s)" % (channels_in, groups_total)
    assert channels_out % groups_total == 0, "grouped_convolution(): no. of output channels (%s) must be a multiplicity of the no. of groups (%s)" % (channels_out, groups_total)
    group_in  = channels_in  // groups_total
    group_out = channels_out // groups_total
    
    # in a grouped convolutional layer, input & output channels are divided into groups and convolutions are performed separately
    # within each group: between k-th input group and k-th output group; outputs are concatenated afterwards
    paths = []
    for k in xrange(groups_total):
        shape = (1, 9) if k < groups_H else (9, 1) if k < groups_H + groups_V else (3, 3)
        start = k * group_in

        # def slice_input(x):
        #     return x[..., start : start + group_in]

        group = Lambda(lambda x: x[..., start : start + group_in])(y)
        layer = Conv2D(group_out, shape, strides = strides, dilation_rate = dilation_rate, padding = 'same')(group)
        paths.append(layer)
        
    return concatenate(paths)


def resnext_unit(y, bottleneck, channels_out, paths, strides = 1, dilation_rate = 1, activation = 'relu'):
    """ResNeXt residual unit, optionally extended with spatial (vertical+horizontal) paths.
       If `paths` is a triple (A,B,C), A is the no. of 1x9 (horizontal) paths, B: 9x1 (vertical), C: 3x3.
    """
    if isstring(activation): activation = Activation(activation)
    depth_in = y.shape[-1]

    shortcut = y
    
    # the residual block is reshaped as a bottleneck + grouped-convolution + rev-bottleneck, which is equivalent
    # to the original formulation as a collection of paths, but makes the network more economical
    y = Conv2D(bottleneck, (1, 1), padding = 'same')(y)                                                 # (1) bottleneck
    y = activation(BatchNormalization()(y))

    # create ResNeXT grouped convolutions (the middle element of paths)
    y = grouped_convolution(y, bottleneck, paths, strides = strides, dilation_rate = dilation_rate)     # (2) grouped convolution
    y = activation(BatchNormalization()(y))

    y = Conv2D(channels_out, (1, 1), padding = 'same')(y)                                               # (3) rev-bottleneck
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = BatchNormalization()(y)

    # if input/output have different dimensions: because of a stride (spatial dimension), or because of a different depth,
    # an extra 1x1 convolution is added on the shortcut connection to perform the adjustment
    if strides not in [None, 1, (1, 1)] or depth_in != channels_out:
        shortcut = Conv2D(channels_out, (1, 1), strides = strides, padding = 'same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = layers_add([shortcut, y])
    y = activation(y)

    return y

