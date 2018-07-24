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
from keras.layers import Conv2D, Activation, Lambda, concatenate, add as layers_add
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_custom_objects

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
    y = layers.LeakyReLU()(y)
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
#####  BLOCKS
#####

def grouped_convolution(y, channels, groups, strides = None):
    """Grouped convolution with `groups` number of groups, between layers of depth: channels[0] to channels[1].
       When `groups`=1 this is just a standard convolution.
       If `channels` is a single number, the same depth on input and output is assumed.
    """
    
    if not istuple(channels): channels = (channels, channels)
    if not istuple(groups):   groups   = (groups, 0, 0)

    (channels_in, channels_out) = channels
    groups_C, groups_H, groups_V = groups
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
    if groups == (1, 0, 0):
        return Conv2D(channels_out, (3, 3), strides = strides, padding = 'same')(y)
    
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
        shape = (3, 3) if k < groups_C else (1, 9) if k < groups_C + groups_H else (9, 1)
        start = k * group_in

        # def slice_input(x):
        #     return x[..., start : start + group_in]

        group = Lambda(lambda x: x[..., start : start + group_in])(y)
        layer = Conv2D(group_out, shape, strides = strides, padding = 'same')(group)
        paths.append(layer)
        
    return concatenate(paths)


def resnext_unit(y, bottleneck, channels_out, paths, strides = None, activation = 'relu'):
    """ResNeXt residual unit, optionally extended with spatial (vertical+horizontal) paths.
       If `paths` is a triple (A,B,C), A is the no. of 3x3 paths, B: 1x9, C: 9x1.
    """
    if isstring(activation): activation = Activation(activation)
    depth_in = y.shape[-1]

    shortcut = y
    
    # the residual block is reshaped as a bottleneck + grouped-convolution + rev-bottleneck, which is equivalent
    # to the original formulation as a collection of paths, but makes the network more economical
    y = Conv2D(bottleneck, (1, 1), padding = 'same')(y)                             # (1) bottleneck
    y = activation(BatchNormalization()(y))

    # create ResNeXT grouped convolutions (the middle element of paths)
    y = grouped_convolution(y, bottleneck, paths, strides = strides)                # (2) grouped convolution
    y = activation(BatchNormalization()(y))

    y = Conv2D(channels_out, (1, 1), padding = 'same')(y)                           # (3) rev-bottleneck
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

