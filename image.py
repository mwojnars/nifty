'''
Basic and advanced image operations. Most of them available in two forms:
 - procedural:  imXXX(...)
 - object-oriented:  methods of class Image, typically wrappers for imXXX() functions: 
                     function imABC(img) transforms to a method call img.ABC().

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''


import numbers, math, scipy
import numpy as np
from numpy import zeros, sqrt, mean, ceil, sum, abs, zeros_like, isnan, pi, exp, log, r_, c_

import matplotlib.pyplot as plt

from numpy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.io import imread       # imread from scipy.misc flips the image upside-down; this one is OK
from skimage.io import imshow

from skimage.color import gray2rgb
from skimage import filter
from scipy.signal import correlate2d
from scipy.ndimage.interpolation import rotate

from util import *


########################################################
### Basic image routines in procedural form.
###

def rgb(i1, i2 = None, i3 = None):
    """
    Either decompose 3D RBG matrix into [R,G,B] (if only i1 is given), 
    or combine R,G,B input matrices into a single 3D output matrix.
    """
    if i2 == None:      # decompose 3D matrix
        return (i1[:,:,0], i1[:,:,1], i1[:,:,2])    # (R,G,B)
    
    if (i1 != None) and (i2 != None) and (i3 != None):        # combine R,G,B into 3D matrix
        (s1,s2,s3) = (i1.shape,i2.shape,i3.shape)
        if s1 != s2 or s2 != s3:
            raise Exception("rgb(), input arrays have different shapes: %s %s %s" % (s1,s2,s3))
        if len(s1) < 2:
            raise Exception("rgb(), input arrays are not images - too few dimensions: %s" % str(s1))
        out = zeros(s1 + (3,))
        out[:,:,0] = i1
        out[:,:,1] = i2
        out[:,:,2] = i3
        return out

    print 'Error. Incorrect number of input arguments'

def im3d(I):
    """
    Ensure that the image is 3D not 2D. If 2D, it's transformed by adding
    3rd dimension of length 1. Returns transformed image
    (this may be a view into original image! not a copy). 
    """
    if I.ndim <= 2:
        return I.reshape(I.shape + (1,))
    return I

def im2d(I):
    """
    Merge all color planes, no matter how many of them there are.
    All planes in 3rd dimension are simply averaged.
    """
    if I.ndim <= 2: return I
    return mean(I,2)

def meancolor(I):
    "Returns average color in I, as a scalar (if I is 2D) or ndarray vector (if 3D)."
    if I.ndim <= 2:
        return mean(I)
    return np.array(mean(mean(I,0),0))

def imsolid(color, H = 0, W = 0, like = None):
    """Creates image of solid color 'color' and size (H,W) or the same as image 'like'.
    'color' can be either a vector (RGB) or a scalar (gray-scale),
    then returned image is either 3D or 2D.
    Warning: dtype of resulting image is the same as type of 'color'! (e.g., int64 if color=0)
    To use different dtype, convert 'color' into desired dtype before calling imsolid().
    Only if 'like' image is given, convertion will be done automatically to "like"'s dtype.
    """
    if H <= 0: H = like.shape[0]
    if W <= 0: W = like.shape[1]
    if like is not None:
        color = like.dtype.type(color)
    if np.isscalar(color):
        return np.tile(color, (H,W))
    else:
        return np.tile(color, (H,W,1))

def stdcolor(I):
    """
    Standard deviation of color in I, as expected NORM of deviation from the mean color.
    Treats all color planes equally, no matter how many of them there are and what they mean.
    Handles also gray images, then it's equal to std(I).
    """
    I = im3d(I)
    nc = I.shape[2]
    v = 0.0
    # sum variances across all color planes
    for c in range(0,nc):
        v += np.var(I[:,:,c])
    return sqrt(v)

def meanimg(imgs, axis = 0):
    """
    Given a list of images of the same shape, 2D or 3D, computes mean image of all of them.
    'imgs' is either a list or ndarray with images indexed by dimension 'axis' (0 by default).
    """
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
        axis = 0
    return mean(imgs, axis)
def sumimg(imgs, axis = 0):
    """
    Given a list of images of the same shape, 2D or 3D, computes sum image of all of them.
    'imgs' is either a list or ndarray with images indexed by dimension 'axis' (0 by default).
    """
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
        axis = 0
    return np.sum(imgs, axis)
def maximg(imgs, axis = 0):
    """
    Given a list of images of the same shape, 2D or 3D, computes maximum image of all of them (pixel-wise max).
    'imgs' is either a list or ndarray with images indexed by dimension 'axis' (0 by default).
    """
    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)
        axis = 0
    return np.max(imgs, axis)


    
########################################################
### Advanced image routines in procedural form.
###

def imresize(I, size = None, H = None, W = None, order = 3, vcrop = None):
    """Resize to a fixed 'size' (tuple) or by a given scaling coefficient ('size' is number)
    or proportionally to a fixed width W, or proportionally to a fixed height H,
    or to a fixed size (H,W) if both given.
    'order' is the order of the spline interpolation, integer in 0-5.
    'vcrop' is (min,max) range of allowed values in the result, e.g. (0,1), or None if no cropping to be done.
    """
    (h,w) = I.shape[:2]
    
    eps = 0.1       # added to numerator when computing 'factor', to ensure correct size after convertion to pixels
    if size is not None:
        if isnumber(size):      # scaling coefficient
            factor = size
        else:                   # tuple
            (H,W) = size
    elif H is None:
        factor = (W+eps) / float(w)
    elif W is None:
        factor = (H+eps) / float(h)
    
#    if not size:
#        if w == W: return I         # no changes to be done - return
#        H = int(h * (W/float(w)))
#        size = (H,W)
#    if size == (h,w): return I
    
    factors = [1] * I.ndim
    if factor:
        factors[0] = factors[1] = factor
    else:
        factors[0] = (H+eps) / float(h)
        factors[1] = (W+eps) / float(w)
    
    J = scipy.ndimage.interpolation.zoom(I, factors, order = order) #@UndefinedVariable
    if vcrop:
        J.clip(vcrop[0], vcrop[1], J)
    
    #J = scipy.misc.pilutil.imresize(I, size, mode='F')      # 'F' = floating-point image, don't make any crazy rescaling of values
    #J = img_as_float(J)

    J = J.astype(I.dtype)           # force the original dtype, otherwise strange things happen with efficiency
    return J 

def imresizeX(I, size = None, W = None):
    """
    OLD imresize(), using scipy.misc.pilutil.imresize().
    Resize to a fixed 'size' (tuple) or by a given scaling coefficient ('size' is number)
    or proportionally to a fixed width W.
    """
    (h,w) = I.shape[:2]
    if not size:
        if w == W: return I         # no changes to be done - return
        H = int(h * (W/float(w)))
        size = (H,W)
    if size == (h,w): return I
        
    J = scipy.misc.pilutil.imresize(I, size)
    J = img_as_float(J)
    #J = J.astype(I.dtype)           # force the original dtype, after crazy pilutil changes
    
    return J 

def imzoom(I, factor = 1.0, order = 3, vcrop = None, fill = None):
    """
    Similar to imresize(), but after rescaling the output image is cropped or padded
    to have the same size as original image.
    'factor' is a floating-point scaling coefficient, 1.0 means original size.
    """
    (H,W) = I.shape[:2]
    J = imresize(I, factor, order = order, vcrop = vcrop)
    if J.shape < I.shape:
        J = impad(J, H, W, fill)
    elif J.shape > I.shape:
        J = imroi(J, H, W)
    J = J.astype(I.dtype)
    return J

def imcrop(I, top=0, bottom=0, left=0, right=0, All=None):
    "Crops the image by a given no. of pixels (if int) or fraction (if float) from each side."
    def toInt(f, base):
        if not isIntegral(f): return int(f*base)
        return f

    if All is not None:
        if top <= 0: top = All
        if bottom <= 0: bottom = All
        if left <= 0: left = All
        if right <= 0: right = All
    
    # convert floats to absolute pixels
    (H,W) = I.shape[:2]
    top = toInt(top, H)
    bottom = toInt(bottom, H)
    left = toInt(left, W)
    right = toInt(right, W)
    
    # all indices count from array beginning, not end
    bottom = H - bottom
    right = W - right
    
    return I[top:bottom,left:right]


def imroi(I, H, W):
    '''
    Cuts off rectangular ROI (Regions Of Interest) of size (H,W,_)
    from the center of I.
    Assumes that (H,W) <= size of I.
    Floating point values of (H,W) are interpreted as scaling factors for I size.
    Value <= 0 or None on any dimension is interpreted as "use current image size". 
    '''
    (h,w) = I.shape[:2]
    # assumption: h >= H, w >= W
    if type(H) is float: H = int(H * h)
    if type(W) is float: W = int(W * w)
    if H <= 0: H = h
    if W <= 0: W = w
    if H > h or W > w:
        raise Exception("Error in imroi(). Size of ROI (%d,%d) larger than image size (%d,%d)" % (H,W,h,w))
    top = (h - H) / 2
    left = (w - W) / 2
    I = I[top:top+H, left:left+W]
    return I

def impad(I, H = 0, W = 0, fill = None):
    '''
    Pad 'I' symmetrically on top+bottom and/or left+right, 
    to obtain height of at least H and width of at least W.
    'fill' (optional) is the color to be used as a filler, mean color used if None.
    '''
    (h,w,c) = (I.shape + (0,))[:3]
    
    H = max(H,h)
    W = max(W,w)
    top = (H - h) / 2
    left = (W - w) / 2
    
    color = fill
    if c == 0:      # 2D image
        if color == None:
            color = mean(I)
        else:
            color = mean(color)     # turn to grayscale, in case if 'color' is accidentally a real color
        J = np.zeros((H,W), dtype=I.dtype) + color
        J[top:top+h, left:left+w] = I
        
    else:           # 3D image
        if color == None:
            color = meancolor(I)    # vector of average color  
        elif np.isscalar(color):
            color = [color] * c
        J = np.zeros((H,W,c), dtype=I.dtype)
        for ic in range(0,c):
            J[:,:,ic] = color[ic]
        J[top:top+h, left:left+w, :] = I
        
    return J

def imrotate(I, angle, reshape = False, fill = None, vcrop = None):
    """
    Rotates the image by 'angle' degrees counter-clockwise around its center.
    'fill' is the color to be used as a filler for missing pixels. If None, mean color is used.
    'vcrop' is (min,max) range of allowed values in the result, e.g. (0,1), or None if no cropping to be done
    (scipy rotation routine tends to extrapolate values to outside the original image range).
    """
    #if fill is None:
    #    fill = mean(self.a)
    if fill is None:
        fill = np.nan
    if (fill is np.nan) and (I.ndim <= 2):
        fill = mean(I)
    
    J = scipy.ndimage.interpolation.rotate(I, angle, reshape=reshape, cval=fill) #@UndefinedVariable
    
    if fill is np.nan:          # for 3D images we have to make some tricks to fill with mean color
        fill = meancolor(I)
        for c in range(0,J.shape[2]):
            L = J[:,:,c]
            J[:,:,c][np.isnan(L)] = fill[c]
    if vcrop:
        J.clip(vcrop[0], vcrop[1], J)
    return J

def imshift(I, dy = 0, dx = 0, fill = None):
    """
    Shifts image I by an integral no. of pixels in right/downwards direction.
    (dx,dy) is either absolute shift (if integral), or a shift factor 
    relative to image width/height (if float), which after conversion to absolute is rounded down.
    'fill' is the color to be used as a filler. If None, mean color is used. 
    """
    def toInt(f, base):
        if f is None: return 0
        if not isIntegral(f): return int(f*base)
        return f
    def z(t):
        return max(t,0)

    # convert floats to absolute pixels
    (H,W) = I.shape[:2]
    dx = toInt(dx, W)
    dy = toInt(dy, H)
    if not (dx or dy): return I
    
    if fill is None:
        fill = meancolor(I)
    elif np.isscalar(fill) and I.ndim > 2:
        fill = np.tile(fill, I.shape[2])        # vector color needed when scalar given - repeat it as many times as needed 
        
    J = imsolid(fill, like=I)
    #print I.shape, J.shape
    
    if I.ndim <= 2:
        J[z(dy):z(H+dy),z(dx):z(W+dx)] = I[z(-dy):z(H-dy),z(-dx):z(W-dx)]   # copy image with shift
    else:
        J[z(dy):z(H+dy),z(dx):z(W+dx),:] = I[z(-dy):z(H-dy),z(-dx):z(W-dx),:]

    ## The routine from SciPy is more complex: handles shifts by non-integral no. of pixels using splines
    #from scipy.ndimage.interpolation import shift
    #J = shift(I, (dx,dy))
    
    return J

#def imtrans(I, rot, shift, scale):
#    from skimage.transform import homography
#    H = []
#    J = homography(I, H)
#    return J

def imdilate(I, size = (3,3), soft = 0.0):
    """
    'soft' is a weight to be used if dilated image is to be combined back with orignal one for smoothing.
    soft=0 means no smoothing (image fully dilated), soft=1 returns original image (no dilation).
    """
    from scipy.ndimage.morphology import grey_dilation
    if I.ndim > len(size):
        size = size + (1,)*(I.ndim - len(size))     # append missing 1's at the end
    J = grey_dilation(I, size)
    if soft:
        J = (1-soft)*J + soft*I
    return J

def imfft(I, doLog = True, merge = True):
    axes = (0,1)
    J = fftshift(abs(fft2(I,axes=axes)), axes)
    if merge and J.ndim > 2:
        J = sqrt(np.sum(J**2,-1))       # merge R,G,B (or other) layers
    if doLog:
        J = log(1+J)                    # log-FFT
    # J[J < 1] = 0                      # thresholding
    # print "fft-log:  mean %f  std %f  min-max %f-%f" % (mean(J), std(J), min(J), max(J))
    #if I.ndim > 2:
    #    J = gray2rgb(J)                # get back to 3-D image, as needed for other routines
    return J                                                    

    
def immean(I, mask, norm = True):
    '''
    1st order local filter: average color in a given location L (subwindow of I),
    over all locations in image I, with shape of L defined by 'mask'.
    Calculated via linear cross-correlation - weighted average of pixels.
    If 'norm'=True, mask is normalized to 1.0 sum.
    Mask is 2D. Image is 2D or 3D. Returned image is 2D or 3D, like original image.
    (last dimension can have length of 1).
    Cross-correlation is done in 'same'/'symm' mode,
    which means that the resulting image has the same size as I,
    but values at the border (where mask went partially out of I) can be disturbed. 
    '''
    is2D = (I.ndim <= 2)
    I = im3d(I)
    J = zeros_like(I)
    nc = I.shape[2]     # no. of color planes
    if norm: mask = mask / float(sum(mask))
    #print minmax(mask)
    
    for c in range(0,nc):
        J[:,:,c] = correlate2d(I[:,:,c], mask, 'same', 'symm')
        #print c, minmax(J[:,:,c])
    
    if is2D: J = J[:,:,0]
    
    # numpy has a bug in correlate2d() and can produce NaN's
    # or very big numbers when mask is much larger than 'I';
    # below is a fix:
    (mi,MI) = minmax(I)
    (mj,MJ) = minmax(J)
    d = (MI-mi)*1000 
    if isnan(mj+MJ) or (mj < mi-d) or (MJ > MI+d):
        print "immean(), Warning: stepped onto correlate2d() bug, fixing..."
        fix = mean(I)
        J[isnan(J)] = fix
        J[J < mi-d] = fix
        J[J > MI+d] = fix
    
    return J

def imvar(I, mask, retMean = False, scalar = True):
    '''
    2nd order local filter: variance of pixel color in a given location L.
    Calculated via multiple application of immean() filter,
    from the equation:  D[L]^2 = E[L^2] - (E[L])^2
    where weights of the mask define prob.distribution of D/E calculation.
    Each color plane is filtered separately.
    If 'scalar' is True, all planes are merged by summing variances of different dimensions (planes).
    '''
    E = immean(I, mask)
    E2 = immean(I**2, mask)
    #print np.min(E), np.max(E)
    D2 = E2 - E**2
    if scalar and D2.ndim >= 3: 
        D2 = np.sum(D2, 2)
    #print "Minimum value in D2:", np.min(D2)
    D2[D2 < 0] = 0      # floating-point errors may result in slightly negative numbers sometimes
    if retMean:
        return (E, D2)  # usually when variance is needed then mean also - don't compute it twice 
    return D2

def imstd(I, mask, retMean = False, scalar = True):
    '''
    2nd order local filter: standard deviation of pixel color in a given location L.
    See 'imvar()' for details.
    '''
    if retMean:
        (E, D2) = imvar(I, mask, retMean, scalar)
        return (E, np.sqrt(D2))
    return np.sqrt(imvar(I, mask, retMean, scalar))

def impattern(I, mask, thresh = 0.2, echo = 1.0, g = 0.5, scalar = True):
    '''
    2nd order local differential filter: tells if a given 2-region pattern exists in a given location. 
    Filters the image twice: using positive (P) and negative (N) mask
    (P = foreground, N = background);
    then subtracts the means, takes norm of the difference 
    and normalizes in each point by local deviation of color
    (mask for calculation of D is a plain average of masks for P & N; each subregion
    has *equal* contribution of 0.5 in distribution of D):
      Y = (E[P] - E[N]) / (2*D[L])
    where 
      L = sum(P,N).
    If L takes on exactly 2 different values determined by P or N, then |Y| = 1.0.
    Otherwise, 0 <= |Y| < 1.0.
    However, this is modified additionally by global deviation of color added in denominator:
    the inter-region deviation must be high not only compared with local deviation,
    but also compared with global deviation in entire image.
    
    'scalar' indicates whether the result should be merged across all color planes (default)
       or should be left separate for each color.
    'thresh' is the minimum output value; lower values are zeroed out.
    'mask' is either a 2D image where sign (positive/negative) of values
       indicates P/N subregion; or a 3D image where R ([0]) plane is the positive region
       and B ([2]) plane is the negative region.
    '''
    def zeroMargin(Y, marginFrac = 3):
        """
        Zeros out margin values of Y, because they may contain artifacts caused by boundary effects.
        By default width of margin is 1/3 of the 'mask' width or height.
        """ 
        (m0,m1) = mask.shape[:2]
        (m0,m1) = (divup(m0,marginFrac), divup(m1,marginFrac)) 
        Y[:m0,:] = 0
        Y[-m0:,:] = 0
        Y[:,:m1] = 0
        Y[:,-m1:] = 0
        return Y
    
    I = im3d(I)
    if mask.ndim <= 2:
        amask = abs(mask)
        P = (mask + amask) / 2      # positive values preserved, negative zeroed out
        N = (amask - mask) / 2      # negative values preserved (turned positive), positive zeroed out
    else:
        P = mask[:,:,0]
        N = mask[:,:,2]
    L = (P + N) / 2.0       # assumption: sum(P) == sum(N)
    
    (EP, VP) = imvar(I, P, retMean = True)
    (EN, VN) = imvar(I, N, retMean = True)
    V = imvar(I, L)
    #D = np.sqrt(V)

    eps = 0.0001
    Icenter = imcrop(I, All = 0.1)
    if not Icenter.size: Icenter = I
    gDev = stdcolor(Icenter) + eps     # global deviation of color (don't count margins, bcs they often disturb dev)
    gDev2 = gDev ** 2
    
    #print P; print N; print L
    #print EP.shape, EN.shape, D.shape

    # the following equality should always hold:
    #   D[L]^2 = (|EP-EN|/2)^2 + (D[P]^2 + D[N]^2)/2
    # that is:
    #   V = Ediff2 + (VP+VN)/2.0
    Ediff2 = normv2(EP - EN, 2)*0.25
    #print V - (Ediff2 + (VP+VN)/2.0)
    
    #Y = normv(EP - EN, 2) / (2*D + gDev)    # pattern occurrence indicator value
    #Y[Y < thresh] = 0                       # zero out low values
    #Y = binarize(Y, 0.1, 0.6)               # softly binarize remaining values
    
    Y = Ediff2 / (V + g*gDev2)                  # pattern occurrence indicator value
    Y[np.maximum(VP,VN) > echo*V] = 0        # zero out if any (P or N) subregion variance is too high, to remove ECHO
    Y = binarize(Y, 0.1, 0.4)                  # softly binarize/enhance remaining values
    
    Y = zeroMargin(Y)
    
    return Y


########################################################
### Image routines in object-oriented form.
### Classes:
###  - Image: represents a single image
###  - Images: represents a list of images, to allow
###            group operations on all of them
###


class Image(object):
    
    # root folder (with trailing "/") where to look for the image file during __init__()
    ROOT = "../data/"
    # if no extension in image filename, add this one
    EXT  = ".jpg"
    
    def __init__(self, img = None):
        if isinstance(img, (str,unicode)):
            # append path?
            if not (img.startswith('.') or img.startswith('/') or img.startswith(self.ROOT)):
                img = self.ROOT + img
            
            # append extension?
            if not '.' in img[-5:]:
                img = img + self.EXT
            
            self.a = _load(img)
        else:
            self.a = np.array(img)      # image data, hopefully
        self.isImage = True         # a type indicator, more reliable than 'isinstance' with ipython reloading  

    def height(self):
        return self.a.shape[0]
    def width(self):
        return self.a.shape[1]
    def hw(self):
        "(height,width) of the image"
        return self.a.shape[:2]
    
    def show(self):
        print "show(), min-max:", np.min(self.a), np.max(self.a)
        if self.a.ndim < 3:
            img = rgb(self.a, self.a, self.a)
        else:
            img = self.a.copy()
        img[img < 0.0] = 0.0
        img[img > 1.0] = 1.0    # to avoid crazy behavior of imshow() to take out-of-bound values with modulo 1.0, instead of truncating
        return imshow(img)            
    
    def layer(self, l):
        "Selected layer of the image"
        return Image(self.a[:,:,l])
    def lmean(self):
        "Layer-wise mean: mean of all layers as a 2D Image."
        return Image(meanimg(self.a,-1))
    def lsum(self):
        "Layer-wise sum: sum of all layers as a 2D Image."
        return Image(sumimg(self.a,-1))
    def lmax(self):
        "Layer-wise maximum: max (pixel-wise) of all layers as a 2D Image."
        return Image(maximg(self.a,-1))
    
    def resize(self, *args, **kwargs):
        return Image(imresize(self.a, *args, **kwargs))
    def scale(self, factor = 1.0):
        "Same as resize(), only explicitly takes a scaling factor"
        return self.resize(float(factor))
    def zoom(self, *args, **kwargs):
        return Image(imzoom(self.a, *args, **kwargs))
    def shift(self, *args, **kwargs):
        return Image(imshift(self.a, *args, **kwargs))
    
    def crop(self, top=0, bottom=0, left=0, right=0):
        "Crops the image by a given no. of pixels (if int) or fraction (if float) from each side."
        return Image(imcrop(self.a, top, bottom, left, right))
    
    def roi(self, H = None, W = None):
        return Image(imroi(self.a, H, W))
    
    def pad(self, *args, **kwargs):
        '''
        Pad 'I' symmetrically on top+bottom and/or left+right, to obtain height of at least H and width of at least W.
        'fill' is the color to be used as a filler. If not given, mean color is used. 
        '''
        return Image(impad(self.a, *args, **kwargs))
    
    def rotate(self, *args, **kwargs):
        """
        Rotates the image by 'angle' degrees counter-clockwise.
        'fill' is the color to be used as a filler. If not given, mean color is used.
        'vcrop' is (min,max) range of allowed values in the result, or None if no cropping to be done
        (scipy rotation routine tends to extrapolate values to outside the original image range).
        """
        return Image(imrotate(self.a, *args, **kwargs))
    
    def times(self, factor):
        return self.vscale(factor)
    def vscale(self, factor = None, sum2 = None, vmax = 1.0, eps = 1e-10):
        '''
        Scale values of all pixels linearly by 'factor';
        or, if 'factor' not given, scale so that sum of squares
        (squared vector norm) equals 'sum2';
        or, if 'sum2' not given, pick factor such that
        maximum absolute value is mapped to 'vmax'.
        '''
        if not factor:
            if sum2:
                factor = sqrt(sum2) / (sqrt(np.sum(self.a ** 2)) + eps)
            else:
                factor = vmax / (np.max(np.abs(self.a)) + eps)
        return Image(self.a * factor)
    
    def vcrop(self, vmin = 0.0, vmax = 1.0):
        return Image(self.a.clip(vmin, vmax))
    
    def fft(self, *args, **kwargs):
        return Image(imfft(self.a, *args, **kwargs))
    
    def canny(self, sigma=1.0, low_threshold=0.1, high_threshold=0.2, mask=None):
        '''
        Locates edge points and returns as an Image.
        Edge point is the one which has low color variantions
        both on the inner circle (region A) and outer ring (region B);
        but has high color variation between inner circle and the ring. 
        '''
        I = self.a
        F = np.zeros_like(I)    # will hold filtered image, all layers
        layers = I.shape[2]
        for l in range(0,layers):
            L = I[:,:,l]
            F[:,:,l] = filter.canny(L, sigma, low_threshold, high_threshold, mask)
        F = np.max(F,2)     # maximum over all layers = logical OR, because canny returns binary pixels
        return Image(gray2rgb(F))

    def edge(self):
        return 

    def pattern(self, mask, merge = None, group = None, **params):
        """
        Applies mask(s) to the image and returns pattern indicator image(s).
        'mask' is either a single Mask instance or a list of Masks.
        'merge' tells if and how to combine multiple pattern images.
        'group' is used with 'merge', to specify grouping of masks for merging: 
           None, by scale ('scale') or by rotation ('rotation').
        If combined, Image() is returned, otherwise Images().
        """
        if getattr(mask, 'isMask', 0):          # a single mask
            return Image(impattern(self.a, mask.a, **params))
        
        # list of masks...
        if merge:
            funs = {'mean':meanimg, 'sum':sumimg, 'max':maximg}
            merge = funs[merge]
            if not group:
                pats = [impattern(self.a, m.a, **params) for m in mask]
                return Image(merge(pats))
            
            # grouping to be applied...
            gpats = {}          # to hold grouped patterns
            for m in mask:
                key = getattr(m, group)
                gpats.setdefault(key, [])
                gpats[key] += [impattern(self.a, m.a, **params)]
            
            # now merge each group...
            merged = [merge(pats) for pats in gpats.values()]
            return Images(merged)

        pats = [impattern(self.a, m.a, **params) for m in mask]
        return Images(pats)

    def binarize(self, *args, **kwargs):
        return Image(binarize(self.a, *args, **kwargs))

    def dilate(self, *args, **kwargs):
        return Image(imdilate(self.a, *args, **kwargs))
    

class Images(object):
    
    def __init__(self, imgs):
        '''
        'imgs' is either a list (of Image instances or filenames),
        then all the images are used;
        or 'imgs' is a single Image (not in a list), 
        then it's decomposed into layers. 
        '''
        if isinstance(imgs, (list, np.ndarray)):
            self.l = []         # holds all Image objects
            for i in imgs:
                if isinstance(i, Images):       # nested list of Images? append entire list
                    self.l += i.l
                    continue
                if not getattr(i, 'isImage', 0):   # wrap in Image() if necessary
                    i = Image(i)
                self.l.append(i)
            #print self.l
        else:
            # 'split' must hold an Image to be decomposed into color layers
            self.l = []
            layers = imgs.a.shape[2]
            for i in range(0,layers):
                self.l.append(Image(imgs.a[:,:,i]))
    
    def __len__(self):
        return len(self.l)
    def __getitem__(self, i):
        if type(i) is int:          # request for single item
            return self.l[i]
        else:                       # request for a slice
            return Images(self.l[i])
    def __setitem__(self, i, v):
        self.l[i] = v
    def __delitem__(self, i):
        del self.l[i]
    def __iter__(self):
        return self.l.__iter__()
            
    def show(self, add = None, cols = 1, fig = None):
        '''
        'add' is an additional list of Images to be shown together with these ones,
        e.g., for comparison. If lists are equal, images will appear in neighboring columns
        (automatically cols = 2).
        '''
        imgs = self.l
        if add:
            imgs = [i for pair in zip(imgs,add) for i in pair]
            cols = 2
        n = len(imgs)
        rows = divup(n, cols)
        plt.clf()
        i = 1
        for img in imgs:
            plt.subplot(rows, cols, i)
            img.show()
            i += 1
        plt.subplots_adjust(0.02,0.02,0.98,0.98,0.1,0.1)

    def merge(self):
        "Merge all images as layers of a single Image. All must be 2D with the same shape."
        if not self.l:
            return None
        sh = self.l[0].a.shape
        if len(sh) < 2:
            raise Exception("Images.merge(), arrays are not images - too few dimensions: %s" % str(sh))
        if len(sh) > 2:
            if sh[2] > 1:
                raise Exception("Images.merge(), can't merge 3D images. Shape: %s" % str(sh))
            del sh[2]
        
        layers = len(self.l)
        out = zeros(sh + (layers,))
        for i in range(0,layers):
            out[:,:,i] = self.l[i].a
        
        return Image(out)

    def layer(self, l):
        "Selected layer of all the images."
        return Images([i.layer(l) for i in self.l])
    def lmean(self):
        return Images([i.lmean() for i in self.l])
    def lsum(self):
        return Images([i.lsum() for i in self.l])
    def lmax(self):
        return Images([i.lmax() for i in self.l])
    
    def resize(self, *args, **kwargs):
        return Images([i.resize(*args, **kwargs) for i in self.l])
    def shift(self, *args, **kwargs):
        return Images([i.shift(*args, **kwargs) for i in self.l])
    def roi(self, H = None, W = None):
        return Images([i.roi(H,W) for i in self.l])
    def pad(self, *args, **kwargs):
        return Images([i.pad(*args, **kwargs) for i in self.l])

    def times(self, *args, **kwargs):
        return Images([i.times(*args, **kwargs) for i in self.l])
    def vscale(self, *args, **kwargs):
        return Images([i.vscale(*args, **kwargs) for i in self.l])

    def fft(self, *args, **kwargs):
        return Images([i.fft(*args, **kwargs) for i in self.l])
    def canny(self, sigma=1.0, low_threshold=0.1, high_threshold=0.2, mask=None):
        return Images([i.canny(sigma, low_threshold, high_threshold, mask) for i in self.l])
    def binarize(self, *args, **kwargs):
        return Images([i.binarize(*args, **kwargs) for i in self.l])
    def dilate(self, *args, **kwargs):
        return Images([i.dilate(*args, **kwargs) for i in self.l])



########################################################
### Mask class and subclasses, for detection of patterns
### in images.
### 

class Mask(Image):
        
    def init(self, mask, scale = None, rotation = None):
        '''
        Decomposes the 'mask' (2-D ndarray) into positive and negative
        layers, which are then inserted into Red [0] and Blue [2] layers
        of the internal 3-D image.
        In this way, the mask can be displayed easily, like an image.
        '''
        a = mask
        R = (a + abs(a)) / 2        # positive values preserved, negative zeroed out
        G = zeros_like(a)
        B = (abs(a) - a) / 2        # negative values preserved (turned positive), positive zeroed out
        self.a = rgb(R, G, B)
        
        self.scale = scale
        self.rotation = rotation
        self.isImage = True
        self.isMask = True          # a type indicator, more reliable than 'isinstance' with ipython reloading  
    
    def _show(self, scale = 1.0):
        '''
        Mask is displayed differently than an Image,
        because negative values must be shown properly,
        and not truncated.
        Values for display are multiplied by 'scale'. 
        '''
        a = self.a * scale
        print "Mask.show(), min-max:", np.min(a), np.max(a)
        R = (a + abs(a)) / 2        # positive values preserved, negative zeroed out
        G = zeros_like(a)
        B = (abs(a) - a) / 2        # negative values preserved (turned positive), positive zeroed out
        
        img = rgb(R, G, B)
        img[img > 1.0] = 1.0    # to avoid crazy behavior of imshow() to take out-of-bound values with modulo 1.0, instead of truncating
        return imshow(img)
    
    def mnormalize(self, mask):
        "Normalizes newly constructed mask 'mask'. Modifies the argument."
        idx = np.nonzero(mask)
        mask[idx] -= mean(mask[idx])        # shift all non-zero values to get zero mean
        S = sum(abs(mask))
        if S < 1e-10:
            raise Exception("Mask is either empty or has only one type of region (positive/negative)")
        mask /= S * 0.5                     # make positive and negative parts sum to +1 and -1
        return mask
    def mrotate(self, mask, rot, symm = True):
        """Rotates mask by 'rot' degrees counter-clockwise, with reshape (!). 
        'symm' says if the mask is symmetrical, so that optimization is used when possible.
        After rotation, the matrix is gently truncated from each side, 
        for efficiency of calculations afterwards."""
        if symm:
            if abs(rot) == 90 or abs(rot) == 270:
                return mask.T
            if abs(rot) == 180:
                return mask
        if rot == 0: return mask
        (H,W) = mask.shape[:2]
        mask = rotate(mask, rot)
        cropH = 1 + int(0.1*H)
        cropW = 1 + int(0.1*W)
        mask = imcrop(mask, cropH, cropH, cropW, cropW)
        return mask
    
    def mcrop(self, mask, t = 0.1):
        """
        Crops the 'mask' on each side to make it smaller (for efficiency),
        in such a way that no more than 't' fraction of total mass is removed.
        'mask' is temporarily normalized beforehand, so that positive and negative masses are equal. 
        """
        def cuts(s, limit):
            (l,r) = (0,0)
            cum = 0.0
            for mass in s:
                cum += mass
                if cum > limit: break
                l += 1 
            cum = 0.0
            for mass in reversed(s):
                cum += mass
                if cum > limit: break
                r += 1 
            return (l,r)
        
        maskn = np.abs(self.mnormalize(mask))
        if maskn.ndim > 2:
            maskn = sum(maskn, 2)
        massLimit = t * sum(maskn) / 4.0
        (left,right) = cuts(sum(maskn, 0), massLimit)
        (top,bottom) = cuts(sum(maskn, 1), massLimit)
        return imcrop(mask, top, bottom, left, right)
        
    
class CrispMask(Mask):
    def __init__(self, width = 10, rot = 0):
        '''
        A 2-D crisp (-1/+1) straight line mask composed of a central positive stripe 
        and two negative stripes running vertically and symmetrically on both sides
        (if rotation is 0).
        'width' = width (in pixels) of the positive stripe
        'rot' = rotation angle in degrees
        Length is always 3 x width.
        '''
        # parameters of the mask
        margin = divup(width, 2)
        cols = width + 2*margin
        rows = (width + 1) * 3
        
        # 1-D prototype of the mask (seed mask)
        mask1d = - np.ones(cols)            # negative weights at the margin
        mask1d[margin:-margin] = 1          # positive weights inside
        
        # replication of the mask into 2D; rotation; value normalization
        mask = np.outer(np.ones(rows), mask1d)      # 2D mask = [1,1,...,1]' * mask1d
        mask = self.mrotate(mask, rot)
        mask = self.mnormalize(mask)
        self.init(mask, width, rot)
    
    @classmethod
    def multi(cls, width = [10], rot = [0]):
        '''
        Returns a multi-mask - a battery of masks for all combinations
        of provided parameters. Each parameter can be a list or vector of values.
        Masks are stacked as images in Images object.
        '''
        masks = []
        for w in width:
            for r in rot:
                masks.append(cls(w,r))
        return Images(masks)
        
class SoftMask(Mask):
    def __init__(self, width = 10, rot = 0):
        """
        A 2-D straight line mask with soft profile based on mexican hat function.
        'width' is the desired width of positive region.
        """
        rows = (width + 1) * 3
        #margin = divup(width,2)*3 + 1
        margin = int(ceil(width*1.4 + 1))       # SoftMask has wider margin than CrispMask
        widthT = width + 2 * margin             # total width, including negative stripes

        left = -widthT * 0.5 + 0.5                  # left-most pixel center X
        top = rows * 0.5 - 0.5                      # top-most pixel center Y
        ys = [(top-i)*1j for i in xrange(0,rows)]   # all Y's of all pixel centers
        xs = [left+i for i in xrange(0,widthT)]     # all X's of all pixel centers
        pixels = np.add.outer(ys, xs)
        mask = self.gen(pixels, width)
        
        mask = self.mrotate(mask, rot)
        mask = self.mnormalize(mask)
        self.init(mask, width, rot)

    def rotateSoft(self, rot):
        # unfinished draft of rotating the mask by rotation of generating function
        """
        # every pixel is a 1x1 box in the C plane;
        # depending on oddity of 'width', pixel corners are based at k*n or k*n-0.5 coordinates
        top = rows * 0.5j - 0.5j        # top line of pixel centers of unrotated mask
        right = widthT * 0.5 - 0.5      # right line of ...
        topleft = top - right           # corner pixels (their centers) of unrotated mask
        topright = top + right
        
        rotation = exp((pi/180) * rot * 1j)     # rotation angle from degs to radians and then to complex number
        transform = 1. / (width * rotation)
        
        # matrix of points where we want to calculate mask function ...
        topleft *= rotation         # rotate corners to see what bounding box we shall choose
        topright *= rotation
        
        self.gen(z * transform)
        """
        pass
    
    @staticmethod
    def gen(z, width = 1.0):
        """
        Generating function of the mask: positive region is a vertical
        stripe centered at (0,0), of unit width (from -0.5 to 0.5).
        Argument is a complex number. Defined on entire C plane.
        """
        if isinstance(z, numbers.Number):   # z is single number
            x = z.real
        else:                               # z is ndarray
            x = np.real(z)
        if width != 1.0:
            x /= width
        return mexican(x)

    @classmethod
    def multi(cls, width = [10], rot = [0]):
        masks = []
        for w in width:
            for r in rot:
                masks.append(cls(w,r))
        return Images(masks)
        

