# -*- coding: utf-8 -*-
'''
Routines for Multiple String Alignment.

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import sys, numpy as np
from itertools import izip, groupby
from copy import copy

try:
    from numba import jit
except:
    def jit(f): return f
    print >>sys.stderr, "nifty.text: numba not found, JIT disabled for multiple string alignment align()"


#########################################################################################################################################################
###
###  CHARSET & FUZZY STRING classes
###

class Charset(object):
    chars = None            # list of characters in this charset:     chars[0..N-1] -> char
    classes = None          # dictionary of class IDs for all chars:  classes[char] -> 0..N-1
    
    def __init__(self, chars = None, text = None):
        "'chars': list or string; if a list, it may contain special pseudo-characters in a form of arbitrary string or object."
        
        if chars is None: chars = []
        if text: chars = list(chars) + sorted(set(text))
        
        assert len(set(chars)) == len(chars)        # make sure there are no duplicates in 'chars'
        self.chars = chars
        self.classes = {char:cls for cls, char in enumerate(chars)}
    
    def size(self): return len(self.chars)
        
    def classOf(self, char):
        "Mapping: char -> 0..N-1 or None if 'char' not in charset."
        return self.classes.get(char)
    
    __len__ = size
    __getitem__ = classOf
    
    def encode(self, s, dtype = 'float', weight = 1):
        "Convert a plain string of characters into a list of one-hot numpy vectors encoding class IDs."
        
        from numpy import zeros
        N = len(self.chars)
        
        def encode_one(char):
            freq = zeros(N, dtype = dtype)
            hot = self.classes.get(char)
            if hot is None: raise Exception("Charset.encode(): trying to encode a character (%s) that is not in charset." % char)
            freq[hot] = weight
            return freq
        
        return map(encode_one, s)
        
    def cost_matrix(self, GAP = '_', cost_base = 2, cost_case = 1, cost_gap = 3, cost_gap_gap = 0, dtype = None):
        "Create a parameterized cost matrix for edit distance."
    
        costs = np.array([cost_base, cost_case, cost_gap, cost_gap_gap])
        # print "[cost_base, cost_case, cost_gap, cost_gap_gap]:", costs
        
        # can dtype be int16 to speed up calculations, reduce memory footprint and avoid rounding errors?
        if dtype is None:
            if np.array_equal(costs, costs.astype('int32')):
                dtype = 'int32'
            else:
                dtype = 'float32'
        
        # misclassficiation cost is normally 'cost_base' everywhere except diagonal, and (-cost_base) on diagonal
        D = cost_base * (1 - np.eye(self.size(), dtype = dtype))
    
        # GAP vs. other
        cls_gap = self.classes[GAP]
        D[cls_gap,:] = cost_gap
        D[:,cls_gap] = cost_gap
        D[cls_gap,cls_gap] = cost_gap_gap
    
        # case difference
        if cost_case != cost_base:
            for ch in self.chars:
                if not isinstance(ch, basestring): continue
                lo = self.classes.get(ch.lower())
                up = self.classes.get(ch.upper())
                if lo == up or cls_gap in (lo,up) or None in (lo,up): continue
                D[lo,up] = D[up,lo] = cost_case
    
        assert 0 <= D.min()
    
        return D
    
        
class FuzzyString(object):
    """
    A string of "fuzzy characters", each being a probability/frequency distribution over a predefined charset.
    
    >>> charset = Charset('abc')
    >>> fuzzy = FuzzyString('aabccc', charset, dtype = int)
    >>> list(fuzzy.chars[2])
    [0, 1, 0]
    >>> fuzzy += 'aaa'
    >>> list(fuzzy.chars[-1])
    [1, 0, 0]
    >>> fuzzy[0] == 'a' and 'a' == fuzzy[0]
    True
    >>> fuzzy[0] != 'a' or 'a' != fuzzy[0]
    False
    >>> fuzzy[0] == 'b'
    False
    >>> fuzzy[0] != 'b'
    True
    >>> fuzzy.discretize()
    'aabcccaaa'
    >>> fuzzy[::-1].discretize()
    'aaacccbaa'
    >>> fuzzy[::2].discretize()
    'abcaa'
    >>> FuzzyString.merge(fuzzy[::2], 'aacc', norm = False).chars
    [array([2, 0, 0]), array([1, 1, 0]), array([0, 0, 2]), array([1, 0, 1]), array([1, 0, 0])]
    >>> FuzzyString.merge(fuzzy[::2], 'aacc', norm = True, dtype = float).chars
    [array([ 1.,  0.,  0.]), array([ 0.5,  0.5,  0. ]), array([ 0.,  0.,  1.]), array([ 0.5,  0. ,  0.5]), array([ 1.,  0.,  0.])]
    """
    
    # __isfuzzy__ = True      # flag to replace isinstance() checks with hasattr()
    
    charset = None          # Charset instance that defines a char-class mapping: char -> 0..N-1
    chars = None            # list of numpy vectors: chars[pos][c] = probability/frequency of character class 'c' on position 'pos' in string
                            # kept as a list, not monolithic 2D array, to enable fast edit operations: character insertion/deletion;
                            # you should treat each array, chars[pos], as IMMUTABLE (!) and make a copy
                            # whenever particular fraquency values need to be modified. The `chars` list itself is mutable (!).
    dtype = None
    
    def __init__(self, text = '', charset = None, dtype = 'int32', chars = None, weight = 1):
        "Convert a crisp string 'text' to fuzzy."
        assert charset is not None
        self.charset = charset
        self.dtype = dtype
        self.chars = charset.encode(text, dtype, weight) if chars is None else chars
        
    def copy(self):
        "Shallow copy of self."
        return copy(self)
        
    def convert(self, text):
        "Create a new FuzzyString, like this one (same charset and dtype), but with a different plain text."
        assert isinstance(text, basestring)
        return FuzzyString(text, self.charset, dtype = self.dtype)
    
    def new(self):
        "Create a FuzzyString like this one (same charset and dtype), but with empty text."
        return self.convert('')
        
    def append(self, other):
        "Append a char or a string, crisp or fuzzy, to the end of this string."
        self.chars = self._concat_R(other)
        
    def discretize(self, minfreq = None, unknown = None):
        """
        On each position in `chars` pick the first most likely crisp character and return concatenated as a crisp string.
        Optionally, apply minimum frequency threshold, if not satisfied insert `unknown` character.
        """
        if minfreq is None:
            return ''.join(self.charset.chars[freq.argmax()] for freq in self.chars)
        else:
            if unknown is None: raise Exception('FuzzyString.discretize(): `unknown` character is missing')
            return ''.join(self.charset.chars[freq.argmax()] if freq.max() >= minfreq else unknown for freq in self.chars)

    def regexify(self, minfreq = 0.0, maxchars = 3, GAP = None, merge = True, merge_stop = [], _escape = set(r'.[]{}()|?\\^$*+-')):
        """
        Encode this FuzzyString as a regex pattern, where alternative characters (freq > maxfreq) on each position
        are encoded as character sets [ABC], uncertainties (all freq <= minfreq) are replaced with a dot '.',
        gaps are converted to optional markers '?' and repeated code points are merged (if merge=True)
        to repetitions {m,n}. If merge_stop is given, characters (code points) from merge_stop are excluded from merging.
        """
        from nifty.math import np_find
        charset_chars = self.charset.chars
        GAP_cls = self.charset.classOf(GAP)
        
        def escape(char): return '\\' + char if char in _escape else char

        codes = []
        modes = []
        
        last_code = None            # recent regex code, pending to be emitted
        last_rep = (0,0)            # (min,max) repetition of the last code
        
        for freq in self.chars:
            idx = list(np_find(freq > minfreq))
            gap = bool(GAP and GAP_cls in idx)              # gap=True if the current character(s) is optional, or there are no characters (skip)
            if gap: idx.remove(GAP_cls)
            
            n = len(idx)
            if n == 0 or n > maxchars:
                # if gap: continue                            # no characters, only a gap? skip without emitting any regex code
                code = '.'
            elif n == 1:
                char = charset_chars[idx[0]]
                code = escape(char)
            else:
                chars = [escape(charset_chars[i]) for i in idx]
                code = '[%s]' % ''.join(chars)
            
            # if code == last_code:
            # if gap: code += '?'
            
            mode = '?' if gap else ''
            codes.append(code)
            modes.append(mode)
        
        if not merge:
            return ''.join(c + m for c, m in izip(codes, modes))
        
        # merge repetitions of the same code
        pos = 0
        codes_final = []
        
        for code, code_group in groupby(codes):
            
            code_group = list(code_group)
            k = len(code_group)
            assert k > 0
            
            mode_group = modes[pos:pos+k]
            pos += k

            # in simple case (low `k`) or character from merge_stop, just copy original <code,mode> pairs to output, no merging
            if k <= 2 or code in merge_stop:
                for c, m in izip(code_group, mode_group):
                    codes_final.append(c + m)
                continue
                
            # merge modes and convert to {repmin,repmax} pair
            repmin = repmax = 0
            for mode in mode_group:
                repmin += int(mode is not '?')
                repmax += 1
            
            repmax = k
            repmin = k - len([m for m in mode_group if m is '?'])           # subtract the no. of optional '?' characters

            # output a single `code` token with appropriate `mode` as obtained from merge
                
            # if repmin == repmax == 1: mode = ''
            # elif repmin == 0 and repmax == 1: mode = '?'
            if repmin == repmax:   mode = '{%s}' % repmax
            elif repmin == 0:      mode = '{,%s}' % repmax
            else:                  mode = '{%s,%s}' % (repmin, repmax)
                
            codes_final.append(code + mode)
        
        pattern = ''.join(codes_final)
        return pattern

    @staticmethod
    def merge(*strings, **params):
        """
        On each position, add corresponding char frequencies/probabilities of fuzzy1 and fuzzy2,
        and normalize to unit sums if norm=True. Return as a new FuzzyString.
        Default params: charset=None, dtype=None, norm=False, weights=None.
        """
        weights = params.pop('weights', None)
        charset = params.pop('charset', None)
        dtype = params.pop('dtype', None)
        norm = params.pop('norm', False)
        
        if weights is not None and len(weights) != len(strings): raise Exception("The number of weights (%s) and strings (%s) differ." % (len(weights), len(strings)))
        
        # infer charset and dtype
        if dtype is None:
            dtypes = [np.dtype(s.dtype) for s in strings if hasattr(s, 'dtype')]
            dtype = max(dtypes) if dtypes else None
            
        if charset is None:
            for s in strings:
                if not isinstance(s, basestring):
                    charset = s.charset
                    break
        
        if charset is None: raise Exception("Cannot infer charset of strings to be combined")
        if dtype is None: raise Exception("Cannot infer dtype of strings to be combined")

        # check compatibility and convert plain strings to fuzzy
        def validate(s):
            if isinstance(s, basestring): return FuzzyString(s, charset = charset, dtype = dtype)
            if not s.charset == charset: raise Exception("Trying to combine FuzzyStrings with different charsets")
            # if not np.dtype(s.dtype) <= dtype: raise Exception("Trying to combine FuzzyStrings with incompatible numeric types: %s, %s" % (s.dtype, dtype))
            return s
        
        fuzzy = map(validate, strings)
        
        # combine numpy arrays on each char position
        v = len(charset)
        n = max(len(s) for s in fuzzy)
        chars = [np.zeros(v, dtype) for _ in xrange(n)]
        
        for i, s in enumerate(fuzzy):
            schars = s.chars
            w = weights[i] if weights is not None else 1
            for j in xrange(len(schars)):
                chars[j] += schars[j] * w
        
        # normalize?
        if norm: chars = [freq / freq.sum() for freq in chars]
        
        return FuzzyString(chars = chars, charset = charset, dtype = dtype)
        
    def dist(self, other, degree = 1):
        "Compute 1-norm or 2-norm distance between frequency vectors of self and 'other', summed up over all characters."
        assert self.charset == other.charset
        
        chars1 = self.chars
        chars2 = other.chars
        n1, n2 = len(chars1), len(chars2)
        
        if n1 == n2 == 1:             # most typical case, handled separately for speed
            if degree == 1: return sum(np.abs(chars1[0] - chars2[0]))
            if degree == 2: return sum((chars1[0] - chars2[0]) ** 2) ** 0.5
        
        if n1 != n2:
            if n2 < n1:
                chars1, chars2 = chars2, chars1
                n1, n2 = n2, n1
            assert len(chars1) < len(chars2)
            
            chars1 = copy(chars1)
            chars1 += [np.zeros_like(chars2[0])] * (n2 - n1)

        if degree == 1:
            _dist = lambda f1, f2: sum(np.abs(f1 - f2))
        elif degree == 2:
            _dist = lambda f1, f2: sum((f1 - f2) ** 2) ** 0.5
        else:
            raise Exception("Unknown norm type: %s" % degree)
            
        return sum(_dist(freq1, freq2) for freq1, freq2 in izip(chars1, chars2))
        
    def mismatch(self, other, degree = 1, is_basestring = None):
        """
        Like dist(), but handles only 1-letter strings, and 'other' can be a plain string.
        If frequency vectors are normalized to unit sum, the distance returned is guaranteed to lie in <0.0,1.0> range.
        """
        assert len(self.chars) == len(other) == 1 and degree in (1,2)
        freq = self.chars[0]
        
        if is_basestring or (is_basestring is None and isinstance(other, basestring)):
            cls = self.charset.classes[other]
            # diff = freq.copy()
            # diff[cls] -= 1
            if degree == 1: return (sum(freq) - freq[cls] + np.abs(freq[cls]-1)) * 0.5                  # same as: sum(abs(diff)) * 0.5
            if degree == 2: return ((sum(freq**2) - freq[cls]**2 + (freq[cls]-1)**2) * 0.5) ** 0.5      # same as: sum(diff**2 * 0.5) ** 0.5

        diff = freq - other.chars[0]
        if degree == 1: return sum(np.abs(diff)) * 0.5
        if degree == 2: return sum(diff**2 * 0.5) ** 0.5

    def mismatch_crisp(self, other, degree = 1):
        "Like mismatch(), for use when `other` is guaranteed to be a crisp string (basestring)."
        return self.mismatch(other, degree = degree, is_basestring = True)

    def __len__(self):
        return len(self.chars)
        
    def __getitem__(self, pos):
        "Character #pos returned as a FuzzyString (!), not a numpy array. Can be safely compared using == or != "
        dup = copy(self)
        dup.chars = self.chars[pos] if isinstance(pos, slice) else [self.chars[pos]]
        return dup
        
    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, basestring):                                       # crisp string?
            other = FuzzyString(other, self.charset, dtype = self.dtype)
        
        if self.charset is not other.charset: return False
        if len(self.chars) != len(other.chars): return False
        
        for v1, v2 in izip(self.chars, other.chars):
            if v1 is v2: continue
            if not np.array_equal(v1, v2): return False
        
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def _concat_R(self, other, inplace = False):
        "Concat raw `chars` of self and `other`."
        if isinstance(other, basestring):                                       # crisp string?
            if inplace:
                self.chars += self.charset.encode(other, self.dtype)
            else:
                return self.chars + self.charset.encode(other, self.dtype)
        else:                                                                   # or FuzzyString?
            # assert isinstance(other, FuzzyString)
            if self.charset is not other.charset: raise Exception("Trying to add two FuzzyStrings with different charsets")
            if inplace:
                self.chars += other.chars
            else:
                return self.chars + other.chars
        
    def _concat_L(self, other):
        assert isinstance(other, basestring)
        return self.charset.encode(other, self.dtype) + self.chars
        
    def __add__(self, other):
        return FuzzyString(chars = self._concat_R(other), charset = self.charset, dtype = self.dtype)
        
    def __radd__(self, other):
        return FuzzyString(chars = self._concat_L(other), charset = self.charset, dtype = self.dtype)

    def __iadd__(self, other):
        self._concat_R(other, inplace = True)
        return self


#########################################################################################################################################################
###
###  STRING ALIGNMENT
###

def align(s1, s2, mismatch = None, GAP = '_', GAP1 = None, GAP2 = None, dtype = 'float32', return_gaps = False, mismatch_pairs = None):
    u"""
    Align two strings, s1 and s2, and compute their Levenshtein distance using Wagner-Fischer algorithm.
    Each of s1/s2 can be a plain character string (str/unicode) or a FuzzyString.
    Return aligned strings and their distance value: (aligned1, aligned2, distance).
    During alignment, GAP character is inserted where gaps are created.
    The cost of each character-level alignment of chars c1 and c2 is evaluated with mismatch(c1,c2) function,
    or a standard 0/1 equality function if mismatch is None.
    The order of characters for mismatch() is always kept the same: c1 from s1 as 1st argument, c2 from s2 as second;
    hence mismatch() can exploit this information in cost calculation, e.g. to weigh differently characters from s1 and s2.
    
    >>> align('', '', dtype = 'int8')
    ('', '', 0)
    >>> align('align two strings', 'align one string', dtype = 'int8')
    ('align two strings', 'align one string_', 4)
    >>> align('algorithm to align', 'align two', dtype = 'int8')
    ('algorithm t_o align', 'al___ign_ two______', 13)
    >>> align('to align', 'align two', GAP = u'_')
    (u'to align____', u'___align two', 7.0)
    >>> align('to align', 'align two', GAP = u'⫠')
    (u'to align\\u2ae0\\u2ae0\\u2ae0\\u2ae0', u'\\u2ae0\\u2ae0\\u2ae0align two', 7.0)
    >>> charset = Charset(text = 'algorithm to align two_')
    >>> fuzzy1 = FuzzyString('to align', charset, int)
    >>> a1, a2, d = align(fuzzy1, 'align two')
    >>> (a1.discretize(), a2, d)
    ('to align____', '___align two', 7.0)
    >>> fuzzy2 = FuzzyString('align two', charset, int)
    >>> a1, a2, d = align('algorithm to align', fuzzy2)
    >>> (a1, a2.discretize(), d)
    ('algorithm t_o align', 'al___ign_ two______', 13.0)
    >>> a1, a2, d = align(fuzzy1, fuzzy2)
    >>> (a1.discretize(), a2.discretize(), d)
    ('to align____', '___align two', 7.0)
    >>> a1, a2, d = align(FuzzyString(charset = charset, dtype = int), 'to align two')
    >>> (a1.discretize(), a2, d)
    ('____________', 'to align two', 12.0)
    >>> charset = Charset('_abc')
    >>> fuzzy = FuzzyString.merge('abbac', 'abcba', 'bcaaa', charset = charset, dtype = float, norm = True)
    >>> a1, a2, d = align(fuzzy, '')
    >>> (a2, d)
    ('_____', 5.0)
    >>> a1, a2, d = align(fuzzy, 'bba')
    >>> (a2, '%.2f' % d)
    ('bb_a_', '3.33')
    >>> a1, a2, d = align(fuzzy, 'bbaccab')
    >>> (a2, '%.2f' % d)
    ('bbaccab', '5.00')
    """
    from numpy import zeros, array, cumsum

    swap = False
    
    # set a default mismatch() function, depending on the types of s1/s2 strings (crisp or fuzzy)
    if mismatch is None:
        if isinstance(s1, FuzzyString) or isinstance(s2, FuzzyString):
            if isinstance(s1, basestring):
                s1, s2 = s2, s1                             # make a swap to always ensure that mismatch() has a FuzzyString as its first argument
                swap = True
            mismatch = FuzzyString.mismatch_crisp if isinstance(s2, basestring) else FuzzyString.mismatch
        else:
            def mismatch(c1, c2): return int(c1 != c2)      # crisp 0/1 character comparison for plain strings
            
    # convert GAP to a FuzzyString if needed, separately for each string (their types can differ: FuzzyString / basestring)
    if GAP1 is None:  GAP1 = s1.convert(GAP) if isinstance(s1, FuzzyString) else GAP
    if GAP2 is None:  GAP2 = s2.convert(GAP) if isinstance(s2, FuzzyString) else GAP
        
    # memorize char-vs-GAP mismatch costs to avoid repeated calculation of the same values
    mismatch_1_GAP = array([0] + [mismatch(c1, GAP2) for c1 in s1], dtype)
    mismatch_GAP_2 = array([0] + [mismatch(GAP1, c2) for c2 in s2], dtype)
    
    # initialize 'dist' array: dist[i,j] => distance between s1[:i] and s2[:j]
    n1, n2 = len(s1), len(s2)
    dist = zeros((n1+1, n2+1), dtype = dtype)
    dist[:,0] = cumsum(mismatch_1_GAP)                      # fill out row #0 and column #0
    dist[0,:] = cumsum(mismatch_GAP_2)
    
    # edit[i,j] = 0/1/2: indicator of the optimal edit operation on (i,j) position when aligning s1[:i] to s2[:j]
    edit = zeros((n1+1, n2+1), dtype = 'int8')
    edit[:,0] = 1
    
    if mismatch_pairs is None:
        mismatch_pairs = np.zeros((n1, n2), dtype = dtype)
        for i in range(n1):
            c1 = s1[i]
            for j in xrange(n2):
                mismatch_pairs[i,j] = mismatch(c1, s2[j])
    
    # fill out the rest of 'dist' and 'edit' arrays, in a separate function to allow speed optimization with Numba
    _align_loop(dist, edit, mismatch_1_GAP, mismatch_GAP_2, mismatch_pairs)

    i, j = n1, n2
    a1 = s1.new() if isinstance(s1, FuzzyString) else ''
    a2 = s2.new() if isinstance(s2, FuzzyString) else ''
    gaps1 = []
    gaps2 = []
    
    # reconstruct aligned strings a1, a2, from the array of optimal edit operations in each step
    while i or j:
        if edit[i,j] == 0:
            if return_gaps: gaps1.append(len(a1))                   # remember position of the gap being inserted
            a1 += GAP1
            a2 += s2[j-1]
            j -= 1
        elif edit[i,j] == 1:
            if return_gaps: gaps2.append(len(a2))                   # remember position of the gap being inserted
            a1 += s1[i-1]
            a2 += GAP2
            i -= 1
        elif edit[i,j] == 2:
            a1 += s1[i-1]
            a2 += s2[j-1]
            i -= 1
            j -= 1
        
    assert len(a1) == len(a2)
    a1 = a1[::-1]
    a2 = a2[::-1]

    if swap: a1, a2 = a2, a1

    if return_gaps:
        gaps1 = [len(a1)-i-1 for i in reversed(gaps1)]              # reverse the order and values of gap indices, like a1/a2 were reversed
        gaps2 = [len(a2)-j-1 for j in reversed(gaps2)]              # reverse the order and values of gap indices, like a1/a2 were reversed
        # assert all(a1[i] == GAP1 for i in gaps1)
        # assert all(a2[j] == GAP2 for j in gaps2)
        return a1, a2, dist[n1,n2], gaps1, gaps2
    else:
        return a1, a2, dist[n1,n2]
    
@jit
def _align_loop(dist, edit, mismatch_1_GAP, mismatch_GAP_2, mismatch_pairs):
    """
    The main loop of align() function. Separated out from the main function to allow Numba JIT compilation,
    which gives approx. 6x speedup. Matrices `dist` and `edit` are in-out arguments: they are modified in place
    and serve as return variables.
    """
    n1, n2 = dist.shape
    for i in range(1, n1):
        for j in xrange(1, n2):
            cost_left = dist[i, j-1] + mismatch_GAP_2[j]    #mismatch(GAP1, s2[j-1]) #+ suffix_cost(lastchar1[i,j-1],GAP) + suffix_cost(lastchar2[i,j-1],s2[j-1])
            cost_up   = dist[i-1, j] + mismatch_1_GAP[i]    #mismatch(s1[i-1], GAP2)
            cost_diag = dist[i-1, j-1] + mismatch_pairs[i-1, j-1]  #mismatch_idx(i-1, j-1) #mismatch(s1[i-1], s2[j-1])
            
            M = dist[i,j] = min(cost_left, cost_up, cost_diag)
            if M == cost_left: step = 0
            elif M == cost_up: step = 1
            else:              step = 2
            edit[i,j] = step
    
    
def align_multiple(strings, mismatch = None, GAP = '_', cost_base = 2, cost_case = 1, cost_gap = 3, cost_gap_gap = 0,
                   weights = None, charset = None, cost_matrix = None, return_consensus = False, verbose = False):
    """
    Multiple Sequence Alignment (MSA) of given strings through the use of incrementally updated FuzzyString consensus.
    
    >>> align_multiple(['abbac', 'abcbaa', 'bcaa', '  ', 'aaaaaa', 'bbbbbb'], cost_gap = 3)
    ['ab_bac', 'abcbaa', '_b_caa', '_ _ __', 'aaaaaa', 'bbbbbb']
    >>> align_multiple(['aabbcc', 'bbccaa', 'ccaabb'], cost_gap = 2)
    ['aabbcc____', '__bbccaa__', '____ccaabb']
    >>> align_multiple(['aabbcc', 'bbccaa', 'ccaabb'], cost_gap = 3)
    ['aabbcc__', '__bbccaa', 'ccaabb__']
    >>> align_multiple(['aabbcc', 'aadcc', 'aaeecc'], cost_gap = 2)
    ['aabbcc', 'aad_cc', 'aaeecc']
    >>> align_multiple(['aabbcc', 'aadcc', 'aaeecc', ''], cost_gap = 3)
    ['aabbcc', 'aad_cc', 'aaeecc', '______']
    >>> align_multiple(['', '   '], cost_gap = 3)
    ['___', '   ']
    """
    from numpy import array, dot, ones
    
    # safety checks
    if len(GAP) != 1: raise Exception(u"Incorrect GAP value, GAP must be a 1-character string.")
    for s in strings:
        if GAP in s: raise Exception(u"align_multiple(): GAP character '%s' occurs in a string to be matched. Use a different GAP." % GAP)
    
    # create charset & cost matrix
    if charset is None: charset = Charset(text = ''.join(strings) + GAP)
    classes = charset.classes

    if cost_matrix is None:
        cost_matrix = charset.cost_matrix(GAP, cost_base = cost_base, cost_case = cost_case, cost_gap = cost_gap, cost_gap_gap = cost_gap_gap)  #dtype = 'float32'
    if verbose: print 'cost_matrix:\n', cost_matrix
    
    if weights is None: weights = ones(len(strings))
    
    maxlen = max(map(len, strings))
    consensus = FuzzyString(strings[0], charset = charset, dtype = 'float32', weight = weights[0])      #'float32'... 'int16' if maxlen*cost_base < 10000 else
    if verbose: print '#1: ', '%8s' % strings[0]

    def mismatch(fuzzy, crisp):
        cls = classes[crisp]
        freq = fuzzy.chars[0]
        # assert freq.min() >= 0
        return dot(cost_matrix[cls,:], freq)
    
    # def get_mismatch_idx(consensus, s):
    #     "Create mismatch_idx(), a partially pre-computed variant of mismatch() function, to speed up the most critical operation in align() calls."
    #     consensus_chars = consensus.chars
    #     classes_s = [classes[c] for c in s]
    #
    #     def mismatch_idx(i, j):
    #         # print 'i,j:', type(i), i, type(j), j
    #         cls = classes_s[j]
    #         freq = consensus_chars[i]
    #         return dot(cost_matrix[cls,:], freq)
    #
    #     return mismatch_idx

    # @jit
    # def _get_mismatch_pairs_loop(consensus_chars, classes_s):
    #     return dot(consensus_chars, cost_matrix[:,classes_s])
    #     # for j, cls in enumerate(classes_s):
    #     #     mismatch_pairs[:,j] = dot(consensus_chars, cost_matrix[:,cls])

    def get_mismatch_pairs(consensus, s, dtype):
        "Create 2D matrix of mismatch() values for all pairs of letters in both strings, to speed up the most critical operation in align() calls."
        n1, n2 = len(consensus), len(s)

        # classes_s = [classes[c] for c in s]
        # return dot(consensus_chars, cost_matrix[:,classes_s])
        
        # classes_s = array([classes[c] for c in s])
        # return _get_mismatch_pairs_loop(consensus_chars, classes_s)
        
        mismatch_pairs = np.zeros((n1, n2), dtype)
        if n1 == 0 or n2 == 0: return mismatch_pairs
        
        consensus_chars = array(consensus.chars)
        for j, c in enumerate(s):
            mismatch_pairs[:,j] = dot(consensus_chars, cost_matrix[:,classes[c]])
            # for i in xrange(n1):
            #     mismatch_pairs[i,j] = dot(cost, consensus_chars[i])

        return mismatch_pairs
    
    # 1st pass: come up with a stable consensus; strings are accumulated through adding, NO normalization
    for i, s in enumerate(strings[1:]):
        count = i + 1
        weight = sum(weights[:count])
        
        GAP1 = consensus.convert(GAP)
        GAP1.chars[0] *= weight             # rescale GAP weight to account for increased total weight of accumulated `consensus`

        # all frequency vectors in consensus must sum up to `weight`
        assert all(np.abs(freq.sum() - weight) < 0.0001 for freq in consensus.chars + GAP1.chars)

        dtype = 'float32' #cost_matrix.dtype
        mismatch_pairs = get_mismatch_pairs(consensus, s, dtype = dtype)
        
        consensus_aligned, string_aligned, dist, c_gaps, s_gaps = \
            align(consensus, s, GAP1 = GAP1, GAP2 = GAP, return_gaps = True, mismatch = mismatch, mismatch_pairs = mismatch_pairs, dtype = dtype)
        
        consensus = FuzzyString.merge(consensus_aligned, string_aligned, weights = [1,weights[count]], norm = False)
        
        if verbose:
            print '#%-3d a:' % (i+2), '%8s' % string_aligned
            print '     c: %8s' % consensus_aligned.discretize()
            # print '  gaps:', s_gaps
            # print '       ', c_gaps
            print
    if verbose: print
    
    
    consensus = FuzzyString.merge(consensus, norm = True, dtype = 'float32')
    # if verbose: print 'avg:', consensus.chars
    
    # 2nd pass: align strings once again to a semi-fixed consensus;
    # only gaps can be added to consensus and to previously aligned strings
    
    dtype = 'float32'
    GAP1 = consensus.convert(GAP)
    def insert_gap(z, pos): return z[:pos] + GAP + z[pos:]

    aligned = []                                        # output list of aligned strings

    for s in strings:
        
        mismatch_pairs = get_mismatch_pairs(consensus, s, dtype)
        
        c_aligned, s_aligned, dist, c_gaps, s_gaps = \
            align(consensus, s, GAP1 = GAP1, GAP2 = GAP, return_gaps = True, mismatch = mismatch, mismatch_pairs = mismatch_pairs, dtype = dtype)
        
        # new gaps have been inserted into consensus (c_aligned)?
        # backpropagate them to already-aligned strings...
        if len(consensus) != len(c_aligned):
            assert len(c_aligned) - len(consensus) == len(c_gaps) > 0
            consensus = c_aligned
            for gap in c_gaps:
                for k in xrange(len(aligned)):
                    aligned[k] = insert_gap(aligned[k], gap)

        aligned.append(s_aligned)
        if verbose: print s_aligned

        assert all(len(a) == len(consensus) for a in aligned)
                    
    if verbose: print

    return (aligned, consensus) if return_consensus else aligned

    
#########################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()

    print
    print "align_multiple..."
    res = align_multiple([u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', 'abcbaabcaa', '  ', u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', u'The following example shows how the Command-Line Interface can be used to compare three different expressions:'])
    for s in res: print s
    print
    
    from timeit import timeit
    print 'timeit...',
    print timeit("""align_multiple([u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', 'abcbaabcaa', '  ', u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', u'The following example shows how the Command-Line Interface can be used to compare three different expressions:'])""",
                 "from __main__ import align_multiple",
                 number = 10
                 )
    