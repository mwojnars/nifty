#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for standard Machine Learning. Some of them implement scikit-learn API.
"""

import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score


#####################################################################################################################################################
#####
#####  ONE-HOT ENCODING/DECODING in different forms
#####


def one_hot(value, length, dtype = bool):
    """
    Encode a given integer `value` (0..length-1) into a numpy one-hot vector, hot,
    such that hot==0 everywhere except hot[value]==1. By default, the one-hot vector has elements of type bool.
    This can be changed with `dtype`.
    """
    
    assert 0 <= value < length
    hot = np.zeros(length, dtype = dtype)
    hot[value] = 1
    return hot


class OneHot:
    """
    Fixed (predefined, non-trainable) one-hot encoding of a single attribute into a numpy "hot" vector.
    Provides decoding (binary or fuzzy) and supports missing/unknown values represented by None:
    - during encoding, None is encoded as all-zeros vector (note: None can't be used as a valid value);
    - during decoding, None is returned when no output exceeds a given threshold.
    Optionally, if encode_unknown=True, unknown input values can be treated as missing values (encoded as all-zeros).
    Encoding/decoding of multiple values in a single vector is supported through encode_multi() and decode_multi().
    All values must be hashable, for use in an internal dict object.
    """
    
    def __init__(self, values, dtype = bool, encode_unknown = False):
        """
        :param values:
        :param dtype:
        :param encode_unknown: if True, out-of-dictionary values will be treated as missing (None) values,
                               i.e., encoded as all-zeros vectors; otherwise, such values will raise exception
        """
        self.values = list(values)
        self.length = len(values)
        self.dtype  = dtype
        self.encode_unknown = encode_unknown
        
        if None in self.values:
            raise Exception("`None` is encoded as all-zeros vector and so it can't be used as a valid value")
        
        self.map_encode = {val: i for i, val in enumerate(self.values)}
        assert len(self.map_encode) == self.length
        
        
    def encode(self, value = None):
        """
        `value` can be None (missing value), in such case it's encoded as all-zeros vector
        """
        hot = np.zeros(self.length, dtype = self.dtype)
        if value is None:
            return hot
        
        pos = self.map_encode.get(value)
        if pos is None:
            if self.encode_unknown:
                return hot
            else:
                raise Exception("Unknown value (%s) passed to OneHot.encode" % value)
        
        hot[pos] = 1
        return hot
    
    
    def encode_multi(self, values):
        """
        :param values: a list or sequence (can be empty) of values to be encoded into the output vector
        :return: binary vector with 1's on positions corresponding to the given values;
                 raises exception if unknown (out-of-dictionary) value is encountered
        """
        hot = np.zeros(self.length, dtype = self.dtype)
        for val in values:
            pos = self.map_encode[val]
            hot[pos] = 1
        return hot
        
    
    def decode(self, hot, binary = False, none_threshold = 0.0):
        """
        `hot` is either a 0/1 binary (or boolean/int/float) vector where the single occurrence of "1" is seeked;
        or a real-valued vector (if binary=False) where the first occurrence of maximum value is decoded.
        """
        
        assert len(hot) == self.length
        
        if binary:
            poss = np.where(hot == 1)[0]
            assert len(poss) == 1
            pos = poss[0]
        else:
            pos = np.argmax(hot)
            val = hot[pos]
            if val <= none_threshold: return None
            
        return self.values[pos]


    def decode_multi(self, hot, threshold = 0.5):
        """
        Returns a list (can be empty) of dictionary values whose corresponding scores in the `hot` vector exceed the `threshold`.
        """
        
        assert len(hot) == self.length
        poss = np.where(hot >= threshold)[0]
        return [self.values[pos] for pos in poss]



class OneHotFrame(TransformerMixin):
    """
    Trainable Transformer that automatically detects (in fit()) which columns of training DataFrame need one-hot encoding and infers
    mappings for them. Then, it converts a test DataFrame to either a DataFrame (transform() or transform_to_frame())
    or to a numpy array (transform_to_array()), with one-hot encoding of non-numeric (categorical) columns;
    other columns are left unchanged; missing values (None, nan) are encoded as all-zeros vectors.
    The set of encoded attributes and/or some of the mappings can be altered (predefined) manually.
    Below, "value" means any dict-encoded value taken on by a categorical column that undergoes encoding.
    """

    conversion = None           # mapping (dict):  column_name_or_ID -> list_of_values
                                # this mapping can be partial (include selected columns only; others will be added here automatically during fit())
    
    include   = None            # list/set of columns to consider for transformation; all columns if None
    exclude   = None            # no transform for columns in this list/set

    NANs = {np.nan, None}       # things that should be treated as a missing value; missing or unknown values are encoded as all-zeros vectors


    def __init__(self, conversion = {}, exclude = set(), include = None):

        self.conversion = self.conversion or conversion or {}
        self.exclude = self.exclude or exclude or set()
        self.include = self.include or include


    def fit(self, frame, max_vals = None, min_freq = None, sort = "count"):
        """
        :param sort: either "count" (sort values by decreasing frequency) or "alpha" (sort alphabetically)
        """

        columns_all = frame.columns if self.include is None else self.include
        
        for col in columns_all:

            # there's already a mapping defined for the column, or the column should be explicitly ignored? skip...
            if col in self.conversion or col in self.exclude:
                continue

            unique = frame[col].unique()
            vals   = sorted(set(unique) - self.NANs)
            dtype  = np.array(vals).dtype

            # numeric column? no transform...
            if np.issubdtype(dtype, np.number):
                continue

            # find the list of (most frequent) values in this column
            if sort == "count" or max_vals or min_freq:
                vals = list(frame[col].value_counts().sort_values(ascending = False).index)

                if max_vals:
                    vals = vals[:max_vals]
                    if sort == "alpha":
                        vals = sorted(vals)

                if min_freq:
                    vals = list(frame[col].value_counts().index[frame[col].value_counts().values > min_freq])
                    if sort == "alpha":
                        vals = sorted(vals)

            self.conversion[col] = vals
            print("%-20s --> %s" % (col, vals))

        return self


    def transform_to_frame(self, frame, _re_chars = re.compile(r'[^a-zA-Z0-9]+')):

        def string(v):
            return _re_chars.sub('_', str(v))

        cols = []

        for col in frame.columns:

            if col not in self.conversion or col in self.exclude:
                cols.append(frame[col])
                continue

            # create a list of one-hot columns to replace the column 'col'
            for idx, value in enumerate(self.conversion[col]):
                hot_col = (frame[col] == value).astype(int)
                hot_col.name = "%s_%s" % (col, string(value))
                cols.append(hot_col)

        return pd.concat(cols, axis = 1)


    def transform_to_array(self, frame):

        return self.transform_to_frame(frame).values


    def transform(self, frame):
        """Alias for transform_to_frame()."""
        return self.transform_to_frame(frame)


#####################################################################################################################################################
#####
#####  VARIA
#####

class Thresholded(BaseEstimator):
    """
    Wrapper around a Scikit estimator that adds thresholding of the output score (in 0.0-1.0) to a binary decision 0/1.
    The best threshold is found through exhaustive search over a range of possible thresholds.
    """
    
    thresh_min  = 0.0
    thresh_max  = 1.0
    thresh_step = 0.01

    score_function = f1_score       # score function to be maximized; F1 by default
    
    
    def __init__(self, model_obj = None, model_cls = None, **params):
        """
        :param model: instantiated Scikit model
        """
        self.model = model_obj if model_obj is not None else model_cls(**params)
        self.thresh = None
        
    def fit(self, X_train, y_train, thresh = None):
        
        self.model.fit(X_train, y_train)
        
        if thresh is not None:
            self.thresh = thresh
        else:
            self.thresh, _ = self._find_threshold(self.model, X_train, y_train)
        
        return self

    def _find_threshold(self, model, X_train, y_train):

        proba = model.predict_proba(X_train)
        proba = pd.DataFrame(proba)[1]

        best_score = 0
        best_thresh = 0
       
        for threshold in np.arange(self.thresh_min, self.thresh_max, self.thresh_step):
            y_pred = proba.apply(lambda x: int(x > threshold))
            score = self.score_function(y_train, y_pred)
            if score > best_score:
                best_thresh = threshold
                best_score = score
                
        return best_thresh, best_score

    def predict(self, X):
        
        proba = self.predict_proba(X)
        proba = pd.DataFrame(proba)[1]
        y_pred = proba.apply(lambda x: int(x > self.thresh))
        return y_pred

    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    def score(self, X, y, sample_weight = None):
        
        y_pred = self.predict(X)
        return self.score_function(y, y_pred)
    
    def get_params(self, deep = True):
        
        params = self.model.get_params(deep = deep).copy()
        params['model_cls'] = self.model.__class__
        return params
        
    def set_params(self, **params):
    
         model_cls = params.pop('model_cls', None) or self.model.__class__
         self.model = model_cls()
         self.model.set_params(**params)
         return self
    
    
