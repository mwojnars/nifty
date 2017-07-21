'''
Data pipes. 
Allow construction of complex networks (pipelines) of data processing units, that combined perform advanced operations, 
and process large volumes of data (data streams) efficiently thanks to streamed processing
(processing one item at a time, instead of a full dataset).

Features:
- Building pipelines:  pipe1 >> pipe2 >> pipe3 >> ....
- Generating and processing multi-element samples in the pipeline:
   ... >> TUPLE(fun1, fun2, fun3) >> ...
  or
   ... >> (fun1, fun2, fun3) >> ...
  applies all the functions to every input item and generates a tuple of atomic results on output.
  If an input item is a tuple itself, functions are applied selectively to corresponding elements of the tuple.

Related topics:
- Dataflow programming: https://en.wikipedia.org/wiki/Dataflow_programming
- Flow-based programming: https://en.wikipedia.org/wiki/Flow-based_programming

DEPENDENCIES: jsonpickle

---
Typical Pipe's life cycle:

init
| setup
| | open / _prolog
| |  | ...
| |  | iter (process / monitor / accept) OR run
| |  | ...
| | close / report / _epilog
| reset

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import sys, heapq, math, random, numpy as np, jsonpickle, csv, itertools, threading
from copy import copy, deepcopy
from time import time, sleep
from Queue import Queue
from itertools import islice, izip
from collections import OrderedDict

# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages;
# only when executed as a standalone file, for unit tests, do an absolute import
if __name__ != "__main__":
    from .. import util
    from ..util import isint, islist, istuple, isstring, issubclass, isfunction, isgenerator, iscontainer, istype, \
                       classname, getattrs, setattrs, Tee, openfile, Object, __Object__
    from ..files import GenericFile, File as files_File, SafeRewriteFile, ObjectFile, JsonFile, DastFile
else:
    from nifty import util
    from nifty.util import isint, islist, istuple, isstring, issubclass, isfunction, isgenerator, iscontainer, istype, \
                       classname, getattrs, setattrs, Tee, openfile, Object, __Object__
    from nifty.files import GenericFile, File as files_File, SafeRewriteFile, ObjectFile, JsonFile, DastFile


#####################################################################################################################################################
###
###   DATA objects
###

class __Data__(type):
    def __init__(cls, *args):
        type.__init__(cls, *args)
        
        # initialize special attribute __metadata__;
        # set default None values for metadata fields
        if hasattr(cls, '__metadata__') and cls.__metadata__ is not None:
            if isstring(cls.__metadata__):
                cls.__metadata__ = cls.__metadata__.split()
            for attr in cls.__metadata__:
                if not hasattr(cls, attr):
                    setattr(cls, attr, None)


class Data(object):
    """A data item. Keeps the core data value (self.value) + any metadata that is produced
       or needs to be consumed at different stages of a data processing pipeline.
       Typically, metadata attributes record the origin or evolution of a data item,
       or contain ground truth information for training/testing of learning models.
       
       Cells, pipes and pipelines can in general operate on data objects of ANY type, not necessarily Data,
       but in many cases it is more convenient to wrap up original objects in Data class
       so as to annotate them with various additional information.
    """
    __metaclass__ = __Data__
    
    # The core data value; an object of any type
    value = None

    # Name of the attribute holding core data
    __value__ = 'value'

    # An optional list of metadata attributes.
    # Inside a subclass definition, it can be initialized with a string of names, which will be automatically
    # converted to a list of names upon creation of the class.
    # Corresponding class-level attributes are created automatically with default None values if missing.
    # If __metadata__ is undefined (None), all other attributes are treated as metadata...
    __metadata__ = None


    def __init__(self, value = None, *meta, **kwmeta):
        
        if value is not None: self.value = value
        if meta:
            for k, v in izip(self.__metadata__, meta):
                setattr(self, k, v)
        if kwmeta:
            for k, v in kwmeta.iteritems():
                setattr(self, k, v)
    
    def get(self):
        """Return 'value' and all metadata as a tuple, in the same order as in __metadata__,
           or sorted alphabetically by name if __metadata__ is None.
        """
        keys = self.__metadata__
        if keys is None:
            keys = [k for k in self.__dict__.iterkeys() if k != 'value'].sort()
        return (self.value,) + tuple(getattr(self, k) for k in keys)
    
    def meta(self):
        "Return all metadata as a newly created dict object."
        d = self.__dict__.copy()  # fast but sometimes incorrect: when a getter is defined for an attribute
        if 'value' in d: del d['value']
        return d
    
    @staticmethod
    def derived(origin, value):
        "Create a Data item with a new 'value'and all metadata copied from a previous Data item, 'origin'."
        data = Data(value)
        for meta in origin.__dict__.iterkeys():
            v = getattr(origin, meta)
            setattr(data, meta, v)
        return data


class __DataTuple__(type):
    def __init__(cls, *args):
        type.__init__(cls, *args)
        
        # initialize a list of attributes, __attrs__;
        # set default None value for an attribute if a default is missing
        if hasattr(cls, '__attrs__') and cls.__attrs__ is not None:
            if isstring(cls.__attrs__):
                cls.__attrs__ = cls.__attrs__.split()
            for attr in cls.__attrs__:
                if not hasattr(cls, attr):
                    setattr(cls, attr, None)

class DataTuple(object):
    """
    An object that can be used like a tuple, with the object's predefined attributes
    being mapped to fixed positions in a tuple.
    Similar to namedtuple(), but namedtuple is a function not a (base) class,
    so it's impossible to check isinstance(obj, namedtuple).
    """
    __metaclass__ = __DataTuple__
    __attrs__ = []
    
    def get(self):
        "Return all __attrs__ values as a tuple."
        return tuple(getattr(self, k) for k in self.__attrs__)

    def set(self, *args, **kwargs):
        if args:
            for k, v in izip(self.__attrs__, args):
                setattr(self, k, v)
        if kwargs:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def __getitem__(self, pos):
        attrs = self.__attrs__[pos]
        if isinstance(attrs, list):
            return [getattr(self, a) for a in attrs]
        return getattr(self, attrs)

    def __setitem__(self, pos, item):
        return getattr(self, self.__attrs__[pos])

    def __iter__(self):
        for a in self.__attrs__: yield getattr(self, a)

    __init__ = set

    # def apply(self, *fun):
    #     """Returns a function g(x) that applies given functions, fun[0], fun[1], ... to a data item 'x'
    #        and returns the results combined into a DataTuple of the same class as self.
    #        If any fun[i] is None, an identity function is used on a given position.
    #     """
    #     def ident(x): return x
    #
    #     fun = tuple(f or ident for f in fun)            # replace Nones with identity function
    #     cls = self.__class__
    #
    #     def g(x):
    #         return cls(*(f(x) for f in fun))
    #
    #     return g
    

#####################################################################################################################################################
###
###   SPACES and KNOBS
###

class Space(object):
    """Any 1D set of values, finite or infinite, typically used to specify allowed values for a knob.
    Can define a probability density function on the space. None value is forbidden, plays a special role. 
    """
    
    def __len__(self):
        "No. of elements in this space, or None if infinite."
        return None
    
    def __iter__(self):
        "Yields all consecutive values of a given space. If possible, in increasing order, or in circular order starting from a given point (for infinite spaces)."

    def random(self, rand):
        "Returns a random value selected from this space."
        
    def cast(self, value):
        "Convert a 'value' that's possibly from outside this space, to a (possibly) nearest value in this space. None if can't convert."
        
    def move(self, start, step):
        "Try to make a move in the space from point 'start' by increment 'step'. Return target point or None if impossible to move."

class Singular(Space):
    "Space with only 1 value. The value can be of any type, no particular properties are required."
    def __init__(self, value): self.value = value
    def __len__(self): return 1
    def __iter__(self): yield self.value
        

class Nominal(Space):
    "A finite discrete set of nominal values of any type (strings, numbers, ...), with no natural ordering. The ordering of values in the collection is used."
    def __init__(self, *values):
        if not len(set(values)) == len(values): raise Exception("Nominal: values provided are not unique, %s" % values)
        self.values = v = list(values)
        self.indices = {v[i]:i for i in range(len(v))}          # for metric operation on values from the space
    
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        for v in self.values: yield v
    def cast(self, value):
        return value if value in self.values else None 
    def move(self, start, step):
        "'start' is any value present on self.values; 'step' is an integer: no. of steps to be done up or down the self.values list (the list is NOT circular)."
        i = self.indices[start]
        j = i + step
        return self.values[j] if 0 <= j < len(self.values) else None

# class Discrete(Space):
#     "A finite discrete subset of real or integer numbers with natural ordering (unlike Nominal)."
#     def __init__(self, values):
#         self.values = values
    
class Cartesian(Space):
    "Cartesian product of multiple Spaces. A multidimensional set of allowed values for a given combination of knobs. Each multi-value is a tuple."
    
    def __init__(self, *spaces):
        "If any given space is a tuple or list instead of a Space instance, it's wrapped up in Nominal or Singular."
        def wrap(space):
            if isinstance(space, Space): return space
            if not islist(space): return Singular(space)
            if len(space) == 1: return Singular(space[0])
            return Nominal(*space)
            
        self.spaces = [wrap(s) for s in spaces]
    
    def __len__(self):
        prod = 1
        for space in self.spaces:
            d = len(space)
            if d in (0, None): return d
            prod *= d
        return prod

    def __iter__(self):
        n = len(self.spaces)
        if not n: return
        for v in self.subiter(n): yield v
        
    def subiter(self, dim):
        "Yields all combinations of values of the lower-dimensional subspace spanned by self.spaces[:dim]."
        space = self.spaces[dim-1]
        if dim == 1: 
            for w in space: yield (w,)
        else:
            for v in self.subiter(dim-1):
                for w in space: yield v + (w,)
    
    def cast(self, value):
        dims = zip(self.spaces, value)
        value = (d[0].cast(d[1]) for d in dims)
        if None in value: return None
        return value
        
    def move(self, start, step):
        "Moves independently along each dimension. 'start' and 'stop' are n-dimentional tuples or lists. If on any dimension fails, all function fails (None returned)."
        dims = zip(self.spaces, start, step)
        stop = (d[0].move(d[1], d[2]) for d in dims)
        if None in stop: return None
        return stop
        
    
class Knobs(OrderedDict):
    """An ordered dictionary of {name:value} pairs for a number of knobs, representing one particular knobs combination.
    Currently equivalent to OrderedDict, with no custom methods.
    While Spaces are just sets of values, unnamed, knobs are *named*: values are linked to specific names.
    """

class KnobSpace(object):
    "A collection of knobs combinations: each value is a Knobs instance. Similar to Space, but with knob names assigned."

class KGrid(KnobSpace):
    "Knobs produced from a grid (cartesian product) of allowed values specified for each knob/dimension."
    def __init__(self, **knobs):
        self.space = Cartesian(*knobs.values())             # value space of all possible knob values
        self.names = knobs.keys()                           # names of knobs
    def __len__(self): return len(self.space)
    def __iter__(self):
        "Generates knob combinations. Every combination is a list of (name,value) pairs, one pair for each knob."
        for vector in self.space:
            yield Knobs(zip(self.names, vector))
            #yield Knobs([(self.runID, self.startID + i)] + zip(self.names, v))

#####################################################################################################################################################
###
###   DATA CELL
###

class __Cell__(__Object__):
    "Metaclass for generating Cell subclasses. Sets up the lists of knobs and inner cells."
    def __init__(cls, *args):
        super(__Cell__, cls).__init__(cls, *args)
        cls.label('__knobs__')
        cls.label('__inner__')
        #print cls, 'knobs:', cls.__knobs__


class Cell(Object):
    """Element of a data processing network. 
    Typically most cells are Pipes and can appear in vertical (pipelines) as well as horizontal (nesting) relationships with other cells.
    Sometimes, there can be cells that are not pipes - they participate only in vertical relationships 
    and expose their own custom API for data input/output (instead of Pipe's streaming API).
    
    Pipes are elements of horizontal structures of data flow. Typically 1-1 or many-1 relationships.
    Cells are elements of vertical structures of control. Typically 1-1 or 1-many relationships.
    
    Every cell can contain "knobs": parameters configured from the outside by other cells, e.g., by meta-optimizers; input parameters of the cell.
    Knobs have a form of attributes located in a particular cell and identified by cell's name/path and attribute's name.
    Knobs enable implementation of generic meta-algorithms which operate on inner structures without exact knowledge of their identity,
    only by manipulation of knob values.
    
    Things to keep in mind:
    - properties defined inside inner classes, __knobs__ and __inner__, are automatically copied to parent class level 
      after class definition.
      
    INITIALIZATION.
    
    Cell base class provides special mechanism for class initialization to enable transparent setting of knobs during __init__,
    without writing custom __init__ in each subclass that would only make a series of dumb assignments: self.X = X; self.Y = Y; ...
    If the subclass doesn't implement __init__, all arguments passed to initializer are interpreted as knobs.
    This is the case for unnamed arguments, too - they are interpreted as knobs passed in the order of their declaration
    in the subclass'es __knobs__ property.
    
    Thus, Cell subclasses can perform initialization in 3 different ways:
    1) Leave it entirely to Cell base initialization mechanism that interpretes all arguments as knobs.
    2) Additionally to (1), override init() method to perform custom post-processing of knob values.
       This shall only be used in final classes, not in classes intended for further subclassing,
       so that calling super().init() can be safely omitted in all final classes.
    3) The subclass can override __init__, possibly calling initKnobs() and init() inside, like Cell.__init__ does,
       but adding custom code in-between; or calling super().__init__(). 
       This is a recommended approach for base classes which are intended for subclassing, but still need to do 
       custom post-processing of knob values or other initialization during __init__.
    
    Note that many types of initialization can be done also in Pipe's setup(), _prolog() or open() - often these places
    are more suitable than __init__, so most classes don't need custom init/__init__ at all.
    
    SERIALIZATION.
    
    Serialization is implemented by inheriting from nifty.util.Object class, 
    which provides __getstate__ and __setstate__ methods.
    You can serialize a cell by calling, for instance: dast.dump(cell, afile).
    
    """
    __metaclass__ = __Cell__

    __knobs__  = []         # names of attributes that serve as knobs of a given class; list, string, or class __knobs__: ...
    __inner__ = []

    name = None             # optional label, not necessarily unique, that identifies this cell instance or a group of cells in signal routing
    owner = None            # the cell which owns 'self' and creates an environment where 'self' lives; typically 'self' is present in owner.__inner__
    verbose = None          # pipe-specific setting that controls how much debug information is printed during pipe operations

    printlock = threading.Lock()          # mutual exclusion of printing (on stdout); assign NoneLock to switch synchronization off

    def __init__(self, *args, **kwargs):
        """The client can pass knobs already in __init__, without manual call to setKnobs. 
        Unnamed args passed down to custom init(), but first interpreted as knobs passed in the order of their declaration
        in the class."""
        self.initKnobs(*args, **kwargs)
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        """Override in subclasses to provide custom initialization, without worrying about calling super __init__.
        Upon call, knobs are already set, so init() can post-process them or use their values to set other attributes.
        It's recommended to read knobs values from 'self' not 'knobs'.
        The 'knobs' dict is given here only to enable passing of regular non-knob arguments, 
        not declared as __knobs__ of the class.
        """

    def initKnobs(self, *args, **knobs):
        self.name = knobs.pop('name', None)         # special knob, not listed in __knobs__
        unnamed = self._setUnnamedKnobs(*args)
        if not knobs: return
        for name in unnamed:                        # check that no unnamed knob value will be overriden by a named one
            if name in knobs: 
                raise Exception("Knob %s passed as both named and unnamed argument to __init__ of %s" % (name, self))
        self.setKnobs(knobs)
        
    def _setUnnamedKnobs(self, *args):
        """Interprets unnamed arguments passed to __init__ as knobs listed in the same order as on the __knobs__ list,
        and assigns them to attributes of self. Returns names of assigned knobs.
        """
        if not args: return []
        if len(args) > len(self.__knobs__): 
            raise Exception("More unnamed arguments than knobs passed to _setUnnamedKnobs() of %s", self)
        knobs = dict(zip(self.__knobs__, args))
        #print self, knobs
        self.setKnobs(knobs)
        #print self.__dict__
        return knobs.keys()

    def copy(self, deep = True):
        """Shorhand for copy(self) or deepcopy(self).
        In Pipes, 'source' is excluded from copying and the returned pipe has source UNassigned, 
        even when deep copy (configured in __transient__ and handled by Object.__getstate__)."""
        if deep: 
            #if self.source: raise Exception("Deep copy called for a data pipe of %s class with source already assigned." % classname(self))
            return deepcopy(self) 
        return copy(self)
        
    def copy1(self):
        """Copying 1 step deeper than a shallow copy. Copies all attribute values, too, so inner collections 
        (lists/dicts) of pipes/knobs can be modified afterwards without affecting original ones."""
        res = copy(self)
        d = res.__dict__
        for k, v in d.iteritems():
            d[k] = copy(v)
        return res
        
    def dup(self, knobs = {}, strict = False, deep = True):
        "copy() + setKnobs() combined, for easy generation of duplicates that differ only in knobs setting."
        dup = self.copy(deep)
        dup.setKnobs(knobs, strict)
        return dup
        
    def getKnobs(self):
        "Dict with current values of all the knobs of 'self'."
        return {name:getattr(self,name) for name in self.__knobs__}

    def setKnobs(self, knobs = {}, strict = False, **kwargs):
        """Set given dict of 'knobs' onto 'self' and sub-cells. In strict mode, all 'knobs' must be recognized 
        (declared as knobs) in self.
        Subclasses should treat objects inside 'knobs' as *immutable* and must not make any modifications,
        since a given knob instance can be reused multiple times by the client.
        """
        if kwargs:
            knobs = knobs.copy()
            knobs.update(kwargs)
        #print id(self), self, '...'
        #print "  setKnobs:", knobs
        for cell in self.inner(): cell.setKnobs(knobs, strict)          # walk the tree of all inner cells first
        if not self.__knobs__ and not strict: return
        for address, value in knobs.iteritems():
            attr = self.findAttr(address)
            if attr is None:
                if strict: raise Exception("Knob '%s' not present in '%s'" % (name, classname(self, full=True)))
            else:
                setattr(self, attr, value)
        #print "  getKnobs:", self.getKnobs()
    
    def findAttr(self, addr):
        """Find attribute name in 'self' that corresponds to a given knob. None if the knob doesn't belong to self 
        (mismatch of class/instance specifier), or is undefined in self."""
        obj, attr = addr.rsplit('.', 1) if '.' in addr else ('', addr)          # knob's owner object name is the substring up to the last dot '.' of 'name'
        if obj and obj not in (classname(self), classname(self, full=True)): return None
        if attr not in self.__knobs__: return None
        return attr
    
    def inner(self):
        """A generator that yields all cells contained in (owned by) this one. 
        Used in methods that need to apply a given operation to all subcells."""
        for name in self.__inner__: 
            cell = getattr(self, name, None)
            if cell is None: continue
            if islist(cell):
                for item in cell: yield item
            else:
                yield cell
    
    def __str__(self):
        if not self.name: return classname(self)
        return "%s [%s]" % (self.name, classname(self))

    def _pushTrace(self, pipe): trace.push(pipe)
    def _popTrace (self, pipe): trace.pop(pipe) if trace else None      # 'if' necessary for unit tests to avoid strange error messages


class OpenPipes(list):
    """For debugging. List of pipes whose __iter__ is currently running. Usually these pipes form a chain and their __iter__ 
    executions form a stack, but this not always must be the case, e.g. with multi-input pipes 
    - that's why pop() takes a pipe object again.
    """
    def push(self, pipe):
        self.append(pipe)
    def pop(self, pipe):
        i = len(self)
        while i:
            i -= 1
            if self[i] is pipe:
                list.pop(self, i)
                return
        raise Exception("OpenPipes.pop, trying to pop an object that's not present on the list: %s" % pipe)
     
    def __str__(self):
        lines = ["Open pipes (%d):" % len(trace)]
        for i, pipe in enumerate(self):
            lines.append("%d %s" % (i+1, pipe))
        return '\n'.join(lines)    

trace = OpenPipes()             # for debugging

    
#####################################################################################################################################################
###
###   DATA PIPE
###

class __Pipe__(__Cell__):
    "Enables chaining (pipelining) of pipe classes, not only instances. >> and << return a Pipeline instance."
    def __rshift__(cls, other):
        if other is RUN: Pipeline(cls).run()
        else: return Pipeline(cls, other)
        
    def __lshift__(cls, other):
        return cls() << other
        if other is PIPE: return Pipeline(cls)
        return Pipeline(other, cls)


class Pipe(Cell):
    """Base class for data processing objects (pipes) that can be chained together to perform streamed data processing, 
    each one performing an atomic operation on the data received from preceding pipe(s), 
    or being an initial source of data (loader, generator).
    Every iterable type - a collection, a generator or any type that implements __iter__() 
    - can be used as a source pipe, too.
    
    Features of data pipes:
    - operator '>>'; can use classes, not only instances, with >>; can use collections, functions, ... with >>
      See also PIPE and RUN tokens and Pipeline class.
    - attributes managed by base classes during execution, can be accessed by subclasses:
        count, yielded, _iterating
    
    See Cell base class for information about knobs and serialization.
    """
    __metaclass__ = __Pipe__
    __transient__ = "source"    # don't serialize 'source' attribute and exclude it from copy() and deepcopy();
                                # __transient__ is handled by Object.__getstate__
    
    source    = None            # source Pipe or iteratable from which input data for 'self' will be pulled
    sources   = None            # list of source pipes; used only in pipes with multiple inputs, instead of 'source'
    count     = None            # no. of input items read so far in this iteration, or 1-based index of the item currently processed; calculated in most standard pipes, but not all
    yielded   = None            # no. of output items yielded so far in this iteration, EXcluding header item

    _created   = False          # has the object been initialized already, in setup()? most pipes have empty setup(), only more complex ones use it for creation of internal structures
    _iterating = False          # flag that protects against multiple iteration of the same pipe, at the same time
    

    def setup(self):
        """Delayed initialization of the model, called just before the first iteration (and before open()).
        Delaying initialization and doing it in setup() instead of __init__() enables knobs 
        to be configured in the meantime, *after* and separately from instantiation of the object.
        If you have to call both setKnobs() and setup(), it's better to first call setKnobs.
        """

    def reset(self):
        """Reverse of setup(). Clears internal structures and brings the pipe back to an uninitialized state, 
        like if setup() were never executed. Knobs and other static settings should be preserved!
        If overriding in subclasses, remember to set self._created=False at the end.
        """

    def open(self):
        """Overridden by end client subclasses to perform custom per-iteration initialization.
        Called at the beginning of __iter__(), in _prolog().
        Classes intended for further subclassing shall override _prolog(), not open(),
        with a call to super(X,self)._prolog() at the end of overriding method.
        If open() returns a not-None result, it is yielded from __iter__ as the 1st data item.
        Typically, this is used for passing header data, like names of columns for CSV printer.
        """
    def close(self):
        """Overridden by end client subclasses to perform custom per-iteration clean-up.
        Called at the end of __iter__(), in _epilog().
        Classes intended for further subclassing shall override _epilog(), not close(),
        with a call to super(X,self)._epilog() at the beginning of overriding method.
        """
    
    def __iter__(self):
        """Main method of every data pipe. Pulls data from the source(s), processes and yields results of processing 
        one by one. Implicitly calls self._prolog() before pulling/yielding the 1st item, and self._epilog()
        at the end of iteration (but only when at least one attempt to pull an item is performed!).
        Subclasses should override either iter() or __iter__(), in the latter case the subclass
        is responsible for calling _prolog and _epilog. Generic subclasses also allow to override open() and close() 
        to perform custom initialization and clean-up on every cycle of iteration.
        """
        header = self._prolog()
        if header is not None: yield header
        try:
            for item in self.iter(): 
                self.yielded += 1
                yield item
        except GeneratorExit, ex:                       # closing the iterator is a legal way to break iteration
            self._epilog()
            raise
        self._epilog()

    def iter(self):
        """Generator that yields subsequent items of the stream, just like __iter__() would do.
        If you subclass Pipe directly rather than via specialized base classes (Transform, Monitor, Filter, ...),
        it's typically better to override iter() not __iter__(), to allow for implicit calls to _prolog/_epilog
        at the start/end of iteration.
        """
        raise NotImplementedError

    def _prolog(self):
        if not self._created:               # call reset/setup() if needed
            self.setup()
            self._created = True
        if self._iterating: raise Exception("Data pipe (%s) opened for iteration twice, before previous iteration has been closed" % self)
        self._iterating = True
        self.yielded = 0
        header = self.open()
        self._pushTrace(self)
        return header

    def _epilog(self):
        """_prolog and _epilog must always be invoked together, otherwise there will be a mismatch between
        opens & closes, and between _pushTraces & _popTraces.
        """
        self._popTrace(self)
        self.close()
        del self._iterating                 # could set self._iterating=False instead, but deleting is more convenient for serialization

    def run(self):
        """Pull all data through the pipe, but don't yield nor return anything. 
        Typically used for Pipelines which end with a sink and only produce side effects."""
        for item in self: pass

    def fetch(self, limit = None):
        """
        Pull all data (or up to 'limit' items if limit != None) through the pipe and return as a list.
        If limit=0, an empty list is returned immediately, otherwise open/close and _prolog/_epilog
        are called and an actual attempt to pull an item through the pipe is made.
        """
        if limit is None:
            return list(self)
        else:
            return list(islice(self, limit))

    def fetch1(self):
        "Return 1 item from the pipe, or None if the pipe is empty."
        items = self.fetch(1)
        return items[0] if items else None

    def __rshift__(self, other):
        """'>>' operator overloaded, enables pipeline creation via 'a >> b >> c' syntax. Returned object is a Pipeline.
        Put RUN of FETCH token at the end: a >> b >> RUN to execute the pipeline immediately after creation."""
        # 'self' is a regular Pipe; specialized implementation for Pipeline defined in the subclass
        if other is RUN:
            Pipeline(self).run()
        else:
            return Pipeline(self, other)

    def __lshift__(self, other):
        if other is PIPE: return Pipeline(self)
        return Pipeline(other, self)

    def __add__(self, other):
        """Addition '+' operator creates a Union node that performs *sequential concatenation* of data streams from both sources into a single output stream.
        Note that in Python shifting operations have lower priority than arithmetic operations, so A+B >> C is interpreted as (A+B) >> C, as expected!"""
        return Union(self, other)

    def __batch_iter__(self, maxsize = 100):
        "Like __iter__, but yields batches of data items instead of single items. Every batch is a list."
        batch = []
        for item in self.__iter__():
            if len(batch) >= maxsize: 
                self.yielded += len(batch)
                yield batch
                batch = []
            batch.append(item)
        if batch: yield batch

    def stats(self):
        """String with detailed statistics of the no. of input & output items that passed through the pipe in the current
        or last iteration. For reporting and debugging. Can be overridden in subclasses to provide details 
        specific to the subclass."""
        return "No. of input/output data items of %s: %s, %s" % (self, self.count, self.yielded)

#     def getSpec(self):
#         "1-line string with technical specification of this pipe: its name and possibly values of its knobs etc."


#####################################################################################################################################################
###
###   TOKENS
###

class Token(Pipe):
    """
    Base class for special predefined pipe objects that can be used in pipelines
    to control pipeline creation and execution.
    """

class _PIPE(Token):                                                         # class of PIPE
    def __rshift__(self, other): return Pipeline(other)

class _RUN(Token): pass

class _FETCH(Token):
    def __init__(self, limit = None): self.limit = limit
    def __call__(self, limit = None): return _FETCH(limit)

class _OFF(Token):                                                          # class of OFF
    def __div__(self, other): return repr(other)
    __truediv__ = __rdiv__ = __div__

class _ON(Token):                                                           # class of ON
    def __div__(self, other): return other
    __truediv__ = __rdiv__ = __div__


# Starts a pipeline. For easy appending of other pipes with automatic type casting: PIPE >> a >> b >> ...
# Doesn't do any processing itself (is excluded from the pipeline).
#
PIPE = _PIPE()

# When put at the end of a pipeline (a >> b >> ... >> RUN) indicates that it should be executed now.
#
RUN = _RUN()

# When used at the end of a pipeline (a >> b >> ... >> FETCH) indicates that it should be
# executed now and all output items should be returned as a list. This is convenient
# for materializing a pipeline into a regular data set:
#    data = (a >> b >> ... >> FETCH)
# which is equivalent to:
#    data = list(a >> b >> ... )
# In some case the former is more readable than the latter.
# FETCH can also be called with a parameter 'limit', like in:
#    a >> b >> ... >> FETCH(100)
# This creates another token object with self.limit defined indicating the maximum no. of values
# that should be fetched from the pipe and returned in a list.
#
FETCH = _FETCH()

# When used in division: OFF/pipelike_object or pipelike_object/OFF - turns off a given element of a pipeline by converting it to a string.
# To be used for easy commenting out of individual pipes in a pipeline.
#
OFF = _OFF()

# Complementar to OFF. When used in division: ON/pipelike_object or pipelike_object/ON - does nothing (that is, turns an element ON again).
# Convenient when you need to turn an element off and on many times: you will first put /OFF, than change it to /ON, and so on.
#
ON = _ON()


# A token used inside Tuple(...) to indicate that a corresponding input element should be passed unchanged
# to the output tuple, without any transformation (empty transform).
FORWARD = object()


#####################################################################################################################################################
###
###   FAMILIES
###

"""
class Operator(Pipe):
    "Any flat pipe: without inner pipes. Can be unary or multiary."

class Container(Pipe):
    "A pipe tha contains inner pipe(s)."
"""

#####################################################################################################################################################
###
###   FUNCTIONAL PIPES
###

class _Functional(Pipe):
    "Base class for Transform, Monitor and Filter. Implements wrapping up a custom python function into a functional pipe."

    class __knobs__:
        fun = None              # plain python function (or lambda) that implements class functionality, if core method not overriden
    
#     def __init__(self, *args, **knobs):
#         "Inner function - if present - must be given as 1st and only unnamed argument. All knobs given as keyword args."
#         if args: self.fun = args[0]
#         super(_Functional, self).__init__(**knobs)
    
class Transform(_Functional):
    """Plain item-wise processing & filtering function, implemented in the subclass by overloading process() 
    and possibly also open/close(). Method process(item) should return an item to be yielded,
    or None to indicate that the item should be dropped entirely (filtered out).
    For operators that only perform filtering, with no modification of items, use Filter class instead.
    """
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header
        if not self.source:
            raise Exception("No source pipe connected (self.source=%s) in a transformative pipe: <%s>" % (self.source, self))
        
        try:
            self.count = 0
            for item in self.source: 
                self.count += 1
                res = self.process(item)
                if res is not None:
                    self.yielded += 1
                    yield res
                # if res is not False:
                #     self.yielded += 1
                #     yield item if res is None else res
        except GeneratorExit, ex:
            self._epilog()
            raise
        #except Exception, ex:
        #    self._sealStackTrace()
        #    raise
        self._epilog()
        
    def process(self, item):
        "Return modified item; or None, interpreted as no result (drop item). Subclasses can read self.count to get 1-based index of the current item."
        return self.fun(item)
    
    def __call__(self, item):
        return self.process(item)
    
    
class Monitor(_Functional):
    """A pipe that (by assumption) doesn't modify input items, only observes the data and possibly produces 
    side effects (collecting stats, logging, final reporting etc). 
    Monitor provides subclasses with an output stream, self.out, that's configurable by the client
    in the 1st argument to __init__ (stdout by default).
    Subclasses override monitor(item) and possibly open/close() or report(). 
    The monitoring function can also be passed as the 2nd argument to __init__."""

    class __knobs__:
        outfiles = None     # list of: <file> or name of file, where logging/reporting should be printed out
        mode     = 'wt'     # mode to be used for opening files

    out = None              # the actual file object to be used for all printing/logging/reporting in monitor() and report();
                            # opened in _prolog(), can stay None if the pipe doesn't need output stream
    mustclose = False       # if True, it means that 'out' was opened here (not outside) and it must be closed here, too
    
#     def __init__(self, outfiles = None, *args, **knobs):
#         """'outfiles' can be: None or '' (=stdout), or a <file>, or a filename, or a list of <file>s or filenames 
#         (None, '' and 'stdout' allowed). 'stdout', 'stderr', 'stdin' are special names, mapped to sys.* file objects."""
#         self.outfiles = outfiles if islist(outfiles) else [outfiles]
#         #print self, self.outfiles
#         super(Monitor, self).__init__(*args, **knobs)

    def __getstate__(self):
        "Handles serialization/copying of file objects in self.outfiles and self.out (standard copy doesn't work for files)."
        state = super(Monitor, self).__getstate__()
        def encode(f, std = {sys.stdout:'stdout', sys.stderr:'stderr', sys.stdin:'stdin'}):
            if not isinstance(f, (file, Tee)): return f
            if f in std: return std[f]
            raise Exception("Monitor.__getstate__, can't serialize/copy a file object: %s" % f)
        
        if 'outfiles' in state: 
            state['outfiles'] = [encode(f) for f in self.outfiles] if islist(self.outfiles) else encode(self.outfiles)
        if 'out' in state: state['out'] = encode(self.out)
        return state
        
    def __setstate__(self, state):
        def decode(f, std = ['stdout', 'stderr', 'stdin']):
            if f in std: return getattr(sys, f)
            return f
        if 'out' in state: state['out'] = decode(state['out'])
        super(Monitor, self).__setstate__(state)
        
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header
        if not self.source:
            raise Exception("No source pipe connected (self.source=%s) in a monitoring pipe: <%s>" % (self.source, self))
        try:
            self.count = 0
            for item in self.source: 
                self.count += 1
                self.monitor(item)
                self.yielded += 1           # unlike in Transform, we don't expect any returned result from monitor()
                yield item                  # however, watch out for bugs: monitor() can implicitly modify internals of 'item' unless 'item' is immutable
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def _prolog(self):
        "Open the output stream, self.out."
        if not islist(self.outfiles): self.outfiles = [self.outfiles]
        if len(self.outfiles) > 1:
            self.out = Tee(*self.outfiles) 
            self.mustclose = True
        elif self.outfiles:
            self.out, self.mustclose = openfile(self.outfiles[0], self.mode)
        return super(Monitor, self)._prolog()

    def _epilog(self):
        "Run report() and close self.out."
        with self.printlock: self.report()
        super(Monitor, self)._epilog()
        if self.mustclose: self.out.close()
        del self.out

    def monitor(self, item):
        "Override in subclasses to process next item during iteration. If printing a log, use self.out as the output stream."
        self.process(item)              # for backward compatibility, process() is still called; TODO: remove process() and leave only monitor() in the future
    def process(self, item):
        "Can return modified item; or None, interpreted as no result (drop item); or True (pass unchanged); or False (drop item)."
        #if self.fun is None: raise Exception("Missing inner function (self.fun) in class %s" % classname(self))
        self.fun(item)

    def report(self):
        "Override in subclasses to print out a report at the end of iterating. Use self.out as the output stream."


class Filter(_Functional):
    """Doesn't change input items, but filters out undesired ones. 
    Method process() must return True (pass the item through) or False (drop the item), not the actual object.
    By default, if no 'fun' is given, filters out false values from the stream.
    """
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header
        if not self.source:
            raise Exception("No source pipe connected (self.source=%s) in a filtering pipe: <%s>" % (self.source, self))
        try:
            self.count = 0
            for item in self.source: 
                self.count += 1
                if self.accept(item): 
                    self.yielded += 1
                    yield item        # unlike in Transform, we expect only True/False from accept/process(), not an actual data object
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def accept(self, item):
        return self.process(item)
    def process(self, item):                            # deprecated in Filter; override accept() instead
        if self.fun is None: return bool(item)
        return self.fun(item)


# class Capacitor(Pipe):
#     "Consumes all input data and only then starts producing output data, possibly of a different type, e.g. aggregates of input items."
#
# class DataPile(Pipe):
#     """Permanent (buffered) storage of data, in memory or filesystem, that can be reused many times after one-time creation.
#     Can provide random access to items via index (dict) or multiple indices.
#     It's intentional that the term 'pile' resembles both 'pipe' and 'file'."""
#     def build(self): pass
#     def append(self): pass
#     def load(self): pass
#     def __getitem__(self, key): pass


class Generator(_Functional):
    """
    Generator of a sequence of data items produced by an external function passed as self.fun, or by an overriden method
    (one of: produce, produceMany or generate) implemented in a subclass.
    """
    
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header

        try:
            self.count = 0
            for item in self.generate():
                self.count += 1
                self.yielded += 1
                yield item
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def produce(self):
        "Override to produce 1 data item at a time. Return None to indicate no more items."
        if self.fun is None: raise NotImplementedError
        return self.fun()

    def produceMany(self):
        "Override to produce a batch of data items. Each batch should be returned as a list, None if no more items."
        while True:
            item = self.produce()
            if item is None: break
            return [item]

    def generate(self):
        "Override to yield a sequence of data items."
        for batch in self.produceMany():
            if batch is None: break
            for item in batch: yield item


#####################################################################################################################################################
###
###   CONCRETE PIPES
###

###  Wrappers for standard Python objects

class Collection(Pipe):
    """Wrapper for plain collections or iterators, to turn them into DataPipes that can be used as sources in a pipeline. 
    In subclasses, set 'self.data' with the iterable to take data from; optionally override open() to initialize 'data' just before iteration starts,
    but note that close() is not called (this would require control over iteration process and yielding items one-by-one, 
    instead of following back on the collection's own iterator).
    """
    def __init__(self, data):
        self.data = data
    def __iter__(self):             # Pipe fields: count, yielded, ... are not used, they will have default (empty) values
        self.open()
        return iter(self.data)

class File(Pipe):
    """Wrapper for a file object opened for reading. Iteration delegates to file.__iter__(). 
    Transparently repositions file pointer when iteration restarts. The file is never closed."""
    def __init__(self, file):
        "'f' is a file object."
        self.file = file
    def __iter__(self):
        if isinstance(self.file, file): self.file.seek(0)
        else: self.file.reopen()
        return iter(self.file)
    
class Function(Transform):
    """Transform OR filter constructed from a plain python function. A wrapper.
    The inner (wrapped) function should return an item to be yielded,
    or True/None to indicate that the same (input) item should be yielded (perhaps with some internals modified),
    or False to indicate that the item should be dropped entirely (filtered out).
    Explicit use of Transform or Filter classes instead of this one is recommended.
    """
    def __init__(self, oper = None):
        "oper=None handles the case when processing function is implemented through overloading of process()."
        self.oper = oper
    def process(self, item):
        if self.oper is None: raise Exception("Missing operator function (self.oper) in %s" % self)
        ret = self.oper(item)
        if ret is True: return item
        if ret is False: return False
        if ret is None: return item         # for monitor- or transform-like functions that don't make final 'return item'
        return ret
    def __str__(self):
        if self.oper and hasattr(self.oper, 'func_name'):
            return "%s %s" % (classname(self), self.oper.func_name)
        return classname(self)


class Tuple(Function):
    """Creates a function-pipe g(x) that applies predefined functions, fun[0], fun[1], ...
       to a data item 'x' and returns atomic results all combined into a tuple.
       If fun[i] is FORWARD or [], an identity function is used on a given position.
       If fun[i] is a list (of functions): fun[i] == [f1,f2,f3...], all the functions will be combined
       into a composite function: fun[i](x) ::= f3(f2(f1(x))) - note the order (!).
       If 'x' is a tuple, 'fun' functions are applied to corresponding subitems of 'x'
       ('x' must be of the same length as 'fun' in such case).
       
       Tuple can be used for easy creation of data tuples of any size/structure during pipeline processing:
       ... >> Tuple(fun1, fun2, fun3) >> ...
       The above can also be written as:
       ... >> (fun1, fun2, fun3) >> ...
       which is automatically converted to a basic form with an explicit Tuple as above.
    """
    def __init__(self, *fun):
        
        def ident(x): return x
        def convert(f):
            if f in (FORWARD, []): return ident
            if isinstance(f, Pipe): return operator(f)
            return f
        
        def composite(fl):
            if not islist(fl): return convert(fl)
            if len(fl) == 1: return convert(fl[0])
            if len(fl) == 0: return ident
            def F(x):
                # 'fl' is a list of functions
                res = x
                for f in fl: res = convert(f)(res)
                return res
            return F
    
        # replace FORWARD tokens with identity functions; convert lists of functions to a composite function
        self.fun = fun = tuple(composite(f) for f in fun)
        self.n   = n   = len(fun)
        self.rng = rng = range(n)
        
    def process(self, x):
        if isinstance(x, tuple):
            assert len(x) == self.n, 'Tuple.process(): expected an input tuple of length %s, ' \
                                     'a tuple of length %s received instead: %s' % (self.n, len(x), x)
            return tuple(self.fun[i](x[i]) for i in self.rng)
        else:
            return tuple(f(x) for f in self.fun)
        

###  Generators

class Empty(Pipe):
    """Generates an empty output stream. Useful as an initial pipe in an incremental sum of pipes: 
    p = Empty; p += X[0]; p += X[1] ..."""
    def __iter__(self):
        return; yield

class Const(Pipe):
    """Infinite stream with the same item repeated again and again. The item can be (deep)copied every time, or not.
    Also, the item can be changed from the outside during iteration, by assigning to self.item
    (useful when Const is used as a data feed inside metapipes).
    >>> Const("sample") >> Limit(2) >> Print >> RUN
    sample
    sample
    """
    class __knobs__:
        item = None                 # the item to be repeated; can be passed during __init__: Const(x)
        makecopy = None             # if 'copy' or 'deepcopy', the item will be copied (deepcopied) before each yield
    def __iter__(self):
        if not self.makecopy:
            while True: yield self.item
        elif self.makecopy == 'copy':
            while True: yield copy(self.item)
        elif self.makecopy == 'deepcopy':
            while True: yield deepcopy(self.item)
        else:
            raise Exception("Incorrect knob value: ")

class Range(Pipe):
    "Generator of consecutive integers, equivalent to xrange(), same parameters."
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return iter(xrange(*self.args))

class Repeat(Pipe):
    "Returns a given item for the specified number of times, or endlessly if times=None. Like itertools.repeat()"
    def __init__(self, item, times = None):
        self.item = item
        self.times = times
    def __iter__(self):
        item = self.item
        if self.times is None:
            while True: 
                self.yielded += 1
                yield item
        else:
            for _ in xrange(self.times): 
                self.yielded += 1
                yield item
    
###  Filters

class Slice(Pipe):
    "Like slice() or itertools.islice(), same parameters. Transmits only a slice of the input stream to the output."
    def init(self, *args):
        self.args = args
    def iter(self):
        return islice(self.source, *self.args)

class Offset(Pipe):
    """Drop a predefined number of initial items.
    >>> PIPE >> [1,2,3,4,5,6] >> Offset(2) >> Offset(offset=2) >> List >> Print >> RUN
    [5, 6]
    """
    class __knobs__:
        offset = 0
    def iter(self):
        count = 0
        for item in self.source: 
            count += 1
            if count <= self.offset: continue
            yield item

class Limit(Pipe):
    "Terminate the data stream after a predefined number of items. 'Head' is an alias."
    class __knobs__:
        limit = 0

    def iter(self):
        self.count = 0
        if self.count >= self.limit: return
        for item in self.source: 
            self.count += 1
            yield item
            if self.count >= self.limit: return

class DropWhile(Pipe):
    "Like itertools.dropwhile()."
class TakeWhile(Pipe):
    "Like itertools.takewhile()."    
class StopOn(Pipe):
    """Terminate the data stream when a given condition becomes True. Condition is a function that takes current item as an argument. 
    This function can also keep an internal state (memory)."""
class Loop(Pipe):
    """Iterate over the source a number of times, concatenating the repeated input streams into one output stream."""

class Subset(Pipe):
    """Selects every 'fraction'-th item from the stream, equally spaced, yielding <= 1/fraction of all data. Deterministic subset, no randomization.
    >>> Range(7) >> Subset(3) >> Print >> RUN
    2
    5
    """
    class __knobs__:
        fraction = 1

    def iter(self):
        frac = self.fraction
        batch = 0                               # current batch size, iterates in cycles from 0 to 'frac', and again from 0 ...
        self.count = 0
        for item in self.source: 
            self.count += 1
            batch += 1
            if batch == frac: 
                yield item
                batch = 0

class Sample(Pipe):
    """Random sample of input items. Every input item is decided independently with a given probability,
    unconditional on what items were chosen earlier."""

    
###  Transforms

class Batch(Pipe):
    """Combine every consecutive group of 'batch' items into a single list and yield as an output item (a batch).
       The last batch can have less than 'batch' items.
       If 'batch' is None, all input items are combined together into one batch.
       If no input items were pulled from the source, no output batch is yielded.
    >>> PIPE >> [1,2,3,4,5,6] >> Batch(4) >> List >> Print >> RUN
    [[1, 2, 3, 4], [5, 6]]
    """
    class __knobs__:
        batch = None
    def iter(self):
        items = []
        self.count = 0
        for item in self.source:
            self.count += 1
            items.append(item)
            if None != self.batch <= len(items):
                yield items
                items = []
        if items: yield items


###  Buffers

class Buffer(Pipe):
    "Buffers all input data in memory, then iterates over it and yields from memory, possibly multiple times."
    def setup(self):
        self.data = list(self.source)

class Random(Buffer):
    """
    Buffers all data in memory, then picks and yields items randomly on each subsequent request.
    Iterates infinitely over the buffered data. If a weighing function is given, 'weigh',
    items will be selected with probability proportional to their weights
    (weights can be any non-negative numbers, they will be normalized to unit sum during buffer setup).
    """
    NONE = object()
    
    class __knobs__:
        seed  = None        # optional random seed
        weigh = None        # optional weighing function for items: weigh(item) >= 0
        
    def setup(self):
        super(Random, self).setup()
        if self.weigh is None:
            self.rand = random.Random(self.seed)
            self.probs = None
        else:
            self.rand = np.random.RandomState(self.seed)        # use numpy's random to be able to choose items with non-uniform distribution
            weights = [self.weigh(item) for item in self.data]
            weights = np.array(weights + [0])                   # append zero weight for NONE below
            self.probs = weights / float(weights.sum())         # normalize weights to unit sum (probabilities)

            # convert self.data from list to array;
            # append NONE with zero weight beforehand to avoid merging individual numpy arrays if present in self.data
            self.data += [Random.NONE]
            self.data = np.array(self.data, dtype = object)

    def iter(self):
        if not len(self.data): return
        choice = self.rand.choice
        
        if self.probs is None:
            while True:
                yield choice(self.data)
        else:
            while True:
                yield choice(self.data, p = self.probs)         # numpy's RandomState accepts probability distribution


class Sort(Pipe):
    """Total or partial in-memory heap sort of the input stream. Buffers items in a heap and when the heap is full, 
    outputs them in sorted order. Heap size can be unlimited (default), which results in total sorting: 
    output items appear only after all input data was consumed; or limited to a predefined maximum size 
    (partial sort, generation of output items begins as soon as the heap achieves its maximum size).
    >>> Collection([2,7,3,6,8,3]) >> Sort(2) >> List >> Print >> RUN
    [2, 3, 6, 7, 3, 8]
    """
    class __knobs__:
        size = None

    def iter(self):
        from heapq import heapify, heappush, heappop
        source = iter(self.source)
        
        # prolog: fill out the heap with initial data and heapify in one step
        heap = list(islice(source, self.size)) if self.size else list(source)
        heapify(heap)
        
        # more input data remains?
        if self.size and len(heap) == self.size:
            for item in source:
                yield heappop(heap)
                heappush(heap, item)
        
        # epilog: flush remaining items
        while heap: yield heappop(heap)


#####################################################################################################################################################
###
###   MONITORS & REPORTING
###

class Print(Monitor):
    """Print items passing through, or value of 'func' function calculated on each item and/or a static message. 
    If outfile is given, additionally print to that file (in such case, stdout can be suppressed).
    Subclasses can override monitor(item) and open/close() to provide custom printing: 
    use self.out as the output stream: 'print >>self.out, ...' 
    - it redirects to stdout and/or file, appropriately."""

    class __knobs__:
        msg     = "%s"      # format string of messages, may contain 1 parameter %s to print the data item in the right place
        func    = None      # optional function called on every item before printing, its output is printed instead of the actual item
        file    = None      # path to external output file or None
        disp    = True      # shall we print to stdout? (can be combined with 'outfile')
        step    = None      # print every step-th item, or all items if None
        index   = False     # shall we print 1-based item number at the begining of a line?

    def __init__(self, *args, **kwargs):
        self.initKnobs(*args, **kwargs)

        if 'subset' in kwargs: self.step = kwargs['subset']         # 'subset' is an alias for 'step'
        if 'count' in kwargs: self.index = kwargs['count']          # 'count' is an alias for 'index'
        self.static = ('%s' not in self.msg)
        self.message = self.msg
        
        # set 'outfiles' for Monitor base class
        if self.file and disp: 
            self.outfiles = [self.file, sys.stdout]
        else:
            self.outfiles = self.file

        self.init(*args, **kwargs)

#     def __init__(self, msg = "%s", func = None, outfile = None, disp = True, count = False, subset = None, *args, **kwargs):
#         """msg: format string of messages, may contain 1 parameter %s to print the data item in the right place.
#         func: optional function called on every item before printing, its output is printed instead of the actual item.
#         outfile: path to external output file or None; disp: shall we print to stdout?; 
#         count: shall we print 1-based item number at the begining of a line?
#         subset: print every n-th item (see Subset), or None to print all items."""
#         if outfile and disp: outfile = [outfile, sys.stdout]
#         kwargs['outfile'] = outfile
#         Monitor.__init__(self, *args, **kwargs)
#         #if '%s' not in msg: msg += ' %s'
#         self.static = ('%s' not in msg)
#         self.message = msg
#         self.func = func
#         self.index = count
#         if subset: self.subset = subset
#         #print "created Print() instance, message '%s'" % self.message

    def monitor(self, item):
        #print "Print.monitor()", self.subset, self.index
        if self.step and self.count % self.step != 0: return
        with self.printlock:
            if self.index: print >>self.out, self.count,
            self.print1(item)
        
    def print1(self, item):
        #print "Print.print1()"
        if self.static: print >>self.out, self.message
        else:
            if self.func is not None: item = self.func(item)
            print >>self.out, self.message % item

def PrintSubset(subset, func = None, count = True, outfile = None, disp = True):
    "A shorthand for Print with 'subset' argument. Differs from Print only in the order of __init__ arguments."
    return Print(subset = subset, func = func, count = count, outfile = outfile, disp = disp)

class Count(Monitor):
    "Counts data items and prints the index before yielding each consecutive item. Indexing is 1-based by default, this can be changed. No newline added by default."
    class __knobs__:
        step  = 1
        msg   = "%d "           # thanks to the space after %d you can use: Count >> Print >> ... and print other data after the count on the same line
        start = 1
    
    def init(self, *args):
        if '%d' not in self.msg: self.msg = '%d ' + self.msg
        if isint(self.step):
            self.step, self.frac = (self.step, None)
        else:
            self.step, self.frac = (None, self.step)
            
    def open(self):
        self.next = self.step or 1

    def monitor(self, _):
        if self.count >= self.next:
            with self.printlock:
                self.out.write( self.msg % self.count )
            if self.step: self.next = self.count + self.step
            else: 
                step = int(self.count * self.frac) + 1
                digits = int(math.log10(step))              # no. of digits after the 1st one - that many should be zeroed out in 'next'
                self.next = self.count + step
                self.next = int(round(self.next, -digits))

class Countn(Count):
    "Like Count, but adds newline instead of space after each printed number."
    class __knobs__:
        msg = '%d\n'


class Progress(Monitor):
    "Like Count, but prints progress as a percentage of 'total' instead of an absolute count. Total no. of items must be known in advance."
    def __init__(self, total, step = 10, msg = "%.0f%%"):
        Monitor.__init__(self)
        self.message = msg
        self.total = total
        self.step = step
    def open(self):
        self.next = self.step
    def monitor(self, _):
        if self.count >= self.next:
            with self.printlock:
                print self.message % (self.count * 100. / self.total)
            self.next = self.count + self.step
    
class Total(Monitor):
    "Print total no. of items at the end of processing."
    class __knobs__:
        msgTotal = "#items:  %d"
    
    def init(self, *args):
        if self.msgTotal is None: return
        try: self.msgTotal % 0
        except: self.msgTotal += " %d"                   # append format character if missing
    def monitor(self, item): pass
    def close(self):
        with self.printlock: print self.msgTotal % self.count

class Time(Monitor):
    "Measure time since the beginning of data iteration. If 'message' is present, print the total time at the end, embedded in 'message'."
    class __knobs__:
        msgTime = "Time elapsed: %.1f s"
        
    start = None                # time when last open() was run, as Unix timestamp
    elapsed = None              # final time elapsed, in seconds, as float

    def open(self):
        self.start = time()
    def current(self):
        return time() - self.start      # current count of time elapsed; can be called during iteration, when 'elapsed' is not yet available
    def close(self):
        self.elapsed = self.current()
        if self.msgTime: 
            with self.printlock: print self.msgTime % self.elapsed
    
class Report(Total, Time):
    "'Total' and 'Time' combined."
    def init(self, header = "\n========================="):
        Total.init(self)
        Time.init(self)
        self.header = header
    def close(self):
        self.elapsed = self.current()
        with self.printlock:
            if self.header: print self.header
            if self.msgTotal: print self.msgTotal % self.count
            if self.msgTime: print self.msgTime % self.elapsed


class Metric(Monitor):
    """Calculates a given metric on each input item and aggregates to an overall metric(s).
    Optionally appends individual metrics to each item (modifies items). 
    Subclasses can implement initialization of internal structures in open() and reporting in report().
    Aggregation can be implemented in either aggregate() (recommended for general-purpose classes, more versatile), 
    or metric(), together with calculating individual values (easier when writing a short class for one-time use).
    Subclasses can use self.size attribute, which holds the no. of individual not-None metrics computed so far.
    """

    class __knobs__:
        fun = None
    
    #metricname = None           # if not-None, metric of each item will be saved in the item under this name
    last = None                 # most recent individual metric value calculated
    size = None                 # no. of individual metrics calculated & aggregated so far excluding Nones

    def _prolog(self):
        self.size = 0
        return super(Metric, self)._prolog()

    def monitor(self, item):
        self.last = metric = self.metric(item)
        #if self.metricname: setattr(item, self.metricname, metric)
        if metric is None: return
        self.aggregate(metric)
        self.size += 1
        
    def metric(self, item):
        "Override in subclasses to calculate individual metric of each item. Can return None if the item should be ignored"
        return self.fun(item)

    def aggregate(self, metric):
        """Override in subclasses to update internal structures for calculation of an aggregated metric, 
        after new individual sample was measured with the result 'metric'."""
        
    def report(self):
        """Override in subclasses to print out calculated metrics at the end of data iteration. 
           Don't use self.printlock! This would cause a deadlock."""

    
class Mean(Metric):
    """Calculates sample mean & std.deviation of values measured for individual items by a given metric.
    The metric is either implemented in overridden metric() method, or given as a function - argument of initialization
    (typically a lambda expression).
    """
    class __knobs__:
        title = None                # leading message when printing the report line
        
    def open(self):
        self.sum = self.sum2 = 0.0
    
    def aggregate(self, metric):
        self.sum += metric
        self.sum2 += metric ** 2

    def mean(self): 
        "Sample mean"
        if self.size <= 0: return None
        return self.sum / float(self.size)

    def deviation(self): 
        "Sample standard deviation"
        if self.size <= 1: return None
        N = float(self.size)
        return np.sqrt((self.sum2 - self.sum/N * self.sum) / (N-1))

    def report(self):
        def _s(x, f): return None if x is None else f % x
        header = "mean +stddev /size:    "
        if self.title: header = self.title + ' ' + header
        if self.size:
            mean = _s(self.mean(), "%.4f")
            dev = _s(self.deviation(), "%.2f")
            print >>self.out, header + "%s +%s /%d" % (mean, dev, self.size)
        else:
            print >>self.out, header + "None +None /%s" % self.size
    

# class Experiment(Monitor):
#     "Provides subclasses with logging facilities."

# class Trainable(object):
#     trained  = False        # is the model trained enough to make predictions? training can be continued in parallel with predicting
#     training = True         # shall the items passing through be used for training, in addition to making predictions if possible?


#####################################################################################################################################################
###
###   FILES
###

class Save(Monitor):
    """Writes all passing items to a predefined file and outputs them unmodified to the receiver. 
    Item type must be compatible with the file's write() method."""
    def monitor(self, item):
        self.out.write(item)
    
class CSV(Monitor):
    "Writing to a CSV file. Wrapper for the standard Python module, 'csv'."    
    class __knobs__:
        mode   = 'wb'
        delim  = ','
        strict = True       # if True, all rows must have the same no. of items, otherwise exception is raised
        
    def open(self):
        self.csv = csv.writer(self.out, delimiter = self.delim)
        self.length = None              # row length
    def monitor(self, row):
        if self.strict:
            if self.length is None: self.length = len(row)
            elif self.length != len(row): 
                raise Exception("CSV, incorrect no. of items in a row: %s instead of %s" % (len(row), self.length))
        self.csv.writerow(row)

class Pile(Pipe):
    """Pipe wrapper around file objects, for use of files in pipelines and with pipe operators.
    Sequence of data items stored in a file. During iteration, either reads and outputs items from the file (if no source pipe connected) 
    or takes items from source, saves to the file (override or append) and outputs unchanged to the caller."""
    
    fileclass = None            # subclass of ObjectFile to be used as an underlying object-oriented file implementation
    file      = None
    
    def __init__(self, f, fileclass = None, append = False, flush = 0, rewrite = False, emptylines = 0):
        """
        f: either a file object (ObjectFile in *closed* state), or a file name (string).
        rewrite: if True, SafeRewriteFile class will be used in write operations, for safe rewrite of an existing file.
        emptylines: no. of extra empty lines after every object.
        """
        if fileclass: self.fileclass = fileclass
        self.file = f
        self.append = append
        self.flush = flush
        self.rewrite = rewrite
        self.emptylines = emptylines
        
    def __iter__(self):
        if self.source: return self._write()
        else: return self._read()
        
    def _write(self):
        if isstring(self.file):
            mode = 'at' if self.append else 'wt'
            rawclass = SafeRewriteFile if self.rewrite else files_File
            f = self.fileclass(self.file, mode = mode, cls = rawclass, flush = self.flush, emptylines = self.emptylines)
        else:
            f = self.file
            f.open()
        
        self.count = 0
        for item in self.source:
            self.count += 1
            f.write(item)
            yield item
        f.close()
        
    def _read(self):
        if isstring(self.file):
            f = self.fileclass(self.file, 'rt')
        else:
            f = self.file
            f.open()
        for item in f: yield item
        f.close()
        
class JsonPile(Pile):
    fileclass = JsonFile

class DastPile(Pile):
    fileclass = DastFile
    

# class Blocks(DataPile):
#     "Saves key-value pairs data to files in chunks (blocks), to enable local modifications of data items and fast read access to any position in data sequence."
# 
# class Retention(DataPile):
#     "Each new chunk stays for some time in memory to allow random-access modifications before being written to disk."


###  Other

class List(Pipe):
    "Combines all input items into a list. At the end, this list is output as the only output item; it's also directly available as self.items property."
    def iter(self):
        self.items = []
        self.count = 0
        for item in self.source:
            self.count += 1
            self.items.append(item)
        yield self.items

        
#####################################################################################################################################################
###
###   PIPELINE
###

class MetaPipe(Pipe):
    "A pipe that internally contains other pipes to provide meta-operations on them."

class Pipeline(Pipe):
    """
    Sequence of data pipes connected sequentially, one after another. Pipeline is a Pipe itself.
    Inner pipes can be accessed by indexing operator: pipeline[3]
    """
    
    pipes    = None         # static list of pipes as passed during pipeline initialization; may contain non-pipe objects
    pipeline = None         # the actual pipes used in iteration, created dynamically in __iter__ or setKnobs() 
    knobs    = None         # knobs to be set before iteration starts; 
                            # for delayed setting of knobs, necessary when some pipes are only templates that require normalization
    
    #__inner__ = "pipeline"
    
    def __init__(self, *pipes, **knobs):
        self.pipes = list(pipes)
        
    def __rshift__(self, other):
        """Append 'other' to the end of the pipeline. Shallow-copy the pipeline beforehand, 
        to avoid in-place modifications but preserve '>>' protocol."""
        if other is RUN:
            self.run()
        elif isinstance(other, _FETCH):         # pull all data and return as a list, possibly with a limit on the no. of items to be fetched
            return self.fetch(other.limit)
        else:
            res = self.copy1()
            res.pipes.append(other)
            return res

    def __lshift__(self, other):
        "Like __rshift__, but 'other' is put at the tail not head of the pipeline."
        res = self.copy1()
        res.pipes = [other] + res.pipes
        return res

    def __getitem__(self, pos):
        """Returns either an operating pipe from self.pipeline, or a static pipe from self.pipes, 
        depending whether called during iteration or not."""
        return self.pipeline[pos] if self.pipeline else self.pipes[pos]
    
    def setKnobs(self, knobs = {}, strict = False, **kwargs):
        if kwargs:
            knobs = knobs.copy()
            knobs.update(kwargs)
        super(Pipeline, self).setKnobs(knobs)           # Pipeline itself has no knobs, but a subclass can define some
        for pipe in self.pipes:                         # set knobs in template pipes (some of them can be actual pipes)
            if isinstance(pipe, Cell): pipe.setKnobs(knobs)
        if self.pipeline: self.setInnerKnobs(knobs)
        else:
            # no normalized pipeline yet? keep the knobs to apply in the future
            if self.knobs: self.knobs.update(knobs)
            else: self.knobs = knobs

    def setInnerKnobs(self, knobs):
        if not knobs: return
        for pipe in self.pipeline:
            pipe.setKnobs(knobs)
#             try:
#                 if hasattr(pipe, 'setKnobs'): pipe.setKnobs(*self.knobs)
#             except:
#                 print pipe
#                 raise

    def setup(self):
        # normalize pipes and connect into a list
        self.pipeline = _normalize(self.pipes)
        self.setInnerKnobs(self.knobs)
        self.knobs = None
#         for pipe in self.pipeline:
#             pipe.setup()

    def reset(self):
        del self.pipeline
        self._created = False

    def iter(self):
        prev = self.source
        for next in self.pipeline:
            if prev is not None: next.source = prev         # 1st pipe can be a generator or collection, not necessarily a Pipe (no .source attribute)
            prev = next
            
        # pull data
        head, tail = self.pipeline[-1], self.pipeline[0]
        for item in head:
            self.count = tail.count                         # update indirectly how many items were read from source
            yield item
        self.count = tail.count        
    
    def flatten(self):
        "Flattened list of all pipes involved in the current self.pipeline, with nested pipelines replaced with lists of their pipes."
        def flat(pipes):
            result = []
            for p in pipes:
                if isinstance(p, Pipeline): result += flat(p.pipeline)
                else: result.append(p)
            return result
        return flat(self.pipeline)
    
    def stats(self):
        """String with detailed statistics of the no. of input & output items in each pipe of the flattened pipeline.
        Should be called during or after iteration, when self.pipeline exists, otherwise only general info is returned.
        """
        if not self.pipeline: return super(Pipeline, self).stats()
        lines = ["No. of data items (#input, pipe, #output) passing through each pipe of: %s" % self]
        for pipe in self.flatten():
            lines += ["%7s %s %s" % (pipe.count, pipe, pipe.yielded)]
        return '\n'.join(lines)
    
    def __str__(self):
        if self.pipeline:
            return "connected Pipeline [" + '] >> ['.join(map(str, self.pipeline)) + ']'
        return "abstract Pipeline [" + '] >> ['.join(map(str, self.pipes)) + ']'
    

#####################################################################################################################################################

class MultiSource(Pipe):
    pass

class Union(MultiSource):
    """Combine items from multiple sources by concatenating corresponding streams one after another: all items from the 1st source; then 2nd... then 3rd...
    TODO: 'mixed' mode (breadth-first, streams interlaced)."""
    def __init__(self, *sources):
        self.sources = _normalize(sources)
    def iter(self):
        return itertools.chain(*self.sources)
    def __add__(self, other):
        res = self.copy1()
        res.sources.append(other)
        return res

class Zip(MultiSource):
    """Combines items from multiple sources. Similar to zip(). Available modes: 
     - long (default): yield tuples of items, one item from each source; put 'fillvalue' (default=None) if a given source is exhausted; like zip_longest()
     - short: like zip(), truncates output stream to the length of the shortest input stream
     - strict: raise exception if one of the sources is exhausted while another one has still some data
    """
    def __init__(self, *sources, **kwargs):
        "kwargs may contain: 'mode' (default 'long'), 'fillvalue' (default None)."
        self.sources = _normalize(sources)
        self.mode = kwargs.get('mode', 'long')
        self.fillvalue = kwargs.get('fillvalue', None)
    #def iter(self): pass
    
class MergeSort(MultiSource):
    """Merge multiple sorted inputs into a single sorted output. Like heapq.merge(), but wrapped up in a Pipe. 
    If only the input streams were fully sorted, the result stream is guaranteed to be fully sorted, too."""
    def __init__(self, *sources):
        if len(sources) == 1 and islist(sources[0]):
            self.sources = sources[0]
        else:
            self.sources = sources
    def iter(self):
        for item in heapq.merge(*self.sources): yield item

class Ensemble(MetaPipe, Transform):   # DRAFT
    ""
    def __init__(self, *algs, **knobs):
        self.pipes = list(algs)
        self.feed = Const()

    def process(self, item):
        self.feed.item = item
        
    def vote(self, item):
        "Override in subclasses to provide custom extraction of vote value from the output item yielded by an algorithm."
        return item
    
    def merge(self, votes):
        "Override in subclasses to provide custom merging of votes into final decision. Default: numerical averaging (mean)."
        return sum(votes) / len(votes)
        
    def output(self, item, vote):
        "Override in subclasses to output final votes in a custom way (typically, to store it inside 'item' instead of yielding directly)."
        return vote
        

class Parallel(MetaPipe):   # DRAFT
    """Connects multiple pipes as parallel routes from a single source and no destination.
    Each parallel route is wrapped up in a Thread object, so that 'push' interface can be used
    to feed data to every route. Output items - if generated by the routes - are ignored.
    Instead, the entire Parallel pipe yields input items on its output.
    Note that there can be a delay in thread execution such that an output item can be yielded
    when some thread hasn't consumed it yet - take this into account when monitoring side effects
    of execution on particular routes. However, it's guaranteed that when all iteration ends,
    all the threads have already finished their execution.
    """
    def __init__(self, *pipes, **knobs):
        self.pipes = list(pipes)

class Serial(MetaPipe): pass
class Sequential(MetaPipe): pass


#####################################################################################################################################################
###
###   STRUCTURAL PIPES
###

class Wrapper(MetaPipe):
    "Provides meta-operations for exactly 1 internal pipe: self.pipe."
    
    pipe = None
    
    def setKnobs(self, knobs={}, strict=False, **kwargs):
        if kwargs:
            knobs = knobs.copy()
            knobs.update(kwargs)
        super(Wrapper, self).setKnobs(knobs, strict)  # Wrapper itself has no knobs, but a subclass can define some
        if self.pipe: self.pipe.setKnobs(knobs, strict)
    
    def _prolog(self):
        "Call inner pipe's setup before starting iteration."
        if self.pipe: self.pipe.setup()
        return super(Wrapper, self)._prolog()


# class Container(MetaPipe):
#     "Base class for classes that contain multiple pipes inside: self.pipes."
#     pipes = []          # any collection of internal pipes
#
#     def copy(self, deep = True):
#         "Smart shallow copy (in addition to deep copy). In shallow mode, copies the collection of pipes, too, so it can be modified afterwards without affecting the original."
#         res = super(Container, self).copy(deep)
#         if not deep: res.pipes = copy(self.pipes)
#         return res
#
#     def setKnobs(self, knobs, strict = False, pipes = None):
#         super(Container, self).setKnobs(knobs, strict)          # Container itself has no knobs, but a subclass can define some
#         if pipes is None: pipes = self.pipes
#         for pipe in pipes:
#             try:
#                 if hasattr(pipe, 'setKnobs'): pipe.setKnobs(knobs, strict)
#             except:
#                 print pipe
#                 raise


#####################################################################################################################################################

class MetaOptimize(Wrapper):
    """Client MUST ensure that internal pipe does NOT modify input items. Otherwise, only the 1st copy of the pipe will work on correct input data, 
    others will receive broken input. Does not yield anything, may only produce side effects."""
    
class Grid(MetaOptimize):
    """
    Executes a given pipe multiple times, for all possible combinations of knob values.
    Runs in mixed serial-parallel mode, depending on 'maxThreads' setting. 
    In parallel mode, the algorithm must NOT deep-modify data items, otherwise there will be interference 
    between concurrent threads (items are only shallow-copied for each thread). 
    For confirmation, after evaluating different knob settings and choosing the best one, you should execute 
    the pipe with this setting outside Grid and see if it produces the same results as inside Grid.
    No output produced, only empty stream.
    """
    
    # TODO: 3rd mode: sequential
    
    runID        = "runID"      # name of a special knob inside 'pipe' that will be set with an ID (integer >= 1) of the current run
    startID      = 0            # ID of the first run, to start counting from
    maxThreads   = 0            # max. no. of parallel threads; <=1 for serial execution; None for full parallelism, with only 1 scan over input data
    threadBuffer = 10           # length of input queue in each thread; 1 enforces perfect alignment of threads execution; 
                                # 0: no alignment, queues can get big when input data are produced faster than consumed
    copyPipe     = True         # shall we make a separate deep copy of the pipe for each run? Applies to serial scan only; in parallel, copy is always done
    copyData     = True         # shall we make separate deep copies of data items for each parallel run? no copy in serial mode
    
    def __init__(self, pipe, **kwargs):
        #"""'space', if present, is a Cartesian or another Space instance, 
        #but typically space=None and knobs are passed as keyword args.
        #"""
        self.pipe = pipe                                    # the pipe(line) to be copied and executed, in parallel, for different combinations of knob values
        self.runID = kwargs.pop('runID', self.runID)
        self.startID = kwargs.pop('startID', self.startID)
        self.maxThreads = kwargs.pop('maxThreads', self.maxThreads)
        self.threadBuffer = kwargs.pop('threadBuffer', self.threadBuffer)
        self.copyPipe = kwargs.pop('copyPipe', self.copyPipe)
        self.copyData = kwargs.pop('copyData', self.copyData)
        
        self.grid = KGrid(**kwargs)
        self.runs = len(self.grid)                          # no. of runs to be done
        self.done = None                                    # no. of runs completed so far
        
#         knobs = kwargs
#         self.space = Cartesian(*knobs.values())             # value space of all possible knob values
#         self.names = knobs.keys()                           # names of knobs
    
#         def setID(i, knobs):
#             return Knobs([(self.runID, self.startID + i)] + knobs.items())
#         knobspace = KGrid(**kwargs)
#         self.knobspace = (setID(i, knobs) for i, knobs in enumerate(knobspace))
    
    def knobspace(self):
        "wrapper for the iterator of self.grid, to append runID"
        for i, knobs in enumerate(self.grid):               # append runID to each knobs combination
            yield Knobs([(self.runID, self.startID + i)] + knobs.items())
    
#     def createKnobs(self, ID, values):
#         "Creates one combination of knobs using given values and returns as a list of (name,value) pairs."
#         return Knobs([(self.runID, ID)] + zip(self.names, values))
        
#     def createPipe(self, knobs, forceCopy = False):
#         if self.copyPipe or forceCopy:          # make a duplicate ...
#             return self.pipe.dup(knobs)
#         self.pipe.setKnobs(knobs)               # ... or use the original pipe again
#         return self.pipe
#         #pipe = self.pipe.copy() if (self.copyPipe or forceCopy) else self.pipe
#         #pipe.setKnobs(knobs)
#         ##pipe.setup()
#         #return pipe
        
    def iter(self):
        self.done = 0
        if self.maxThreads is None or self.maxThreads > 1:
            self.iterParallel()
        else:
            self.iterSerial()
        return; yield                                       # to make this method work as a generator (only an empty one)
        
    def iterSerial(self):
        with self.printlock: print "Grid: %d serial runs to be executed..." % self.runs
        #for i, value in enumerate(self.space):
        #    knobs = self.createKnobs(self.startID + i, value)
        for knobs in self.knobspace():
            self.scanSerial(knobs)
            self.done += 1
        
    def scanSerial(self, knobs):
        """Single scan over input data, with items passed directly to the single pipe being executed. 
        No parallelism, no multi-threading, no pipe copying.
        The same pipe object is reused in all runs - watch out against interference between consecutive runs.
        """
        with self.printlock: print "Grid, starting next serial scan for run ID=%s..." % knobs[self.runID]
        self.printKnobs(knobs)
        pipe = self.pipe.copy() if self.copyPipe else self.pipe
        pipe.setKnobs(knobs)
        #pipe = self.createPipe(knobs)
        PIPE >> self.source >> pipe >> RUN
        self.count = pipe.count
        self.report(pipe)
    
    def iterParallel(self):
        scans = util.divup(self.runs, self.maxThreads) if self.maxThreads else 1
        with self.printlock: print "Grid: %d runs to be executed, in %d scan(s) over input data..." % (self.runs, scans)
        
#         def knobsStream():
#             "Generates knob combinations. Every combination is a list of (name,value) pairs, one pair for each knob."
#             for i, v in enumerate(self.space):
#                 yield self.createKnobs(self.startID + i, v)

        def knobsGroups():
            "Partitions stream generated by knobsStream() into groups of up to 'maxThreads' size each."
            if not self.maxThreads:                     # yield all knobs combinations at once?
                yield list(self.knobspace())
                #yield list(knobsStream())
                return
            group = []
            for knobs in self.knobspace():              # make groups of 'maxThreads' knob combinations each
            #for knobs in knobsStream():
                group.append(knobs)
                if len(group) >= self.maxThreads:
                    yield group
                    group = []
            if group: yield group

        for kgroup in knobsGroups():
            self.scanParallel(kgroup)
            self.done += len(kgroup)

    def scanParallel(self, knobsGroup):
        "Single scan over input data, with each item fed to a group of parallel threads."
        with self.printlock: print "Grid, starting next parallel scan for %d runs beginning with ID=%s..." % (len(knobsGroup), knobsGroup[0]['runID'])
        threads = self.createThreads(knobsGroup)
        duplicate = deepcopy if self.copyData else lambda x:x

        self.count = 0
        for item in self.source: 
            self.count += 1
            for _, thread in threads:                           # feed input data to each pipe in parallel
                thread.put(duplicate(item))
        
        self.closeThreads(threads)

    def createThreads(self, knobsGroup):
        threads = []                                            # list of pairs: (knobs, pipe_thread)
        for knobs in knobsGroup:                                # create a copy of the template pipe for each combination of knob values; wrap up in threads
            pipe = self.pipe.dup(knobs)
            thread = Thread(pipe, self.threadBuffer)
            thread.start()
            threads.append((knobs, thread))
        return threads
        
    def closeThreads(self, threads):
        for _, thread in threads:
            thread.emptyFeed()              # wait until all pipes eat up all remaining input items;
        sleep(1)                            # note: some pipes may still be processing the last item! thus sleep()
        
        for _, thread in threads:           # if verbose, print stats of #items 
            self.report(thread.pipe)
        
        with self.printlock: print "Grid, %d runs done." % (self.done + len(threads))
        for knobs, thread in threads:
            self.printKnobs(knobs)
            thread.end()
            thread.join()

    def report(self, pipe):
        if not self.verbose: return
        with self.printlock: print pipe.stats()
        
    def printKnobs(self, knobs):
        with self.printlock:
            print "----------------------------------------------------------------"
            print ' '.join("%s=%s" % knob for knob in knobs.iteritems())        # space-separated list of knob values
    

class Evolution(MetaOptimize):
    "Evolutionary algorithm for (meta-)optimization of a given signal of a pipe through tuning of its knobs."
    
    
#####################################################################################################################################################
###
###   Functional wrappers (operators)
###

class Controller(Wrapper):
    """A wrapper that controls input and output of a given pipe, by manually feeding individual items 
    to its input and manually retrieving items from the output, with possibly some additional operations
    performed before/during/after the item is processed.
    In default implementation of process() method, the inner pipe is expected to pull exactly 1 item at a time
    from the source, otherwise an exception will be raised (typically the pipe is an Transform).
    However, if a subclass overrides process(), it can request the inner pipe to exhibit any other type 
    of input-output behavior.
    """

    class NoData(Exception):
        """Raised when there is no input data to fulfill the request, which indicates that input-output
        relationship implemented by Controller's inner pipe doesn't match the relationship expected by process()."""
        
    class Feed(Pipe):
        """A proxy that enables an external agent to supply input data to the pipe(line) 1 item at a time, 
        or in multiple separate batches of arbitrary size. When the pipeline reads the data, the agent must supply 
        new batch with set() or setList(), otherwise Feed.NoData exception is raised."""
        data = None
        def set(self, *items):
            self.data = items
        def setList(self, items):
            self.data = items
        def __iter__(self):
            while True:
                items = self.data
                self.data = None
                if items is None: raise Controller.NoData()
                for item in items: yield item

    
    def __init__(self, pipe, *args, **kwargs):
        self.pipe = pipe                                # client can use this for later introspection of the inner pipe(s)
        super(Controller, self).__init__(*args, **kwargs)
    
    def _prolog(self):
        self.feed = Controller.Feed()
        self.iterator = (self.feed >> self.pipe).__iter__()
        return super(Controller, self)._prolog()
        
    def _epilog(self):
        super(Controller, self)._epilog()
        self.iterator.close()
        self.feed = None

    def iter(self):
        """Subclasses can override this method to handle other types (not 1-1) of input-output relationship
        implemented by the inner pipe. In such case, process() can also be overridden, or left unused.
        """
        for item in self.source: 
            res = self.process(item)
            if res is not False:                            # False interpreted as "drop this item"
                self.yielded += 1
                yield item if res is None else res          # None interpreted as "yield the same item"
    
    def process(self, item):
        """Subclasses can override this method to perform additional operations before/during/after 
        the item is processed. In such case, remember to call self.put(item) and self.get() in appropriate places,
        or alternatively self.push(item), which runs put() and get() altogether."""
        return self.push(item)
    
    def push(self, item):
        "put + get in one step"
        self.put(item)
        return self.get()

    def put(self, item): self.feed.set(item)
    def get(self): return self.iterator.next()

    def __getitem__(self, pos):
        "Delegates self[] to self.pipe[] - useful when 'pipe' is a pipeline, to access individual pipes directly via self[i]"
        return self.pipe[pos]

#     def reopen(self):
#         self.close()
#         self.iterator = self.pipeline.__iter__()


def operator(pipe):
    """
    A wrapper that turns a pipe (pipeline) back to a regular input-output function
    that can be fed with data manually, one item at a time.
    During processing, the original pipe can still be accessed via 'pipe' property of the wrapper.

    >>> pipeline = Function(lambda x: 2*x) >> Function(lambda x: str(x).upper())
    >>> fun = operator(pipeline)
    >>> print fun(3)
    6
    >>> print fun(['Ala'])
    ['ALA', 'ALA']
    >>> print fun.pipe[0]
    Function <lambda>

    Typically the pipe is an instance of Transform. Even if not, it still must pull exactly 1 item at a time
    from the source, otherwise an exception will be raised.
    Can be used in the same caller's thread, no need to spawn a new thread.
    """
    controller = Controller(pipe)
    controller._prolog()
    def f(x): return controller.process(x)
    f.pipe = pipe
    return f


class operator_pipe(Controller):
    "Same as operator(), but implemented as a callable pipe, a subclass of Controller, rather than a function."
    def __init__(self, pipe):
        super(operator, self).__init__(pipe)
        self._prolog()
    
    # processing can be invoked with op(item), like a function, in addition to op.process(item) defined in base class
    __call__ = Controller.process


#####################################################################################################################################################

class Thread(threading.Thread, Wrapper):
    """A thread object that executes given pipe(line) in a separate thread.
    Can serve also as a pipe and be included in a pipeline, but only if the inner pipe takes no input data (has no source).
    Instantly after creation, the thread itself starts internally pulling output items from the pipe,
    which can block the thread until input items are fed - see run().
    This effectively makes the interface to be a "PUSH" one, not "pull" one, unlike in all other Pipes,
    which makes Thread very distinct from other pipes and which is useful for impelementing parallelism of pipes.
    """
    
    END = object()      # token to be put into a queue (input or output) to indicate end of data stream
    
    class Feed(Pipe):
        "A data-pipe wrapper around threading queue for input data."
        def __init__(self, queue):
            self.queue = queue
        def __iter__(self):
            while True: 
                item = self.queue.get()
                self.queue.task_done()
                if item is Thread.END: break
                yield item
        def join(self):
            "Block until all items in the feed have been retrieved. Note: more items can still be added afterwards."
            self.queue.join()
    
    def __init__(self, pipe, insize = None, outsize = None):
        """insize, outsize: maximum capacity of in/out queues, 0 for unlimited, None for no queue (default). 
        If insize=None, input items must be generated by the 'pipe' itself.
        If outsize=None, output items are dropped.
        """
        threading.Thread.__init__(self)
        self.pipe = pipe                # pipe(line) to be executed in the thread
        self.input = Queue(insize) if insize is not None else None
        self.output = Queue(outsize) if outsize is not None else None
        self.feed = None
    
    def run(self):
        # connect the pipe with input data feed
        if self.input:
            self.feed = Thread.Feed(self.input)
            pipeline = Pipeline(self.feed, self.pipe)
        else:
            pipeline = self.pipe
            
        # run the pipeline, optionally pushing output items to self.output
        if self.output:
            for item in pipeline: self.output.put(item)
            self.output.put(Thread.END)
        else:
            for item in pipeline: pass
    
    # to be used by calling thread...
    
    def put(self, *args): self.input.put(*args)
    def get(self, *args): return self.output.get(*args)
    def end(self): 
        "Terminates the stream of input data"
        self.input.put(Thread.END)                          # put a token that marks end of input data
    
    def emptyFeed(self): self.feed.join()
    
    def iter(self):
        if self.source: 
            raise Exception("Pipe of class Thread can't be used with a source attached. It can't synchronize input and output by itself.")
        while True:
            item = self.get()
            if item is Thread.END: break
            yield item
        

#####################################################################################################################################################

def _normalize(pipes):
    """Normalize a given list of pipes. Remove None's and strings (used for commenting out), 
    instantiate Pipe classes if passed instead of an instance, wrap up functions, collections and files.
    """
    def convert(pos_h):
        pos, h = pos_h
        if issubclass(h, Pipe): return h()
        if isfunction(h): return Function(h)
        if istuple(h): return Tuple(*h)
        if (iscontainer(h) or isgenerator(h)): return Collection(h)
        if isinstance(h, (file, GenericFile, Tee)): return File(h)
        if isstring(h): return None                             # strings can be inserted in a pipeline; treated as comments (ignored)
        return h
    
    return filter(None, map(convert, enumerate(pipes)))


#####################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()

    
