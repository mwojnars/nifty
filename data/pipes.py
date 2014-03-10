'''
Data pipes. 
Allow construction of complex networks (pipelines) of data processing units, that combined perform advanced operations, 
and process large volumes of data (data streams) efficiently thanks to pipelining (processing one item at a time, instead of a full dataset).

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import sys, heapq, math, numpy as np, jsonpickle, itertools
from copy import copy, deepcopy
from itertools import islice
from time import time, sleep

from threading import Thread, Lock
from Queue import Queue

from nifty.util import isint, islist, isstring, issubclass, isfunction, iscontainer, istype, unique, classname, getattrs, setattrs, divup, Tee, NoneLock
from nifty.files import GenericFile, File as files_File, SafeRewriteFile, ObjectFile, JsonFile, DastFile


#####################################################################################################################################################
###
###   SPACE and GRID of values
###

class Space(object):
    """A 1D set of values allowed for a given knob. Typically used to restrict the set of all real numbers to a discrete (finite or infinite) evenly-spaced subset.
    None is disallowed as a value (plays a special role). Can define also a probability density function on the space.
    """
    name = None                 # optional name of the parameter represented by this space
    
    def getKnob(self):
        "Return an empty (value = None) Knob instance that's compatible with this space and can subsequently be filled in with values of this space."
    
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

class Nominal(Space):
    "A finite discrete set of nominal values of any type (strings, numbers, ...), with no natural ordering. The ordering of values in the collection is used."
    def __init__(self, *values, **kwargs):
        if kwargs: self.name = kwargs['name']
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
#     "A finite discrete subset of real or integer numbers. Natural ordering."
#     def __init__(self, values):
#         self.values = values
    
class Grid(Space):
    "Cartesian product of multiple Spaces. A multidimentional set of allowed values for a given combination of knobs. Each multi-value is a tuple."
    
    def __init__(self, *spaces, **namedSpaces):
        "If on the 'spaces' list a tuple or list is given instead of a Space instance, it's wrapped up in Nominal space."
        spaces = [s if isinstance(s, Space) else Nominal(s) for s in spaces]
        namedSpaces = [Nominal(*values, name = name) for name, values in namedSpaces.iteritems()]
        self.spaces = spaces + namedSpaces
        self.names = [s.name for s in self.spaces]
        if None in self.names: self.names = None            # if any subspace is unnamed, entire grid is treated as unnamed
    
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
            
    
#####################################################################################################################################################
###
###   KNOBS & LABELS
###

# class Label(object):
#     "A special attribute in a class that keeps meta data about a given object, for reporting purposes, e.g.: algorithm name, execution ID etc."
# 
# class Knob(object):
#     """Knob - a tunable named parameter that can be set on an object by the client. Identified by name, can be assigned specifically to many objects at once.
#     Also, knobs can keep special information, for reporting purposes, like algorithm name, execution ID etc.
#     In general, you should avoid using None as a legal knob value, since None is reserved for internal purposes to indicate no value.
#     """
#     def __init__(self, name, value = None):
#         self.name = name                                # full name of the knob; can have any form and contain any characters, only the last dot has special meaning (separates owner name from owner's attribute name)
#         self.value = value                              # value to be assigned to object attribute when setting the knob
#         self.owner, self.attr = self._owner(name)       # owner (class/object name) and attribute name parts of full 'name'
#         
#     def _owner(self, name):
#         "Extracts the owner (class/object name) part of 'name'. Owner name is the substring up to the last dot '.', or '' if no dot present."
#         if '.' not in name: return '', name
#         return name.rsplit('.', 1)
# 
#     def set(self, value):
#         "Set new value of the knob. Name of the knob is kept unchanged."
#         self.value = value
#         
# 
# class Knobs(list):
#     "List of Knob instances."
#     def __init__(self, *knobs, **namedKnobs):
#         "'knobs': a sequence of Knob objects, and/or (name,value) pairs to be converted into Knob-s, and/or just names (strings) of empty knobs."
#         for knob in knobs:
#             if isstring(knob): knob = Knob(knob)
#             elif not isinstance(knob, Knob): knob = Knob(*knob)
#             self.append(knob)
#         for name, value in namedKnobs.iteritems():
#             self.append(Knob(name, value))
# 
#     def set(self, values):
#         "Set new values of all knobs in 'self'; names kept unmodified."
#         if len(self) != len(values): raise Exception("Knobs.set(), the no. of new values to assign differs from the no. of knobs")
#         for i, v in enumerate(values):
#             self[i].set(v)
#         

"""
Special class attributes that can be read/written in massive way (in one go) in nested structures (pipelines, metapipes, ...).
 - Knobs:    (input parameters) passed from outside to inside; allow for massive write in.
 - Signals:  (output parameters) passed from inside to outside; allow for massive read out
 - Controls: (input meta-parameters) like knobs, but not directly involved in data processing, only controling the environment of algorithm execution; e.g., no. of threads to use, log file, ...
 - Labels:   (output meta-parameters) like signals, but containing meta-information unrelated to data processing

An attribute can serve as a knob and as a label at the same time.
"""

class Signaling(object):
    """Base class for all classes that contain labels or knobs and thus enable reporting (of labels) and parameterization with meta optimizers (of knobs).
       Subclasses should treat incoming knobs (passed from client in setKnobs) as *immutable* and must not make any modifications,
       since a given knobs instance can be reused multiple times by the client.
    """
    
    __labels__ = []         # names of attributes that serve as labels of a given class
    __knobs__  = []         # names of attributes that serve as knobs of a given class

    def setKnobs(self, knobs, strict = False):
        "Set given dict of 'knobs' onto 'self' and sub-cells. In strict mode, all 'knobs' must be recognized (declared as knobs) in self."
        if not self.__knobs__ and not strict: return
        for address, value in knobs.iteritems():
            attr = self.findAttr(address)
            if attr is None:
                if strict: raise Exception("Knob '%s' not present in '%s'" % (name, classname(self, full=True)))
            else:
                setattr(self, attr, value)
    
    def getInnerSignals(self, names = None):
        "Return dictionary of signals and their values present in 'self' or below."
    def getOuterSignals(self, names = None):
        "Signals from the environment: owner and beyond."
    def getSignals(self, names = None):
        return self.getInnerSignals(names) + self.getOuterSignals(names)
        
    def findAttr(self, addr):
        """Find attribute name in 'self' that corresponds to a given knob. None if the knob doesn't belong to self 
        (mismatch of class/instance specifier), or is undefined in self."""
        owner, attr = addr.rsplit('.', 1) if '.' in addr else ('', addr)        # owner name is the substring up to the last dot '.' of 'name'
        if owner and owner not in (classname(self), classname(self, full=True)): return None
        if attr not in self.__knobs__: return None
        return attr
    
#####################################################################################################################################################
###
###   DATA CELL
###

class __DataCell__(type):
    "Metaclass that generates DataCell subclasses, with reorganization of signals when needed."
    def __init__(cls, *args):
        cls.arrange('__knobs__')
        cls.arrange('__inner__')
        
    def arrange(cls, attrname):
        # create list of knobs/signals/... declared in this class
        items = getattr(cls, attrname, [])                          # list of attribute names representing items of a given type (knobs/signals/inner-cells...)
        if istype(items):                                           # inner class instead of a list?
            attrs = getattrs(items)
            setattrs(cls, attrs)                                    # copy all attrs from the inner class to top class level
            items = attrs.keys()                                    # collect attr names
        elif isstring(items):                                       # space-separated list of knob names?
            items = items.split()
    
        # append all declarations of knobs/signals/... from superclasses
        baseitems = [getattr(base, attrname, []) for base in cls.__bases__]
#         print cls.__name__, baseitems, items
        items = reduce(lambda x,y:x+y, baseitems) + items
        items = unique(items, order = True)
#        print cls.__name__, items
        setattr(cls, attrname, items)

class DataCell(Signaling):
    """Element of a data processing network. Participates in signalling pathways.
    Typically most cells are DataPipes and can appear in vertical (pipelines) as well as horizontal (nesting) relationships with other cells.
    Sometimes, there can be cells that are not pipes - they participate only in vertical relationships 
    and exhibit their own custom API for data input/output (instead of DataPipe's API).
    
    Pipes are elements of horizontal structures of data flow. Typically 1-1 or many-1 relationships.
    Cells are elements of vertical structures of control. Typically 1-1 or 1-many relationships.
    
    Every cell can contain "knobs": parameters configured from the outside by other cells (input parameters of the cell).
    Every cell can produce "signals" that can be read by other cells in DPN (output parameters of the cell). 
    Knobs and signals have a form of attributes located in a particular cell and identified by cell's name/path and attribute's name.
    """
    __metaclass__ = __DataCell__

    __signals__ = []
    __inner__ = []

    name = None                 # optional label, not necessarily unique, that identifies this cell instance or a group of cells during signal routing
    owner = None                # the cell that owns this one and which creates an environment where 'self' lives

    printlock = Lock()          # mutual exclusion of printing (on stdout); assign NoneLock to switch synchronization off

    def inner(self):
        "A generator or a list of all cells contained in (owned by) this one. Used in methods that need to apply a given operation to all subcells."
    
    def __str__(self):
        if not self.name: return classname(self)
        return "%s[%s]" % (self.name, classname(self))

    def _logStackTrace(self, method = None):
        dataStackTrace.log(self, method)
    def _sealStackTrace(self):
        dataStackTrace.dirty = True
    
class DataStackTrace(object):
    def __init__(self):
        self.dirty = False
        self.stack = []         # stack is a list of entries, one for each traced data cell + method
        self.exception = None   # the current exception being propagated right now; if a log with another exception comes in, the stack is reset to empty list
    def log(self, obj, method = None):
        #if ex is not self.exception: self.stack = []
        if self.dirty: 
            self.stack = []
            self.dirty = False
        entry = (obj, method)
        self.stack.append(entry)
    def __str__(self):
        if not self.stack: return "  Empty stack trace"
        def compile(entry):
            obj, method = entry
            if method is None: method = "<unknown>"
            return "  %s.%s" % (classname(obj), method)
        return '\n'.join(map(compile, self.stack))
    
dataStackTrace = DataStackTrace()

    
#####################################################################################################################################################
###
###   DATA PIPE
###

class __DataPipe__(__DataCell__):
    "Enables chaining of pipe classes (automatically instantiated without args), not only pipe instances."
    def __rshift__(cls, other):
        return cls() >> other
    def __lshift__(cls, other):
        return cls() << other


class DataPipe(DataCell):
    """Base class for data processing objects (pipes) that can be chained together to perform pipelined processing, 
    each one performing an atomic operation on the data received from preceding pipe(s), or being an initial source of data (loader, generator).
    Every iterable type - a collection, a generator or any type that implements __iter__() - can be used as a source pipe, too.
    
    Features of data pipes:
    - operator >>
    - use classes, not only instances, with >>
    - use collections, functions, ... with >> 
    """
    __metaclass__ = __DataPipe__
    
    source = None               # source DataPipe or iteratable from which input data for 'self' will be pulled
    count  = None               # number of input items read so far in a given execution, or 1-based index of the item currently being processed; tracked in most standard pipes (not all)
    iterating = False           # flag that protects against multiple iteration of the same pipe, at the same time
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __iter__(self):
        """Main method of every data pipe. Pulls data from the source(s), processes and yields results of processing one by one. 
        Subclasses should override either iter() or __iter__(), in the latter case the subclass is responsible for calling _prolog and _epilog.
        Generic subclasses also allow to override open() and close() to perform custom initialization and clean-up on every cycle of iteration.
        """
        self._prolog()
        try:
            for item in self.iter(): yield item
        except GeneratorExit, ex:                       # closing the iterator is a legal way to break iteration
            self._epilog()
            raise
        self._epilog()

    def iter(self):
        raise NotImplemented()

    def _prolog(self):
        if self.iterating: raise Exception("Data pipe %s opened for iteration twice, before previous iteration has been closed" % self)
        self.iterating = True
        self.count = 0
        self._logStackTrace("__iter__")
        
        self.open()

    def _epilog(self):
        self.close()
        self.iterating = False

    def open(self):
        "Called at the beginning of __iter__(). Can be overloaded in subclasses to perform per-cycle initialization."
    def close(self):
        "Called at the end of __iter__(). Can be overloaded in subclasses to perform per-cycle clean-up."
    
    def copy(self, deep = True):
        if deep: 
            if self.source: raise Exception("Deep copy called for a data pipe of %s class with source already assigned." % classname(self))
            return deepcopy(self) 
        return copy(self)
        
    def execute(self):
        """Pull all data through the pipe, but don't yield nor return anything. 
        Typically used for Pipelines which end with a sink and only produce side effects."""
        for item in self: pass

    def loop(self, times = None):
        """Executes this pipe multiple times, or infinitely if times=None. Yields items from all iterations as a single stream of data. 
        Client can read self.nloops - the no. of loops completed so far - to find out which loop is being executed right now.
        There is NO restarting between loops."""
        self.nloops = 0
        while True:
            if times != None and self.nloops >= times: break
            if self.nloops > 0: self.reopen()
            empty = True
            for item in self: 
                empty = False
                yield item
            self.nloops += 1
            if times == None and empty: raise Exception("Infinite loop over empty dataset")

    def reopen(self):
        "Override in subclasses to perform custom setup before next loop of processing is executed in loop()."
        
    def __rshift__(self, other):
        """'>>' operator overloaded, enables pipeline creation via 'a >> b >> c' syntax. Returned object is a Pipeline.
        Put RUN token at the end: a >> b >> RUN - to execute the pipeline immediately after creation."""
        # 'self' is a regular DataPipe; specialized implementation for Pipeline defined in the subclass
        if other is RUN:
            Pipeline(self).execute()
        else:
            return Pipeline(self, other)

    def __lshift__(self, other):
        if other is PIPE: return Pipeline(self)
        return Pipeline(other, self)

    def __add__(self, other):
        """Addition '+' operator creates a Union node that performs *sequential concatenation* of data streams from both sources into a single output stream.
        Note that in Python shifting operations have lower priority than arithmetic operations, so A+B >> C is interpreted as (A+B) >> C, as expected!"""
        return Union(self, other)

    def __mult__(self, other):
        """Multiplication '*' operator creates a Zip node that combines items from all input sources, 1 from each, into output tuples."""
        return Zip(self, other)

    def getSpec(self):
        "1-line string with technical specification of this pipe: its name and possibly values of its knobs etc."


# TOKENS

class _PIPE(DataPipe):
    def __rshift__(self, other): return Pipeline(other)

# Starts a pipeline. For easy appending of other pipes with automatic type casting: PIPE >> a >> b >> ...
# Doesn't do any processing itself (is excluded from the pipeline). 
PIPE = _PIPE()

# Invokes execution of a pipeline. When put at the end of a pipeline (a >> b >> ... >> RUN) indicates that it should be executed now. 
RUN = DataPipe()


#####################################################################################################################################################
###
###   Functional Pipes (base abstract classes)
###

class Operator(DataPipe):
    """Plain item-wise processing & filtering function, implemented in the subclass by overloading process() and possibly also open/close().
    Method process(item) returns modified version of the item or None to indicate that the item should be dropped (filtered out).
    For operators that only perform filtering, with no modification of items, see Filter class."""
    
    count = None            # during iteration holds the number of input items read so far; 1-based index of the item being currently processed
    
    def __iter__(self):
        self._prolog()
        try:
            for item in self.source: 
                self.count += 1
                res = self.process(item)
                if res is not None: yield res
        except GeneratorExit, ex:
            self._epilog()
            raise
        #except Exception, ex:
        #    self._sealStackTrace()
        #    raise
        self._epilog()
        
    def process(self, item):
        "Return modified item; or None, interpreted as no result (drop item). Subclasses can read self.count to get 1-based index of the current item."
        raise NotImplemented()
    
class Monitor(Operator):
    """An operator that (by assumption) doesn't modify input items, only observes the data and possibly produces side effects (collecting stats, logging etc).
    Subclasses override monitor(item) and possibly open/close() or report(); or monitoring function is passed as 'oper' upon Monitor instantiation."""
    def __iter__(self):
        self._prolog()
        self.count = 0
        try:
            for item in self.source: 
                self.count += 1
                self.monitor(item)          # unlike Operator, we don't expect any returned result from monitor()
                yield item                  # however, watch out for bugs: monitor() can implicitly modify internals of 'item' unless 'item' is immutable
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def monitor(self, item):
        self.process(item)              # for backward compatibility, process() is still called; TODO: remove process() and leave only monitor() in the future
    def process(self, item):
        "Can return modified item; or None, interpreted as no result (drop item); or True (pass unchanged); or False (drop item)."
        if self.oper is None: raise Exception("Missing operator function (self.oper) in class %s" % classname(self))
        self.oper(item)

    def close(self):
        with self.printlock: self.report()
    def report(self): pass


class Filter(Operator):
    "Doesn't change input items, but filters out undesired ones. process() must return True (pass the item through) or False (drop the item), not the actual object."
    def __iter__(self):
        self._prolog()
        try:
            for item in self.source: 
                self.count += 1
                if self.accept(item): yield item        # unlike Operator, we expect only True/False from accept/process(), not an actual data object
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        #except Exception, ex:
        #    self._sealStackTrace()
        #    raise
        self._epilog()

    def accept(self, item):
        return self.process(item)
    def process(self, item):                            # deprecated in Filter; override accept() instead
        return self.oper(item)


# class Sink(DataPipe):
#     "A pipe that only consumes data and never outputs any items."
#     def __iter__(self):
#         self.run()
#     def run(self):
#         "This method shall be overloaded in subclasses. You must iterate yourself over input items from self.source."
# 
# class Capacitor(DataPipe):
#     "Consumes all input data and only then starts producing output data, possibly of a different type, e.g. aggregates of input items."

class DataPile(DataPipe):
    """Permanent (buffered) storage of data, in memory or filesystem, that can be reused many times after one-time creation.
    Can provide random access to items via index (dict) or multiple indices.
    It's intentional that the term 'pile' resembles both 'pipe' and 'file'."""
    def build(self): pass
    def append(self): pass
    def load(self): pass
    def __getitem__(self, key): pass


#####################################################################################################################################################
###
###   Specialized Pipes (concrete classes)
###

###  Wrappers for standard Python objects

class Collection(DataPipe):
    """Wrapper for plain collections or iterators, to turn them into DataPipes that can be used as sources in a pipeline. 
    In subclasses, set 'self.data' with the iterable to take data from; optionally override open() to initialize 'data' just before iteration starts,
    but note that close() is not called (this would require control over iteration process and yielding items one-by-one, 
    instead of following back on the collection's own iterator).
    """
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        self.open()
        return iter(self.data)

class File(DataPipe):
    """Wrapper for a file object opened for reading. Iteration delegates to file.__iter__(). 
    Transparently repositions file pointer when iteration restarts. The file is never closed."""
    def __init__(self, file):
        "'f' is a file object."
        self.file = file
    def __iter__(self):
        if isinstance(self.file, file): self.file.seek(0)
        else: self.file.reopen()
        return iter(self.file)
    
class Function(Operator):
    "An operator constructed from a plain python function. A wrapper."
    def __init__(self, oper = None):
        "oper=None handles the case when processing function is implemented through overloading of process()."
        self.oper = oper
    def process(self, item):
        "Can return modified item; or None, interpreted as no result (drop item); or True (pass unchanged); or False (drop item)."
        if self.oper is None: raise Exception("Missing operator function (self.oper) in %s" % self)
        ret = self.oper(item)
        if ret is True: return item
        if ret is False: return None
        return ret


###  Generators

class Empty(DataPipe):
    "Generates an empty output stream. Useful as an initial pipe in an incremental sum of pipes: p = Empty; p += X[0]; p += X[1] ..."
    def __iter__(self):
        return; yield

class Repeat(DataPipe):
    "Returns a given item for the specified number of times, or endlessly if times=None. Like itertools.repeat()"
    def __init__(self, item, times = None):
        self.item = item
        self.times = times
    def __iter__(self):
        item = self.item
        if self.times is None:
            while True: yield item
        else:
            for _ in xrange(self.times): yield item
    
class Range(DataPipe):
    "Generator of consecutive integers, equivalent to xrange(), same parameters."
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return iter(xrange(*self.args))


###  Filters

class Slice(DataPipe):
    "Like slice() or itertools.islice(), same parameters. Transmits only a slice of the input stream to the output."
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return islice(self.source, *self.args)

class Offset(DataPipe):
    "Drop a predefined number of initial items"
    def __init__(self, offset):
        self.offset = offset
    def __iter__(self):
        count = 0
        for item in self.source: 
            count += 1
            if count <= self.offset: continue
            yield item

class Limit(DataPipe):
    "Terminate the data stream after a predefined number of items. 'Head' is an alias."
    def __init__(self, limit):
        self.limit = limit
    def __iter__(self):
        count = 0
        if count >= self.limit: return
        for item in self.source: 
            yield item
            count += 1
            if count >= self.limit: return
Head = Limit

class DropWhile(DataPipe):
    "Like itertools.dropwhile()."
class TakeWhile(DataPipe):
    "Like itertools.takewhile()."    
class StopOn(DataPipe):
    """Terminate the data stream when a given condition becomes True. Condition is a function that takes current item as an argument. 
    This function can also keep an internal state (memory)."""


class Subset(DataPipe):
    """Selects every 'fraction'-th item from the stream, equally spaced, yielding <= 1/fraction of all data. Deterministic subset, no randomization.
    >>> Range(7) >> Subset(3) >> Print >> RUN
    2
    5
    """
    def __init__(self, fraction):
        self.fraction = fraction
    def __iter__(self):
        frac = self.fraction
        count = 0                               # count iterates in cycles from 0 to 'frac', and again from 0 ...
        for item in self.source: 
            count += 1
            if count == frac: 
                yield item
                count = 0

class Sample(DataPipe):
    "Random sample of input items. Every input item is decided independently with a given probability, unconditional on what items were chosen earlier."
    

###  Buffers

class Buffer(DataPipe):
    "Upon build(), buffer all input data in memory. Then, when data is buffered, can iterate (multiple times) over it and yield from memory."

class Sort(DataPipe):
    """Total or partial in-memory heap sort of the input stream. Buffer items in a heap and when the heap is full, output them in sorted order. 
    Heap size can be unlimited (default), which results in total sorting: output items appear only after all input data was consumed; 
    or limited to a predefined maximum size (partial sort, generation of output items begins as soon as the heap achieves its maximum size).
    >>> Collection([2,7,3,6,8,3]) >> Sort(2) >> List >> Print >> RUN
    [2, 3, 6, 7, 3, 8]
    """
    def __init__(self, size = None):
        self.size = size
    def __iter__(self):
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


###  Reporting

class Print(Monitor):
    """Print items passing through, or value of 'func' function calculated on each item and/or a static message. If outfile is given, additionally print to that file (in such case, stdout can be suppressed).
    Subclasses can override monitor(item) and open/close() to provide custom printing: use self.out as the output stream: 'print >>self.out, ...' 
    - it redirects to stdout and/or file, appropriately."""

    outfile = None
    subset = None

    def __init__(self, msg = "%s", func = None, outfile = None, disp = True, count = False, subset = None):
        """msg: format string of messages, may contain 1 parameter %s to print the data item in the right place.
        func: optional function called on every item before printing, its output is printed instead of the actual item.
        outfile: path to external output file or None; disp: shall we print to stdout?; count: shall we print 1-based item number at the begining of a line?
        subset: print every n-th item (see Subset), or None to print all items."""
        Monitor.__init__(self)
        #if '%s' not in msg: msg += ' %s'
        self.static = ('%s' not in msg)
        self.message = msg
        self.func = func
        if outfile: self.outfile = outfile
        self.disp = disp
        self.index = count
        if subset: self.subset = subset
        #print "created Print() instance, message '%s'" % self.message

    def open(self):
        if self.outfile and self.disp: self.out = Tee(self.outfile) 
        else: self.out = open(self.outfile, 'wt') if self.outfile else sys.stdout
        self.open1()
    
    def open1(self):
        "Override in subclasses to extend opening procedure with own operations."

    def monitor(self, item):
        #print "Print.monitor()", self.subset, self.index
        if self.subset and self.count % self.subset != 0: return
        with self.printlock:
            if self.index: print >>self.out, self.count,
            self.print1(item)
        
    def print1(self, item):
        #print "Print.print1()"
        if self.static: print >>self.out, self.message
        else:
            if self.func is not None: item = self.func(item)
            print >>self.out, self.message % item

    def close(self):
        if self.out != sys.stdout: self.out.close()

def PrintSubset(subset, func = None, count = True, outfile = None, disp = True):
    "A shorthand for Print with 'subset' argument. Differs from Print only in the order of __init__ arguments."
    return Print(subset = subset, func = func, count = count, outfile = outfile, disp = disp)

class Count(Monitor):
    "Counts data items and prints the index before yielding each consecutive item. Indexing is 1-based by default, this can be changed. No newline added by default."
    def __init__(self, step = 1, msg = "%d", start = 1):
        "If step is float in (0,1), adaptive step is used: next step is 'step' fraction of the total no. of items read so far."
        Monitor.__init__(self)
        if '%d' not in msg: msg = '%d ' + msg
        self.message = msg
        if isint(step):
            self.step, self.frac = (step, None)
        else:
            self.step, self.frac = (None, step)
    def open(self):
        self.next = self.step or 1
    def monitor(self, _):
        if self.count >= self.next:
            with self.printlock:
                print self.message % self.count,
            if self.step: self.next = self.count + self.step
            else: 
                step = int(self.count * self.frac) + 1
                digits = int(math.log10(step))              # no. of digits after the 1st one - that many should be zeroed out in 'next'
                self.next = self.count + step
                self.next = int(round(self.next, -digits))
def Countn(*args, **kwargs):
    "'Count with newline': like Count, but adds newline instead of space after each printed number."
    return Count(*args, msg = '%d\n', **kwargs)
        
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
    def __init__(self, msg = "Total data items: %d"):
        Monitor.__init__(self)
        self.msgTotal = msg
        if msg:
            try: msg % 0
            except: raise Exception("Incorrect message string, must contain %%d parameter: %s" % msg)
    def monitor(self, item): pass
    def close(self):
        with self.printlock: print self.msgTotal % self.count

class Time(Monitor):
    "Measure time since the beginning of data iteration. If 'message' is present, print the total time at the end, embedded in 'message'."
    def __init__(self, message = "Time elapsed: %.1f s"):
        Monitor.__init__(self)
        self.msgTime = message
        self.start = None
    def __iter__(self):
        self.start = time()
        for item in self.source: yield item
        if self.msgTime: 
            with self.printlock: print self.msgTime % self.elapsed()
    def elapsed(self):
        "Time elapsed in seconds, as float."
        return time() - self.start
    
class Report(Total, Time):
    "'Total' and 'Time' combined."
    def __init__(self, header = "\n========================="):
        Total.__init__(self)
        Time.__init__(self)
        self.header = header
    def __iter__(self):
        self.start = time()
        self.total = 0
        for item in self.source: 
            self.total += 1
            yield item
        with self.printlock:
            if self.header: print self.header
            if self.msgTotal: print self.msgTotal % self.total
            if self.msgTime: print self.msgTime % self.elapsed()

class Metric(Operator):
    "Calculates a given metric on input samples and appends to them on the output. Typically used for evaluation."

class Experiment(Monitor):
    "Provides subclasses with logging facilities."


###  File access

class Save(Monitor):
    "Writes all passing items to a predefined file and outputs them unmodified to the receiver. Item type must be compatible with the file's write() method."
    def __init__(self, outfile):
        self.outfile = outfile
    def monitor(self, item):
        self.outfile.write(item)
        

class Pile(DataPile):
    """DataPipe wrapper around file objects, for use of files in pipelines and with pipe operators.
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
        if self.source: return self._write(self.source)
        else: return self._read()
        
    def _write(self, source):
        if isstring(self.file):
            mode = 'at' if self.append else 'wt'
            rawclass = SafeRewriteFile if self.rewrite else files_File
            f = self.fileclass(self.file, mode = mode, cls = rawclass, flush = self.flush, emptylines = self.emptylines)
        else:
            f = self.file
            f.open()
        
        for item in source:
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
        
#     def _write(self, source):
#         mode = 'at' if self.append else 'wt'
#         fileclass = SafeRewriteFile if self.rewrite else open
#         out = fileclass(self.filename, mode)
#         
#         self.flushcount = self.flush
#         for item in source:
#             self.write1(out, item)
#             self.flushcount -= 1
#             if self.flushcount == 0:
#                 out.flush()
#                 self.flushcount = self.flush
#             yield item
#         out.close()
#     def write1(self, out, item):
#         "Override in subclass to perform write of a single data item to an open file 'out'."        
        
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

class List(DataPipe):
    "Combines all input items into a list. At the end, this list is output as the only output item; it's also directly available as self.items property."
    def __iter__(self):
        self.items = []
        for item in self.source:
            self.items.append(item)
        yield self.items

        
#####################################################################################################################################################
###
###   Structural Pipes
###

class MetaPipe(DataPipe):
    "Wraps up a number of pipes to provide meta-operations on them."

class Container(MetaPipe):
    "Base class for classes that contain multiple pipes inside: self.pipes."
    
    pipes = []          # any collection of internal pipes
    
    def copy(self, deep = True):
        "Smart shallow copy (in addition to deep copy). In shallow mode, copies the collection of pipes, too, so it can be modified afterwards without affecting the original."
        res = super(Container, self).copy(deep)
        if not deep: res.pipes = copy(self.pipes)
        return res
        
    def setKnobs(self, knobs, strict = False):
        super(Container, self).setKnobs(knobs, strict)          # Container itself has no knobs, but a subclass can define some
        for pipe in self.pipes:
            try:
                if hasattr(pipe, 'setKnobs'): pipe.setKnobs(knobs, strict)
            except:
                print pipe
                raise


class Wrapper(MetaPipe):
    "Like Container, but always has only 1 internal pipe: self.pipe."
    
    pipe = None
    
    def setKnobs(self, knobs, strict = False):
        super(Wrapper, self).setKnobs(knobs, strict)            # Wrapper itself has no knobs, but a subclass can define some
        self.pipe.setKnobs(knobs, strict)
        

#####################################################################################################################################################

class Pipeline(Container):
    "Sequence of data pipes connected sequentially, one after another. Pipeline is a MetaPipe and a DataPipe itself."
    
    def __init__(self, *pipes, **kwargs):
        "'kwargs' may contain connected=True to indicate that pipes are already connected into a list or a tree (must be sorted in topological order!)."
        if pipes and islist(pipes[0]): pipes = pipes[0]
        self.pipes = list(pipes)
        #if kwargs.get("connected"): return
        
    def __rshift__(self, other):
        "Append 'other' to the end of the pipeline. Shallow-copy the pipeline beforehand, to avoid in-place modifications and preserve '>>' protocol."
        if other is RUN:
            self.execute()
        else:
            res = self.copy(False)
            res.pipes.append(other)
            return res

    def __lshift__(self, other):
        "Like __rshift__, but 'other' is put at the tail not head of the pipeline."
        res = self.copy(False)
        res.pipes = [other] + res.pipes
        return res

    def setKnobs(self, knobs, strict = False):
        self.pipes = _normalize(self.pipes)                 # if knobs to be configured, pipes must be normalized beforehand (normally __iter__ does this)
        super(Pipeline, self).setKnobs(knobs, strict)

    def __iter__(self):
        # normalize pipes; connect into a list; connect entire pipeline with the source
        self.pipes = _normalize(self.pipes)
        prev = self.source
        for next in self.pipes:
            if prev is not None: next.source = prev         # 1st pipe can be a generator or collection, not necessarily a DataPipe (no .source attribute)
            prev = next
            
        # pull data
        head = self.pipes[-1]
        for item in head: yield item
    
    def __str__(self):
        return "Pipeline " + ' >> '.join(map(str, self.pipes))
    

#####################################################################################################################################################

class MultiSource(DataPipe):
    def copy(self, deep = True):
        "Shallow copy does copy the list of pipes too (list can be modified afterwards without affecting the original)."
        if deep: return deepcopy(self)
        res = copy(self)
        res.sources = copy(self.sources)
        return res

class Parallel(MultiSource):
    "Connects multiple pipes as parallel routes between a single source and a single destination."
    def __init__(self, source, handlers):
        if handlers and islist(handlers[0]): handlers = handlers[0]
        self.pipes = _normalize(handlers)
        for h in handlers:
            h.source = source

class Hub(MultiSource):
    "Branching of pipes: a central pipe (hub) with multiple parallel sources."
    def __init__(self, sources, hub):
        self.sources = hub.sources = _normalize(sources)
        self.hub = hub
        self.pipes = self.sources + [hub]            # for open() & close()

class Union(MultiSource):
    """Combine items from multiple sources by concatenating corresponding streams one after another: all items from the 1st source; then 2nd... then 3rd...
    TODO: 'mixed' mode (breadth-first, streams interlaced)."""
    def __init__(self, *sources):
        self.sources = _normalize(sources)
    def __iter__(self):
        return itertools.chain(*self.sources)
    def __add__(self, other):
        res = self.copy(False)
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
    def __iter__(self): pass
    
class MergeSort(MultiSource):
    """Merge multiple sorted inputs into a single sorted output. Like heapq.merge(), but wrapped up in a DataPipe. 
    If only the input streams were fully sorted, the result stream is guaranteed to be fully sorted, too."""
    def __init__(self, *sources):
        if len(sources) == 1 and islist(sources[0]):
            self.sources = sources[0]
        else:
            self.sources = sources
    def __iter__(self):
        for item in heapq.merge(*self.sources): yield item

#####################################################################################################################################################

class MetaOptimize(Wrapper):
    """Client MUST ensure that internal pipe does NOT modify input items. Otherwise, only the 1st copy of the pipe will work on correct input data, 
    others will receive broken input. Does not yield anything, may only produce side effects."""
    
#class GridSearch(MetaOptimize):
#    "Sequential exhaustive search over space of all possible knob values. Input data is reloaded from scratch for every new algorithm execution."

class GridSearch(MetaOptimize):
    """Exhaustive search over space of all possible knob values, executed in mixed serial-parallel mode, depending on 'maxThreads' setting. 
    In parallel mode, the algorithm must NOT modify data items, otherwise there will be interference between concurrent threads.
    No output produced, only empty stream.
    """
    
    runID        = "runID"              # name of a special knob inside 'pipe' that will be set with an ID (integer >= 1) of the current run
    startID      = 0                    # ID of the first run, to start counting from
    maxThreads   = 0                    # max. no. of parallel threads; <=1 for serial execution; None for full parallelism, with only 1 scan over input data
    threadBuffer = 10                   # length of input queue in each thread; 1 enforces perfect alignment of threads execution; 
                                        # 0: no alignment, queues can get big when input data are produced faster than consumed
    
    def __init__(self, pipe, *args, **kwargs):
        "'grid' is either a Grid or another Space instance, or a list of tuples to be converted into a Grid."
        self.pipe = pipe                                    # the pipe(line) to be copied and executed, in parallel, for different combinations of knob values
        self.runID = kwargs.pop('runID', self.runID)
        self.startID = kwargs.pop('startID', self.startID)
        self.maxThreads = kwargs.pop('maxThreads', self.maxThreads)
        self.threadBuffer = kwargs.pop('threadBuffer', self.threadBuffer)
        
        if len(args) == 1 and isinstance(args[0], Space) and not kwargs:
            self.grid = args[0]
        else:
            self.grid = Grid(*args, **kwargs)               # value space of all possible knob values
        self.done = None                                    # no. of runs completed so far
    
    def __iter__(self):
        self.done = 0
        if self.maxThreads is None or self.maxThreads > 1:
            self.iterParallel()
        else:
            self.iterSerial()
        return; yield                                       # to make this method work as a generator (only an empty one)
        
    def iterSerial(self):
        with self.printlock: print "GridSearch: %d serial runs to be executed..." % len(self.grid)
        for i, value in enumerate(self.grid):
            self.scanSerial(value, self.startID + i)
            self.done += 1
        
    def iterParallel(self):
        scans = divup(len(self.grid), self.maxThreads) if self.maxThreads else 1
        with self.printlock: print "GridSearch: %d runs to be executed, in %d scan(s) over input data..." % (len(self.grid), scans)
        
        def knobsStream():
            for i, v in enumerate(self.grid):
                yield self.createKnobs(self.startID + i, v)

        def knobsGroups():
            "Take the 'knobsStream' and partition into groups of up to 'maxThreads' size each."
            if not self.maxThreads: 
                yield knobsStream()
                return
            group = []
            for knob in knobsStream():
                group.append(knob)
                if len(group) >= self.maxThreads:
                    yield group
                    group = []
            if group: yield group

        for kgroup in knobsGroups():
            self.scanParallel(kgroup)
            self.done += len(kgroup)

    def createKnobs(self, ID, values):
        return [(self.runID, ID)] + zip(self.grid.names, values)
        
    def createPipe(self, knobs, copy = True):
        pipe = self.pipe.copy() if copy else self.pipe
        pipe.setKnobs(dict(knobs))
        return pipe        
        
    def scanSerial(self, value, ID):
        """Single scan over input data, with items passed directly to the single pipe being executed. No parallelism, no multi-threading, no pipe copying.
        Note that the same pipe object will be reused in all runs - watch out against interference between consecutive runs.
        """
        with self.printlock: print "GridSearch, starting next serial scan for run ID=%s..." % ID
        knobs = self.createKnobs(ID, value)
        pipe = self.createPipe(knobs, copy = False)
        self.printKnobs(knobs)
        PIPE >> self.source >> pipe >> RUN
    
    def scanParallel(self, knobsGroup):
        "Single scan over input data, with each item fed to a group of parallel threads."
        with self.printlock: print "GridSearch, starting next parallel scan for %d runs beginning with ID=%s..." % (len(knobsGroup), knobsGroup[0][0][1])
        self.count = 0
        threads = self.createThreads(knobsGroup)
        for item in self.source: 
            self.count += 1
            for _, pipe in threads: pipe.put(item)              # feed input data to each pipe in parallel
        self.closeThreads(threads)
        
    def createThreads(self, knobsGroup):
        threads = []                                            # list of pairs: (knobs, pipe_thread)
        for knobs in knobsGroup:                                # create a copy of the template for each combination of knob values; wrap up in threads
            pipe = self.createPipe(knobs)
            thread = DataThread(pipe, self.threadBuffer)
            thread.start()
            threads.append((knobs, thread))
        return threads
        
    def closeThreads(self, pipes):
        for _, pipe in pipes:
            pipe.emptyFeed()                    # wait until all pipes eat up all remaining input items; note: some pipes may still be processing the last item!
        sleep(1)
        
        with self.printlock: print "GridSearch, %d runs done." % (self.done + len(pipes))
        for knobs, pipe in pipes:
            self.printKnobs(knobs)
            pipe.end()
            pipe.join()

    def printKnobs(self, knobs):
        with self.printlock:
            print "----------------------------------------------------------------"
            print ' '.join("%s=%s" % knob for knob in knobs)            # space-separated list of knob values
    

class Evolution(MetaOptimize):
    "Evolutionary algorithm for (meta-)optimization of a given signal of a pipe through tuning of its knobs."
    
    
#####################################################################################################################################################
###
###   Functional wrappers (operators)
###

# def pipeline(*pipes):
#     "Connect the pipeline and execute it, in one step."
#     Pipeline(pipes).execute()

class NoData(Exception):
    "Raised to indicate that there is no input data to fulfill the request in operator()."

class DataFeed(DataPipe):
    data = None
    def set(self, *items):
        self.data = items
    def setList(self, items):
        self.data = items
    def __iter__(self):
        while True:
            items = self.data
            self.data = None
            if items is None: raise NoData
            for item in items: yield item

# def operator(*pipes):
#     """Functional wrapper for a pipe or pipeline that enables manual pushing of items to the pipe input, 1 at a time. Can be used in the same thread as a caller. 
#     Pipes must pull exactly 1 item at a time from the pipeline source. If they pull more or less than this, exception will be raised.
#     Pipes must not rely on being properly closed, as operator access prevents correct closing of the pipeline.
#     """
#     feed = DataFeed()
#     pipeline = Pipeline([feed] + pipes).__iter__()
#     def process(item):
#         feed.set(item)
#         return pipeline.next()
#     return process

class operator(object):
    """Functional wrapper for a pipe or pipeline: enables manual pushing of items to pipe input, 1 at a time, like to a function (operator),
    and reading the output as a returned value. Can be used in the same caller's thread, no need to spawn new thread. 
    Pipes must pull exactly 1 item at a time from the pipeline source. If they pull more or less than this, exception will be raised.
    """
    def __init__(self, *pipes):
        self.feed = DataFeed()
        self.pipeline = Pipeline([self.feed] + pipes)
        self.process = self.pipeline.__iter__()
    def __call__(item):
        self.feed.set(item)
        return self.process.next()
    def close(self):
        self.process.close()
    def reopen(self):
        self.close()
        self.process = self.pipeline.__iter__()
        
    write = push = send = __call__              # aliases, when explicit method call, like op.write(item), is more suitable than op(item)


#####################################################################################################################################################

class DataThread(Thread, Wrapper):
    END = object()      # token to be put into a queue (input or output) to indicate end of data stream
    
    class Feed(DataPipe):
        "A data-pipe wrapper around threading queue for input data."
        def __init__(self, queue):
            self.queue = queue
        def __iter__(self):
            while True: 
                item = self.queue.get()
                self.queue.task_done()
                if item is DataThread.END: break
                yield item
        def join(self):
            "Block until all items in the feed have been retrieved. Note: more items can still be added afterwards."
            self.queue.join()
    
    def __init__(self, pipe, insize = None, outsize = None):
        """insize, outsize: maximum capacity of in/out queues, 0 for unlimited, None for no queue (default). 
        If insize=None, input items must be generated by the 'pipe' itself.
        If outsize=None, output items are dropped.
        """
        Thread.__init__(self)
        self.pipe = pipe                # pipe(line) to be executed in the thread
        self.input = Queue(insize) if insize is not None else None
        self.output = Queue(outsize) if outsize is not None else None
        self.feed = None
    
    def run(self):
        # connect the pipe with input data feed
        if self.input:
            self.feed = DataThread.Feed(self.input)
            pipeline = Pipeline(self.feed, self.pipe)
        else:
            pipeline = self.pipe
            
        # run the pipeline, optionally pushing output items to self.output
        if self.output:
            for item in pipeline: self.output.put(item)
            self.output.put(DataThread.END)
        else:
            for item in pipeline: pass
    
    # to be used by calling thread...
    
    def put(self, *args): self.input.put(*args)
    def get(self, *args): return self.output.get(*args)
    def end(self): 
        "Terminates the stream of input data"
        self.input.put(DataThread.END)                      # put a token that marks end of input data
    
    def emptyFeed(self): self.feed.join()
    

#####################################################################################################################################################

def _normalize(pipes):
    """Normalize a given list of pipes. Remove None's and strings (used for commenting out), 
    instantiate DataPipe classes if passed instead of an instance, wrap up functions, collections and files."""
    def convert(h):
        if issubclass(h, DataPipe): return h()
        if isfunction(h): return Function(h)
        if iscontainer(h): return Collection(h)
        if isinstance(h, (file, GenericFile)): return File(h)
        if isstring(h): return None
        return h
    return filter(None, map(convert, pipes))
    
#####################################################################################################################################################
###
###   NUMPY ARRAYS
###

import numpy.linalg as linalg

norm = linalg.norm

###   2D DATA   ###

def zeroSum(X):
    "Shift values in each row to zero out in-row sums."

def unitSum(X):
    "Scale 1D vector X, or all rows of 2D array X, to unit sum. All sums must be originally non-zero."
    if X.ndim == 1:
        return X / np.sum(X)
    if X.ndim == 2:
        scale = 1. / np.sum(X,1)
        scale = scale[:,np.newaxis]
        return X * scale                # numpy "broadcasting" activates here, it automatically copies 'scale' to all columns
    
def unitNorm(X, p = 2):
    ""

#####################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()

    
