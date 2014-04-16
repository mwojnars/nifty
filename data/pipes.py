'''
Data pipes. 
Allow construction of complex networks (pipelines) of data processing units, that combined perform advanced operations, 
and process large volumes of data (data streams) efficiently thanks to pipelining (processing one item at a time, instead of a full dataset).

- All functional pipes (Operator, Monitor, Filter and subclasses) accept plain python functions 
  as an argument to initializer. This function is called if the core method was left unimplemented.

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import sys, heapq, math, numpy as np, jsonpickle, csv, itertools, threading
from copy import copy, deepcopy
from time import time, sleep
from Queue import Queue
from itertools import islice
from collections import OrderedDict

from nifty.util import isint, islist, isstring, issubclass, isfunction, iscontainer, istype, \
                       classname, getattrs, setattrs, divup, Tee, openfile
from nifty.util import Object, __Object__, NoneLock
from nifty.files import GenericFile, File as files_File, SafeRewriteFile, ObjectFile, JsonFile, DastFile


#####################################################################################################################################################
###
###   SPACE and GRID of values
###

class Space(object):
    """A 1D set of values allowed for a given knob. Typically used to restrict the set of all real numbers to a discrete (finite or infinite) evenly-spaced subset.
    None is disallowed as a value (plays a special role). Can define also a probability density function on the space.
    """
    #name = None                 # optional name of the parameter represented by this space
    
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

class Singular(Space):
    "Space with only 1 value. The value can be of any type, no particular properties are required."
    def __init__(self, value): self.value = value
    def __len__(self): return 1
    def __iter__(self): yield self.value
        

class Nominal(Space):
    "A finite discrete set of nominal values of any type (strings, numbers, ...), with no natural ordering. The ordering of values in the collection is used."
    def __init__(self, *values, **kwargs):
        #self.name = kwargs.pop('name')
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
    
class Cartesian(Space):
    "Cartesian product of multiple Spaces. A multidimentional set of allowed values for a given combination of knobs. Each multi-value is a tuple."
    
    def __init__(self, *spaces, **namedSpaces):
        "If any given space is a tuple or list instead of a Space instance, it's wrapped up in Nominal or Singular."
        def wrap(space):
            if isinstance(space, Space): return space
            if not islist(space): return Singular(space)
            if len(space) == 1: return Singular(space[0])
            return Nominal(*space)
            
        self.spaces = [wrap(s) for s in spaces]
#         namedSpaces = [Nominal(*values, name = name) for name, values in namedSpaces.iteritems()]
#         self.spaces = spaces + namedSpaces
#         self.names = [s.name for s in self.spaces]
#         if None in self.names: self.names = None            # if any subspace is unnamed, entire grid is treated as unnamed
    
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

class Knobs(OrderedDict):
    "An ordered dictionary of {name:value} of knobs, representing one particular knobs combination."


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
Special class properties that can be extracted or assigned in a massive way (in one go) in nested structures (pipelines, metapipes, ...),
without the need to specify exact access paths to target objects, only with global addressing (naming) of the elements of the structure.
 - Knobs:    (input parameters) passed from outside to inside; allow for massive write in.
 - Signals:  (output parameters) passed from inside to outside; allow for massive read out
 - Controls: (input meta-parameters) like knobs, but not directly involved in data processing, only controling the environment of algorithm execution; e.g., no. of threads to use, log file, ...
 - Labels:   (output meta-parameters) like signals, but containing meta-information unrelated to data processing itself

An attribute can serve as a knob and as a label at the same time.
"""

#####################################################################################################################################################
###
###   DATA CELL
###

class __Cell__(__Object__):
    "Metaclass for generating Cell subclasses. Sets up the lists of knobs and inner cells."
    def __init__(cls, *args):
        super(__Cell__, cls).__init__(*args)
        cls.label('__knobs__')
        cls.label('__inner__')


class Cell(Object):
    """Element of a data processing network. Participates in signalling pathways.
    Typically most cells are Pipes and can appear in vertical (pipelines) as well as horizontal (nesting) relationships with other cells.
    Sometimes, there can be cells that are not pipes - they participate only in vertical relationships 
    and exhibit their own custom API for data input/output (instead of Pipe's API).
    
    Pipes are elements of horizontal structures of data flow. Typically 1-1 or many-1 relationships.
    Cells are elements of vertical structures of control. Typically 1-1 or 1-many relationships.
    
    Every cell can contain "knobs": parameters configured from the outside by other cells, e.g., by meta-optimizers; input parameters of the cell.
    Every cell can produce "signals" that can be read by other cells in DPN; output parameters of the cell. 
    Knobs and signals have a form of attributes located in a particular cell and identified by cell's name/path and attribute's name.
    Knobs and signals enable implementation of generic meta-algorithms which operate on inner structures without exact knowledge of their identity,
    only by manipulation of knobs and signals.
    
    Addressing.
    - "experiment/model/submodel.knob" - slash indicates inclusion: inner cell included in an outer cell
    - "experiment/*"  - star indicates the current cell and all inner cells, but not nested ones
    - "experiment/**" - like '*', additionally includes nested cells
    - "experiment/+"  - like '*', but excludes current (outer) cell
    - "experiment/++"
    
    Things to keep in mind:
    - properties defined inside inner classes, __knobs__ and __inner__, are automatically copied to parent class level 
      after class definition.
    """
    __metaclass__ = __Cell__

    __labels__ = []         # names of attributes that serve as labels of a given class
    __knobs__  = []         # names of attributes that serve as knobs of a given class; list, string, or class __knobs__: ...
    __signals__ = []
    __inner__ = []

    name = None             # optional label, not necessarily unique, that identifies this cell instance or a group of cells in signal routing
    owner = None            # the cell which owns 'self' and creates an environment where 'self' lives; typically 'self' is present in owner.__inner__
    verbose = None          # pipe-specific setting that controls how much debug information is printed during pipe operations

    printlock = threading.Lock()          # mutual exclusion of printing (on stdout); assign NoneLock to switch synchronization off

    def __init__(self, *args, **knobs):
        """The client can pass knobs already in __init__, without manual call to setKnobs. 
        Unnamed args passed down to custom init()."""
        self.name = knobs.pop('name', None)
        if knobs: self.setKnobs(knobs)
        self.init(*args)

    def init(self):
        "Override in subclasses to provide custom initialization, without worrying about calling super __init__."

    def copy(self, deep = True):
        """Shorhand for copy(self) or deepcopy(self).
        In Pipes, 'source' is excluded from copying and the returned pipe has source UNassigned, 
        even when deep copy (configured in __transient__ and handled by Object.__getstate__)."""
        if deep: 
            #if self.source: raise Exception("Deep copy called for a data pipe of %s class with source already assigned." % classname(self))
            return deepcopy(self) 
        return copy(self)
        
    def copy1(self):
        """Copying 1 step deeper than a shallow copy. Copies all attribute values, too, 
        so collections (pipes, knobs) can be modified afterwards without affecting original ones."""
        res = copy(self)
        d = res.__dict__
        for k, v in d.iteritems():
            d[k] = copy(v)
        return res
        
    def getKnobs(self):
        "Dict with current values of all the knobs of 'self'."
        return {name:getattr(self,name) for name in self.__knobs__}

    def setKnobs(self, knobs, strict = False):
        """Set given dict of 'knobs' onto 'self' and sub-cells. In strict mode, all 'knobs' must be recognized 
        (declared as knobs) in self.
        Subclasses should treat objects inside 'knobs' as *immutable* and must not make any modifications,
        since a given knob instance can be reused multiple times by the client.
        """
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
    
    # DRAFT
    def walkInner(self, oper, **kwargs):
        "Executes given 'oper' in self and all inner cells, visiting cells in post-order."
    def walkOuter(self, oper, **kwargs): pass
    
    def getInnerSignals(self, names = None):
        "Return dictionary of signals and their values present in 'self' or below."
    def getOuterSignals(self, names = None):
        "Signals from the environment: owner and beyond."
    def getSignals(self, names = None):
        return self.getInnerSignals(names) + self.getOuterSignals(names)
        

    def save(self):
        "Serialization of the cell."
        raise NotImplemented()
    def load(self):
        raise NotImplemented()
    
    def __str__(self):
        if not self.name: return classname(self)
        return "%s[%s]" % (self.name, classname(self))

    def _pushTrace(self, pipe): trace.push(pipe)
    def _popTrace (self, pipe): trace.pop(pipe) if trace else None      # 'if' necessary for unit tests to avoid strange error messages
    
# class StackTrace(list):
#     def __init__(self):
#         self.dirty = False
#         self.stack = []         # stack is a list of entries, one for each traced data cell + method
#         self.exception = None   # the current exception being propagated right now; if a log with another exception comes in, the stack is reset to empty list
#     def log(self, obj, method = None):
#         #if ex is not self.exception: self.stack = []
#         if self.dirty: 
#             self.stack = []
#             self.dirty = False
#         entry = (obj, method)
#         self.stack.append(entry)
#     def __str__(self):
#         if not self.stack: return "  Empty stack trace"
#         def compile(entry):
#             obj, method = entry
#             if method is None: method = "<unknown>"
#             return "  %s.%s" % (classname(obj), method)
#         return '\n'.join(map(compile, self.stack))

class OpenPipes(list):
    """List of pipes whose __iter__ is currently running. Usually these pipes form a chain and their __iter__ 
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

trace = OpenPipes()

    
#####################################################################################################################################################
###
###   DATA PIPE
###

class __Pipe__(__Cell__):
    "Enables chaining (pipelining) of pipe classes, not only instances. >> and << return a Pipeline instance."
    def __rshift__(cls, other):
        if other is RUN: Pipeline(cls).execute()
        else: return Pipeline(cls, other)
        
    def __lshift__(cls, other):
        return cls() << other
        if other is PIPE: return Pipeline(cls)
        return Pipeline(other, cls)


class Pipe(Cell):
    """Base class for data processing objects (pipes) that can be chained together to perform pipelined processing, 
    each one performing an atomic operation on the data received from preceding pipe(s), or being an initial source of data (loader, generator).
    Every iterable type - a collection, a generator or any type that implements __iter__() - can be used as a source pipe, too.
    
    Features of data pipes:
    - operator '>>'
    - can use classes, not only instances, with >>
    - can use collections, functions, ... with >> 
    
    Serialization is implemented by inheriting from Object class.
    """
    __metaclass__ = __Pipe__
    __transient__ = "source"    # don't serialize 'source' attribute and exclude it from copy() and deepcopy();
                                # __transient__ is handled by Object.__getstate__
    
    source    = None            # source Pipe or iteratable from which input data for 'self' will be pulled
    sources   = None            # list of source pipes; used only in pipes with multiple inputs, instead of 'source'
    count     = None            # no. of input items read so far in this iteration, or 1-based index of the item currently processed; tracked in most standard pipes (not all)
    yielded   = None            # no. of output items yielded so far in this iteration, EXcluding header item

    created   = False           # has the object been initialized already, in setup()? most pipes have empty setup(), only more complex ones use it for creation of internal structures
    iterating = False           # flag that protects against multiple iteration of the same pipe, at the same time
    

    def setup(self):
        """Delayed initialization of the model, called just before the first iteration (and before open()).
        Delaying initialization and doing it in setup() instead of __init__() enables knobs 
        to be configured in the meantime, *after* and separately from instantiation of the object.
        If you have to call both setKnobs() and setup(), it's better to first call setKnobs.
        """

    def reset(self):
        """Reverse of setup(). Clears internal structures and brings the pipe back to an uninitialized state, 
        like if setup() were never executed. Knobs and other static settings should be preserved!
        If overriding in subclasses, remember to set self.created=False at the end.
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
        one by one. Subclasses should override either iter() or __iter__(), in the latter case the subclass 
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
        """If you subclass Pipe directly, not via specialized base classes (Operator, Monitor, Filter, ...),
        better override iter() not __iter__()."""
        raise NotImplemented()

    def _prolog(self):
        if not self.created:            # call reset/setup() if needed
            self.reset()
            self.setup()
            self.created = True
        if self.iterating: raise Exception("Data pipe %s opened for iteration twice, before previous iteration has been closed" % self)
        self.iterating = True
        self.count = self.yielded = 0
        header = self.open()
        self._pushTrace(self)
        return header

    def _epilog(self):
        """_prolog and _epilog must always be invoked together, otherwise there will be a mismatch between
        opens & closes, and between _pushTraces & _popTraces.
        """
        self._popTrace(self)
        self.close()
        del self.iterating                  # could set self.iterating=False instead, but deleting is more convenient for serialization

    def execute(self):
        """Pull all data through the pipe, but don't yield nor return anything. 
        Typically used for Pipelines which end with a sink and only produce side effects."""
        for item in self: pass

#     def loop(self, times = None):
#         """Executes this pipe multiple times, or infinitely if times=None. Yields items from all iterations as a single stream of data. 
#         Client can read self.nloops - the no. of loops completed so far - to find out which loop is being executed right now.
#         There is NO restarting between loops."""
#         self.nloops = 0
#         while True:
#             if times != None and self.nloops >= times: break
#             if self.nloops > 0: self.reopen()
#             empty = True
#             for item in self: 
#                 empty = False
#                 yield item
#             self.nloops += 1
#             if times == None and empty: raise Exception("Infinite loop over empty dataset")
# 
#     def reopen(self):
#         "Override in subclasses to perform custom setup before next loop of processing is executed in loop()."
        
    def __rshift__(self, other):
        """'>>' operator overloaded, enables pipeline creation via 'a >> b >> c' syntax. Returned object is a Pipeline.
        Put RUN token at the end: a >> b >> RUN - to execute the pipeline immediately after creation."""
        # 'self' is a regular Pipe; specialized implementation for Pipeline defined in the subclass
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

#     def __mult__(self, other):
#         """Multiplication '*' operator creates a Zip node that combines items from all input sources, 1 from each, into output tuples."""
#         return Zip(self, other)

#     def getSpec(self):
#         "1-line string with technical specification of this pipe: its name and possibly values of its knobs etc."


# TOKENS

class _PIPE(Pipe):
    def __rshift__(self, other): return Pipeline(other)

# Starts a pipeline. For easy appending of other pipes with automatic type casting: PIPE >> a >> b >> ...
# Doesn't do any processing itself (is excluded from the pipeline). 
PIPE = _PIPE()

# Invokes execution of a pipeline. When put at the end of a pipeline (a >> b >> ... >> RUN) indicates that it should be executed now. 
RUN = Pipe()


#####################################################################################################################################################
###
###   FUNCTIONAL PIPES
###

class _Functional(Pipe):
    "Base class for Operator, Monitor and Filter. Implements wrapping up a custom python function into a functional pipe."

    fun = None              # plain python function to be used as the class implementation, if core method not overriden
    
    def __init__(self, *args, **knobs):
        "Inner function - if present - must be given as 1st and only unnamed argument. All knobs given as keyword args."
        if args: self.fun = args[0]
        super(_Functional, self).__init__(**knobs)
    
class Operator(_Functional):
    """Plain item-wise processing & filtering function, implemented in the subclass by overloading process() 
    and possibly also open/close(). Method process(item) returns modified version of the item,
    or None to indicate that the item should be dropped (filtered out).
    For operators that only perform filtering, with no modification of items, see Filter class."""
    
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header
        try:
            for item in self.source: 
                self.count += 1
                res = self.process(item)
                if res is not None: 
                    self.yielded += 1
                    yield res
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
    
class Monitor(_Functional):
    """A pipe that (by assumption) doesn't modify input items, only observes the data and possibly produces 
    side effects (collecting stats, logging, final reporting etc). 
    Monitor provides subclasses with an output stream, self.out, that's configurable by the client
    in the 1st argument to __init__ (stdout by default).
    Subclasses override monitor(item) and possibly open/close() or report(). 
    The monitoring function can also be passed as the 2nd argument to __init__."""

    mode     = 'wt'         # mode to be used for opening files
    outfiles = []           # list of: <file> or name of file, where logging/reporting should be printed out
    out      = None         # the actual file object to be used for all printing/logging/reporting in monitor() and report();
                            # opened in _prolog(), can stay None if the pipe doesn't need output stream
    
    def __init__(self, outfiles = None, *args, **kwargs):
        """'outfiles' can be: None or '' (=stdout), or a <file>, or a filename, or a list of <file>s or filenames 
        (None, '' and 'stdout' allowed). 'stdout', 'stderr', 'stdin' are special names, mapped to sys.* file objects."""
        self.outfiles = outfiles if islist(outfiles) else [outfiles]
        #print self, self.outfiles
        super(Monitor, self).__init__(*args, **kwargs)

    def __getstate__(self):
        "Handles serialization/copying of file objects in self.outfiles and self.out."
        state = super(Monitor, self).__getstate__()
        def encode(f, std = {sys.stdout:'stdout', sys.stderr:'stderr', sys.stdin:'stdin'}):
            if not isinstance(f, (file, Tee)): return f
            if f in std: return std[f]
            raise Exception("Monitor.__getstate__, can't serialize/copy a file object: %s" % f)
        
        if 'outfiles' in state: state['outfiles'] = [encode(f) for f in self.outfiles]
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
        self.count = 0
        try:
            for item in self.source: 
                self.count += 1
                self.monitor(item)
                self.yielded += 1           # unlike in Operator, we don't expect any returned result from monitor()
                yield item                  # however, watch out for bugs: monitor() can implicitly modify internals of 'item' unless 'item' is immutable
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def _prolog(self):
        "Open the output stream, self.out."
        if len(self.outfiles) > 1:
            self.out = Tee(*self.outfiles) 
        elif self.outfiles:
            self.out = openfile(self.outfiles[0], self.mode)
        return super(Monitor, self)._prolog()

    def _epilog(self):
        "Run report() and close self.out."
        with self.printlock: self.report()
        super(Monitor, self)._epilog()
        if self.out not in [None, sys.stdout, sys.stderr]:
            self.out.close()
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
    Method process() must return True (pass the item through) or False (drop the item), not the actual object."""
    
    def __iter__(self):
        header = self._prolog()
        if header is not None: yield header
        try:
            for item in self.source: 
                self.count += 1
                if self.accept(item): 
                    self.yielded += 1
                    yield item        # unlike in Operator, we expect only True/False from accept/process(), not an actual data object
        
        except GeneratorExit, ex:
            self._epilog()
            raise
        self._epilog()

    def accept(self, item):
        return self.process(item)
    def process(self, item):                            # deprecated in Filter; override accept() instead
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
    
class Function(Operator):
    """An operator OR a filter constructed from a plain python function. A wrapper.
    Explicit use of Operator or Filter classes instead of this one is recommended."""
    def __init__(self, oper = None):
        "oper=None handles the case when processing function is implemented through overloading of process()."
        self.oper = oper
    def process(self, item):
        "Can return modified item; or None, interpreted as no result (drop item); or True (pass unchanged); or False (drop item)."
        if self.oper is None: raise Exception("Missing operator function (self.oper) in %s" % self)
        ret = self.oper(item)
        if ret is True: return item
        if ret is False: return None
        if ret is None: return item         # for monitor- or operator-like functions that don't make final 'return item'
        return ret


###  Generators

class Empty(Pipe):
    "Generates an empty output stream. Useful as an initial pipe in an incremental sum of pipes: p = Empty; p += X[0]; p += X[1] ..."
    def __iter__(self):
        return; yield

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
    
class Range(Pipe):
    "Generator of consecutive integers, equivalent to xrange(), same parameters."
    def __init__(self, *args):
        self.args = args
    def __iter__(self):
        return iter(xrange(*self.args))


###  Filters

class Slice(Pipe):
    "Like slice() or itertools.islice(), same parameters. Transmits only a slice of the input stream to the output."
    def init(self, *args):
        self.args = args
    def iter(self):
        return islice(self.source, *self.args)

class Offset(Pipe):
    "Drop a predefined number of initial items"
    class __knobs__:
        offset = 0
    def init(self, offset):
        self.offset = offset
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
    def init(self, limit):
        self.limit = limit
    def iter(self):
        count = 0
        if count >= self.limit: return
        for item in self.source: 
            yield item
            count += 1
            if count >= self.limit: return
Head = Limit

class DropWhile(Pipe):
    "Like itertools.dropwhile()."
class TakeWhile(Pipe):
    "Like itertools.takewhile()."    
class StopOn(Pipe):
    """Terminate the data stream when a given condition becomes True. Condition is a function that takes current item as an argument. 
    This function can also keep an internal state (memory)."""


class Subset(Pipe):
    """Selects every 'fraction'-th item from the stream, equally spaced, yielding <= 1/fraction of all data. Deterministic subset, no randomization.
    >>> Range(7) >> Subset(3) >> Print >> RUN
    2
    5
    """
    class __knobs__:
        fraction = 1
    def init(self, fraction):
        self.fraction = fraction
    def iter(self):
        frac = self.fraction
        count = 0                               # count iterates in cycles from 0 to 'frac', and again from 0 ...
        for item in self.source: 
            count += 1
            if count == frac: 
                yield item
                count = 0

class Sample(Pipe):
    "Random sample of input items. Every input item is decided independently with a given probability, unconditional on what items were chosen earlier."
    

###  Buffers

#class Cache(Pipe):
class Buffer(Pipe):
    """Upon setup(), buffers all input data in memory. Then, when data is buffered, 
    can iterate (multiple times) over it and yield from memory."""

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
    def init(self, size = None):
        self.size = size
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

    subset = None

    def __init__(self, msg = "%s", func = None, outfile = None, disp = True, count = False, subset = None, *args, **kwargs):
        """msg: format string of messages, may contain 1 parameter %s to print the data item in the right place.
        func: optional function called on every item before printing, its output is printed instead of the actual item.
        outfile: path to external output file or None; disp: shall we print to stdout?; count: shall we print 1-based item number at the begining of a line?
        subset: print every n-th item (see Subset), or None to print all items."""
        if outfile and disp: outfile = [outfile, sys.stdout]
        Monitor.__init__(self, outfile, *args, **kwargs)
        #if '%s' not in msg: msg += ' %s'
        self.static = ('%s' not in msg)
        self.message = msg
        self.func = func
        self.index = count
        if subset: self.subset = subset
        #print "created Print() instance, message '%s'" % self.message

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
    def init(self, msg = "Data items:   %d"):
        if msg is None: return
        try: msg % 0
        except: msg += " %d"                    # append format character if missing
        self.msgTotal = msg
    def monitor(self, item): pass
    def close(self):
        with self.printlock: print self.msgTotal % self.count

class Time(Monitor):
    "Measure time since the beginning of data iteration. If 'message' is present, print the total time at the end, embedded in 'message'."
    def init(self, message = "Time elapsed: %.1f s"):
        self.msgTime = message
        self.start = None               # time when last open() was run, as Unix timestamp
        self.elapsed = None             # final time elapsed, in seconds, as float
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

    #metricname = None           # if not-None, metric of each item will be saved in the item under this name
    last = None                 # most recent individual metric value calculated
    size = None                 # no. of individual metrics calculated & aggregated so far excluding Nones; 0-based

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
        
class Mean(Metric):
    """Calculates sample mean & std.deviation of values measured for individual items by a given metric.
    The metric is either implemented in overridden metric() method, or given as a function - argument of initialization.
    """
    
    def open(self):
        self.sum = self.sum2 = 0.0
    
    def aggregate(self, metric):
        self.sum += metric
        self.sum2 += metric ** 2

    def mean(self): 
        "Sample mean"
        return self.sum / float(self.size)

    def deviation(self): 
        "Sample standard deviation"
        N = float(self.size)
        return sqrt((self.sum2 - self.sum/N * self.sum) / (N-1))


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
        for item in self.source:
            self.items.append(item)
        yield self.items

        
#####################################################################################################################################################
###
###   STRUCTURAL PIPES
###

class MetaPipe(Pipe):
    "Wraps up a number of pipes to provide meta-operations on them."

# class Container(MetaPipe):
#     "Base class for classes that contain multiple pipes inside: self.pipes."
#     
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

class Wrapper(MetaPipe):
    "Like Container, but always has only 1 internal pipe: self.pipe."
    
    pipe = None
    
    def setKnobs(self, knobs, strict = False):
        super(Wrapper, self).setKnobs(knobs, strict)            # Wrapper itself has no knobs, but a subclass can define some
        if self.pipe: self.pipe.setKnobs(knobs, strict)
        
    def _prolog(self):
        "Call inner pipe's setup before starting iteration."
        if self.pipe: self.pipe.setup()
        return super(Wrapper, self)._prolog()

#####################################################################################################################################################

class Pipeline(MetaPipe):
    """Sequence of data pipes connected sequentially, one after another. Pipeline is a MetaPipe and a Pipe itself.
    Inner pipes can be accessed by indexing operator: pipeline[3]
    """
    
    pipes    = None         # static list of pipes as passed during pipeline initialization; may contain non-pipe objects
    pipeline = None         # the actual pipes used in iteration, created dynamically in __iter__ or setKnobs() 
    knobs    = None         # knobs to be set before iteration starts; 
                            # for delayed setting of knobs, necessary when some pipes are only templates that require normalization
    
    #__inner__ = "pipeline"
    
    def __init__(self, *pipes, **kwargs):
        "'kwargs' may contain connected=True to indicate that pipes are already connected into a list or a tree (must be sorted in topological order!)."
        if pipes and islist(pipes[0]): pipes = pipes[0]
        self.pipes = list(pipes)
        #if kwargs.get("connected"): return
        
    def __rshift__(self, other):
        """Append 'other' to the end of the pipeline. Shallow-copy the pipeline beforehand, 
        to avoid in-place modifications but preserve '>>' protocol."""
        if other is RUN:
            self.execute()
        else:
            res = self.copy1()
            res.pipes.append(other)
            return res

    def __lshift__(self, other):
        "Like __rshift__, but 'other' is put at the tail not head of the pipeline."
        res = self.copy1()
        res.pipes = [other] + res.pipes
        return res

    def setKnobs(self, knobs, strict = False):
        super(Pipeline, self).setKnobs(knobs)           # Pipeline itself has no knobs, but a subclass can define some
        for pipe in self.pipes:                         # set knobs in template pipes (some of them can be actual pipes)
            if isinstance(pipe, Cell): pipe.setKnobs(knobs)
        if self.pipeline: self.setPipelineKnobs(knobs)
        else:
            # no normalized pipeline yet? keep the knobs to apply in the future
            if self.knobs: self.knobs.update(knobs)
            else: self.knobs = knobs

    def setPipelineKnobs(self, knobs):
        if not knobs: return
        for pipe in self.pipeline:
            pipe.setKnobs(knobs)
#             try:
#                 if hasattr(pipe, 'setKnobs'): pipe.setKnobs(*self.knobs)
#             except:
#                 print pipe
#                 raise

    def setup(self):
        #self.pipes = _normalize(self.pipes)
        self.pipeline = _normalize(self.pipes)
        self.setPipelineKnobs(self.knobs)
        self.knobs = None
#         for pipe in self.pipeline:
#             pipe.setup()

    def iter(self):
        # normalize pipes; connect into a list; connect entire pipeline with the source
        #self.pipeline = _normalize(self.pipes)
        prev = self.source
        for next in self.pipeline:
            if prev is not None: next.source = prev         # 1st pipe can be a generator or collection, not necessarily a Pipe (no .source attribute)
            prev = next
            
        # pull data
        head = self.pipeline[-1]
        for item in head: yield item
    
    def close(self):
        if not self.verbose: return
        with self.printlock:
            print "Data counts [input, pipe name, output] of", self
            for pipe in self.pipeline:
                print "%7d" % pipe.count, pipe, pipe.yielded
    
    def __getitem__(self, pos):
        """Returns either an operating pipe from self.pipeline, or a static pipe from self.pipes, 
        depending whether called during iteration or not.
        """
        return self.pipeline[pos] if self.iterating else self.pipes[pos]
    
    def __str__(self):
        if self.iterating:
            return "Open pipeline " + ' >> '.join(map(str, self.pipeline))
        return "Pipeline " + ' >> '.join(map(str, self.pipes))
    

#####################################################################################################################################################

class MultiSource(Pipe):
    pass
#     def copy(self, deep = True):
#         "Shallow copy does copy the list of pipes too (list can be modified afterwards without affecting the original)."
#         if deep: return deepcopy(self)
#         res = copy(self)
#         res.sources = copy(self.sources)
#         return res

# class Parallel(MultiSource):
#     "Connects multiple pipes as parallel routes between a single source and a single destination."
#     def __init__(self, source, handlers):
#         if handlers and islist(handlers[0]): handlers = handlers[0]
#         self.pipes = _normalize(handlers)
#         for h in handlers:
#             h.source = source
# 
# class Hub(MultiSource):
#     "Branching of pipes: a central pipe (hub) with multiple parallel sources."
#     def __init__(self, sources, hub):
#         self.sources = hub.sources = _normalize(sources)
#         self.hub = hub
#         self.pipes = self.sources + [hub]            # for open() & close()

class Union(MultiSource):
    """Combine items from multiple sources by concatenating corresponding streams one after another: all items from the 1st source; then 2nd... then 3rd...
    TODO: 'mixed' mode (breadth-first, streams interlaced)."""
    def __init__(self, *sources):
        self.sources = _normalize(sources)
    def __iter__(self):
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
    
    runID        = "runID"          # name of a special knob inside 'pipe' that will be set with an ID (integer >= 1) of the current run
    startID      = 0                # ID of the first run, to start counting from
    maxThreads   = 0                # max. no. of parallel threads; <=1 for serial execution; None for full parallelism, with only 1 scan over input data
    threadBuffer = 10               # length of input queue in each thread; 1 enforces perfect alignment of threads execution; 
                                    # 0: no alignment, queues can get big when input data are produced faster than consumed
    copyPipe     = True             # shall we make a separate deep copy of the pipe for each run?
    copyData     = True             # shall we make separate deep copies of data items for each parallel run? no copy in serial mode
    
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
        
        knobs = kwargs
        self.space = Cartesian(*knobs.values())             # value space of all possible knob values
        self.names = knobs.keys()                           # names of knobs
        self.done = None                                    # no. of runs completed so far
    
    def iter(self):
        self.done = 0
        if self.maxThreads is None or self.maxThreads > 1:
            self.iterParallel()
        else:
            self.iterSerial()
        return; yield                                       # to make this method work as a generator (only an empty one)
        
    def iterSerial(self):
        with self.printlock: print "Grid: %d serial runs to be executed..." % len(self.space)
        for i, value in enumerate(self.space):
            self.scanSerial(value, self.startID + i)
            self.done += 1
        
    def iterParallel(self):
        scans = divup(len(self.space), self.maxThreads) if self.maxThreads else 1
        with self.printlock: print "Grid: %d runs to be executed, in %d scan(s) over input data..." % (len(self.space), scans)
        
        def knobsStream():
            "Generates knob combinations. Every combination is a list of (name,value) pairs, one pair for each knob."
            for i, v in enumerate(self.space):
                yield self.createKnobs(self.startID + i, v)

        def knobsGroups():
            "Partitions stream generated by knobsStream() into groups of up to 'maxThreads' size each."
            if not self.maxThreads: 
                yield knobsStream()
                return
            group = []
            for knobs in knobsStream():
                group.append(knobs)
                if len(group) >= self.maxThreads:
                    yield group
                    group = []
            if group: yield group

        for kgroup in knobsGroups():
            self.scanParallel(kgroup)
            self.done += len(kgroup)

    def createKnobs(self, ID, values):
        "Creates one combination of knobs using given values and returns as a list of (name,value) pairs."
        return Knobs([(self.runID, ID)] + zip(self.names, values))
        
    def createPipe(self, knobs, forceCopy = False):
        pipe = self.pipe.copy() if (self.copyPipe or forceCopy) else self.pipe
        pipe.setKnobs(knobs)
        #pipe.setup()
        return pipe
        
    def scanSerial(self, value, ID):
        """Single scan over input data, with items passed directly to the single pipe being executed. 
        No parallelism, no multi-threading, no pipe copying.
        The same pipe object is reused in all runs - watch out against interference between consecutive runs.
        """
        with self.printlock: print "Grid, starting next serial scan for run ID=%s..." % ID
        knobs = self.createKnobs(ID, value)
        pipe = self.createPipe(knobs)
        self.printKnobs(knobs)
        PIPE >> self.source >> pipe >> RUN
    
    def scanParallel(self, knobsGroup):
        "Single scan over input data, with each item fed to a group of parallel threads."
        with self.printlock: print "Grid, starting next parallel scan for %d runs beginning with ID=%s..." % (len(knobsGroup), knobsGroup[0][0][1])
        self.count = 0
        threads = self.createThreads(knobsGroup)
        duplicate = deepcopy if self.copyData else lambda x:x

        for item in self.source: 
            self.count += 1
            for _, pipe in threads:                             # feed input data to each pipe in parallel
                pipe.put(duplicate(item))
        
        self.closeThreads(threads)
        
    def createThreads(self, knobsGroup):
        threads = []                                            # list of pairs: (knobs, pipe_thread)
        for knobs in knobsGroup:                                # create a copy of the template pipe for each combination of knob values; wrap up in threads
            pipe = self.createPipe(knobs, forceCopy = True)
            thread = Thread(pipe, self.threadBuffer)
            thread.start()
            threads.append((knobs, thread))
        return threads
        
    def closeThreads(self, pipes):
        for _, pipe in pipes:
            pipe.emptyFeed()                    # wait until all pipes eat up all remaining input items; note: some pipes may still be processing the last item!
        sleep(1)
        
        with self.printlock: print "Grid, %d runs done." % (self.done + len(pipes))
        for knobs, pipe in pipes:
            self.printKnobs(knobs)
            pipe.end()
            pipe.join()

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
    In default implementation of process(), the inner pipe is expected to pull exactly 1 item at a time
    from the source, otherwise an exception will be raised (typically the pipe is an Operator).
    However, if a subclass overrides process(), it can expect the inner pipe to exhibit any other type 
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

    
    def __init__(self, pipe):
        self.pipe = pipe                        # client can use this for later introspection of the inner pipe(s)
    
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
        for item in self.source: yield self.process(item)
    
    def process(self, item):
        """Subclasses can override this method to perform additional operations before/during/after 
        the item is processed. In such case, remember to call self.put(item) and self.get() in appropriate places,
        or alternatively super(X,self).process(item), which runs put() and get() altogether."""
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


class operator(Controller):
    """
    A wrapper that turns a pipe (pipeline) back to a regular input-output function 
    (a callable, to be precise) that can be fed with data manually, one item at a time.
    During processing, the original pipe can still be accessed via 'pipe' property of the wrapper.
    
    >>> pipeline = Function(lambda x: 2*x) >> Function(lambda x: str(x).upper())
    >>> fun = operator(pipeline)
    >>> print fun(3)
    6
    >>> print fun(['Ala'])
    ['ALA', 'ALA']
    >>> print fun.pipe[0]
    Function
    
    Typically the pipe is an instance of Operator. Even if not, it still must pull exactly 1 item at a time
    from the source, otherwise an exception will be raised.
    Can be used in the same caller's thread, no need to spawn a new thread.
    """

    def __init__(self, pipe):
        super(operator, self).__init__(pipe)
        self._prolog()

    # processing can be invoked with op(item), like a function, not only op.process(item)
    __call__ = Controller.process
    

#####################################################################################################################################################

class Thread(threading.Thread, Wrapper):
    """A thread object that executes given pipe(line) in a separate thread.
    Can serve also as a pipe and be included in a pipeline, if only the inner pipe takes no input data (has no source)."""
    
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
        if self.source: raise Exception("Pipe of class Thread can't be used with a source attached. It can't synchronize input and output by itself.")
        while True:
            item = self.get()
            if item is Thread.END: break
            yield item
        
    

#####################################################################################################################################################

def _normalize(pipes):
    """Normalize a given list of pipes. Remove None's and strings (used for commenting out), 
    instantiate Pipe classes if passed instead of an instance, wrap up functions, collections and files."""
    def convert(h):
        if issubclass(h, Pipe): return h()
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

    
