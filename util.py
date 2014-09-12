# -*- coding: utf-8 -*-
'''
Collection of short but frequently used routines and classes - shorthands for different daily tasks.

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import __builtin__, os, sys, glob, types as _types, re, numbers, json, time, datetime, calendar
import logging, random, math, collections, unicodedata, heapq, threading, inspect
from StringIO import StringIO


#####################################################################################################################################################
###
###   TYPE CHECKING
###

def isint(x): return isinstance(x, numbers.Integral)
isintegral = isinteger = isint
def isnumber(x):   return isinstance(x, numbers.Number)
def isstring(s):   return isinstance(s, basestring)
def isdict(x):     return isinstance(x, dict)
def istype(x):     return isinstance(x, type) or isinstance(x, _types.ClassType)        # recognizes old-style classes, too

def islist(x, orTuple = True):
    if orTuple: return isinstance(x, (list,tuple))
    return isinstance(x, list)
def istuple(x):
    return isinstance(x, tuple)
def iscontainer(x):
    "True if x is a container object (list,tuple,dict,set), but NOT a string or custom iterable."
    return isinstance(x, collections.Container) and not isinstance(x, basestring)
def isiterable(x):
    "True if x is *any* iterable: list, tuple, dict, set, string (!), any object with __iter__ or __getitem__ method."
    return isinstance(x, collections.Iterable) #and not isinstance(x, basestring)
def isregex(x):
    return isinstance(x, re._pattern_type)
# def isarray(x) - defined in 'math' module

def isfunction(x):
    "True if x is any kind of a 'syntactic' function: function, method, built-in; but NOT any other callable (object with __call__ method is not a function)."
    return isinstance(x, (_types.FunctionType, _types.BuiltinFunctionType, _types.MethodType, _types.BuiltinMethodType, _types.UnboundMethodType))
def isgenerator(x):
    return isinstance(x, _types.GeneratorType)
def isbound(method):
    "True if a given method is bound, i.e., assigned to an instance (with 'self'), not a class method."
    return method.im_self is not None

# Environment checks:
def islinux():
    "Is the operating system posix-type: linux, unix, Mac OS"
    return os.name == "posix"


########################################################################################################################################################
###
###  CONVERSIONS & COMMAND-LINE
###

# Conversions of un-structured values, typically strings received from the console (sys.argv), to different types of structured objects.
# If an input value is already a structured one (None value in particular), it is returned unchanged.

def asbool(s):
    if isstring(s): s = s.tolower()
    if s in [False, 0, 0.0, "0", "", "false", "no", "n"]: return False
    if s in [True, 1, 1.0, "1", "true", "yes", "y"]: return True
    raise Exception("Unrecognized value passed to asbool(): %s" % s)

def asint(s):
    if not isstring(s): return s
    return int(s)

def asnumber(s):
    if not isstring(s): return s
    try: return int(s)
    except: pass
    return float(s)

def asobject(name, context = {}):
    "Find an object defined inside 'context' (dict, object, module) by its name."
    if not isstring(name): return name
    if not isdict(context): context = context.__dict__
    if name in context: return context[name]
    raise Exception("Object can't be found: '%s'" % name)
    

def runCommand(context = {}, args = None):
    """Takes from 'sys' all command-line arguments passed to the script and interprets them as a name 
    of a callable (function) from 'context' (module or dict, typically globals() of the caller), 
    and possibly its parameters; finds the function, executes with given parameters (passed as unnamed strings) 
    and returns its result. If the command is not present in 'context' and there are no parameters,
    pass it to eval(), which is more general and can execute an arbitrary expression, not only a global-scope function. 
    If 'args' list is present, use it as arguments instead of sys.argv[1:]. 
    Note: the called function should convert internally the parameters from a string to a proper type and 
    this conversion is done in a local context of the function, so it may be hard to pass variables as parameters.
    """
    if args is None: args = sys.argv[1:]                        # argv[0] is the script name, omit
    if not args: return None                                    # no command? do nothing 
    if not isdict(context): context = context.__dict__
    cmd = args[0]
    params = tuple(args[1:])
    if cmd in context:
        fun = context[cmd]
        return fun(*params)
    elif not params:
        return eval(cmd, context)
    raise Exception("Object can't be found: '%s'" % cmd)


#####################################################################################################################################################
###
###  CLASSES
###

def issubclass(x, cls):                         #@ReservedAssignment
    "True if x is a class and subclass of cls, False otherwise. Overrides built-in issubclass() which raised exception if 'x' was not a class (inconvenient in many cases); this function accepts non-classes too."
    return isinstance(x, type) and __builtin__.issubclass(x, cls)

def classname(obj = None, full = False, cls = None):
    "Return (fully qualified) class name of the object 'obj' or class 'cls'."
    if cls is None: cls = obj.__class__
    if full: return cls.__module__ + "." + cls.__name__
    return cls.__name__
    
def types(obj):
    "Finds the type and all base types of a given object. Like baseclasses(), but includes also own type()."
    t = type(obj)
    return [t] + baseclasses(t)

def baseclasses(cls, include_self=False):
    "Finds all base classes of a given class, also indirect ones, by recursively looking up __bases__. 'object' base is excluded."
    if cls is object: return []
    l = []
    for base in cls.__bases__:
        l.extend(baseclasses(base, True))
    if include_self:
        l.append(cls)
    return l
bases = baseclasses         # alias

def subclasses(cls, include_self=False):
    "Finds all subclasses of a given class, also indirect ones, by recursively calling __subclasses__()"
    l = []
    for child in cls.__subclasses__():
        l.extend(subclasses(child, True))
    if include_self:
        l.append(cls)
    return l

#####################################################################################################################################################
###
###   COLLECTIONS & DATA STRUCTURES
###

def unique(seq, order = False):
    "List of elements of 'seq' with duplicates removed. If order=True, preserves original order (from: http://stackoverflow.com/a/480227/1202674)."
    if not seq: return []
    #if not order: return list(set(seq))
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]

def flatten(*seq):
    """List of all atomic elements of 'seq' (strings treated as atomic) together with all elements of sub-iterables of 'seq', recursively.
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, ('a','string')], (8, 9))
    [1, 2, 3, 42, None, 4, 5, 6, 7, 'a', 'string', 8, 9]
    """
    result = []
    if len(seq) == 1: seq = seq[0]
    for x in seq:
        if hasattr(x, "__iter__"): result += flatten(x)
        else: result.append(x)
    return result

def list2str(l, sep = " ", f = str):
    "Convert all items of list 'l' into strings of type 'f' (str) and concatenate into one string separated by 'sep'. 'f' can also be any mapping function."
    return sep.join(map(f, l))

def str2list(s, sep = None):
    """Return s.split(sep), but first check if 's' is not already a list or None (return unchanged in such case). 
    For convenient definition of string lists: either as lists or as sep-separated strings of words."""
    if s is None or islist(s): return s
    return s.split(sep)

def obj2dict(obj):
    'Recursively convert a tree of nested objects into nested dictionaries. Iterables converted to lists.'
    if hasattr(obj, "__iter__"):
        return [obj2dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return dict([(k, obj2dict(v)) for k,v in obj.__dict__.iteritems() if not callable(v) and not k.startswith('_')])
    else:
        return obj
        
def dict2obj(d, cls, obj = None):
    '''Converts dictionary 'd' to an object of class 'cls' (instantiated via cls()); 
    or if 'obj' is given then sets attributes of this existing object (like setattrs()). Returns the object'''
    if not obj:
        obj = cls()
    setattrs(obj, d)
    return obj

def class2dict(cls, exclude = "__", methods = False):
    """Retrieves all attributes of a class, possibly except methods if methods=False (default), and returns as a dict.
    Similar to getattrs() called with names=None, but detects inherited class attributes, too."""
    names = filter(lambda n: not n.startswith(exclude), dir(cls))
    if methods: return {n: getattr(cls, n) for n in names}
    d = {}
    for n in names: 
        v = getattr(cls, n)
        if not isfunction(v): d[n] = v
    return d

def subdict(d, keys, strict = False, default = False):
    "Creates a sub-dictionary from dict 'd', by selecting only the given 'keys' (list). If strict=True, all 'keys' must be present in 'd'."
    if isstring(keys): keys = keys.split()
    if strict: return dict((k,d[k]) for k in keys)
    if default: return dict((k,d.get(k)) for k in keys)
    if len(keys) <= len(d): return dict((k,d[k]) for k in keys if k in d)
    return dict(item for item in d.iteritems() if item[0] in keys)

def splitkeys(d):
    """Split multi-name keys of dictionary 'd' and return as a new dictionary. If 'd' contains string keys of the form 'key1 key2 key3 ...' 
    (several keys merged into one string, sharing the same value), they will be split on whitespaces, creating separate keys with the same value assigned. 
    All keys in 'd' must be strings, or exception is raised."""
    d2 = {}
    for key, val in d.iteritems():
        for k in key.split():
            d2[k] = val
    return d2    

def lowerkeys(d):
    "Copy dictionary 'd' with all keys changed to lowercase. Class of 'd' is preserved (can be other than dict)."
    return d.__class__((k.lower(), v) for k,v in d.iteritems())

def getattrs(obj, names = None, exclude = "__", default = None, missing = True, usedict = False):
    """Similar to built-in getattr(), but returns many attributes at once, as a dict.
    Attribute names are given in 'names' as a list of strings, or a string with 1+ space-separated names.
    By default, attributes are retrieved using getattr(), which detects class attributes, 
    fires up descriptors (if any) and returns methods as <unbound method> not <function>. 
    Only if names=None and usedict=True, attributes are taken directly from __dict__,
    which can be faster, but in some cases behaves differently than getattr().  
    For missing attributes: returns None if missing=True (default), skips if missing=None, raises an exception if missing=False. 
    """
    if names is None:                                                               # retrieving all attributes?
        if usedict:                                                                 # use faster but less correct approach: __dict__
            d = obj.__dict__.copy()
            if exclude is None: return d
            for k in d.keys():
                if k.startswith(exclude): del d[k]
            return d
        
        # proceed to a slower but fully correct approach: getattr() ...
        if exclude is None: names = obj.__dict__.keys()
        else: names = filter(lambda n: not n.startswith(exclude), obj.__dict__.keys())
    
    if isstring(names):                                                             # retrieving an explicit list of attributes?
        if ' ' not in names: return {names: getattr(obj, names)}                    # a single name given
        names = names.split()                                                       # multiple names
    
    d = {}
    if missing:
        for k in names: d[k] = getattr(obj, k, default)
    elif missing == False:
        for k in names: d[k] = getattr(obj, k)
    else:
        for k in names: 
            if hasattr(obj, k): d[k] = getattr(obj, k)
    return d

def setattrs(obj, d, values = None):
    """Similar to built-in setattr(), but takes entire dictionary 'd'; or a list of names 'd' and list of values 'values' - and sets many attributes at once.
    'values' can also be a single non-list value, in which case it will be assigned to all attributes."""
    #obj.__dict__.update(d)
    if values:
        if not islist(values): values = [values] * len(d)
        pairs = zip(d,values)
    else:
        pairs = d.iteritems()
    for k,v in pairs:
        setattr(obj, k, v)
    return obj

def copyattrs(dest, src, names = None, missing = False):
    "Like setattrs() above, but sets attributes by copying (shallow copy) all attributes from another object or dict 'src' (see getattrs())."
    if isdict(src): attrs = subdict(src, names) if names is not None else src
    else: attrs = getattrs(src, names, missing = missing)
    setattrs(dest, attrs)
    return dest

#def retype(obj, newtype):
#    ""
#    obj2 = newtype()

def setdefaults(d, keys = '', default = ''):
    '''
    Checks keys in dictionary 'd' and inserts default values 
    if a given key is missing or None.
    '''
    for k in keys if islist(keys) else keys.split():
        d.setdefault(k, default)
        if d[k] is None: d[k] = default
    return d

def get(d, key, default = ''):
    '''
    Similar to dict.get(), but returns 'default' also when the key is defined, 
    but value is empty, e.g. None.
    '''
    v = d.get(key)
    if not v: return default
    return v

class ObjDict(dict):
    """A dictionary whose items can be accessed like object properties (d.key), in addition to standard access (d['key']). Be careful with keys named like standard dict properties.
    Keys starting with '__' can't be accessed in this way."""
    def __getattr__(self, name): 
        if name.startswith('__'): raise AttributeError(name)
        return self[name]
    def __setattr__(self, name, value): 
        if name.startswith('__'): raise AttributeError(name)
        self[name] = value
    def __delattr__(self, name): 
        if name.startswith('__'): raise AttributeError(name)
        del self[name]

class ComparableMixin:
    "Base class (mixin) that implements all comparison operators in terms of __lt__()."
    def __eq__(self, other):
        return not self < other and not other < self
    def __ne__(self, other):
        return self < other or other < self
    def __gt__(self, other):
        return other < self
    def __ge__(self, other):
        return not self < other
    def __le__(self, other):
        return not other < self

from heapq import heappush, heappop

class Heap(object):
    "An object-oriented wrapper for standard heapq module. Additionally allows custom comparison key to be provided."
    def __init__(self, items = None, key = None):
        self.key = key          # function key(item) that generates key value
        self.items = []
        if items:
            if key: self.items = [(key(item), item) for item in items]          # insert pairs to the heap, to enable custom comparison 
            else: self.items = list(items)                                      # copy the original list
            heapq.heapify(self.items)

    def push(self, item):
        if self.key: heappush(self.items, (self.key(item), item))
        else: heappush(self.items, item)

    def pop(self):
        if self.key: return heappop(self.items)[1]
        else: return heappop(self.items)
        
    def __len__(self):
        return len(self.items)

def heapmerge(*inputs):
    """Like heapq.merge(), merges multiple sorted inputs (any iterables) into a single sorted output, but provides more convenient API:
    each input is a pair of (iterable, label) and each yielded result is a pair of (item, label of the input) - so that it's known what input a given item originates from.
    Labels can be any objects (e.g., object that produced the input stream)."""
    def entries(iterable, label):
        for obj in iterable: yield (obj, label)
    iterables = [entries(*inp) for inp in inputs]
    return heapq.merge(*iterables)


#####################################################################################################################################################
###
###   OBJECTS
###

class __Labelled__(type):
    "Metaclass that implements labels for the actual class: inheritable lists of attributes that exhibit a special behavior."

    def __init__(cls, *args):
        cls.__labels__ = []                         # names of attributes that represent labels in this class

    def labels(cls, names): #@NoSelf
        for name in str2list(names): cls.label(name)
        
    def label(cls, name): #@NoSelf
        "Declare 'name' as a label and set up the list of labelled attributes, the list to be stored under 'name'."
        cls.normLabel(name)                         # convert cls's own labelling to canonical representation
        cls.inheritList(name)                       # inherit labellings from superclasses
        cls.__labels__.append(name)                 # mark 'name' as a label
    
    def normLabel(cls, label): #@NoSelf
        "Normalize a list of labelled attributes declared in this class, by converting it from a string or inner class if necessary."
        attrs = getattr(cls, label, [])                     # list of names of attributes labelled by 'label'
        if istype(attrs):                                   # inner class?
            inner = attrs
            vals = getattrs(inner)
            for name in vals.iterkeys():                    # check that all attrs can be safely copied to top class, without overwriting regular attr
                if not name in cls.__dict__: continue 
                raise Exception("Attribute %s appears twice in %s: as a regular attribute and inside label class %s" %
                                (name, cls, label))
            setattrs(cls, vals)                             # copy all attrs from the inner class to top class level
            attrs = cls._getattrs(inner)
            #attrs = vals.keys()                            # collect attr names
        elif isstring(attrs):                               # space-separated list of attribute names?
            attrs = attrs.split()
        setattr(cls, label, attrs)

    def inheritList(cls, label): #@NoSelf
        "If 'label' is the name of a special attribute containing a list of items, append lists from base classes to cls's list."
        #"""Find out what attributes are labelled by 'label' in superclasses and label them in this class, too. 
        #'label' is the name of attribute that keeps a list of labelled attrs of a given class."""
        baseitems = [getattr(base, label, []) for base in cls.__bases__]        # get lists defined in base classes
        baseitems = reduce(lambda x,y:x+y, baseitems)                           # combine into one list
        items = getattr(cls, label)
        combined = unique(items + baseitems, order = True)                      # add cls's items at the BEGINNING and uniqify
        setattr(cls, label, combined)

    def _getattrs(outer, cls): #@NoSelf
        """Get names of all attributes of a given class, arrange them in the same ORDER as in the source code,
        and return together with their values as an Ord.
        Warning: only the attributes that appear at the beginning of their line are detected.
        For example, if attribubes are defined like this:
            x = y = 0
        only 'x' will be detected, 'y' will be missed.
        """
        from tokenize import generate_tokens
        import token
        
        src = outer._getsource(cls.__name__)
        tokens = generate_tokens(StringIO(src).readline)
        tokens = [(t[1], t[4]) for t in tokens if t[0] == token.NAME]               # pairs (name, line) for all NAME tokens
        attrs = [name for (name,line) in tokens if line.strip().startswith(name)]   # only take names that start the line
        attrs = unique(attrs, order = True)                                         # remove duplicates
        
        attrdict = getattrs(cls)
        attrs = [name for name in attrs if name in attrdict]        # remove names that don't appear in 'attrdict'
        for name in attrdict:                                       # append names that don't appear in 'attrs'
            if name not in attrs: attrs.append(name)
        
        return attrs

    def _getsource(outer, name): #@NoSelf
        """Improved variant of inspect.getsource(), corrected for inner classes.
        Standard getsource() works incorrectly when two outer classes in the same file have inner classes with the same name.
        """
        outsrc = inspect.getsource(outer)           # all source of the outer class, contains somewhere the inner class 'name'
        pat = re.compile(r'^(\s*)class\s*' + name + r'\b')
        lines = outsrc.splitlines()
        for i in range(len(lines)):                 # find the 1st line with "class 'name'"
            match = pat.match(lines[i])
            if not match: continue
            indent = match.group(1)
            indent1 = indent + ' '
            indent2 = indent + '\t'
            j = i + 1
            while j < len(lines):                   # extract all lines of the block following "class 'name'"
                line = lines[j]
                sline = line.strip()
                if line.startswith(indent1) or line.startswith(indent2) or sline == '' or sline.startswith('#'): j += 1
                else: break
            return '\n'.join(lines[i+1:j])
        return outsrc                               # as a fallback, return all 'outsrc'


class __Object__(__Labelled__):
    "Metaclass for Object. Implements __transient__ label."
    def __init__(cls, *args): #@NoSelf
        super(__Object__, cls).__init__(cls, *args)
        cls.label('__transient__')                  # declare '__transient__' as a label and set up the list of labelled attributes, cls.__transient__


class Object(object):
    """For easy creation of objects that can have assigned any attributes, unlike <object> instances. For example: 
         obj = Object(); obj.x = 21
         obj = Object(x = 21, y = 'ala')
         obj = Object({'x':21, 'y':'ala'}) 
       With base <object> this is impossible - a subclass, even if with empty implementation, is required to assign to attributes.
       
       Additionally, Object implements:
       - equality '==' operator __eq__ that performs deep comparison by comparing __dict__ dictionaries, not only object IDs.
       - __str__ that prints the class name and its __dict__, with full recursion like for nested dict's (__repr__ == __str__).
       - __getstate__ that understands the __transient__ list of attributes and excludes them from serialization.
       - copy() and deepcopy() also honor __transient__, since they utilize __getstate__ unless custom __copy__/__deepcopy__ 
         is implemented in a subclass.
       
       When subclassing Object:
       - Some attributes can be labelled as "transient", by adding their names to subclass'es __transient__ list.
         __transient__ can also be given as a space-separated string "name1 name2 ...", which will be converted automatically 
         into a list by the metaclass, after subclass definition. Additionally, the metaclass automatically extends the list 
         with names declared as transient in superclasses.
         __transient__ is typically a class-level attribute, but can be overriden in instances to modify 
         serialization behavior on per-instance basis.
       - If you provide custom metaclass for your Object subclass, remember to inherit that metaclass from __Object__ and call
         super(X, cls).__init__ in your __init__(cls).
       - Subclasses can easily add their own labels, by implementing a metaclass that subclasses __Object__ 
         and invokes cls.label('labelName') in __init__. New labels will be automatically provided with 
         conversions and inheritance, like __transient__ is.
    """
    __metaclass__ = __Object__
    __transient__ = []                      # list of names of attributes to be excluded from serialization 
    
    def __init__(self, __dict__ = {}, **kwargs):                                        #@ReservedAssignment
        self.__dict__.update(__dict__)
        self.__dict__.update(kwargs)
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    def __str__(self):
        items = ["%s = %s" % (k,repr(v)) for k,v in self.__dict__.iteritems()]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(items))        #str(self.__dict__)
    __repr__ = __str__
    
    def __getstate__(self):
        """Representation of this object for serialization. Returns a copy of __dict__ with transient attributes removed, 
        or just original __dict__ if no transient attrs defined."""
        if self.__transient__:
            state = self.__dict__.copy()
            for attr in self.__transient__: state.pop(attr, None)
            return state
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state
    

class NoneObject(Object):
    "Class for mock-up objects that - like None - evaluate to False in bool(), but additionally can hold any data inside or provide other custom behavior."
    def __bool__(self): return False
    __nonzero__ = __bool__
    
    
#####################################################################################################################################################
###
###   STRINGS & TEXT
###

def merge_spaces(s, pat = re.compile(r'\s+')):
    "Merge multiple spaces, replace newlines and tabs with spaces, strip leading/trailing space. Similar to normalize-space() in XPath."
    return pat.sub(' ', s).strip()

def escape(s):
    "Slash-escape (or encode) non-printable characters, including \n and \t."
    return s.encode('unicode_escape')

def ascii(text):
    """ASCII-fication of a given unicode 'text': national characters replaced with their non-accented ASCII analogs. 
    See http://stackoverflow.com/a/1383721/1202674, function bu(), for possible improvements."""
    if isinstance(text, str): text = text.decode("UTF-8")
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')

def prefix(sep, string):
    "Adds a prefix 'sep' to 'string', but only if 'string' is non-empty and not None. Otherwise empty string."
    if string: return sep + str(string)
    return ''

def indent(text, spaces=8, fill=None, strip=False):
    "Inserts 'fill' or a given no. of spaces at the beginning of each line in 'text'. Can strip the text beforehand"
    if fill is None: fill = ' ' * spaces
    if strip: text = text.strip()
    return fill + text.replace('\n', '\n' + fill)


### See also jsonpickle (http://jsonpickle.github.com)

class JsonObjEncoder(json.JSONEncoder):
    """Extends JSON serialization to custom classes. Serializes any non-json-serializable object by outputing its __json__() or __getstate__() or __dict__.
    Sets converted to lists. Good for printing, but not reversible: info about class of the object gets lost."""
    def default(self, obj):
        if isinstance(obj, set): return list(obj)
        if hasattr(obj, '__json__'): return obj.__json__()
        if hasattr(obj, '__getstate__'): return obj.__getstate__()
        try:
            return obj.__dict__
        except:
            return str(obj)

def dumpJson(obj):
    return json.dumps(obj, cls = JsonObjEncoder)
def printJson(*objs):
    for obj in objs: print json.dumps(obj, indent = 4, cls = JsonObjEncoder)
jsondump = dumpjson = jsonDump = dumpJson
jsonprint = printjson = jsonPrint = printJson

class JsonReversibleEncoder(json.JSONEncoder):      ###  DRAFT
    def default(self, obj):
        self.classesHandled = {}
        if isinstance(obj, self.classesHandled.values()):
            key = '__%s__' % obj.__class__.__name__
            d = {'/cls': classname(obj, full=True)}
            d.update(obj.__dict__)
            return d
        return json.JSONEncoder.default(self, obj)
class JsonReversibleDecoder(json.JSONDecoder):      ###  DRAFT
    pass

class JsonDict(dict):
    """A dictionary that's linked to a JSON file on disk: initial data is loaded from file upon __init__; 
    sync() and close() save dict contents back to the file, by re-opening and rewriting all file contents. 
    The file is closed between syncs."""
    def __init__(self, filename, load = True, indent = 2, **json_kwargs):
        super(JsonDict, self).__init__()
        self.filename = filename
        self.json_kwargs = json_kwargs
        self.json_kwargs['indent'] = indent
        if load and os.path.exists(self.filename): self.load()
    def load(self):
        with open(self.filename, 'rt') as f:
            state = json.load(f)
            self.update(state)
    def save(self):
        with open(self.filename, 'wt') as f:
            json.dump(self, f, **self.json_kwargs)
    def sync(self):  self.save()
    def close(self): self.save()


class JSON(object):         ###  DRAFT
    "JSON printer & parser. Customizable."
    
    metadata = False        # if True, printing includes additional info in JSON output, to enable proper restoring of all classes and objects from primitives 
    handlers = {}           # custom handlers for specific types, implemented as external functions rather than __getstate__/__setstate__
    indent = None
    separators = (', ', ': ')
    sort_keys = False
    
    def dumps(self, obj): 
        return json.dumps(obj, cls = JsonObjEncoder, indent = self.indent, separators = self.separators)
    
    def loads(self, s): 
        primitive = json.loads(s, object_pairs_hook = collections.OrderedDict)
        return primitive
    
    def encode(self, obj):
        "Encode 'obj' into a primitive (but possibly complex) value."
        
    def decode(self, primitive):
        "Decode primitive value with metadata into a complex object."
        # obj = cls.__initstate__(state)
    
    
### DAST printing ###

def dumpdast(obj, **kwargs):
    from .data import dast
    return dast.encode(obj, **kwargs)

def printdast(obj, **kwargs):
    from .data import dast
    print dast.encode(obj, **kwargs)


#####################################################################################################################################################
###
###   NUMBERS
###

def minmax(*args):
    if len(args) == 1: args = args[0]                   # you can pass a single argument containing a sequence, or each value separately as multiple arguments
    if len(args) == 2: return args if args[0] <= args[1] else (args[1],args[0])
    return (min(args), max(args))
    
def percent(x, y, ifzero = 0):
    '''
    Returns percentage value of x in y, or 'ifzero' when y==0.
    Return type (int/float) is the same as the type of arguments.
    '''
    return (100 * x / y) if y != 0 else ifzero

def bound(val, minn = 0.0, maxx = 1.0):
    "Restricts 'val' to the range of [minn,maxx]"
    return max(min(val, maxx), minn)

def divup(x, y):
    div = x / y
    if div * y == x: return div
    return div + 1

def noise(scale = 0.1):
    "Symmetric uniform random noise in the range: [-scale, +scale)"
    return (random.random()-0.5)*2 * scale
def mnoise(scale = 1.1):
    "Multiplicative random noise in the range: [exp(-ln(scale)), +scale); symmetric in log-scale, uniform, scale should be >1.0. For example, mnoise(2) is in the range [0.5,2.0)"
    return math.exp(noise(math.log(scale)))

def parseint(s):
    "Flexible parsing of integers from real-world strings. String may contain thousand separators (spaces, commas, dots) or parentheses."
    if not s: return None
    s = s.translate(None, ',. \n()')
    return int(s)
    

#####################################################################################################################################################
###
###   DATE & TIME
###

class Timer(object):
    def __init__(self): self.start = time.time()
    def reset(self):     self.start = time.time()
    def elapsed(self): return time.time() - self.start          # time elapsed until now, in seconds, floating-point
    def seconds(self): return self.elapsed()
    def minutes(self): return self.elapsed() / 60
    def hours(self):   return self.elapsed() / (60 * 60)
    def __str__(self):
        d = self.elapsed()
        if d < 60*10: return "%.1f s" % d
        if d < 60*60*10: return "%.1f min" % (d / 60)
        return "%.1f hours" % (d / (60*60))

"""
Different ways to represent date and time in python:
- timestamp (1360847012.038727): no. of seconds since the Epoch 
  - time.time() returns timestamp in local timezone
- time.struct_time - equivalent of C struct tm. No timezone info, but keeps day of week, day of year, daylight saving.
- datetime.datetime
- datetime.timedelta: difference between two 'datetime' objects
"""

# different time periods in seconds; for use with functions that operate on seconds, like time.time() or time.sleep()
MINUTE = 60
HOUR   = 60*60
DAY    = 60*60*24
WEEK   = 60*60*24*7
YEAR   = 60*60*24*365.2425

# current date in structural form, as datetime.date
def today():                              return datetime.date.today()
def todayString(fmt = '%Y-%m-%d'):        return datetime.date.today().strftime(fmt)

# current date+time in structural form, as datetime.datetime; use time.time() for flat form of #seconds from Epoch
def now():                                return datetime.datetime.now()
def nowString(fmt = '%Y-%m-%d %H:%M:%S'): return datetime.datetime.now().strftime(fmt)
def utcnow():                             return datetime.datetime.utcnow()

def formatDate(dt): return dt.strftime('%Y-%m-%d')               # the most typical format for date printout; 'dt' can be datetime or date
def formatDatetime(dt): return dt.strftime('%Y-%m-%d %H:%M:%S')  # the most typical format for datetime printout, with NO milliseconds, unlike str(dt)

def timestamp(t, tZone = 'UTC'):
    """Converts datetime or struct_time object 't' into Unix timestamp (int, in seconds).
    tZone specifies in what timezone 't' is in and can be either 'UTC' or 'local'. Should hold:
    >>> int(time.time()) == timestamp(datetime.datetime.utcnow()) == timestamp(datetime.datetime.now(), 'local')
    True
    """
    if isinstance(t, datetime.datetime):
        t = t.timetuple()
    if tZone == 'UTC':
        return calendar.timegm(t)
    elif tZone == "local":
        return int(time.mktime(t))      # here, more precise floating point result might be possible
    raise Exception("timestamp(), incorrect timezone specifier: " + tZone + ". Only 'UTC' or 'local' allowed.")

def asdatetime(d):
    "Convert date to a datetime with hour=seconds=0. If 'd' is already a datetime, return 'd' without changes."
    if isinstance(d, datetime.datetime): return d
    return datetime.datetime(d.year, d.month, d.day)

# Precise measurement of t2-t1 time difference, at the most fine-grained level (microseconds), only rescaled to desired units: minutes, hours, ... 
# Timezone-agnostic. Arguments can be 'datetime' or 'date' objects, but both must have the same type. 
# All the methods return floating point numbers, not integers. Values can be negative: if t2 < t1.

def secondsBetween(t1, t2):
    #if timezone: return timestamp(t2) - timestamp(t1)
    if t1 is None or t2 is None: return None
    return (t2 - t1).total_seconds()
def minutesBetween(t1, t2):
    sec = secondsBetween(t1, t2)
    return None if sec is None else sec / 60.
def hoursBetween(t1, t2):
    sec = secondsBetween(t1, t2)
    return None if sec is None else sec / (60. * 60)
def daysBetween(t1, t2):
    sec = secondsBetween(t1, t2)
    return None if sec is None else sec / (60. * 60 * 24)
def weeksBetween(t1, t2):
    sec = secondsBetween(t1, t2)
    return None if sec is None else sec / (60. * 60 * 24 * 7)
def yearsBetween(t1, t2):
    sec = secondsBetween(t1, t2)
    return None if sec is None else sec / (60. * 60 * 24 * 365.2425)

def secondsSince(t):
    return secondsBetween(t, datetime.datetime.now())
def minutesSince(t):
    return minutesBetween(t, datetime.datetime.now())
def hoursSince(t):
    return hoursBetween(t, datetime.datetime.now())
def daysSince(t):
    return daysBetween(t, datetime.datetime.now())
def yearsSince(t):
    return yearsBetween(t, datetime.datetime.now())

# import pytz
def convertTimezone(t, fromzone, tozone, tzinfo = False):
    """Convert datetime 't' from timezone 'fromzone' to 'tozone' and return as a new object ('t' is not modified). 
    Exising timezone information in 't' is ignored. Zones are objects from 'pytz' module: pytz.utc, pytz.timezone('US/Eastern'), ... 
    If country name is included in call to timezone(), daylight saving is taken into account accordingly.
    If tzinfo is True, timezone information will be left in the returned object; removed otherwise."""
    t = fromzone.localize(t).astimezone(tozone)
    if tzinfo: return t
    return t.replace(tzinfo = None)

# def convertTimezone(t, zone1, zone2 = None):
#     """Convert datetime 't' from timezone zone1 to zone2 (if zone2 != None), or from timezone set internally in 't' to zone1 (if zone2 = None).  
#     Zones are objects from 'pytz' module: pytz.utc, pytz.timezone('US/Eastern'), ... 
#     If country name is included in call to timezone(), daylight saving is taken into account accordingly."""
#     if zone2: return zone1.localize(t).astimezone(zone2)
#     return t.astimezone(zone2)

#####################################################################################################################################################
###
###   FILES 
###

# aliases for most commonly used standard methods, imported here for easy access:
fileexists = os.path.exists
normpath = os.path.normpath             # path normalization: convert to shortest string (no trailing slashes! no '.' etc.)

def filesize(name): return os.stat(name).st_size            # file size in bytes
def filetime(name): return os.stat(name).st_mtime           # file modification time (mtime); can use os.path.getmtime() instead
def filectime(name): return os.stat(name).st_ctime          # file metadata modification (Unix) or creation (Win) time (ctime); can use os.path.getctime() instead

def filedatetime(name, typ = "m"):
    "Return m-time (modification, default) or c-time (creation or inode change) of the file as a datetime object. 'typ' is either 'm' or 'c'"
    t = filetime(name) if typ=='m' else filectime(name)
    return datetime.datetime.fromtimestamp(t)
    # OR: return time.ctime(t) - time formatted as a string

def readfile(filename, mode = "rt"):
    "Open, read and close the file - all in one step."
    with open(filename, mode) as f:
        return f.read()

def writefile(filename, obj, mode = "wt"):
    "Open, write and close the file - all in one step. 'obj' can be a string or any other object, then str(obj) is written."
    if not isstring(obj): obj = str(obj)
    with open(filename, mode) as f:
        f.write(obj)

def dirname(path, level = 1):
    """Like standard os.path.dirname returns the directory component of the 'path', but additionally accepts the 'level' 
    parameter which indicates how many levels to go up the directory tree (default: 1). With level=0, 'path' is returned.
    >>> dirname("/home/user/docs/project/file.txt", 3)
    '/home/user'
    """
    for _ in range(level): path = os.path.dirname(path)
    return path

def normdir(folder, full = False):
    "Ending of folder path normalized to always contain trailing slash: 'some/dir/'. If full=True, all path will also be normalized to shortest form with normpath()."
    if full: return normpath(folder) + os.path.sep
    if folder.endswith(os.path.sep): return folder 
    return folder + os.path.sep

def listdir(root, onlyfiles = False, onlydirs = False, recursive = False, fullpath = False):
    """Generic routine for listing directory contents: files or subfolders or both. More versatile than standard os.listdir(), 
    can be used as a replacement. For large folders, with many files/dirs, listing all items is much faster 
    than 'onlyfiles' or 'onlydirs' (file/dir check for each item is very time consuming).
    """
    root = normdir(root)
    if not onlyfiles and not onlydirs:  items = os.listdir(root)
    elif onlyfiles:                     items = os.walk(root).next()[2]
    elif onlydirs:                      items = os.walk(root).next()[1]
    else:                               items = []
    if fullpath: items = [root + item for item in items]
    return items

def listdirs(root, recursive = False, fullpath = False):
    "List all subfolders of 'folder', excluding . and .."
    return listdir(root, onlydirs = True, recursive = recursive, fullpath = fullpath)
def listfiles(root, recursive = False, fullpath = False):
    "List all regular files in 'folder', no subfolders."
    return listdir(root, onlyfiles = True, recursive = recursive, fullpath = fullpath)

def findfiles(pattern):
    "Return a list of files and folders that match a given shell pattern, possibly with wildcards. Just an alias for glob.glob()."
    return glob.glob(pattern)
def ifindfiles(pattern):
    "Like findfiles(), but in a form of a generator. Uses glob.iglob()."
    return glob.iglob(pattern)


def getfile(f):
    """Returns a file object corresponding to a given special name: 'stdout', 'stderr' or 'stdin'.
    None and '' are mapped to stdout, too. Also, 'f' can be already a Tee object, a StringIO or an open file,
    in which case it's returned unchanged. Otherwise, None is returned."""
    if f in [None, '', 'stdout']: return sys.stdout
    if f in ['stderr', 'stdin']: return getattr(sys, f)
    if isinstance(f, Tee): return f
    if isinstance(f, (file, StringIO)) and not f.closed: return f
    return None

def openfile(f, mode = 'wt'):
    """Smart open(). If f is already an open file, returns it without changes. If f is a string, opens the file path
    denoted by the string ('wt' mode by default). Recognizes special names 'stdout', 'stderr' and 'stdin',
    and returns corresponding file objects in such cases. Empty string and None map to stdout file.
    Returns a tuple: the file object and a boolean to indicate if the file was opened here and must be closed by the client.
    """
    g = getfile(f)
    if g is not None: return (g, False)     # 'f' was an existing file; client should not attempt to close it
    if isstring(f):
        return (open(f, mode), True)        # 'f' was not an existing file; must open it here and let the client know that it must be closed at the end
    raise Exception("Object of incorrect type passed as a file(name), or a closed file: %s" % f)

def resource(reference, name):
    """Takes the directory part of 'reference' path (usually __file__ of the calling function) and appends 'name'
    (a *relative* file path) to obtain full path to a given resource file, located inside the directory tree 
    of the application source code. Doesn't work in python zip packages. See also pkgutil.get_data()."""
    folder = os.path.split(reference)[0]
    return folder + '/' + name


class Tee(object):
    """For duplicating output to several streams, typically stdout and a file. Usage:
         out = Tee(logfile)        # includes stdout by default, whenever only 1 argument (or none) is given
         print >>out, "Message"
    """
    def __init__(self, *files):
        """If only 1 file is given, stdout is appended automatically. If names not files are given, 
        they will be opened in 'wt' mode (possible erasure if file exists!).
        None and '' denote stdout. 'stdout', 'stderr', 'stdin' are mapped to sys.* file objects.
        """
        self.files = [openfile(f, 'wt') for f in files]                 # list of (file-object, must-close) pairs
        if len(files) <= 1: self.files.append((sys.stdout, False))
    def write(self, obj):
        for f, _ in self.files: f.write(obj)
    def flush(self):
        for f, _ in self.files: f.flush()
    def close(self):
        "Close all files that were opened here in this object."
        for f, mustclose in self.files:
            if mustclose: f.close()


#####################################################################################################################################################
###
###   CONCURRENCY
###

class Lock(object):
    """Built-in lock is of type thread.LockType, which can't be inherited from. 
       If you need to subclass LockType, use this class as a base instead."""
    def __init__(self):
        self.lock = threading.Lock()
    def acquire(self): self.lock.acquire()
    def release(self): self.lock.release()
    def locked(self): return self.lock.locked()
    def __enter__(self): return self.lock.__enter__()
    def __exit__(self, *args): return self.lock.__exit__(*args)
    def reacquire(self, delay = 0.0001):
        "Convenient shorthand for controlled re-acquiring of the lock, to let other threads execute, but still keep the lock afterwards."
        self.lock.release()
        time.sleep(delay)       # to release GIL and let another thread jump in
        self.lock.acquire()

class NoneLock(NoneObject):
    "Implements API of Lock that does nothing. Evaluates to False. Can be used in place of Lock to switch off synchronization, but keep validity of client code that uses Lock's API."
    def acquire(self): pass
    def release(self): pass
    def locked(self): return False
    def __enter__(self): pass
    def __exit__(self, *args): pass
    def reacquire(self, delay = None): pass


#####################################################################################################################################################
###
###   LOGGING & other stuff
###

class Logger(object):
    """
    Generates log files. Features:
    
    - Levels.
    - Thread safety. Optional mutual exclusion between all threads using the same Logger instance.
    - Customizable message format.
    - Customizable time format.
    - Raw printing.
    - Block printing.
    
    - Context. Every message can be accompanied with context data: a dict of key-value pairs, like time or place 
      of execution that caused this message, or data being processed when the event happened.
      Context can be passed on per-message basis and/or set globally for all subsequent messages.
      When a message is to be logged, named parameters in the message string are filled with context values.
    
    - Context stack. Value of a given item in global context can be replaced with another value and pushed to stack using push(),
      to be recovered again using pop(). This enables easy context management in nested execution.
    
    Other:
    - a shorthand: can use log(...) instead of log.message(...)
    """
    
    # class-level constants, mostly the same values as in standard 'logging' module, with small exceptions
    EMPTY, DEBUG, INFO, WARN, WARNING, ERROR, CRITICAL, FATAL = 0, 10, 20, 30, 30, 40, 50, 50
    levels = {0: '', 10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'CRITICAL',
              None: 0, '': 0, 'DEBUG': 10, 'INFO': 20, 'WARN': 30, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50, 'FATAL': 50}
    
    # defaults of instance-level properties
    mutex      = NoneLock()                         # logger can be made thread-safe (no messages mixed up) by assigning a Lock object to 'mutex'
    out        = sys.stderr                         # any file-like object with write() method
    format     = '%(time)s %(level)-5s  %(message)s'                                                    #@ReservedAssignment
    timeformat = '%Y-%m-%d %H:%M:%S'
    context    = {}                                 # global context, applicable to all messages; selected values can be overriden during log call
    minlevel   = None                               # if set, a number with minimum message level to be printed out; lower levels and None-level will be ignored

    content    = None                               # a StringIO buffer that keeps everything written to the logger; only if buffer=True in __init__
    
    def __init__(self, format = None, timeformat = None, minlevel = None, out = None, buffer = False, mutex = NoneLock(), **ctx):                  #@ReservedAssignment

        # set output stream
        self.out = out if out else sys.stderr
        if buffer:                                      # buffer all output stream in 'content'?
            self.content = StringIO()
            self.out = Tee(self.out, self.content)
            
        # set other parameters    
        if format: self.format = format
        if timeformat: self.timeformat = timeformat
        self.minlevel = minlevel
        self.context = {'level':''}
        self.context.update(ctx)
        self.stacks = collections.defaultdict(list)     # stacks of previous context values stored by push(), old value can be recovered by pop()
        if mutex is True: mutex = Lock()
        self.mutex = mutex
    
    def __setitem__(self, key, val):    self.context[key] = val
    def __getitem__(self, key):         return self.context[key]
    def __delitem__(self, key):         del self.context[key]

    def push(self, key, val):
        if key in self.context:
            self.stacks[key].append(self.context[key])
        self.context[key] = val
    
    def pop(self, key):
        if self.stacks[key]:
            self.context[key] = self.stacks[key].pop()
        else: del self.context[key]
    
    def acquire(self): self.mutex.acquire()             # for manual control over thread separation; alternatively you can use 'log.mutex' directly
    def release(self): self.mutex.release()
        
    def _print(self, msg, lock = True):
        if lock and self.mutex:
            with self.mutex:
                self.out.write(msg)
        else: self.out.write(msg)
        
    def message(self, *args, **kwargs):
        "Returns True if the message was actually printed, or False if level was too low."
        if self.minlevel is not None: 
            level = kwargs.get('level')
            if level is None: return False
            if isstring(level): level = self.levels[level]      # convert level name to number
            if level < self.minlevel: return False
        
        format = kwargs.pop('format', self.format)                                                        #@ReservedAssignment
        timeformat = kwargs.pop('timeformat', self.timeformat)
        end = kwargs.pop('end', '\n')
        lock = kwargs.pop('lock', True)
        
        # compile message string
        if args and args[0] and isstring(args[0]):              # print leading newlines before the actual message
            args = list(args)
            while args[0].startswith('\n'):
                print >>self.out
                args[0] = args[0][1:]
            if not args[0]: args = args[1:]
        msg = ' '.join(unicode(arg) for arg in args)
        
        ctx = self.context.copy()
        ctx.update(kwargs)
        ctx['message'] = msg
        if '%(time)s' in format:
            ctx['time'] = datetime.datetime.strftime(now(), timeformat)
        msg = format % ctx + end
        
        self._print(msg, lock)
        return True

    # instead of log.message(...) you can use a shorthand: log(...)    
    __call__ = message
    
    def debug(self, *args):    return self.message(*args, level = 'DEBUG')
    def info(self, *args):     return self.message(*args, level = 'INFO')
    def warn(self, *args):     return self.message(*args, level = 'WARN')
    def error(self, *args):    return self.message(*args, level = 'ERROR')
    def critical(self, *args): return self.message(*args, level = 'CRITICAL')
    warning = warn

    def raw(self, *args, **kwargs):
        """Print raw string: no formatting (attributes format, timeformat, context ignored), no implicit newlines. 
        Locking disabled by defult, pass lock=True to enable."""
        lock = kwargs.get('lock', False)
        self._print(' '.join(unicode(arg) for arg in args), lock)

    def block(self, *args, **kwargs):
        "Like raw(), but indents all lines by 'indent' spaces (2 by default) and prints newline ('end'='\n' by default) at the end."
        end = kwargs.pop('end', '\n')
        lock = kwargs.get('lock', False)
        _indent = kwargs.get('indent', 2)
        msg = ' '.join(unicode(arg) for arg in args)
        msg = indent(msg, _indent) + end
        self._print(msg, lock)
        
        
defaultLogger = Logger()

logging._levelNames[0] = ''
logging._levelNames[''] = 0
logging._levelNames[None] = 0


###  DRAFT below  ###

# TODO: *context* can be attached to log messages and to Logger2 itself; then used for filtering messages ......

class Logger2(logging.Logger):
    "Extends standard Logger with more convenient handling of context parameters: passed to methods directly as named attributes instead of wraping up in 'extra' dictionary"
    def debug(self, msg, *args, **kwargs):
        super(Logger2, self).debug(msg, *args, **self._rewrite(kwargs))
    def info(self, msg, *args, **kwargs):
        super(Logger2, self).info(msg, *args, **self._rewrite(kwargs))
    def warning(self, msg, *args, **kwargs):
        super(Logger2, self).warning(msg, *args, **self._rewrite(kwargs))
    warn = warning
    def error(self, msg, *args, **kwargs):
        super(Logger2, self).error(msg, *args, **self._rewrite(kwargs))
    def critical(self, msg, *args, **kwargs):
        super(Logger2, self).critical(msg, *args, **self._rewrite(kwargs))
    def exception(self, msg, *args, **kwargs):
        super(Logger2, self).exception(msg, *args, **self._rewrite(kwargs))
    def log(self, level, msg, *args, **kwargs):
        super(Logger2, self).log(level, msg, *args, **self._rewrite(kwargs))
        
    def _rewrite(self, d, known = ['exc_info','extra']):
        "move unknown (extra) parameters from 'd' level to d['extra'] level"
        extra = {}
        for k in list(d.iterkeys()):
            if k not in known:
                extra[k] = d[k]
                del d[k]
        if extra: d['extra'] = extra        # previous 'extra' dictionary is lost, so don't mix this method with the standard one 
        return d
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None):
        "Fixes records generated by original makeRecord() by replacing standard __dict__ with defaultdict, to handle missing parameter values instead of raising exceptions"
        record = super(Logger2, self).makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra)
        d = collections.defaultdict(str)
        d.update(record.__dict__)
        record.__dict__ = d
        return record

class LoggerContext(object):
    """Wrapper (view) for a Logger2 object, which transparently appends predefined parameters to each call to logging methods.
    One Logger2 object can have multiple LoggerContext wrappers at the same time.
    !!! TODO: check standard LoggerAdapter class, which seems to do the same thing!!! http://docs.python.org/2/howto/logging-cookbook.html#context-info
    """
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.context = kwargs


# default format for logging, for use when creating loggers
logging_formatter = logging.Formatter('%(asctime)s %(task) %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')

# default logging handler: prints all messages (level=DEBUG) to stderr
logging_handler = logging.StreamHandler()
logging_handler.setLevel(logging.DEBUG)
logging_handler.setFormatter(logging_formatter)

# default logger: prints all messages (level=DEBUG) to stderr; if needed, this object can be modified (other handlers, ...)
#defaultLogger = Logger2("", logging.DEBUG)
#defaultLogger.addHandler(logging_handler)

# dummy logger that does no logging (high level, no handlers)
noLogger = Logger2("", logging.CRITICAL + 1)


from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler      
# see: http://stackoverflow.com/questions/8467978/python-want-logging-with-log-rotation-and-compression#comment10473658_8468041 
#      http://docs.python.org/library/logging.handlers.html


#####################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()
