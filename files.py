# -*- coding: utf-8 -*-
'''
High-level API for object-oriented file access.
DRAFT, work in progress, some docs below are incorrect.

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

import os, shutil, jsonpickle
from copy import deepcopy
from itertools import count

from nifty.util import classname, fileexists, filesize


#####################################################################################################################################################
###
###   Files
###

"""
class GenericFile:
    "Abstract file. Becomes open upon instantiation, but can be *reset* many times to its inital state, and being open doesn't mean that
    underlying physical file(s) is also open - rather to the contrary, physical file(s) is typically closed until real I/O operation is performed."
    
    __init__(mode) - make the file ready for use in a given mode; do NOT open the physical file yet, only when the actual I/O operation begins;
                     physically create the file if it doesn't exist yet in the underlying filespace
    
    __iter__ - iterate over all items in the file starting from the current position; upon close of iteration, file is reset to its initial state
    read - read next item(s) using the internal, implicitly created, __iter__ iterator; after EOF returns None(s); call explicitly seek() or reset() to start reading again;
           base class implementation uses __iter__ internally
    write - 

    reset(mode) - finish all previous I/O operations (typically by closing physical file(s)) and turn the file back to initial state, to enable new series of I/O operations
    
    flush
    close - mark that this file object won't be used anymore, make all necessary clean up to allow *re-instantiation* of this file as a new object

    -- state configuration:
    setmode
    getmode
    seek
    tell

class FileSpace:
    open(name) - create a file object for an *existing* file; there can be more than 1 file object present at a given time for a given filename - behavior undefined
    create(name) - create a new file, possibly overriding an existing one; return this new file object

"""

class GenericFile(object):
    """Interface exhibited by all file classes, including multilayered file classes. Similar to standard file's API, but with several extensions:
     - file can be in a 'closed' state
     - reopening is possible (close+open, method reopen()), without recreating the file object and passing again the same arguments
     - allows instatiation of non-initialized files (arg open=False) and opening it later with open()
     - (in subclasses) allows other classes than standard 'file' to be used as an underlying file object
    Methods intended for overriding in subclasses:
     - primary:    isopen, _open, _close, _read, write (?)
     - secondary:  __init__, _prolog, _epilog
    """

    filespace = None                # owning filespace of this file
    basespace = None                # underlying space of 'filespace'
    iterating = False               # flag to be used in __iter__ to protect against multiple iteration of the same file, at the same time
    _iterator = None                # internal read iterator created implicitly in read()
    closed    = True
    
    def __init__(self, name, mode = 'r', *args, **kwargs):
        "Arguments: mode='r', open=True"
        self.name = name
        self.mode = mode
        self.args = args
        self.kwargs = kwargs
        #kwargs.pop('mode', None)
        self.filespace = kwargs.pop('filespace', None)
        self.basespace = self.filespace.base if self.filespace else None
        if not self.basespace: self.basespace = rawFileSpace
        if kwargs.pop('open', True): self.open()
        
    # abstract and wrapper-abstract methods, to be overriden in subclasses:
    def isopen(self): return not self.closed

    def open(self, mode = None):
        "If 'mode' is given, it overrides 'mode' setting from __init__ and is kept in self.mode for all future re-openings of the file."
        if mode: self.mode = mode
        #self.kwargs['mode'] = self.mode
        #print "open() in", classname(self), self.args, self.kwargs
        self._open()
        self.closed = False
        
    def _open(self):
        "This method should be overriden in subclasses instead of open(). Implementation should read self.mode for mode parameters."
        raise NotImplemented()

    def reopen(self):
        self.close()
        self.open()
    #def reset(self): self.reopen()

    def close(self):
        if self.closed: raise Exception("Trying to close an already closed file '%s' of class %s" % (self.name, classname(self)))
        self._close()
        self.closed = True
        
    def _close(self):
        "This method should be overriden in subclasses instead of close()."
        raise NotImplemented()

    def __iter__(self):
        """Iterator over all contents of the file. File is opened at the beginning if needed. At the end, initial state (open/closed) is restored.
        Unlike standard file.__iter__, this method accepts closed file (will be opened and closed at the end), 
        and for open file rewinds file pointer to the beginning after iteration is completed, by reopening the file.
        In subclasses, override _read() method instead of this one.
        """
        self._prolog()
        try:
            it = self._read()
            for x in it: yield x
        except GeneratorExit as ex:                       # closing the iterator is a legal way to break the iteration
            self._epilog()
            raise
        self._epilog()

    def _read(self):
        """Return an iterator to be used by __iter__() and, consequently, by read(). 
        Upon call, 'self' can be in any (open/closed) state and must be left in the same state when iterator stops.
        Alternatively, proper opening/closing should be implemented by overloading _prolog and _epilog.
        This is the primary method that should be overriden in subclasses in order to implement __iter__() and read() methods.
        """
        raise NotImplemented()
    
    def _prolog(self):        
        if self.iterating: raise Exception("%s '%s' opened for iteration twice, before previous iteration has completed" % (classname(self), self.name))
        self.iterating = True
    def _epilog(self):
        self.iterating = False

    def read(self, size = None):
        """Read next item from the file using an internal __iter__ iterator. 
        Client doesn't have to create and keep the iterator explicitly, instead the iterator is created implicitly by this file object.
        If 'size' is not None, a 'size' number of items are read and returned as a list (even if size=1 or 0). 
        Less than 'size' items can be returned if EOF is reached. StopIteration is raised if no item can be returned at all.
        Note that behavior of this method differs significantly from standard file.read(): primarily because it operates on objects, not bytes.
        In subclasses, override _read() method instead of this one.
        """
        self._init_read()
        if size is None: return self._iterator.next()
        res = []
        try:
            while size > 0:
                res.append(self._iterator.next())
                size -= 1
        except StopIteration as e:
            if not res: raise
        return res
        
    def readall(self):
        "Materialized list of all objects from the file."
        return list(self)
    
    def _init_read(self):
        if not self._iterator or self._iterator.closed:
            self._iterator = self.__iter__()

    def write(self, s): raise NotImplemented()
    def flush(self): raise NotImplemented()
    
    def __call__(self, *args, **kwargs):
        """A GenericFile object can be used to instantiate a new object with the same properties, overriden afterwards with args and kwargs (if present).
        Yields the same effect as calling FileSubclass(...) but without the need to provide appropriate subclass, class instance is enough."""
        dup = copy(self)
        dup.__init__(*args, **kwargs)
        return dup
        
    def __deepcopy__(self, memo):
        if not self.closed: raise Exception("Can't deep copy an open %s object." % classname(self))
        return deepcopy(self, memo)
            

#     def exists(self): return
#     def tell(self):
#         "Current position in the file. In all types of files, this must be the final position of the previous object written, and at the same time the beginning position of the next object."

#####################################################################################################################################################

class File(GenericFile):
    """Wrapper around standard 'file' class. Note that standard read() method is renamed here to readbytes(), 
    while read() returns line(s), for consistency with iterator-based interface (method __iter__) which iterates over lines."""

    file = None                     # if None, 'self' object represents a closed file; open file otherwise

#     def isopen(self):
#         isopen = (self.file is not None)
#         assert not isopen or isinstance(self.file, file)
#         return isopen

    def _open(self):
        self.file = open(name = self.name, mode = self.mode, *self.args, **self.kwargs)
    
    def _close(self):
        self.file.close()
        self.file = None
    
    def readbytes(self, size = -1):
        if self.iterating: raise Exception("Method read() called on the File '%s' when the file is being iterated over with __iter__()" % self.name)
        return self.file.read(size)
    def write(self, s):
        self.file.write(s)
    def flush(self):
        self.file.flush()
        
    def readchars(self):
        "In the future, this method will read characters in Unicode-aware - or other (en)coding-aware - way. Encoding will be specified as a file parameter."
        raise NotImplemented()
        
    def _read(self):
        return self.file.__iter__()
#     def _prolog(self):        
#         super(File, self)._prolog()
#         self._iter_openfile = self.isopen()
#         if not self._iter_openfile: self.open()         # open the underlying file if closed
#     def _epilog(self):
#         self.close()
#         if self._iter_openfile: self.open()             # if the file was initially open, let's reopen it at the end; file pointer put at the beginning
#         super(File, self)._epilog()


class SafeRewriteFile(File):
    """File with safe rewrite: all write operations go to another file (*.rewrite) and only at the end 
    - when properly closed! - the new file gets renamed to the base name.
    Thus, if any error occurs during writing, the original (old) file is preserved. 
    In append mode, the new file is initialized as a copy of the original file.
    The rewrite file is created in all write modes, even if the original file doesn't exist yet. 
    No rewrite file in read mode. Provides also reopen() method, from File.
    
    Replacing files by the "rename" call is guaranteed to be atomic by POSIX standards! See: http://en.wikipedia.org/wiki/Ext4
    """
    
    EXT = '.rewrite'
    realname = basename = None
    
    def __init__(self, name, rewrite = True, **kwargs):
        "rewrite: if False, safe rewriting is switched off and this object behaves just like a regular file (useful for subclassing)."
        mode = kwargs.get('mode', 'r')
        if not rewrite:
            super(SafeRewriteFile, self).__init__(name, mode, **kwargs)
            return
        nflags = ('r' in mode) + ('w' in mode) + ('a' in mode)
        if nflags > 1: raise Exception("SafeRewriteFile.__init__, more than 1 r/w/a flag specified, this is forbidden")
        
        self.basename = name
        self.realname = name + self.EXT if ('w' in mode) or ('a' in mode) else name
        
        super(SafeRewriteFile, self).__init__(self.realname, **kwargs)
    
    def _open(self):
        if fileexists(self.realname) and filesize(self.realname) > 0: 
            raise Exception("SafeRewriteFile.open(), the rewrite file '%s' already exists, can't override. Possibly previous write operations were not closed properly" % self.realname)
        if 'a' in self.mode: shutil.copy(self.basename, self.realname)
        super(SafeRewriteFile, self)._open()                 # open the file named self.realname
    
    def _close(self):
        super(SafeRewriteFile, self)._close()
        if self.realname != self.basename:
            os.rename(self.realname, self.basename)

#####################################################################################################################################################

class FileWrapper(GenericFile):
    """Wrapper around GenericFile object. 
    Doesn't add any functionality by itself, only delegates all method calls to internal self.file. Functionality can be added by subclasses.
    Closed state is represented by a closed state of self.file object, rather than by self.file=None (unlike in the File class)."""

    file = None             # this file object is created already during initialization and is guaranteed to always exist

    def __init__(self, cls = None, *args, **kwargs):
        filespace = kwargs.pop('filespace', None)
        basespace = filespace.base if filespace else None
        basespace = basespace or rawFileSpace
        cls = cls or basespace.open
        
        self.file = cls(*args, **kwargs)
        super(FileWrapper, self).__init__(*args, **kwargs)        

#     def isopen(self):
#         return self.file.isopen()
    def _open(self):
        self.file.open(mode = self.mode)
    def _close(self):
        self.file.close()
    def _read(self):
        return self.file.__iter__()
#     def read(self, size = -1):
#         return self.file.read(size)
    def write(self, s):
        self.file.write(s)
    def flush(self):
        self.file.flush()


class ObjectFile(FileWrapper):
    """File with a list of serialized objects, written and read 1 at a time using a predefined serialization method,
    implemented by subclasses in _read and _write methods. 
    In read access, entire object can be used as an iterator, or read() can be called, which behaves like iterator's next() method."""

    def __init__(self, name, cls = None, flush = 0, emptylines = 0, **kwargs):
        """
        cls: what class to be used as an underlying raw file implementation.
        flush: if >0, flush() will be called automatically after every 'flush' number of write() calls.
        emptylines: no. of extra empty lines after every object.
        """
#         self.file = None                                    # file of any class 'cls', not necessarily standard 'file'
#         self.filename = name
        self.flushfreq = flush
#         self.cls = cls
        self.emptylines = emptylines
        super(ObjectFile, self).__init__(name = name, cls = cls, **kwargs)
    
    def _open(self):
        super(ObjectFile, self)._open()
#         if mode: self.mode = mode
#         if self.cls:
#             self.file = self.cls(self.filename, mode = self.mode)
#         else:
#             self.file = self.basespace.open(self.filename, mode = self.mode)
        self.flushcount = self.flushfreq
    
    def write(self, item):
        self._write(item)
        if self.emptylines: self.file.write('\n' * self.emptylines)
        self.flushcount -= 1
        if self.flushcount == 0:
            self.file.flush()
            self.flushcount = self.flushfreq
    
#     def _prolog(self):        
#         if self.iterating: raise Exception("File '%s' opened for iteration twice, before previous iteration has completed" % self.name)
#         self.iterating = True
#         self._iter_openfile = self.isopen()
#         if not self._iter_openfile: self.open()         # open the file if closed
#     def _epilog(self):
#         self.close()
#         if self._iter_openfile: self.open()             # if the file was initially open, let's reopen it at the end; file pointer put at the beginning
#         self.iterating = False
        

class JsonFile(ObjectFile):
    def _write(self, item):
        self.file.write(jsonpickle.encode(item) + "\n\n")
    def _read(self):
        "Generator that reads from an already-open self.file."
        for line in self.file:
            if not line.strip(): continue
            yield jsonpickle.decode(line)
            
class DastFile(ObjectFile):
    def __init__(self, filename, mode = 'r', cls = File, flush = 0, emptylines = 0, **dastArgs):
        super(DastFile, self).__init__(filename, cls, flush, emptylines, mode = mode)

        from nifty.data.dast import DAST
        self.dast = DAST(**dastArgs)

    def _write(self, item):
        self.dast.dump(item, self.file, newline = True)
        
    def _read(self):
        return self.dast.decode(self.file.file)
        #raise NotImplemented()
        #for item in []: yield item
    
            
class PagedFile(GenericFile):
    """Logical object file partitioned into a number of separate files (pages), named *.1, *.2, ... 
    (TODO:) On write, new part is created after size threshold is reached."""
    
    new  = "new"        # name to be used for the new page (not yet completed) during write; when done, renamed to its ultimate name
    last = "new"        # name of the last file to be tried during reading, when no more regular IDs are present; None if nothing more should be tried
    
    def __init__(self, pattern, start = 1, stop = None, ids = None, **kwargs):
        "Example 'pattern': data.%s, data.%s.json. 'ids' (optional) is a list of file IDs to be used instead of (start,stop) range."
        self.pattern = pattern
        self.start = start
        self.stop = stop                    # 'stop' INclusive, unlike in standard range()
        self.ids = ids
#         self.page = None                # page counter: name (index) of the current page
#         self.file = None                # base file containing the current page, always in open state if present; None if 'self' is closed
        #if not '%s' in pattern: pattern += '.%s'
        super(PagedFile, self).__init__(pattern, **kwargs)
    
    def _open(self):
        "Invariant of an open file: self.file holds the current page file to be read from, or None if no more pages to be read."
        self.pages = iter(self.ids) if self.ids != None \
                     else xrange(self.start, self.stop+1) if self.stop != None \
                     else count(self.start)
        self.infinite = isinstance(self.pages, count)   # iterating over infinite range of pages? missing page allowed after 1st one
        self.file = None                                # base file with the current page
        self.filename = None
        self.openNext()                                 # open 1st page
        
    def _close(self):
        if self.file: self.file.close()
        del self.file, self.infinite, self.pages
        #self.file = self.page = None
        
    def _read(self):
        while True:
            if not self.file: break                         # we're at the end of data, no more page file to read
            assert not self.file.closed
            for item in self.file: yield item
            assert not self.file.closed
            if not self.openNext(): break

    def openNext(self):
        "Close the current page and open the next one. Return True if succeeded, False if no more pages, exception when no pages present at all."
        first = True
        if self.file:
            self.file.close()
            self.file = None
            first = False
        try:
            filename = self.pattern % self.pages.next()
        except StopIteration as e:
            return self.openLast()                                      # no more pages? try once again with the 'last' name
            
#         if self.infinite and self.last and not self.basespace.exists(filename): 
#             filename = self.pattern % self.last
        try:
            self.file = self.basespace.open(filename, mode = self.mode)
            self.filename = filename
            #print "=====  PAGE %s  =====" % self.page
            return True
        except IOError as e:
            if self.openLast(): return True                             # try once again with the 'last' name
            if self.filename is None: raise                             # didn't manage to open ANY file yet? 'self' file doesn't exist, re-raise
            return False
            #if self.infinite and not first: return False

    def openLast(self):
        "Try to open the file with self.last ID."
        if self.last == None: return False
        filename = self.pattern % self.last
        if filename == self.filename: return False                      # avoid opening the last file multiple times
        try:
            self.file = self.basespace.open(filename, mode = self.mode)
            self.filename = filename
            return True
        except IOError as e:
            return False

#####################################################################################################################################################
###
###   FILE SPACE 
###

"""
Types of files:
- object file - keeps a stream of chars/objects; sequential read from the beginning; sequential write at the end;
  Implements seek/tell and seek+write (position in file is the raw position in underlying character file),
  but doesn't guarantee consistency: it's client's responsibility that data stays consistent when using seek().
  - json (object per line)
  - dast
  - ... (?)
- character file - like object file, but additionally, is position-aware (seek/tell) and rewritable (can overwrite existing content: seek + write)

Basic filespaces:
- gzip
- paged - data split over multiple files, numbered 1,2,...
- safe rewrite
- tee

Block files, memory management:
- removable - enables removal of an object from inside the file, by marking in metadata that it's removed, the main file kept untouched; for data written once, never changed, with ability to remove
- mapped - object file with a full map of object positions and unused regions; allows removal and rewriting of arbitrary objects; behaves like a random-access memory
- indexed - object file with dense index of contained objects; index kept in a separate special file; index translates object key to a file pointer
- minmax - keeps stats on min/max values of selected properties of the stored objects, incl. special '#' attribute counting object number in the stream
- 
Ex:
- filespace = paged + json + tee + rewrite + gzip
- filespace = indexed >> paged
"""

class __FileSpace__(type):
    "Metaclass for FileSpace. Enables chaining (stacking) of filespaces without their explicit instantiation, only using class names."
    def __div__(cls, other):
        return cls() / other            # instantiate the class without arguments and use the basic __div__ implementation in FileSpace
    

class FileSpace(object):
    """Abstract namespace of files and folders, with operations like: create, open, rename ... 
    Actual input/output operations implemented in the embedded File class. Subclasses may define their own File subclasses.
    FileSpaces can be stacked on top of each other, with each one adding another layer of functionality and possibly mapping file names in some way.
    """
    __metaclass__ = __FileSpace__
    
    base = None                 # underlying filespace of this space (base space), to be used for opening and accessing lower-level files
    File = None                 # class of all files returned by this filesystem
    
    def __init__(self, base = None, **kwargs):
        self.base = base
        self.kwargs = kwargs
        
    def open(self, name, **kwargs):
        if kwargs: 
            kw = self.kwargs.copy()
            kw.update(kwargs)
        else:
            kw = self.kwargs
        return self.File(name, filespace = self, **kw)
    
    __call__ = open
    
    def __div__(self, other):
        "Shorthand for addBase(), to write a stack of spaces like: files = Space3/Space2/Space1."
        self.addBase(other)
        return self
        
    def addBase(self, base):
        "Assign 'base' filespace as the most low-level base of the stack of filespaces having 'self' at the top."
        space = self
        while space.base: space = space.base
        space.base = base() if isinstance(base, type) else base
    
    
class RawSpace(FileSpace):
    File = File

rawFileSpace = RawSpace()               # default global filespace to be used when filespace is None

    
class Paged(FileSpace):
    File = PagedFile
    
class Json(FileSpace):
    File = JsonFile

class Dast(FileSpace):
    File = DastFile


class ObjectFiles(FileSpace):
    class File(File):
        def clear(self, endpos):
            "Mark the region from current position till 'endpos' as unused, so that new object can be written in this place."

class Indexed(FileSpace):
    """
    Space of object files that keep indices of their contents and enable random access to objects based on key value
    that gets translated by Indexed.File to a raw position in the underlying character file.
    A special case of key is the object ID in the file, '#'.
    """

