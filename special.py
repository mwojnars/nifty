# -*- coding: utf-8 -*-
'''
Various functions and classes, like in 'util' but more advanced, possibly with special dependencies (not self-contained).

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

import os, collections, json, re, datetime

from util import jsondump, getattrs, isfunction
from web import isxdoc, noscript


########################################################################################################################################################
###
###  Exceptions & Logging
###

class Message(object):
    "Atomic piece of information that can be printed out as a log event and than re-created back from the text form into object representation. May have complex internal structure."
class Snapshot(Message):
    "A snapshot of runtime context, for logging purposes."

class RichException(Snapshot, Exception):
    "Exception with advanced context support and JSON output of the traceback."
    
    order = "url sql args html".split()             # parameters inside 'context' will be kept in this order, for pretty printing
    
    def __init__(self, msg = None, cause = None, obj = None, cls = None, **kwargs):
        """
        Typical arguments (as standalone args or in kwargs dictionary):
        'cause': exception or a list of exceptions that caused this one (list can be used in variant execution, when the code can check different execution paths) 
        'obj': 'self' of the method where error occured
        'cls': class which implements the code that raised exception (typically a base class of 'obj' class, but not necessarily the same)
        'url': URL being processed when the error occured 
        'html': HTML being parsed
        'sql': SQL query being executed
        'args': arguments of the SQL query; or arguments of method call, ...
        Traceback is stored automatically.
        
        On request can store entire exception, with all context objects etc., in a JSON file.
        To conveniently view and browse this file you can use, for example, JSONView add-on for Firefox.        
        """
        self.msg = ""
        if msg: 
            maxlen = 500
            self.msg = str(msg)
            if len(self.msg) > maxlen:
                self.msg = self.msg[:maxlen] + "...<truncated>"
        self.cause = cause
        self.obj = obj
        self.cls = cls
        
        # add items into 'context' in a predefined order, for pretty printing
        self.context = collections.OrderedDict()
        if kwargs:
            for param in self.order:            # first insert common parameters, from 'order' list
                if param in kwargs: self.context[param] = kwargs.pop(param)
            self.context.update(kwargs)         # then add all remaining parameters
        
        
    def __str__(self):
        items = []
        if self.msg: items.append(self.msg)
        if self.context: items.append(jsondump(self.context)[:300])
        return " ".join(items)
        
    def html(self):
        "HTML context of this exception, as string (if present), or None."
        html = self.context.get('html')
        if html is None: return None
        return str(html)
    
    def dump(self, path = None):
        return json.dumps(self, indent = 2, cls = RichException.JsonEncoder(path))
#        def dump(name, val):
#            return '"%s": ' % name + jsondump(val)
#        
#        lines = [dump('msg', self.msg)]
#        if self.cause: lines.append(dump('cause', self.cause))
#            #if isinstance(self.cause, RichException):
#            #    line = '"cause": ' + self.cause.dump()
#        for k,v in self.context.iteritems():
#            lines.append(dump(k, v))
#            
#        return "{\n%s\n}" % ',\n'.join(lines)
    
    def save(self, path):
        "Stores full JSON snapshot of this exception on disk, in a given folder 'path', file 'exception.json' and possibly also 'exception.html' (context.html)."
        path = os.path.normpath(path)
        with open(path + "/exception.json", 'wt') as f:
            f.write(self.dump(path))

        html = self.html()
        if html is None: return
        with open(path + "/exception.html", 'wt') as f:
            f.write(html)


    class JsonEncoder(json.JSONEncoder):
        """RichException's own JSON encoder. Custom serialization of:
            * RichException objects
            * xdoc objects: HTML wrote to an external file; json output contains link to the file and raw text
            * classes
            * other objects: outputs __dict__ truncated to regular properties.
        For printing, not reversible.
        """
        def __init__(self, path = None):
            self.path = path            # if not-None, additional files will be created for each xdoc sub-documents (see makefile())
            self.nfiles = 0             # no. of files created in makefile() method
            
        def __call__(self, *args, **kwargs):
            json.JSONEncoder.__init__(self, *args, **kwargs)
            return self
            
        def default(self, obj):
            if isinstance(obj, RichException): return self.richexception(obj)
            if isinstance(obj, Exception): return self.exception(obj)
            if isinstance(obj, re._pattern_type): return self.regex(obj)
            if isxdoc(obj): return self.xdoc(obj)
            if isinstance(obj, type): return self.anyclass(obj)
            return self.anyobject(obj)

        def richexception(self, ex):
            "Encode RichException object by converting its top level contents and 'context' into a dictionary."
            d = collections.OrderedDict()
            d['EXCEPTION'] = self.classname(ex) #, full = False)
            d['message'] = "=== %s ===" % ex.msg
            if ex.cls is not None:
                d['in code of'] = ex.cls.__name__
            if ex.obj is not None: 
                #d['obj.class'] = self.classname(ex.obj)
                d["in 'self'"] = ex.obj
            if ex.cause: 
                d['caused by exception(s)'] = ex.cause
            #preference = []
            if ex.context:
                d['context of execution'] = ex.context
            #d.update(ex.context)
            return d

        def exception(self, ex):
            "Encode standard Exception object"
            d = collections.OrderedDict()
            d['EXCEPTION'] = self.classname(ex)
            d.update(self.getattrs(ex, ['message', 'args']))        # standard Exception doesn't have __dict__ that's why common attrs must be listed manually
            d.update(self.getattrs(ex))
            if not d['message']: del d['message']                   # don't include empty values
            if not d['args']: del d['args']
            return d
            
        def regex(self, rx):
            return rx.pattern
        
        def xdoc(self, doc):
            #d = collections.OrderedDict()
            #d['OBJECT'] = self.classname(obj)
            html = noscript(doc.html())                             # disable javascript code
            #html = merge_spaces(html)
            if self.path:
                fname = self.makefile(html)
                return "file://" + fname
            else:
                return html
        
        def anyclass(self, cls):
            d = collections.OrderedDict()
            d['CLASS'] = self.classname(cls = cls)
            try: d.update(self.getattrs(cls))
            except: pass
            return d

        def anyobject(self, obj):
            "Encode any other object"
            asstring = (datetime.datetime, datetime.date, datetime.time)
            if isinstance(obj, asstring): return str(obj)
            
            d = collections.OrderedDict()
            d['OBJECT'] = self.classname(obj)
            try:
                d.update(self.getattrs(obj))
            finally:
                return d
                #return str(obj)
        
        def classname(self, obj = None, cls = None):
            "Similar to util.classname, but the string returned has different form"
            if cls is None: cls = obj.__class__
            return cls.__module__ + ". " + cls.__name__
        
        def getattrs(self, obj, names = None):
            "Like util.getattrs, but additionally filters out or converts attributes based on their value type."
            attrs = getattrs(obj, names, '_')
            for attr, val in attrs.items():
                if isfunction(val): del attrs[attr]; continue
            return attrs
            
        def makefile(self, html):
            self.nfiles += 1
            fname = self.path + "exception_%d.html" % self.nfiles
            with open(fname, 'wt') as f:
                if isinstance(html, unicode): html = html.encode("utf-8")
                f.write(html)
            return fname


