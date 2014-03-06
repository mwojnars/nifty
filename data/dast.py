# -*- coding: utf-8 -*-
"""
DAST (DAta STorage) file format. 
Allows easy, object-oriented, stream-oriented, human-readable serialization and de-serialization of any data structures.

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
"""


import re
from StringIO import StringIO
from itertools import izip
from datetime import datetime, date, time
from collections import OrderedDict, defaultdict, namedtuple, Iterator

from nifty.util import isstring, classname, subdict, Object
from nifty.text import regex

########################################################################################################################################################

class DAST_SyntaxError(Exception):

    # detailed info on location of the error, like in standard SyntaxError class    
    lineno = None           # line number, from 1
    offset = None           # column number, from 1
    text   = None           # portion of the input (last token) that caused the error
    
    def __init__(self, msg, token):
        """'token' is a 4-tuple as produced by Tokenizer. 'msg' may contain a %s parameter where the string that caused the error will be inserted.
        Location info (line, column) is automatically appended to the message.
        """
        self.text, self.lineno, self.offset = token[1:]
        if '%s' in msg: msg = msg % self.text
        msg += " on line %d, column %d" % (self.lineno, self.offset)
        super(DAST_SyntaxError, self).__init__(msg)


########################################################################################################################################################
###
###  Hand-crafted PARSER - 40x faster than Waxeye-based one
###

class Tokenizer(object):
    "The concept is based on the example from Python docs, or here: http://stackoverflow.com/a/14919449/1202674"

    # Tokens and their corresponding regexes...
    
    _noalpha = r'(%s)(?!\w)'                                        # checks that no alpha-numeric character follows after the match (expected whitespace or special)
    tokens = [
        # all special chars: newline, parentheses, separators
        ('SPEC' , r'[\n\(\)\[\]\{\}:,=]'),
        
        # atomic values
        ('FLOAT',  _noalpha % (regex.float + r'|([+-]?[iI]nf)|NaN|nan')),       # any floating-point number, or Inf, or NaN
        ('INT'  ,  _noalpha % regex.int),
        ('STR'  ,  regex.escaped_string),
        ('NONE' ,  _noalpha % r'None|null|~|-'),
        ('BOOL' ,  _noalpha % r'[tT]rue|[fF]alse'),
        #('INDENT',  r'[ \t]*'),                  # indentation at the beginning of a line; handled in a special way, thus not included in 'tokens', but can be returned from tokenize()
        
        # objects; chars allowed in type and key names: [a-zA-Z0-9_.] (yes, dots allowed in key names, that's useful for config files)
        ('KEY',  r'[\w\.]+(?=\s*=)'),      # key of a "key = value" pair: an identifier followed by '=' (only ident. is consumed)
        ('OBJ',  r"[\w\.]+(?=\()"),        # closed object: identifier followed immediately by '('. Space before ( interpreted as open obj with a tuple arg (!)
        ('OPEN', r"[\w\.]+"),              # open object
    ]
        
    regex = '|'.join([r'(?P<%s>%s)' % pair for pair in tokens])
    regex = r'[ \t\v]*(?:%s)' % regex             # every token can have a leading whitespace
    regex = re.compile(regex)

    
    @staticmethod
    def tokenize(text, line = 1, getindent = re.compile(r'[ \t]*').match, gettoken = regex.match):
        "'text' should be a single line, \n-terminated. 'line' is the current line number, counted from 1."
        
        pos = 0                                     # no. of input characters consumed till now
        match = getindent(text, pos)
        value = match.group()
        column = match.start() + 1
        pos = match.end()
        yield 'INDENT', value, line, column

        match = gettoken(text, pos)
        while match is not None:
            token = match.lastgroup
            value = match.group(token)
            column = match.start() + 1
            pos = match.end()
            yield token, value, line, column

            #if token == 'NEWLINE':
            #    linestart = pos
            #    line += 1
            #    indent, pos = matchIndent()                 # match and yield indentation at the beginning of the next line
            #    yield indent
            
            match = gettoken(text, pos)
        
        if pos != len(text):
            raise DAST_SyntaxError("Unexpected character '%s'", (None, text[pos], line, pos + 1))
        
        ## wrapping up in Token slows down the routine substantially!    Token = namedtuple('Token', ['name', 'value', 'line', 'column'])
        #yield Token('NEWLINE', '\n', line, pos + 1)
        #yield Token('EOL', '\n', line, pos + 1)     
        yield ('EOL', '\n', line, pos + 1)          # superflous \n at the end to allow arbitrary consumption (or not) of trailing \n (the true one may be consumed by tokenizer)


class Analyzer(object):
    "General concept based on: http://effbot.org/zone/simple-iterator-parser.htm"
    
    def IncorrectKey(self, token): return DAST_SyntaxError("Incorrect type of key '%s' in key=value pair, only identifiers allowed,", token)
    
    EMPTY = object()            # output token that indicates an empty line, without any objects
    
    def __init__(self, decode):
        self.decode = decode
        self.linenum = 1        # current line number, for error messages
        
    # TODO: turn off assertions to speed up
        
    def parse(self, line, tokenize = Tokenizer.tokenize):
        """Parses the next line (input string must be a single line, \n-terminated). Returns a tuple: (indent, isopen, ispair, value),
        where 'value' is the final fully decoded object, except for the case when the line is open, 
        then 'value' is an intermediate tuple (typename, args, kwargs) to be extended with data from subsequent lines and then instantiated.
        """
        self.next = next = tokenize(line, self.linenum).next
        self.isopen = False
        
        indent = next()
        assert indent[0] == "INDENT"
        indent = indent[1]
        if len(indent) == len(line) - 1: 
            assert line[-1] == '\n'
            return indent, False, False, self.EMPTY
        
        item = self.lineitem(next())
        
        # try reading the terminating \n, sometimes it remains in the input stream despite the item being parsed
        #try: end = next()
        #except StopIteration, e: end = None
        end = next()
        if end[1] != '\n': raise DAST_SyntaxError("Too many elements in line, expected newline instead of '%s'", end)
        
        self.linenum += 1
        return (indent, self.isopen) + item

    def lineitem(self, token):
        "Either a value or a pair, that consumes all the input (i.e., till the end of line). Always returns a tuple: ('value', x) or ('pair', (k,v))."
        val1 = self.value(token)
        if self.isopen: return False, val1          # for open values, don't even try to parse 'sep', it will raise exception (EOL passed already)
        sep = self.next()
        if sep[1] not in ':=\n': raise DAST_SyntaxError("Expected a key-value separator ':' or '=', or a newline, but found '%s' instead,", sep)
        if sep[1] == '\n': 
            return False, val1
        val2 = self.value(self.next())
        return True, (val1, val2)
    
    def value(self, token):
        "Returns the object (atomic or composite value, but NOT a pair) that starts with 'token' at the current position."
        name, val = token[:2]
        n = name[0]
        
        # atomic value
        if n in 'SIFKNB':
            if n == 'S' and name == 'STR':
                return val[1:-1].decode("string-escape")
            if n == 'I' and name == 'INT':
                return int(val, 0)
            if n == 'F' and name == 'FLOAT':
                return float(val)
            if n == 'K' and name == 'KEY':
                return val
            if n == 'N' and name == 'NONE':
                return None
            if n == 'B' and name == 'BOOL':
                return val[0] in 'tT'
        
        # collection
        if val in '([{':
            if val == '(':
                items = self.sequence(self.value, ')')
                return tuple(items)
            if val == '[':
                return self.sequence(self.value, ']')
            if val == '{':
                pairs = self.sequence(self.pair, '}')
                return dict(pairs)
        
        # closed object
        if name == 'OBJ':
            token = self.next()
            assert token[1] == '('
            token = self.next()
            args, kwargs = self.itemseq(token, ')')
            return self.decode(val, args, kwargs)
        
        # open object
        if name == 'OPEN':
            self.isopen = True
            token = self.next()
            if token[1] == ':':                             # open object of the form "typename:\n", return immediately
                #token = self.next()
                #if token[1] != '\n': raise DAST_SyntaxError("Open object of the form 'typename:' followed by a token other than newline: '%s',", token)
                return val, [], {}                          # (typename, args, kwargs)
            else:
                args, kwargs = self.itemseq(token, '\n')
                return val, args, kwargs                    # (typename, args, kwargs)
        
        raise DAST_SyntaxError("malformed expression (%s)" % (token,))

    def pair(self, token):
        iskey = (token[0] == 'KEY')
        key = self.value(token)
        sep = self.next()
        if sep[1] not in ':=': raise DAST_SyntaxError("Expected a key-value separator ':' or '=' but found '%s' instead", sep)
        if sep[1] == '=' and not iskey: raise self.IncorrectKey(sep)
        val = self.value(self.next())
        return (key, val)
    
    def sequence(self, parseFun, stop):
        out = []
        token = self.next()
        while token[1] != stop:
            out.append(parseFun(token))
            token = self.next()
            if token[1] == ',': token = self.next()
        return out

    def itemseq(self, token, stop):
        "Sequence of items: values and/or pairs, mixed, returned in two separate lists."
        vals = []
        pairs = {}
        while token[1] != stop:
            iskey = (token[0] == 'KEY')
            val1 = self.value(token)
            #print token, val1
            token = self.next()
            if token[1] == '=':
                if not iskey: raise self.IncorrectKey(val1)
                token = self.next()
                val2 = self.value(token)
                token = self.next()
                pairs[val1] = val2
            else:
                vals.append(val1)
            if token[1] == ',': token = self.next()
        return vals, pairs


########################################################################################################################################################
###
###  ENCODER
###

class Encoder(object):
    """New encoder is created for every new record, to enable use of instance variables as current state of the encoding, in thread-safe way."""
    
    _params = "indent listsep dictsep maxindent mode1".split()          # only these parameters will be copied during initialization, for later use
    
    def __init__(self, out, params): #indent, listsep, dictsep, maxindent, mode1):
        #self.indent, self.listsep, self.dictsep, self.maxindent, self.mode1  =  indent, listsep, dictsep, maxindent, mode1
        #if (set(params.keys()) - self._params): raise Exception("One of the parameters is unrecognized: %s" % params.keys())
        params = subdict(params, self._params)
        self.__dict__.update(params)
        self.out = out                                      # a file-like object where output code will be written to
    
    def encode(self, obj, mode = 2, level = 0, **kwargs):
        "level, mode - *initial* level and mode for 'obj' encoding, used for the root node of object hierarchy and modified along the way."
        if kwargs: self.__dict__.update(kwargs)
        self._encode(obj, mode, level)
        
    def _encode(self, obj, mode = 0, level = None):
        """Encode object hierarchy rooted at 'obj' and write the output code to self.out.
        Output code shall never include leading/trailing whitespace, even in mode 2 (no trailing \n after the last line), 
        it's client's responsibility to add whitespace whenever necessary.
        
        level: current nesting level, for proper indentation
        mode: 0-inline (bounded), 1-endline (unbounded), 2-multiline
        """
        
        if mode >= 2 and level >= self.maxindent: mode = 1              # downgrade 'mode' if level is large already
        if mode == 1 and not self.mode1: mode = 0
        
        t = type(obj)
        encode = self.encoders.get(t) or Encoder._obj
        encode(self, obj, mode, level)
        
    def _write(self, s):
        self.out.write(s)
    def _indent(self, s):
        prefix = self.indent * self.level
        return prefix + s.replace('\n', '\n' + prefix)
    
    def _none (self, x, m, l): self._write('-')
    def _bool (self, x, m, l): self._write(str(x))
    def _int  (self, x, m, l): self._write(str(x))
    def _float(self, x, m, l): self._write(str(x))
    
    def _str(self, s, mode, level):
        if mode == 0: self._write('"' + encode_basestring(s) + '"')
        else:
            s = encode_basestring_multiline(s)
            s = s.replace('\n', '\n' + self.indent * level + ' ')
            self._write('"' + s + '"')
    
    def _list(self, x, mode, level):
        if mode == 0:
            self._write('[')
            self._sequence0(x)
            self._write(']')
        elif mode == 1:
            self._write('list ')
            self._sequence0(x)
        else:
            self._write('list:\n')
            self._sequence2(x, level + 1)
    
    def _tuple(self, x, mode, level):
        if mode == 0:
            self._write('(')
            self._sequence0(x)
            self._write(')')
        elif mode == 1:
            self._write('tuple ')
            self._sequence0(x)
        else:
            self._write('tuple:\n')
            self._sequence2(x, level + 1)
    
    def _set(self, x, mode, level):
        if mode == 0:
            if x:
                self._write('{')
                self._sequence0(x)
                self._write('}')
            else:
                self._write('set()')
        elif mode == 1:
            self._write('set ')
            self._sequence0(x)
        else:
            self._write('set:\n')
            self._sequence2(x, level + 1)
    
    def _dict(self, d, mode, level):
        if mode <= 1: self._dict0(d)
        else: self._dict2(d, level)
        
    def _dict0(self, d):
        self._write('{')
        lsep, dsep = self.listsep, self.dictsep
        first = True
        for k, v in d.iteritems():
            if not first: self._write(lsep)
            self._encode(k)
            self._write(dsep)
            self._encode(v)
            first = False
        self._write('}')
    
    def _dict2(self, d, level, typename = 'dict', fmt = {}):
        "Optional fmt[key] is mode number (0/1/2) that shall be used for the value of a given key."
        self._write(typename + ":\n")
        level += 1
        sep = self.dictsep
        indent = self.indent * level
        first = True
        for k, v in d.iteritems():
            if not first: self._write('\n')
            self._write(indent)
            self._encode(k)
            self._write(sep)
            mode = fmt.get(k, 2)
            self._encode(v, mode, level)
            first = False
    
    def _obj(self, x, mode, level):
        typename = classname(x, full=True)
        try: xrepr = x.__dict__
        except:
            raise Exception("dast.Encoder, can't encode object of type <%s>, unable to retrieve its __dict__ property" % typename)
        if mode <= 1: 
            self._write(typename + " ")
            self._dict0(xrepr)
        else: 
            fmt = getattr(x, '__dast_format__', {})
            self._dict2(xrepr, level, typename, fmt)

    
    def _sequence0(self, l):
        first = True
        for i in l:
            if not first: self._write(self.listsep)
            self._encode(i)
            first = False
    def _sequence2(self, l, level):
        indent = self.indent * level
        first = True
        for i in l:
            if not first: self._write('\n')
            self._write(indent)
            self._encode(i, 2, level)
            first = False


    def _datetime(self, d, m, l):
        self._write('datetime "' + str(d) + '"')
    def _date(self, d, m, l):
        self._write('date "' + str(d) + '"')
    def _time(self, t, m, l):
        self._write('time "' + str(t) + '"')
        

    # encoders for standard types
    encoders = {int:_int, float:_float, bool:_bool, str:_str, unicode:_str, type(None):_none, 
                list:_list, tuple:_tuple, set:_set, 
                dict:_dict, OrderedDict:_dict, defaultdict:_dict,
                datetime:_datetime, date:_date, time:_time}


########################################################################################################################################################
###
###  DECODER
###

def _tuple(*args): return args
def _list(*args): return list(args)
def _dict(*pairs): return dict(pairs)
def _set(*args): return set(args)

def _datetime(s): 
    if '.' in s: return datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
def _date(s): 
    return datetime.strptime(s, '%Y-%m-%d').date()
def _time(s):
    if '.' in s: return datetime.strptime(s, '%H:%M:%S.%f').time()
    return datetime.strptime(s, '%H:%M:%S').time()


class Decoder(object):
    "Decoder of a particular encoded string (file or list of DAST-encoded lines). Creates buffer on top of the string to easily iterate over lines with 1 line lookahead."

    #EOF = object()              # token that indicates end of file OR end of current block during decoding
    dicttype = OrderedDict
    decoders = {'tuple':_tuple, 'list':_list, 'dict':_dict, 'set':_set, 'datetime':_datetime, 'date':_date, 'time':_time}

    #nocompile = False           # if True, decode() will return syntax trees instead of compiled objects

    def __init__(self, input, decoders = None):
        "'decoders': dict of type decoders to override the default Decoder.decoders."
        self.decoders = decs = Decoder.decoders.copy()             # the dict of decoders may get modified during operation, thus shallow-copying
        if decoders: decs.update(decoders)
        
        # replace names of classes/functions, present as values in 'decoders', with actual class/func objects;
        # rewrite values in 'decoders' to include info whether it's a class or just a function
        for name, dec in decs.iteritems():
            if isstring(dec): decs[name] = dec = _import(dec)
            isclass = isinstance(dec, type)
            decs[name] = (dec, isclass)                     # now every value in 'decoders' is a pair: (decoder, isclass)
        
        # make an iterator from 'input'
        if isstring(input):
            self.input = iter(input.splitlines(True))       # must keep newline characters, thus 'True'
        elif isinstance(input, Iterator):
            self.input = input
        else:
            self.input = iter(input)
        
        self.parser = Analyzer(self.decodeType)
        self.line = None                    # the next line to be decoded, in a parsed form; client can read it directly for a preview of the next line; must explicitly call move() afterwards
        self.linenum = 0                    # no. of the current line ('line'), counting from 1
        self.move()
        
    def move(self):
        "Load next line to the buffer."
        try:
            line = self.input.next()
            self.linenum += 1
            self.line = self.parser.parse(line)                 # a tuple: (indent, isopen, ispair, value)
        except StopIteration, e:
            self.line = None

    def hasnext(self, indent):
        "Check if the buffered line (next to be parsed) exists AND is indented MORE than 'indent' (part of a block with header's indentation of 'indent')."
        if self.line is None: return False
        if indent is None: return True
        nextIndent = self.line[0]
        return len(nextIndent) > len(indent) and nextIndent.startswith(indent)

    def skipempty(self, indent):
        "Skip empty lines; all of them must be properly indented (!). Return True if stopped at a non-empty line. False if no more lines in this block."
        while self.hasnext(indent):
            if self.line[3] is not Analyzer.EMPTY: return True
            self.move()
        return False

    def decodeItem(self, indent):
        """Recursive decoding of a single item nested inside an item indented by 'indent'. The item being decoded must be indented MORE than 'indent'.
        Returns a pair: (obj, ispair), if ispair=True it means that 'obj' is a key:value pair instead of an object.
        Returns None instead of a pair if no more items present in the current block specified by 'indent'.
        """
        hasnext = self.skipempty(indent)
        if not hasnext: return None                                 # no more items in this block?
        indent, isopen, ispair, obj = self.line                     # obj can be an open object in intermediate form: (typename, args, kwargs)
        if ispair: key, obj = obj                                   # decode key from key:value pair
        linenum = self.linenum                                      # must read self.line and self.linenum now, bcs they may change in next operations
        self.move()

        # check subsequent lines to collect all arguments of an open object
        while True:
            arg = self.decodeItem(indent)
            if arg is None:                                         # end of block? return the final complete object
                if isopen: obj = self.decodeType(*obj)              # open object? can now instantiate from: obj == (typename, args, kwargs)
                if ispair: return (key, obj), True
                return obj, False
            
            if not isopen:
                raise Exception("Incorrect DAST format, trying to append a sub-object (indented next line) to a raw value or a closed object at line %s" % linenum)
            #assert isinstance(obj, tuple) and len(obj) == 3        # obj == (typename, args, kwargs)

            argObj, argIspair = arg
            if argIspair:                                           # argument is a k:v pair, add to 'kwargs' dict
                k, v = argObj
                obj[2][k] = v
            else:                                                   # argument is a plain object, add to 'args' list
                obj[1].append(argObj)
            
    
    def decodeType(self, typename, args, kwargs):
        "Decode typename extracted from a DAST file (map to a corresponding type or callable), and instantiate with given arguments."
        # find the right decoder for a given class
        decoder, isclass = self.decoders.get(typename, (None,None))
        if decoder is None:                                         # decoder not found? must load a class
            decoder = _import(typename)
            isclass = True
            self.decoders[typename] = (decoder, isclass)            # keep the loaded type for future reference
         
        if hasattr(decoder, '__dast_init__'):                       # 'decoder' is a class with custom deserializer __dast_init__
            return decoder.__dast_init__(*args, **kwargs)           # use __dast_init__ = __init__ to direct decoding to standard initializer
        elif isclass:
            # no __dast_init__? __dict__ will be recreated automatically, on UNinitialized object;
            # if you need something special to be done both for __init__ and DAST loading, try to put it inside custom __new__() of the class,
            # as *args are passed to new(), while **kwargs are copied directly to __dict__.
            obj = decoder.__new__(decoder, *args)
            obj.__dict__.update(kwargs)
            return obj
        else:
            return decoder(*args, **kwargs)                         # decoder is a function, don't bother with __new__ and __dict__
        
        return decoder(*args, **kwargs)
    
    def decode(self):
        while True:
            item = self.decodeItem(None)
            if item is None: break
            yield item[0]
    

def _import(path):
    """Load the module and return the class/function/var, given its full package/module path (there always must be a module name in the path).

    >>> _import('nifty.data.dast.Decoder')
    <class 'nifty.data.dast.Decoder'>
    >>> _import('__builtin__.int')()
    0
    """
    if '.' not in path: raise Exception("Can't import an object without module/package name: %s" % path)
    mod, name = path.rsplit('.', 1)
    module = __import__(mod)
    return getattr(module, name)


########################################################################################################################################################
###
###  DAST
###

class DAST(object):
    """Keeps global parameters of encoding, common to all records to be encoded.
    
    Special properties and methods that can be present in encoded/decoded classes:
     __dast_format__ - dict of {attr: mode} pairs that defines what mode level should be used for encoding of specified attributes of this class'es objects.
     __dast__ - custom encoding method; returns an object that will be subsequently DAST-encoded in usual (recursive) way.
     __dast_init__ - method to be called instead of __init__ during decoding and instantiating of the class.
     
    Usage:
    - Can't encode volatile objects, like: generators, files, ...
    """
    
    # basic parameters
    indent = "  "
    listsep = ", "
    dictsep = ": "
    maxindent = 3
    mode1 = True            # use mode-1 when possible (True) or mode-0 instead (False)

    # initial mode and level, for encoding root node of object hierarchy
    mode = 2
    level = 0

    # custom encoders / decoders - only the overrides that will replace or extend default Encoder.encoders or Decoder.decoders
    decoders = {}

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)                    # one-time assignment of all properties
        #self.parser = WaxeyeParser()

    def encode(self, obj, **kwargs):
        "Encode object hierarchy rooted at 'obj' and return the resulting code as a string."
        output = StringIO()
        self.dump(obj, output, newline = False, **kwargs)
        return output.getvalue()
    
    def dump(self, obj, output, newline = True, **kwargs):
        "Encode object hierarchy rooted at 'obj' and save to 'output' file-like object, with newline(s) at the end if newline=True or 1+."
        params = DAST.__dict__.copy()
        params.update(self.__dict__)
        params.update(kwargs)
        encoder = Encoder(output, params)
        #self.__dict__.update(kwargs)
        #encoder = Encoder(output, self.indent, self.listsep, self.dictsep, self.maxindent, self.mode1)
        encoder.encode(obj, params['mode'], params['level'])
        if newline: output.write('\n' * int(newline))

    def decode(self, input):
        """Generator. Yields consecutive objects decoded from 'input'. 'input' is either a file (an object that iterates over lines), 
        or a string, in such case it will be split into lines beforehand."""
        return Decoder(input, self.decoders).decode()
        #return Decoder(input, self.parser, self.decoders).decode()

    def decode1(self, input):
        "Decode only the 1st object, ignore the rest. Exception if no object present."
        return self.decode(input).next()

    
#####################################################################################################################################################

dast = DAST()

def encode(obj, **kwargs):  return dast.encode(obj, **kwargs)
def decode(input):  return dast.decode(input)
def decode1(input): return dast.decode1(input)

#####################################################################################################################################################

class ObjectStream(object):   # DRAFT
    """Character stream that represents a stream of hierarchical objects, encoded (serialized). Client can read/write objects without worrying about delimiters
    between different objects and parts - this is managed entirely by the stream, by proper encoding and adding delimiters."""
    
    EOS = object()      # "End Of Stream" token, for use in read() to signal no more data at the current nesting level; analog of EOF
    level = 0           # nesting level (in object hierarchy) where current write/read is taking place - for proper delimiting and EOS signaling
    
    def write(self, obj):
        pass
    def read(self):
        return ObjectStream.EOS
    def empty(self):
        "True if the next read() is going to return EOS."

class Substream(object):
    "A view linked to an ObjectStream that gives access to all objects serialized at a given level, via generator."


#####################################################################################################################################################
###
###   PRIMITIVES 
###

### string encoding, source code from standard lib: json/encoder.py

ESCAPE0 = re.compile(r'[\x00-\x1f\\"\b\f\n\r\t]')
ESCAPE0_DCT = {'\\': '\\\\', '"': '\\"', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
ESCAPE2 = re.compile(r'[\x00-\x1f\\"\b\t]')
ESCAPE2_DCT = {'\\': '\\\\', '"': '\\"', '\b': '\\b', '\t': '\\t'}

UNESCAPE_DCT = {'\\\\': '\\', '\\"': '"', "\\'": "'", '\\b': '\b', '\\f': '\f', '\\n': '\n', '\\r': '\r', '\\t': '\t'}

for i in range(0x20):
    ESCAPE0_DCT.setdefault(chr(i), '\\u{0:04x}'.format(i))
    ESCAPE2_DCT.setdefault(chr(i), '\\u{0:04x}'.format(i))


def encode_basestring(s):
    "Return a JSON representation of a Python string"
    def replace(match): return ESCAPE0_DCT[match.group(0)]
    return ESCAPE0.sub(replace, s)

def encode_basestring_multiline(s):
    "Like encode_basestring(), but leave newlines untouched."
    def replace(match): return ESCAPE2_DCT[match.group(0)]
    return ESCAPE2.sub(replace, s)


ESCAPE_ASCII = re.compile(r'([\\"]|[^\ -~])')
HAS_UTF8 = re.compile(r'[\x80-\xff]')

def encode_basestring_ascii(s):
    """Return an ASCII-only JSON representation of a Python string"""
    if isinstance(s, str) and HAS_UTF8.search(s) is not None:
        s = s.decode('utf-8')
    def replace(match):
        s = match.group(0)
        try:
            return ESCAPE0_DCT[s]
        except KeyError:
            n = ord(s)
            if n < 0x10000:
                return '\\u{0:04x}'.format(n)
                #return '\\u%04x' % (n,)
            else:
                # surrogate pair
                n -= 0x10000
                s1 = 0xd800 | ((n >> 10) & 0x3ff)
                s2 = 0xdc00 | (n & 0x3ff)
                return '\\u{0:04x}\\u{1:04x}'.format(s1, s2)
                #return '\\u%04x\\u%04x' % (s1, s2)
    return '"' + str(ESCAPE_ASCII.sub(replace, s)) + '"'


# Assume this produces an infinity on all machines (probably not guaranteed)
INFINITY = float('1e66666')
FLOAT_REPR = repr


#####################################################################################################################################################

# for testing...

class RichText(Object):
    def __init__(self, text):
        self.content = text

class Comment(Object):
    def __init__(self, text, dt):
        self.text = RichText(text)
        self.date = dt
        
class User(Object):
    def __init__(self):
        comm1 = Comment("Ala ma kota", datetime(2013,4,20,10,10,10))
        comm2 = Comment("Ala ma kota", datetime(2013,4,20,10,10,10))
        comm3 = Comment("Ala ma kota", datetime(2013,4,20,10,10,10))
        self.comments = [comm1, comm2, comm3]
        self.pagesVisited = {123,643,76554}


if __name__ == "__main__":
    from nifty.util import printjson
    
    in1  = [1,2,3,['ala','kot','pies'],5,6]
    out1 = encode(in1, mode = 1)
    print out1
    print encode({1:'ala', 2:'kot', 3:['ala',{'i','kot'}, {'jaki':'burek',10:'as'}], 4:None}, mode = 2)

    print "---"
    x = {3,5,7,'ala'}
    y = encode(x, mode=1);      print y
    z = decode1(y);     print z
    print
    
    x = User()
    y = encode(x);      print y
    z = decode1(y);     print z
    print
    
    y = encode(x, mode = 1);    print y
    z = decode1(y);             print z
    print

    print "\ndone"
    

    