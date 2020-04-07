# -*- coding: utf-8 -*-
"""
DAST (DAta STorage) file format. 
Allows easy, object-oriented, streamed, human-readable serialization and de-serialization of any data structures.

Main classes, methods and functions:
- DAST - main class implementing DAST language
- DAST.dump() - serialize an object into a DAST stream
- DAST.load() - decode a DAST stream and yield consecutive deserialized objects
- dump - shorthand for DAST().dump(), uses a global DAST instance created during module initialization
- load - shorthand for DAST().load(), uses a global DAST instance created during module initialization

Instead of dump/load you can also use encode/decode. The only difference is that dump() adds a newline at the end by default,
while encode() not - this can be changed by explicitly setting 'newline' argument.

To encode a custom class, DAST tries the following approaches, in this order:
- x.__getstate__()
- x.__dict__; if x.__transient__ list of attribute names is present, these attributes are excluded from the state.
Regardless of how the state was retrieved, arguments for new() are retrieved from __getnewargs__() if present
and serialized as unnamed arguments, too.


SYNTAX.

There are 3 types (modes) of value formatting:
- inline (bounded/flow) - the code occupies a part of a line; only inline code can be embedded in other inline code: [(2,3),4]
- endline (unbounded/open) - occupies the rest of line up to the nearest \n: list (2,3), 4
- outline (multiline/block) - can occupy multiple lines:
   list
     (2,3)
     4

Indented format:
- atomic values, inline:
  123  123.45  123.  True/False  None/null/-/~           -- float always with a dot . to discern from int

- strings:
  "longer text"  'text'  u"unicode text"  U'unicode text'

  multi-line, no \n encoding, newlines preserved during decoding:
  |this and
     this
    and ...

  newlines removed during decoding (but other spaces except indent preserved!):
  >this and
     this
    and ...

- binary:
  b"R0lGODlhDAAMAIQAAP"         -- values are base64 encoded
  |R0lGODlhDAAMAIQAAP
    ...

- list bounded (space-delimited, tab-delimited, comma-delimited ...):
  [x1, x2, x3, ...]

  unbounded:
  list x1, x2, x3, ...

  multi-line:
  list:
    x1
    x2
    ...

- tuple: (x1 x2 x3)
- set:
  set()
  {x1, x2, x3, ...}
  set x1, x2, x3, ...

- dictionary, bounded; pairs delimited by space/tab/comma, key/value separated by ':' or ': ' or '=' or ' ':
  {k1:v1 k2:v2 k3:v3}
  [k1:v1 k2:v2 k3:v3]            -- to enforce OrderedDict upon load
  dict(k1=v1, k2=v2)

  unbounded:
  dict k1:v1 k2:v2 k3:v3
  odict ...                      -- to enforce OrderedDict upon load

  multi-line:
  dict:
    w1: v1                    -- key can be any value of any type, but strings must be quoted:  "thisIsKey": "thisIsValue"
    w2: v2
    k1 = v3                   -- key is an identifier (string with only alpha-num chars and '.'), thus unquoted:  thisIsKey = "thisIsValue"
    k2 = v4
    ...

- date/time:
  datetime "2013-8-17 16:40:34 [TMZ?]"
  dttm "2013-8-17 16:40:34"
  date "2013-8-17"
  time "16:40:34"

- numpy array:
  array "float" x1 x2 x3 ...      -- 1D array with dtype=float, "float" is optional (default float if missing)
  array "float":                  -- 2D array
    x11 x12 ...
    x21 x22 ...
    ...

  array:                          -- multi-dimensional array, decomposed
    array:
      x111 x112 ...
      x121 x122 ...
    array:
      ...

- custom-class object; arguments to be passed in *args and **kwargs:

  module.classname()  OR  module.classname          -- empty instance (inline or endline), no args  
  module.classname(w1, w2, k1=v1, k2=v2)            -- inline; NO SPACE between classname and '(', otherwise will be treated as a single tuple
  module.classname w1, w2, k1=v1, k2=v2             -- endline
  module.classname w1, w2, k1=v1, k2=v2 ...         -- endline + outline; list of items continued on subsequent lines...
    q1 = x1
    q2 = x2

  module.classname:  OR  module.classname ...       -- outline
    q1 = x1
    q2 = x2

- tokens (reserved words): True, False, str, uni, list, dict, set, datetime, date, time, array, ...

- hooks (custom tokens with user-defined decoders, with mapping defined on app level, in both the reader and the writer):
    MyClass, ...     -- shorthand for package.module.MyClass, for use with frequent classes or mapping functions (any callable can be assigned)

- rewrites: mappings can be defined to reinterpret mod.class calls and replace them with any custom class or callable, on runtime during data read;
    this allows reading code to be reorganized after data being written, without losing access to data that used old code structure

- object IDs for de-duplication:
  &123 "my string"        -- assignment of ID to a new object
  &123 mod.cls x y z ...
  *123                    -- reference to a previously defined object; in YAML: & and *

- comments:
  # a comment until newline ...
  

>>> in1  = [1,2,3,['ala','kot','pies'],5,6]
>>> out1 = dump(in1, mode = 1)
>>> out1
'list 1, 2, 3, ["ala", "kot", "pies"], 5, 6\\n'

>>> print(encode({1:'ala', 2:'kot', 3:['ala',{'i kot'}, {'jaki':'burek',10:'as'}], 4:None}, mode = 2))
dict:
  1: "ala"
  2: "kot"
  3: list:
    "ala"
    set:
      "i kot"
    dict:
      "jaki": "burek"
      10: "as"
  4: ~


---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
"""


from __future__ import absolute_import
import re, codecs, numpy as np
from six import StringIO, PY2, PY3
from datetime import datetime, date, time
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Iterator

# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages;
# only when executed as a standalone file, for unit tests, do an absolute import
if __name__ != "__main__":
    from ..util import isstring, isdict, isbound, classname, subdict, Object
    from ..text import regex
else:
    from nifty.util import isstring, isdict, isbound, classname, subdict, Object
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
    
    _noalpha = r'(%s)(?!\w)'                    # checks that no alpha-numeric character follows after the match (expected whitespace or special)
    tokens = [
        # all special chars: newline, parentheses, separators
        ('SPEC' , r'[\n\(\)\[\]\{\}:,=]'),
        
        # atomic values
        ('FLOAT',  _noalpha % (regex.float + r'|([+-]?[iI]nf)|NaN|nan')),       # any floating-point number, or Inf, or NaN
        ('INT'  ,  _noalpha % regex.int),
        ('STR'  ,  regex.escaped_string),
        ('UNI'  ,  r'[uU]' + regex.escaped_string),                             # unicode string
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

    # for unescaping \x characters in encoded input string
    ESCAPE_SEQUENCE_RE = re.compile(r'''
        ( \\U........      # 8-digit hex escapes
        | \\u....          # 4-digit hex escapes
        | \\x..            # 2-digit hex escapes
        | \\[0-7]{1,3}     # Octal escapes
        | \\N\{[^}]+\}     # Unicode characters by name
        | \\[\\'"abfnrtv]  # Single-character escapes
        )''', re.UNICODE | re.VERBOSE)


    def decode_escapes(s):
        def decode_match(match):
            return codecs.decode(match.group(0), 'unicode-escape')
    
        return ESCAPE_SEQUENCE_RE.sub(decode_match, s)
    def __init__(self, decode):
        self.decode = decode
        self.linenum = 1        # current line number, for error messages
        
    # TODO: turn off assertions to speed up
        
    def parse(self, line, tokenize = Tokenizer.tokenize):
        """Parses the next line (input string must be a single line, \n-terminated). Returns a tuple: (indent, isopen, ispair, value),
        where 'value' is the final fully decoded object, except for the case when the line is open, 
        then 'value' is an intermediate tuple (typename, args, kwargs) to be extended with data from subsequent lines and then instantiated.
        """
        self.next = next = tokenize(line, self.linenum).__next__
        self.isopen = False
        
        indent = next()
        assert indent[0] == "INDENT"
        indent = indent[1]
        if len(indent) >= len(line) - 1 and (len(indent) == len(line) or line[-1] == '\n'):
            return indent, False, False, self.EMPTY
        
        item = self.lineitem(next())
        
        # try reading the terminating \n, sometimes it remains in the input stream despite the item being parsed
        try: 
            end = next()
            if end[1] != '\n': raise DAST_SyntaxError("Too many elements in line, expected newline instead of '%s'", end)
        except StopIteration as e: pass
        
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
    
    if PY2:
        @staticmethod
        def _unescape(text, utf8):
            s = text.decode("string-escape")
            if utf8: return s.decode("utf-8")
            return s
    else:
        @staticmethod
        def _decode_match(match):
            return codecs.decode(match.group(0), 'unicode-escape')
        @staticmethod
        def _unescape(text, utf8):
            return Analyzer.ESCAPE_SEQUENCE_RE.sub(Analyzer._decode_match, text)
            
    def value(self, token):
        "Parses an object (atomic or composite value, but NOT a pair) that starts with 'token' at the current position."
        name, val = token[:2]
        n = name[0]
        
        # atomic value
        if n in 'SUIFKNB':
            if n == 'S' and name == 'STR':
                return self._unescape(val[1:-1], False)
                # return val[1:-1].decode("string-escape")
            if n == 'U' and name == 'UNI':
                return self._unescape(val[2:-1], True)
                # return val[2:-1].decode("string-escape").decode("utf-8")
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
                items = self.valueSeq(')')
                return tuple(items)
            if val == '[':
                return self.valueSeq(']')
            if val == '{':
                #pairs = self.valueSeq(self.pair, '}')
                token = self.next()
                vals, pairs = self.itemSeq(token, '}')
                if not vals: return dict(pairs)
                if pairs: raise DAST_SyntaxError("mixed atomic values and key-value pairs inside {...} expression (%s)", token)
                return set(vals)
        
        # closed object
        if name == 'OBJ':
            token = self.next()
            assert token[1] == '('
            token = self.next()
            args, kwargs = self.itemSeq(token, ')')
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
                args, kwargs = self.itemSeq(token, '\n')
                return val, args, kwargs                    # (typename, args, kwargs)
        
        raise DAST_SyntaxError("malformed expression", token)

    def pair(self, token):
        "Parses a key:value or key=value pair."
        iskey = (token[0] == 'KEY')
        key = self.value(token)
        sep = self.next()
        if sep[1] not in ':=': raise DAST_SyntaxError("Expected a key-value separator ':' or '=' but found '%s' instead", sep)
        if sep[1] == '=' and not iskey: raise self.IncorrectKey(sep)
        val = self.value(self.next())
        return (key, val)
    
    def valueSeq(self, stop):
        "Sequence of atomic values (no pairs) returned as a list."
        out = []
        token = self.next()
        while token[1] != stop:
            val = self.value(token)
            out.append(val)
            token = self.next()
            if token[1] == ',': token = self.next()
        return out

    def itemSeq(self, token, stop):
        "Sequence of items: values and/or pairs, mixed, returned in two separate lists."
        vals = []
        pairs = {}
        while token[1] != stop:
            #print token
            iskey = (token[0] == 'KEY')
            val1 = self.value(token)
            token = self.next()
            if token[1] in ':=':                                    # key-value pair?
                if token[1] == '=' and not iskey: raise self.IncorrectKey(val1)
                token = self.next()
                val2 = self.value(token)
                token = self.next()
                pairs[val1] = val2
            else:                                                   # atomic value
                vals.append(val1)
            if token[1] == ',': token = self.next()
        return vals, pairs


########################################################################################################################################################
###
###  ENCODER
###

class Encoder(object):
    """New encoder is created for every new record to be encoded, to enable use of instance variables 
    as current state of the encoding, in thread-safe way."""

    # only these parameters will be copied during initialization, for later use    
    _params = "indent listsep dictsep keysep0 keysep2 none maxindent mode1".split()
    
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
        encode = self.encoders.get(t) or Encoder._object
        encode(self, obj, mode, level)
        
    def _write(self, s):
        self.out.write(s)
    def _indent(self, s):
        prefix = self.indent * self.level
        return prefix + s.replace('\n', '\n' + prefix)
    
    def _none (self, x, m, l): self._write(self.none)
    def _bool (self, x, m, l): self._write(str(x))
    def _int  (self, x, m, l): self._write(str(x))
    def _float(self, x, m, l): self._write(str(x))
    
    def _datetime(self, d, m, l):
        fmt = 'datetime "%s"' if m >= 1 else 'datetime("%s")'
        self._write(fmt % d)
    def _date(self, d, m, l):
        fmt = 'date "%s"' if m >= 1 else 'date("%s")'
        self._write(fmt % d)
    def _time(self, t, m, l):
        fmt = 'time "%s"' if m >= 1 else 'time("%s")'
        self._write(fmt % t)

    def _str(self, s, mode, level, asunicode = False):
        if asunicode:
            self._write('u')											# prefix for unicode strings
            s = s.encode("utf-8")
        if mode == 0: self._write('"' + encode_basestring(s) + '"')
        else:
            s = encode_basestring(s) #encode_basestring_multiline(s)
            s = s.replace('\n', '\n' + self.indent * level + ' ')
            self._write('"' + s + '"')
    
    def _unicode(self, *args):
        self._str(*args, asunicode = True)

    def _type(self, t, m, l):
        name = t.__module__ + "." + t.__name__
        self._generic_object(m, l, "type", args0 = (name,))
    
    def _list(self, x, mode, level):
        if mode == 0:
            self._write('[')
            self._sequence0(x)
            self._write(']')
        elif mode == 1:
            self._write('list ')
            self._sequence0(x)
        else:
            self._write('list:')
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
            self._write('tuple:')
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
            self._write('set:')
            self._sequence2(x, level + 1)
    
    def _dict(self, d, mode, level):
        if mode <= 1:                           # {k1:v1, k2:v2, ...}
            self._write('{')
            self._pairs0(d)
            self._write('}')
        else:                                   # multiline encoding
            self._write("dict:")
            self._pairs2(d, level + 1)
        
    def _defaultdict(self, d, mode, level):
        self._generic_object(mode, level, "defaultdict", args0 = (d.default_factory,), args2 = (dict(d),))
        
    def _object(self, x, mode, level):
        """Encode object of an arbitrary class. The following approaches to get the state are tried, in this order:
        - x.__getstate__()
        - x.__dict__; if x.__transient__ list of attribute names is present, these attributes are excluded from the state.
        Regardless of how the state was retrieved, arguments for new() are retrieved from __getnewargs__() 
        if present and serialized as unnamed arguments, too.
        """
        def getstate(x):
            getstate = getattr(x, '__getstate__', None)
            if getstate is None: return None
            if not isbound(getstate): return None               # 'x' can be a class! then __getstate__ won't work
            state = getstate()
            if isdict(state): return state
            return {'__state__': state}                         # wrap up a non-dict state in dict
        
        def getnewargs(x):
            getnewargs = getattr(x, '__getnewargs__', None)
            if getnewargs is None: return ()
            if not isbound(getnewargs): return ()
            return getnewargs()
        
        # discover typename
        typename = classname(x, full=True)
        
        # extract newargs
        newargs = getnewargs(x)
        
        # extract state
        state = getstate(x)                                     # try to pick object's state from __getstate__
        if state is None:                                       # otherwise use __dict__
            try: 
                state = x.__dict__
            except:
                raise Exception("dast.Encoder, can't encode object %s of type <%s>, "
                                "unable to retrieve its __dict__ property" % (repr(x), typename))
            
            trans = getattr(x, "__transient__", None)           # remove attributes declared as transient
            if trans:
                assert isinstance(trans, list)
                state = state.copy()
                for attr in trans: state.pop(attr, None)
        
        # parse __dast_format__
        if mode == 2:
            fmt = getattr(x, '__dast_format__', {})
            fmt_self = fmt.get('__self__', None)                # what format to use for the object itself
            if fmt_self is not None:
                mode = fmt_self
        else:
            fmt = None
        
        self._generic_object(mode, level, typename, args2 = newargs, kwargs2 = state, fmt = fmt)
        
        
    def _array(self, x, mode, level):
        dtype = str(x.dtype)
        data = x.tolist()
        self._generic_object(mode, level, "array", args0 = [dtype], args2 = data)

    # for internal use

    def _generic_object(self, mode, level, typename, args0 = (), args2 = (), kwargs0 = {}, kwargs2 = {}, fmt = {}):
        "Encode anything in an object-like format. Each piece of data to be written passed as a separate argument."
        self._write(typename)
        if mode <= 1: 
            if kwargs0 and kwargs2:
                kwargs = kwargs0.copy()
                kwargs.update(kwargs2)
            else:
                kwargs = kwargs0 or kwargs2
            self._state0(args0 + args2, kwargs, mode)
        else:
            self._state2(level, args0 = args0, args2 = args2, kwargs0 = kwargs0, kwargs2 = kwargs2, fmt = fmt)
    
    def _state0(self, args, kwargs, mode):
        "Encode state of an arbitrary object (everything after typename), mode 0 or 1."
        if mode == 0:
            self._write("(")
            self._arglist0(args, kwargs)
            self._write(")")
        elif args or kwargs:
            self._write(" ")
            self._arglist0(args, kwargs)
    def _state2(self, level, args0 = (), args2 = (), kwargs0 = {}, kwargs2 = {}, fmt = {}):
        "Encode state of an arbitrary object, mode 2, with selected arguments (args0, kwargs0) encoded in mode 1"
        inline = (args0 or kwargs0)
        outline = (args2 or kwargs2)
        if inline:
            self._write(" ")
            self._arglist0(args0, kwargs0)
        if outline:
            if not inline: self._write(":")
            self._sequence2(args2, level + 1)
            self._keywords2(kwargs2, level + 1, fmt)

    def _pairs0(self, d):
        "Encode list of pairs given as a dictionary 'd', in mode 0, without boundaries: k1:v1, k2:v2, ..."
        lsep, dsep = self.listsep, self.dictsep
        first = True
        for k, v in d.items():
            if not first: self._write(lsep)
            self._encode(k)
            self._write(dsep)
            self._encode(v)
            first = False
    def _pairs2(self, d, level):
        """Encode list of pairs given as a dict, in mode 2, without header but including leading newline:
                k1: v1
                k2: v2
                ...
        """
        sep = self.dictsep
        indent = self.indent * level
        for key, val in d.items():
            self._write('\n' + indent)
            self._encode(key)
            self._write(sep)
            self._encode(val, 2, level)
    
    def _keywords0(self, d):
        """Like _pairs0, but uses keyword notation for each pair: k1=v1, k2=v2, ... 
        All keywords must be proper identifiers, otherwise the output will be syntactically incorrect."""
        if not d: return
        lsep, dsep = self.listsep, self.keysep0
        first = True
        for k, v in d.items():
            if not first: self._write(lsep)
            self._write(k)
            self._write(dsep)
            self._encode(v)
            first = False
    def _keywords2(self, d, level, fmt = {}):
        """Like _pairs2, but uses keyword notation for each pair: k=v.
        Optional fmt[key] is mode number (0/1/2) that shall be used for the value of a given key.
        """
        if not d: return
        sep = self.keysep2
        indent = self.indent * level
        for key, val in d.items():
            self._write('\n' + indent)
            self._write(key)
            self._write(sep)
            mode = fmt.get(key, 2)
            self._encode(val, mode, level)    
    
    def _arglist0(self, args, kwargs):
        "Encodes argument list, with unnamed and keyword arguments, mode 0 or 1, no boundaries: v1, v2, k1=w1, k2=w2, ..."
        if args:
            self._sequence0(args)
            if kwargs: self._write(self.listsep)
        self._keywords0(kwargs)

    def _sequence0(self, l):
        "Encode a sequence of items 'l' in mode 0/1 (a list without boundaries): x1, x2, x3, ..."
        first = True
        for i in l:
            if not first: self._write(self.listsep)
            self._encode(i)
            first = False
    def _sequence2(self, l, level):
        """Encode a sequence of items 'l' in mode 2, without header but with leading newline:
            x1
            x2
            ...
        """
        indent = self.indent * level
        for i in l:
            self._write('\n')
            self._write(indent)
            self._encode(i, 2, level)


    # encoders for standard types
    encoders = { int:_int, float:_float, bool:_bool, str:_str, type(None):_none,
                 datetime:_datetime, date:_date, time:_time,
                 type:_type, list:_list, tuple:_tuple, set:_set, 
                 dict:_dict, OrderedDict:_dict, defaultdict:_defaultdict,
                 np.float16:_float, np.float32:_float, np.float64:_float, getattr(np, 'float128', np.float):_float,
                 np.ndarray:_array,
                }
    
    # Python 2 types:
    try:
        encoders[long] = _int
        encoders[unicode] = _unicode
    except:
        pass

########################################################################################################################################################
###
###  DECODER
###

class Decoder(object):
    "Decoder of a particular encoded string (file or list of DAST-encoded lines). Creates buffer on top of the string to easily iterate over lines with 1 line lookahead."

    def _type (name): return _import(name)
    def _tuple(*args): return args
    def _list (*args): return list(args)
    def _set  (*args): return set(args)
    #def _dict(*pairs): return dict(pairs)
    
    def _datetime(s): 
        if '.' in s: return datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    def _date(s): 
        return datetime.strptime(s, '%Y-%m-%d').date()
    def _time(s):
        if '.' in s: return datetime.strptime(s, '%H:%M:%S.%f').time()
        return datetime.strptime(s, '%H:%M:%S').time()
    
    def _defaultdict(*args): return defaultdict(*args)
    def _array(dtype, *data):
        return np.array(data, dtype = dtype)
    
    #EOF = object()              # token that indicates end of file OR end of current block during decoding
    dicttype = OrderedDict
    decoders = {'type':_type, 'tuple':_tuple, 'list':_list, 'set':_set, 'datetime':_datetime, 'date':_date, 'time':_time, 
                'defaultdict':_defaultdict, 'array':_array}

    #nocompile = False           # if True, decode() will return syntax trees instead of compiled objects

    def __init__(self, input, decoders = None):
        """'decoders': dict of type decoders, {typename: decoder}, to override default Decoder.decoders.
        Decoder can be a function that's fed with all arguments read from the file: decoder(*args, **kwargs).
        OR, decoder can be a class that's instantiated with __new__(*args) - only unnamed arguments passed!
        - and then the object's __dict__ is updated with kwargs.
        """
        self.decoders = decs = Decoder.decoders.copy()             # the dict of decoders may get modified during operation, thus shallow-copying
        if decoders: decs.update(decoders)
        
        # replace names of classes/functions, present as values in 'decoders', with actual class/func objects;
        # rewrite values in 'decoders' to include info whether it's a class or just a function
        for name, dec in decs.items():
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
            line = next(self.input)
        except StopIteration as e:
            self.line = None
            return
        #print('--', line)
        self.linenum += 1
        self.line = self.parser.parse(line)                 # a tuple: (indent, isopen, ispair, value)
        #print('++', self.line)

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
        """Decode typename extracted from a DAST file (map to a corresponding type or callable), 
        and instantiate with given arguments."""
        
        if typename == 'dict':                                      # 'dict' is special: may have arbitrary objects as keys (non-identifiers), so we can't do **kwargs to call decoder
            return kwargs
        
        # find the right decoder for a given class
        decoder, isclass = self.decoders.get(typename, (None,None))
        if decoder is None:                                         # decoder not found? must import appropriate class first
            decoder = _import(typename)
            isclass = True
            self.decoders[typename] = (decoder, isclass)            # keep the loaded type for future reference
         
        if isclass:
            obj = decoder.__new__(decoder, *args)
            if hasattr(decoder, '__dast_init__'):                   # has custom deserializer __dast_init__
                decoder.__dast_init__(obj, *args, **kwargs)
            else:
                # no __dast_init__? __dict__ will be recreated automatically, on UNinitialized object;
                # if you need something special to be done both for __init__ and DAST loading, try to put it inside 
                # custom __new__() of the class, as *args are passed to new(), while **kwargs are copied directly to __dict__.
                state = kwargs.pop('__state__', None)
                if kwargs: obj.__dict__.update(kwargs)
                if state: obj.__setstate__(state)
            return obj

        return decoder(*args, **kwargs)                         # decoder is a function, don't bother with __new__ and __dict__
    
    def decode(self):
        while True:
            item = self.decodeItem(None)
            if item is None: break
            yield item[0]
        

def _import(path):
    """Load the module and return the class/function/var, given its full package/module path.
    If no module name is present, __main__ is used.

    >>> _import('nifty.util.Object')
    <class 'nifty.util.Object'>
    """
    if '.' not in path:
        mod, name = '__main__', path
        #raise Exception("Can't import an object without module/package name: %s" % path)
    else:
        mod, name = path.rsplit('.', 1)
    module = __import__(mod, fromlist = [mod])
    #print(mod, name)
    #print(module)
    #print(module.__dict__)
    return getattr(module, name)


########################################################################################################################################################
###
###  DAST
###

class DAST(object):
    """Keeps global parameters of encoding, common to all records to be encoded.
    
    Special properties and methods that can be present in encoded/decoded classes:
     __transient__   - list of attribute names to be excluded from __dict__ during encoding
     __dast_format__ - dict of {attr: mode} pairs with mode level to be used for encoding of specified attributes of this class'es objects;
                       __dast_format__['__self__'] is mode level for the instance itself.
     __dast__        - custom encoding method; returns an object that will be subsequently DAST-encoded in usual (recursive) way. [TODO]
     __dast_init__   - method to be called instead of __init__ during decoding and instantiating of the class.
     
    Usage:
    - Can't encode volatile objects, like: generators, files, ...
    """
    
    # basic parameters
    indent  = "  "
    listsep = ", "
    dictsep = ": "
    keysep0 = "="           # separator for keyword notation of pairs: key=value, used in modes 0 and 1 
    keysep2 = " = "         # separator for keyword notation of pairs: key=value, used in mode 2
    none    = "~"           # what string to use for Nones; only '~', '-', 'null' or 'None' allowed
    maxindent = 3           # no. of nesting levels before the encoder turns from mode-2 to mode-1 or 0 
    mode1 = True            # use mode-1 when possible (True) or mode-0 instead (False)

    # initial mode and level, for encoding root node of object hierarchy
    mode = 2
    level = 0

    # custom encoders / decoders - only the overrides that will replace or extend default Encoder.encoders or Decoder.decoders
    decoders = {}

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)                    # one-time assignment of all properties

    def dump(self, obj, out = None, newline = True, **kwargs):
        """Encode object hierarchy rooted at 'obj' and write to 'out' file-like object, or return as a string if out=None. 
        Add newline(s) at the end of produced code if newline=True (default) or 1+."""
        return self.encode(obj, out, newline = newline, **kwargs)
    
    def load(self, input):
        """Generator. Yields consecutive objects decoded from 'input'. 
        'input' is either a file object, or a name of file to be opened.
        If you have a string with encoded data, not a file, use decode() instead."""
        if isstring(input): input = open(input, 'rt')
        return self.decode(input)
    
    def encode(self, obj, out = None, newline = False, **kwargs):
        "Like dump(), only newline=False by default. Used internally by dump()."
        if out is None:
            out = StringIO()
            string = True
        else:
            string = False
            
        params = DAST.__dict__.copy()
        params.update(self.__dict__)
        params.update(kwargs)
        encoder = Encoder(out, params)
        encoder.encode(obj, params['mode'], params['level'])
        if newline: out.write('\n' * int(newline))
        if string: return out.getvalue()

    def decode(self, input):
        return Decoder(input, self.decoders).decode()
        #return Decoder(input, self.parser, self.decoders).decode()

    def decode1(self, input):
        "Decode only the 1st object, ignore the rest. Exception if no object present."
        try:
            return next(self.decode(input))
        except StopIteration as e:
            raise Exception("No object decoded")

    
#####################################################################################################################################################
###
###  HELPER functions
###

dast = DAST()

def dump(obj, **kwargs):  return dast.dump(obj, **kwargs)
def load(input):          return dast.load(input)
def loads(input):         return dast.decode1(input)

def encode(obj, **kwargs):  return dast.encode(obj, **kwargs)
def decode(input):          return dast.decode(input)
def decode1(input):         return dast.decode1(input)

def printdast(obj, **kwargs): print(encode(obj, **kwargs))


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

if __name__ == "__main__":
    # import doctest
    # print(doctest.testmod())

    class RichText(Object):
        def __init__(self, text):
            self.content = text
            self.info = "nothing special"
    
    class Comment(Object):
        def __init__(self, text, dt):
            self.text = RichText(text)
            self.date = dt
            
    class User(Object):
        __transient__ = ["generator"]
        def __init__(self):
            comm1 = Comment("Ala ma kota", datetime(2013,1,20,10,10,10))
            comm2 = Comment("Ala ma kota", datetime(2013,2,20,10,10,10))
            comm3 = Comment("Ala ma kota", datetime(2013,3,20,10,10,10))
            self.comments = [comm1, comm2, comm3]
            self.pagesVisited = {123,643,76554}
            self.generator = (x for x in range(5))
    
    class Class(Object): pass
    
    Point = namedtuple("Point", "x y") 
    
    
    def test1(x, verbose = True, **kwargs):
        "Test that consists of encoding 'x' and then decoding back from the resulting code."
        if verbose: print("input:   ", x)
        y = encode(x, **kwargs)
        if verbose: 
            print("encoded: ",)
            if '\n' in y: print()
        print(y)
        z = decode1(y)
        if verbose: print("decoded: ", z)
    
        # for comparing x and z, we first must remove transient attributes if present in x
        trans = getattr(x, '__transient__', [])
        for name in trans:
            if hasattr(x, name): delattr(x, name)
    
        same = (x == z)
        if isinstance(same, np.ndarray): same = np.all(same)
        if verbose: 
            print("OK" if same else "DIFFERENT")
            print()
        if not same:
            print("decoded:", z)
            raise Exception("test1, decoded object differs from the original one")
    
    def test(x):
        print()
        test1(x, mode=0)
        test1(x, mode=1)
        test1(x, mode=2)

#     class _ndarray_(object):
#         def __eq__(self, other):
#             boolarray = np.ndarray.__eq__(self, other)
#             return np.all(boolarray)
#     np.ndarray.__eq__ = _ndarray_.__eq__


    ### testing...
    test("ala ma kota")
    test(u"żźąłćę “ĄŻÓĘŁ“ 'SuÁrez'")
    test([True, 8.23, "x y z \n abc"])
    test({3,5,7,'ala'})
    test({5: 643, "pies i kot": None, None: True})
    test(Comment("Ala ma kota", datetime(2013,4,20,10,10,10)))
    test(User())                                # user-defined object; contains transient attribute
    test(np.array([1, 2, 3.0]))                 # numpy arrays
    test(np.array([[]]))
    test(np.array([[3, 4], [2, 1], [8, 9]]))
    test([int, float, dict, defaultdict])       # standard types
    # test(Point(2,3))                            # instance of <namedtuple>
    test(defaultdict(int, {3:'x', 'ala ma':'kota'}))

    # test(x for x in range(5))                   # generator object... how to handle?
    test(Class)                                 # user-defined type with custom (inherited) __metaclass__

    print("\ndone")
    
    #__main__.Comment(date = datetime "2013-04-20 10:10:10", text = __main__.RichText(content = "Ala ma kota"))


