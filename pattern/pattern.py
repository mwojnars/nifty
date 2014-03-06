# -*- coding: utf-8 -*-
'''
Dependencies: waxeye 0.8.0, regex 2013-03-11

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

import re, regex as re2 #@UnresolvedImport
import copy, urllib
from collections import namedtuple
from datetime import datetime

import nifty.util as util
from nifty.util import isstring, islist, isdict, istuple, setattrs, prefix, ObjDict, lowerkeys, classname
from nifty.text import merge_spaces, decode_entities
from nifty.web import urljoin, xdoc

import nifty.parsing.parsing as parsing
import pattern_parser


########################################################################################################################################################
###
###  PARSER
###

class Context(parsing.Context):
    previous = None                                 # previous sibling node inside expression, None if the current node is the 1st child 
    strict = False                                  # shall the current node be interpreted in strict mode?
    repeated = False                                # is the current node included in a repeated expression, like {*VAR expr}? IN (top-down) variable
    longfill = False                                # is the current node a longfill (...) or includes longfill in the subtree? OUT (bottom-up) variable 

    #Variable = namedtuple('Variable', 'repeated longfill')
    class Variable(object):
        def __init__(self, repeated = False, longfill = False):
            self.repeated, self.longfill = repeated, longfill
    
    class Data(object):
        "Global semantic data collected by a Context object when passing through the tree. Kept in separate class so that context.copy() still references the same data as 'context'."
        def __init__(self):
            self.variables = {}         # name --> Variable
    
    def addVar(self, name):
        "Add named variable (regex group) to the global list of variables. Must be called *after* the variable's subtree has been analyzed, not before."
        var = self.data.variables.get(name)
        if not var:
            var = Context.Variable(self.repeated, self.longfill)
        else:
            var.repeated |= self.repeated                               # don't override with a new value if True already
            var.longfill |= self.longfill
        self.data.variables[name] = var
    
BaseTree = parsing.WaxeyeTree

class Tree(BaseTree):

    Parser = pattern_parser.PatternParser
    Context = Context

    class _node_(BaseTree._node_):
        def analyse(self, ctx):
            if len(self.children) == 0: return ctx
            if len(self.children) == 1: return self.children[0].analyse(ctx)
            longfill = False
            for c in self.children: 
                ctx2 = c.analyse(ctx.copy())
                longfill |= ctx2.longfill
            ctx.longfill = longfill
            return ctx

    _static_ = BaseTree._static_
    _expr_ = _node_                                 # "Generic expression - a sequence of subnodes. Consecutive expressions can be flatten: merged into one."

    class xwordfill(_node_):
        def compile(self): return r'[^<>\s]*?'                                                      #@ReservedAssignment
    class xshortfill(_node_):
        def compile(self): return r'[^<>]*?'                                                        #@ReservedAssignment
    class xlongfill(_node_):
        display = "..."
        def compile(self): return r'.*?'                                                            #@ReservedAssignment
        def analyse(self, ctx):
            ctx.longfill = True
            return ctx
    class xjustadot(_static_):
        def compile(self): return r'\.'                                                             #@ReservedAssignment

    class space(_node_):
        display = " "
    class xspace0(space):
        "Maybe-space. Can match some spaces, but not obligatory"
        def compile(self): return r'\s*'                                                            #@ReservedAssignment
    class xspace1(space):
        "Must-space. Matches 1 or more spaces."
        def compile(self): return r'\s+'                                                            #@ReservedAssignment
    class xspaceX(space):
        """Filler-space. Like short-filler, but 1st and last char must be one of ['"\s=]; or empty."""
        def compile(self): return r'''(\s*|(['"\s=][^<>]*?['"\s=]))'''                                           #@ReservedAssignment

    class xword(_static_):
        "Static string without spaces. In regex, special characters are escaped, non-printable characters are encoded."
        def compile(self):                                                                          #@ReservedAssignment
            s = self.value.encode('string_escape')
            escape = r".^$*+?{}()[]|"       # no backslash \, it's handled in encode() above
            s = ''.join('\\' + c if c in escape else c for c in s)
            return s
    class xstatic(_expr_): pass
    class xwordB(_expr_):
        """Like 'word', but modifies output regex to allow spaces around '=' and substitution of " with ' or the other way round."""
        def compile(self):                                                                          #@ReservedAssignment
            r = super(Tree.xwordB, self).compile()
            r = r.replace('=', r'\s*=\s*')
            r = r.replace('"', "'")
            r = r.replace("'", r'''("|')''')
            return r

    class xexpr(_expr_): pass
    xexprA = xexprB = xexpr
    
    class xtagname(_node_): pass
    class xnoname(_node_):
        "Match any word as a tag name, with optional leading '/'; 'tag' node below will ensure that the name matched here is followed only by space or end of tag."
        display = "."
        def compile(self): return r'/?\w+'                                                            #@ReservedAssignment
    class xtag(_node_):
        name = expr = closing = None
        def __init__(self, waxnode, tree):
            self.init(waxnode, tree)
            self.name, self.expr, self.closing = self.children[:3]
            
        def __str__(self): return '<%s%s%s' % (self.name, prefix(' ', self.expr), self.closing)
        def compile(self):                                                                          #@ReservedAssignment
            def comp(node): return node.compile() if node else ''
            gap = r'(?=\s|/|>)'                 # checks for separator between tag name and attribute list
            end = r'''['"\s=]'''
            spaceL = r'([^<>]*?%s|)' % end      # match attributes on the left of 'expr', or nothing
            spaceR = r'(%s[^<>]*?|)' % end      # match attributes on the right of 'expr', or nothing
            expr = comp(self.expr)
            expr = spaceL + expr if expr else ''
            return r'(|(?<=>)\s*)' + '<' + comp(self.name) + gap + expr + spaceR + comp(self.closing)
    
    class xrepeat(_static_): pass
    class xregex(_static_): pass
    class xvarname(_static_): pass
    class xvar(_node_):
        "A {xxx} element - named group and/or raw regex. If doesn't contain any expression, a shortfill '.' is used; put '...' inside {} to get a longfill."
        repeat = name = regex = expr = None
        
        def __init__(self, waxnode, tree):
            self.init(waxnode, tree)
            for c in self.children:
                if c.type == 'repeat': self.repeat = c
                elif c.type == 'varname': self.name = c
                elif c.type == 'regex': self.regex = c
                elif isinstance(c, tree.xexpr): self.expr = c
                
        def __str__(self): return '{%s%s%s}' % (self.name, prefix('~', str(self.regex)), prefix(' ', str(self.expr)))
        def analyse(self, ctx):
            #print "analyse() of", self.name.compile()
            if self.repeat: ctx.repeated = True
            ctx = super(Tree.xvar, self).analyse(ctx)
            if self.name: ctx.addVar(self.name.compile())
            return ctx
        
        def compile(self):                                                                          #@ReservedAssignment
            def comp(node): return node.compile() if node else ''
            repeat = comp(self.repeat)
            #if repeat: repeat += "+"        # possessive quantifier, to speed up matching
            regex = comp(self.regex)
            expr = comp(self.expr) 
            name = comp(self.name)
            lookahead = r'(?=%s)' % regex if (regex and expr) else ''   # only if both the regex and expr are present, regex is interpreted as lookahead assertion; regular pattern otherwise
            expr = lookahead + (expr or regex or r'[^<>]*?')
            name = r'?P<%s>' % name if name else ''
            return r'(%s%s)%s' % (name, expr, repeat)
    xvarA = xvarB = xvar
    
    class xoptional(_node_):
        "A [xxx] element. Resolves into a *greedy* optional match of 'xxx' pattern."
        def __init__(self, waxnode, tree):
            self.init(waxnode, tree)
            self.expr = self.children[0]
        def __str__(self): 
            return '[%s]' % self.expr
        def compile(self): #@ReservedAssignment
            return r'(%s)?' % self.expr.compile()
    xoptionalA = xoptionalB = xoptional

    class xatomic(_node_):
        "A {> xxx} element. Resolves into atomic grouping (?>...) that limits backtracking during regex matching, see: www.regular-expressions.info/atomic.html."
        def __init__(self, waxnode, tree):
            self.init(waxnode, tree)
            self.expr = self.children[0]
        def __str__(self): 
            return '{> %s}' % self.expr
        def compile(self): #@ReservedAssignment
            return r'(?>%s)' % self.expr.compile()

    #############################################
    
    def compile(self): #@ReservedAssignment
        "Compile 'tree' into regex. Return (regex2, semantics), where 'regex' is a standard regex compiled by 're' module, 'regex2' is compiled by extended 're2' ('regex') module"
        regpat, semantics = self._compile()
        try:
            #regex = re.compile(regpat, re.DOTALL)       # DOTALL: dot will match newline, too
            # (!) correct pattern which causes error in standard 're' module: re.compile("((?P<TIME>[^<>]*?))?")
            regex2 = re2.compile(regpat, re2.DOTALL | re2.VERSION1)
        except:
            print "exception raised when compiling regex:"
            print regpat
            raise
        return regex2, semantics

    
########################################################################################################################################################
###
###  PATTERN
###

class MetaPattern(type):
    def __new__(cls, name, bases, attrs):
        cls = type.__new__(cls, name, bases, attrs)
        if cls.autotest and 'test1' in cls.__dict__:
            cls().testAll(cls.__dict__)                 # instantiate the class and run unit tests defined at class level
        return cls

    def __add__(self, other):
        "For adding two Pattern subclasses, before their instantiation."
        raise NotImplemented()
    

class Pattern(object):
    """
    This doc is partially OUTDATED (!).
    
    Flexible context-based pattern matching in html/xml/text documents based on robust and simple syntax that matches all markup block, with multiple variables, at once.
    Pattern defines layout of a document: subsequent portions of text that must be present in given locations, 
    and possibly places where - if matched - substrings should be extracted for further analysis. 
    Patterns are written in a special custom language and converted to a regular expression during Pattern.__init__(). 
    Typically applied to HTML, for easy scraping of web content. Single Pattern can substitute a dozen of XPaths.
    If you need online interactive testing of regex-es for debugging, see http://gskinner.com/RegExr/

    How to write a pattern that matches a given html snippet:
     - In the snippet, replace all occurences of the values to be extracted with {NAME} or {NAME subpattern} or {*NAME...} (items of a list).
       Remember to put longfill '...' inside {} if you want to match strings containing tags; a shortfill '.' is the default when no expression is given inside {}.
     - Cut out unnecessary text replacing it with ~ (word fill) or . (short fill) or ... (long fill):
       - raw text between tags replace with . (matches all chars except <>)
       - rich text containing tags replace with ... (matches all chars)
       - tag contents inside <> replace with space or . or empty string (inside <>, all of them match all chars except <>, space matches on boundaries of tag attributes)
       - sequence of non-space chars replace with ~ (matches all except <> and spaces)
       Ensure that every variable {} is still surrounded by enough non-optional text (REFERENCE POINTS / GUARDS) to uniquely identify its location in the document.
     - Insert a space wherever 0+ spaces may occur in the document, except between > and < chars (neighboring tags) where spaces will be matched automatically. 
     - Mark non-obligatory parts of the pattern by surrounding them with []. If the part starts with . or  ..., include them inside [] if only possible
       (this is necessary even if the [] block is the 1st entity of the pattern).
       Remember to leave a CLOSING ANCHOR after []: a static text just after the closing bracket ], to avoid under-matching of the [] block (enforce maximum match).
     - If the Pattern works too slow, especially on negative examples (!), add more reference points or atomic grouping {>...}.
     
    Tips & tricks:
     - when extracting URLs or their portions, ALWAYS use either url() or url_unquote() converters, to provide proper unquoting of extracted strings (!)

    Pattern behavior:
     - leading and trailing whitespaces in the pattern are removed
     - pattern match must occur from the *beginning* of the text, but don't need to match the end
     - whitespace in the pattern matches:
       - outside tags: 1+ spaces if inside static text; 0+ spaces otherwise (e.g., on tag boundary)
       - inside tags: 0+ non-tag (<> excluded) characters, but 1st and last char being a space if present
     - {var} converted to (?P<var>.*) (a named group that will be available under the name 'var' after matching)
     - {var expr}, where expr is interpreted like any top-level expression 
     - {var~regex} or {~regex} converted to a group with 'regex' pasted as-is (plain regex pattern in standard notation)
     - {*...} and {+...} - resolves into repetition operator around inner expression; extracted value is a list of strings, also for all nested variables
     - {*+...} and {++...} - possessive quantifiers are allowed, see: www.regular-expressions.info/possessive.html
     - {>...} - atomic grouping is allowed, see: www.regular-expressions.info/atomic.html
     - [expression] converted to (...)? - optional expression with maximum (longest) possible matching
     - [] and {} can be nested
     - no charsets [abc] (square bracket reserved for "optional"), except for negative ones [^...] or inside regexes {~regex}, {var~regex}
     - ~ (tilde) converted into a wordfill [^<>\s]*? (lazy matching of a word, no tags and no spaces) 
     - after parsing, all variable names will be returned as lowercase, so in pattern and epilog() you can use uppercase names for better readability
    
    Practical guidelines for writing Patterns:
     * REFERENCE POINTS - characteristic and *always* present (non-optional) substrings in several different places of the pattern, 
        which help situate optional subpatterns relative to each other.
     * GUARDS around variables - to precisely position the variable; avoid *over-matching*; kill spaces!
     * CLOSING ANCHOR inside optional blocks - to force maximum match of the last included optional/variable; avoid *under-matching* 
     
    TAG MODE:
    .              [^<>]*   all except tags (entirely inside or entirely outside tags)
    ... or ..      ???      tag-aware "all": '<' and '>' can only go together (complete tags)
    space          \s+      - between 2 statics
    space          \s*      - between non-static elements (max 1 static)
    _              _|\s*
    <xxx>          ~= <xxx .>     <xxx\s*(\s[^>]*)?>       match tag name as given, and any attributes that exist; like <xxx.> with obligatory \s after xxx  
    <.xxx>         ~= <.xxx.>     <\w+\s[^>]*xxx[^>]*>     match substring anywhere inside attribute list
    <xxx.yyy.zzz>  == <xxx yyy zzz> == <xxx .yyy.zzz.>
    <xxx.../>      == <xxx.>...</xxx> | <xxx./>
    <xxx y z .../> == <xxx y z.>...</xxx>   -- ... means not .* but (.(?!<xxx(\s|/|>)))* -- not allowed to open the same tag 2nd time inside ...
    {VAR} -> {VAR.}         extract text up to the 1st tag
    {VAR...}                extract text with tags
    - spaces around < and >
     
    Pattern class uses an extended version of 're' module, the 'regex' - see http://pypi.python.org/pypi/regex

    >>> p = Pattern("{* ... {A}}{B}")
    >>> v = p.semantics.variables; A, B = v['A'], v['B']
    >>> A.longfill, A.repeated, B.longfill, B.repeated
    (False, True, False, False)
    >>> print p.regex.pattern
    (.*?(?P<A>[^<>]*?))*(?P<B>[^<>]*?)
    
    >>> p2 = Pattern("{> [ala]} ala")
    >>> print p2.regex.pattern
    (?>(ala)?)\\s*ala
    >>> print p2.match1("ala")
    None
    """
    __metaclass__ = MetaPattern         # responsible for executing unit tests defined at class level in subclasses
    MISSING = object()                  # token to indicate that true test outcome is unspecified
    
    url = None
    
    # defaults
    pattern   = None    # source code of the patter, will be compiled into regex and matched against document text in match()
    path      = None    # optional XPath string; if present, pattern will be matched only against node(s) selected by this path - document must be an xdoc or will be parsed as HTML
    regex     = None    # compiled regex object, generated from pattern using the exnhanced 're2' ('regex') module
    semantics = None    # Context.Data object with global information about the pattern tree, collected during semantic analysis
    variables = None    # list of names of variables present in the pattern, extracted from 'semantics'
    convert   = {}      # converters or types that the extracted values shall be casted onto (type(val), only not-None values!); dict;
                        # ... can contain multikeys: "key1 key2 key3":value ; keys may contain wildcard '*' to match many variable names, like "URL_*":url
                        # ... converter can be a Pattern object: it will be applied to extracted text and a sub-dict/obj will be returned instead
                        # ... it's more efficient to supply Pattern object than class, though, to avoid multiple compilation of the same pattern string
    extract   = {}      # stand-alone extractors: functions that take an entire document and return extracted value or object for a given item; dict
    html      = True    # shall match() perform HTML entity decoding and normalization of spaces in extracted items? (done before extractors/converters)
    tolower   = False   # shall all item names be converted to lowercase at the end of parsing?
    mapping   = {}      # mapping of item names, for easier integration with other parts of the application (currently unused!)
    dicttype  = ObjDict # what type of dictionary to return; ObjDict allows .xxx access to values in addition to standard ['xxx']
    model     = None    # class to be used as a wrapper for the dictionary of matched fields passed in kwargs: __init__(**items) 
    verbose   = False   # if True, Pattern.__init__ will print out debug information
    autotest  = True    # if True, unit tests (see below) will be executed automatically upon class declaration
    
    # optional unit tests defined in subclasses, named 'testN' and 'goalN', or just 'testN' (no ground truth, only print the output)
    # ... 

    # TODO: implement error detection: binary search for the smallest element of the pattern that causes it to break on a given input string

    def __init__(self, pattern = None, extract = None, convert = None, html = None, tolower = None):
        if self.verbose: print util.classname(self)
        
        if pattern is not None: self.pattern = pattern
        if self.pattern is None: self.pattern = self.__class__.__doc__          # subclasses can define patterns in pydocs, for convenience
        if extract is not None: self.extract = extract
        if convert is not None: self.convert = convert
        if html is not None: self.html = html
        if tolower is not None: self.tolower = tolower

        parser = parsing.WaxeyeParser(Tree)
        self.regex, self.semantics = parser.compile(self.pattern)               # compile source pattern into regex
        
        # decode compact notation of keys in 'convert': split multi-name keys, resolve wildcard keys
        self.convert = util.splitkeys(self.convert)
        self.variables = self.semantics.variables.keys()
        for name, conv in self.convert.items():
            if '*' not in name: continue
            pat = name.replace('*', '.*') + "$"
            for name2 in self.variables:
                if re.match(pat, name2): self.convert[name2] = conv
            del self.convert[name]
        
        if self.verbose: 
            print self.variables
            print self.regex.pattern
        
#    def __call__(self):
#        "Trying to instantiate an already instantiated Pattern subclass? Do nothing."
#        return self        
        
    def testAll(self, tests):
        """'tests' is a dictionary of unit tests, every test is a pair of items in dict named testK and goalK, where K is 1,2,3...
        testK is a string or other type of a document: input for match() method; 
        goalK is the expected result, as a dict, list of strings or string; in the latter cases every string has a form of "name value", 
        with name and value of the variable respectively. 
        """
        if not tests: return
        def asdict(goal):
            if not goal: return goal
            if isstring(goal): goal = [goal]
            # 'goal' is a list or tuple...
            res = {}
            for s in goal:
                if isdict(s):                   # 's' is a dictionary?
                    res.update(s)
                    continue
                split = s.split(None, 1)        # 's' is a string...
                if len(split) > 1: var,val = split[:2]
                else: var,val = split[0], None
                res[var] = val
            return res
        
        # decode input and output part of each test; execute the test 
        k = 0
        while True:
            k += 1
            sk = str(k)
            if 'test' + sk not in tests: break
            test = tests['test' + sk]
            goal = tests.get('goal' + sk, self.MISSING)
            if goal != self.MISSING and not isdict(goal): goal = asdict(goal)
            self.testSingle(test, goal, k)
        
    def testSingle(self, text, goal, testID = "?"):
        def issubdict(d1, d2):
            "Are all keys and values from d1 also present in d2? None values in d1 treated as no values at all"
            if not isdict(d1): return False
            for k,v in d1.iteritems():
                if k not in d2 or (v != None and v != d2[k]): return False
            return True
        
        # extraction
        out = self.match(text, testing = True)
        
        # is OK?
        if out == goal: return True
        
        print "%s.test%s," % (util.classname(self), testID),
        
        # if no goal provided, print all extracted data, in a form suitable for inclusion in the code: strings as list, other values as dict, on new line
        if goal is self.MISSING:
            print "output:"
            if out: 
                textItems = ["%s %s" % item for item in out.iteritems() if isstring(item[1])]
                objItems  = {key:val for key,val in out.iteritems() if not isstring(val)}
                lines = []
                if textItems: lines.append(" " + str(textItems)[1:-1])
                if objItems:  lines.append(" " + str(objItems))
                print ", \\\n".join(lines)
            else:
                print "", out
            return
        
        # not OK...
        print "incorrect result of the matching:"
        #print "- Document to search in:"
        #print text
        #print "- Match pattern, its regex and extractors/converters:"
        #print self.pattern
        #print self.regex.pattern
        #print self.extract, self.convert
        print "expected output:"
        util.printJson(goal)
        #print goal
        print "actual output:"
        util.printJson(out)
        #print out
        
        # which values are different?
        if out is None or goal is None: return False
        print "differences:"
        for k in util.unique(goal.keys() + out.keys()):
            if k in goal and k not in out: print " %s - missing field" % k
            elif k not in goal and k in out: print " %s - unexpected field" % k
            elif goal[k] != out[k]: print " %s - incorrect value" % k
        print
        
        return False

    def _matchRaw(self, doc, startPos = None):
        """Apply the regex and return a tuple: matched items as a dictionary + position in 'doc' where the match ends; no post-processing. 
        'startPos' is the offset from the beginning of 'doc' where to start searching for a match."""
        
        # first, try the standard 're' module to see if there is any match
#        match = self.regex.search(doc, startPos)
#        if match is None: return None
#        items = match.groupdict()
#        return items, match.end()
        
        # match exists, now match with the enhanced 'regex' module to correctly extract repeated groups
        if 1 and self.regex:
            if startPos is None:
                match = self.regex.match(doc)
            else:
                match = self.regex.search(doc, startPos)
            if match is None: return None, None
            #print "matched by regex:", match.groupdict()
            
            # extract variables (groups), possibly repeated - each value is a list, singleton in case of non-repeated groups
            items = self.dicttype()
            for key in match.groupdict().keys():        # TODO: use capturesdict() instead
                items[key] = match.captures(key)
    
            # flatten singleton lists of non-repeated variables AND of groups defined inside raw regexes {~regex} rather than via {VAR ...}
            var = self.semantics.variables
            for name, val in items.iteritems():
                if not var[name].repeated: items[name] = val[0] if val else None
                
        return items, match.end()
        
    def _convert(self, items, doc, model = None, baseurl = None, testing = False):
        "Can modify 'items' dictionary."
        def clean(val, isstring = None):
            if not isstring and islist(val): return [clean(s,True) for s in val]
            return decode_entities(merge_spaces(val))
        #def absolute(url, isstring = None):
        #    if baseurl is None: return url
        #    if not isstring and islist(url): return [absolute(u,True) for u in url]
        #    return urljoin(baseurl, url)
        
        def convert1(val, fun):
            if val is None: return None
            if fun == url: return url(val, baseurl)                                 # 'url' is the convertion function url() defined below
            if isinstance(fun, Pattern): return fun.match(val, baseurl = baseurl)   # 'fun' can be a Pattern instance
            return fun(val)
        def convertAll(vals, fun):
            return [convert1(v,fun) for v in vals]
        
        try:
            # match regex pattern
            if items is None: return None
            
            # clean extracted text, but only for variables which don't contain longfills "..." in their pattern; 
            # longfills match HTML tags, therefore simple cleaning can be incorrect, because entity decoding should be accompanied by tag stripping  
            if self.html:
                var = self.semantics.variables
                for name, val in items.iteritems():
                    if val and not var[name].longfill: items[name] = clean(val)
            
            # run standalone per-item extractors
            for name, fun in self.extract.iteritems():
                val = items.get(name)
                arg = val if val != None else doc
                items[name] = fun(arg)
            
            # perform type casting / convertions and absolutization of URLs
            for name, fun in self.convert.iteritems():
                val = items.get(name)
                if val is None: continue
                if islist(val): items[name] = convertAll(val, fun)
                else: items[name] = convert1(val, fun)
                
            # run epilog
            epilog = self.epilog(items, doc)
            if epilog is not None: items = epilog
            
            # postprocessing begins...
            if testing: return items
            
            # rename items
            if self.tolower: items = lowerkeys(items)
            
            # convert to object
            model = model or self.model
            if model: return model(**items)
            
            return items
        
        except Exception, ex:
            # append name of this class to the error message, for easier tracking down of the problem
            if hasattr(ex, 'args') and hasattr(ex, 'message') and istuple(ex.args) and isstring(ex.message) and ex.args[0] == ex.message:
                prefix = classname(self, False) + " > "
                ex.message = prefix + ex.message
                ex.args = (ex.message,) + ex.args[1:]
            raise
    
    def match(self, doc, path = None, model = None, baseurl = None, testing = False):
        """Matches the pattern against document 'doc'. On successful match returns self.dicttype (ObjDict by default) with extracted and converted values,
        possibly wrapped up in model() if not-None ('model' can be given here as function argument, or as a property of the object or class); 
        or matched string if the pattern doesn't contain any variables. None on failure.
        Typically 'doc' is a string, but it can also be any other type of object convertable into str() - this is useful if custom extractors are to be used
        that require another type of object. If 'baseurl' is given, all extracted URLs will be turned into absolute URLs based at 'baseurl'.
        'testing': True in unit tests, indicates that final processing (item names renaming, class wrapping) should be skipped and 'path' shall not be used."""
        path = path or self.path
        if path and not testing: 
            if isstring(doc): doc = xdoc(doc)
            doc = doc.node(path)
        if self.variables:
            items, _ = self._matchRaw(unicode(doc))
            return self._convert(items, doc, model, baseurl, testing)
        else:
            match = self.regex.match(unicode(doc))
            if match is None: return None
            return match.captures()[0]
    
    def match1(self, *args, **kwargs):
        """Shorthand for patterns that extract exactly 1 variable: returns *value* of this variable, as an atomic value rather than a dictionary. None when no matching found.
        If no variables are present, returns the string matched by entire pattern, like if the pattern were enclosed in {VAR ...}."""
        vals = self.match(*args, **kwargs)
        if isstring(vals) or vals is None: return vals
        if len(vals) != 1: raise Exception("Pattern.match1: incorrect pattern, %d variables extracted instead of 1." % len(vals), self.variables, vals)
        return vals.values()[0]
        
    
    def matchAll(self, doc, path = None, **kwargs):
        """Like match(), but finds all non-overlapping matches of the pattern in 'doc', not only 1, and returns as a list of result sets (dicts/objs), possibly empty.
        If path or self.path is set, matches pattern to all nodes selected by this path (possibly nested, depending on the path), ignoring results of unmatching nodes."""
        path = path or self.path
        if path: return self._matchAllXPath(doc, path, **kwargs)
        return self._matchAllRegex(doc, **kwargs)
        
    def _matchAllRegex(self, doc, **kwargs):
        data = []; pos = 0
        udoc = unicode(doc)
        while True:
            items, pos = self._matchRaw(udoc, pos)
            if items is None: return data
            data.append(self._convert(items, doc, **kwargs))
    
    def _matchAllXPath(self, doc, path, **kwargs):
        data = [] 
        for node in doc.nodes(path):
            items, _ = self._matchRaw(node)
            if items != None: data.append(self._convert(items, node, **kwargs))
        return data
            
    
    def epilog(self, items, doc):
        """Any special extraction/convertion/rewriting operations to be done at the end of parsing on the 'items' dictionary. Nothing by default. 
        When overriding, this method can modify 'items' in place (and return None) or return a new dictionary.
        This method CANNOT change output of the parser to None - returning None will be treated as indicator of no changes to current output."""
        pass

    
    def __add__(self, other): return PatternAdd(self, other)
    def __and__(self, other): return PatternAnd(self, other)        # This overloads "&" operator, not "and" !!! (PAT1 & PAT2) is a more strict version of (PAT1 + PAT2)
    def __or__(self, other):  return PatternOr(self, other)         # This overloads "|" operator, not "or" !!!
    

########################################################################################################################################################

def matchPattern(pat, doc, **kwargs):
    "For easy construction of 1-liners that match short custom pattern (string 'pat') to a given text. Watch out: the pattern is compiled from scratch on every call."
    return Pattern(pat).match(doc, **kwargs)

def matchPattern1(pat, doc, **kwargs):
    "Like matchPattern, but calls Pattern.match1 instead of Pattern.match. For patterns that extract single variable."
    return Pattern(pat).match1(doc, **kwargs)


########################################################################################################################################################
###
###  MultiPatterns
###

class MultiPattern(Pattern):
    def __init__(self, *patterns):
        self.patterns = patterns

class PatternSum(MultiPattern):
    """Base for 'and' and '+' operators. Combines results from several patterns into one dictionary. Further patterns in the list may override values from previous ones,
    but only if new values are non-empty."""
    def match(self, *args, **kwargs):
        items = self.dicttype()
        for pat in self.patterns:
            res = pat.match(*args, **kwargs)
            if res is None:
                if self.strict: 
                    self.failed = pat
                    return None
                continue
            for key, val in res.iteritems():
                if val in (None,[]) and key in items: continue           # don't override existing values with Nones or []
                items[key] = val
        return items        
        
class PatternAdd(PatternSum):
    "Matching never fails, even if some of the subpatterns don't match. If all patterns return None, empty dict is returned as a result."
    strict = False
    def __add__(self, other): return PatternAdd(*(self.patterns + (other,)))
    
class PatternAnd(PatternSum):
    "Matching fails with None if any of the subpatterns return None. In such case, the failing pattern can be found in self.failed. This overloads '&' operator, not 'and' !!!"
    strict = True
    failed = None               # in case of failure of the matching, the pattern from self.patterns that failed
    def __and__(self, other): return PatternAnd(*(self.patterns + (other,)))
    
class PatternOr(Pattern):
    "Alternative of several patterns returned by Pattern.__or__. Applies patterns one by one and returns the 1st not-None result. This overloads '|' operator, not 'or' !!!"
    def __or__(self, other): return PatternOr(*(self.patterns + (other,)))
    
    def match(self, *args, **kwargs):
        for pat in self.patterns:
            res = pat.match(*args, **kwargs)
            if res is not None: return res
        return None
        

########################################################################################################################################################
###
###  Standard converters
###

from nifty.text import html2text             # use html2text in Pattern.convert if you want extracted HTML code to be converted into raw text
from decimal import Decimal                 # for parsing floating-point numbers without rounding errors

def url(s, baseurl):
    """Turn the URL 's' into absolute URL anchored at 'baseurl'. Do NOT unquote! (If you need unquoting, do it manually afterwards or use url_unquote() instead). 
    When used in Pattern.convert, 'baseurl' will be supplied at parsing time by the match() method itself."""
    if baseurl is None: return s
    return urljoin(baseurl, s)

def url_unquote(s, baseurl = None):
    """Unquotes the URL 's'. Optionally can also perform absolutization like url(), but NOT when used in Pattern.convert where baseurl is set to None (unlike url()).
    Use ALWAYS when extracting portions of text (IDs, names) from href anchors, which won't be used as URLs themselves, but MUST be properly unquoted (!)."""
    s = urllib.unquote(s)
    if baseurl is None: return s
    return urljoin(baseurl, s)

def pdate(s):
    "Parse date string 's' and return as a datetime.date object (class date in module datetime). Try several different formats. None if no format works."
    def check(*formats):
        for f in formats:
            try: return datetime.strptime(s, f).date()
            except: pass
        return None
    
    if '-' in s: return check("%Y-%m-%d", "%Y-%m")                         # 2010-11-25, 2010-11
    if '.' in s: return check("%d.%m.%Y", "%d.%m.%y")                      # 25.11.2010, 25.11.10
    if '/' in s: return check("%m/%d/%Y", "%m/%d/%y")                      # 11/25/2010, 11/25/10 (US style: day in the middle, year at the end)
    if ' ' in s: return check("%B %d, %Y", "%B %Y")                        # November 25, 2010; November 2010
    return check("%Y", "%Y%m", "%Y%m%d")                                   # 2010; 201011; 20101125

def pdatetime(s):
    "Parse date+time string and return as a datetime.datetime object. Try several different formats. None if no format works."
    def check(*formats):
        for f in formats:
            try: return datetime.strptime(s, f)
            except: pass
        return None
    
    return check("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p")

# Caution: in functions below, the client must make sure beforehand that comma is really a thousands separator, not fractional part separator!
def pint(s):
    "Parse integer, but first remove spaces and commas used as thousands separators."
    s = s.replace(',', '').replace(' ', '')
    return int(s)
def pfloat(s):
    "Parse floating-point number, but first remove spaces and commas used as thousands separators."
    s = s.replace(',', '').replace(' ', '')
    return float(s)
def pdecimal(s):
    "Parse Decimal, but first remove spaces and commas used as thousands separators."
    s = s.replace(',', '').replace(' ', '')
    return Decimal(s)

def percent(s):
    "Parse percent string, like 32.1% or 32.1 (% sign is optional) into float and divide by 100 to make a real fraction."
    if '%' in s: s = s[:s.find('%')]
    return float(s) * 0.01

#def table(s):
#    "Parse code of an HTML table (enclosed in <table> tag) and turn into ??? representation."


########################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()
    
