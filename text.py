# -*- coding: utf-8 -*-
'''
Text representation, text processing, text mining. Regular expressions.
Basic routines for HTML processing (see also nifty.web module for more).

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import re, math
import HTMLParser
from collections import defaultdict
from array import array
from itertools import imap

if __name__ != "__main__":
    from .util import isstring, islist, bound, flatten, merge_spaces
    from . import util
else:
    from nifty.util import isstring, islist, bound, flatten, merge_spaces
    from nifty import util


#########################################################################################################################################################
###
###  Subclasses of <str> and <unicode> that add extra functionality for pattern matching
###

class xbasestring(basestring):
    """Extends standard string classes (str, unicode, basestring) with several convenient methods."""
    _std = basestring                   # override this in subclasses with specific string class: str/unicode
    def __new__(cls, s = ""):
        "You can call xbasestring(s) to convert standard string to xstr or xunicode, depending on the type of 's' - appropriate type is picked automatically."
        if isinstance(s, str): return xstr(s)
        elif isinstance(s, unicode): return xunicode(s)
        return basestring.__new__(cls, s)

    def after(self, sep):
        "Substring that follows the first occurrence of 'sep', or emtpy string if 'sep' doesn't exist."
        return self.__class__(self.partition(sep)[2])
        #parts = self.split(sep, 1)
        #if len(parts) < 2: return self
        #return self.__class__(parts[1])
    
    def before(self, sep):
        "Substring that precedes the first occurrence of 'sep', or original string if 'sep' doesn't exist."
        return self.__class__(self.partition(sep)[0])

    def sub(self, pat = re.compile(r'\s\s+'), repl = '', count = 0):
        "Replaces all (or up to 'count') occurences of pattern 'pat' with replacement string 'repl'. 'pat' is a regex (compiled or raw string)"
        if isinstance(pat, basestring):
            pat = re.compile(pat)
        return self.__class__(pat.sub(repl, self, count))

    def re(self, regex, asList = False):
        """If asList = False, returns the first substring that matches a given regex/group, or empty string.
        If asList = True, returns a list of all matches (can be empty).
        See extract_regex() for more details on what is extracted.
        """
        matches = extract_regex(regex, self)
        if asList:          return map(self.__class__, matches)
        elif len(matches):  return self.__class__(matches[0])
        else:               return self._empty

    def replace(self, old, new, count = -1):
        return self.__class__(self._std.replace(self,old,new,count))
    def split(self, sep = None, maxsplit = -1):
        return map(self.__class__, self._std.split(self,sep,maxsplit))

class xstr(xbasestring, str):
    _std = str
    def __new__(cls, s = ""): return str.__new__(cls, s)
class xunicode(xbasestring, unicode):
    _std = unicode
    def __new__(cls, s = ""): return unicode.__new__(cls, s)

xstr._empty = xstr("")
xunicode._empty = xunicode("")


#########################################################################################################################################################
###
###  Basic text processing: extraction, cleansing, restructuring etc.
###

# def merge_spaces() -- from 'util' module

def trim_text(text, mlen, pat = re.compile(r"\W+")):
    "Trim 'text' to maximum of 'len' characters, at word boundary (no partial words left). Do nothing if mlen < 0."
    if mlen >= len(text) or mlen < 0: return text
    cut = text[:(mlen+1)]
    splits = pat.findall(cut)
    if not splits: return ""        # only 1 word in whole 'cut'
    p = cut.rfind(splits[-1])      # p = position of the right-most split point
    return cut[:p]

#########################################################################################################################################################
###
###  HTML processing
###

def decode_entities(s, h = HTMLParser.HTMLParser()):
    "Turn HTML entities (&amp; &#169; ...) into characters. 's' string does NOT have to be a correct HTML, any piece of text can be decoded."
    return h.unescape(s)

def striptags(html, remove = [], allow = [], replace = '', ignorecase = True):
    """Regex-based stripping of HTML tags. Optional 'remove' is a list of tag names to remove. 
    Optional 'allow' is a list of tag names to preserve (remove all others). 
    At most one of remove/allow parameters can be non-empty. If both are empty, all tags are removed.
    
    >>> striptags("  <html><HTML><A><a><a/><a /><img></img><image> <form><script><style><!-- ala --><?xml ?><![CDATA[...]]></body>")        # spaces preserved
    '   '
    >>> striptags("< b></ body>")        # leading spaces inside tags are not allowed
    '< b></ body>'
    >>> striptags("<i>one</i><u>two</u>")
    'onetwo'
    >>> html = "<html><A> <a href='http://xyz.com/'>ala ma <i>kota</i></a><a /><img></img><image><form><!-- ala -->"
    >>> striptags(html, allow = 'a i u')
    "<A> <a href='http://xyz.com/'>ala ma <i>kota</i></a><a />"
    >>> striptags(html, remove = ['i', 'html', 'form', 'img'])
    "<A> <a href='http://xyz.com/'>ala ma kota</a><a /><image><!-- ala -->"
    >>> striptags("<a href = 'http://xyz.com/?q=3' param=xyz boolean> ala </a>", remove = ['a'])
    ' ala '
    """
    if remove:
        pat = regex.tags(remove)
    elif allow:
        pat = regex.tags_except(allow)
    else:
        pat = regex.tag
    #print pat
    text = re.sub(pat, replace, html, flags = re.IGNORECASE if ignorecase else 0)
    return text

def html2text(html, sub = '', parser = HTMLParser.HTMLParser()):
    "Simple regex-based converter. Strips out all HTML tags, decodes HTML entities, merges multiple spaces. No HTML parsing is performed, only regexes."
    if not html: return ''
    s = regex.tag_re.sub(sub, html)                     # strip html tags, replace with 'sub' (use sub=' ' to avoid concatenation of neighboring words)
    s = parser.unescape(s)                              # turn HTML entities (&amp; &#169; ...) into characters
    return merge_spaces(s)                              # merge multiple spaces, remove newlines and tabs

#########################################################################################################################################################
###
###  REGEX-es
###

class regex(object):
    """Common regex patterns. See also:
        - http://gskinner.com/RegExr/  -- "Community" tab, thousands of regexes
        - http://www.regular-expressions.info/examples.html
        - http://regexlib.com/
    """
    _B = r'\b%s\b'              # for patterns bounded by word boundaries (alphanumeric or underscore character preceded/followed by a character from outside this class)
    
    int   = r'[+-]?\d+'         # can include 0 as the first char (!), sometimes this is recognized as octal; don't include hexadecimal integers   @ReservedAssignment
    float = r'[+-]?((\.\d+)|(\d+\.\d*))([eE][+-]?\d+)?'             # floating-point number in any form recognized by Python, except NaN and Inf   @ReservedAssignment
    escaped_string = r'"(?:[^"\\]|\\.)*"' + r"|'(?:[^'\\]|\\.)*'"   # "string" or 'string', with escaped characters, like \" and \' or other
    
    ident = r"\b\w+\b"          # identifier, only a-zA-Z0-9_ characters (all alphanumeric, maybe more depending on locale), with word boundaries
    word  = r"\S+"              # word, a sequence of any non-whitespace characters; no boundaries (any length)
    text  = r"\S.+\S|\S"        # free text, possibly with spaces, only must begin and end with a non-whitespace character; no boundaries

    comment      = r"(\A|(?<=\s))#.+$"              # "# ..." until the end of a line; "#" is either the 1st char on the line, or preceded by a whitespace 
    html_comment = r"<!--([^\-]|-(?!->))*-->"       # <!-- .. -->, with no '-->' inside
    
    ISSN = issn = _B % r'\d{4}-\d{3}[\dXx]'
    IP   = ip   = _B % r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'                            # IP number
    
    email        = _B % r"[\w\-\._\+%]+@(?:[\w-]+\.)+[\w]{2,6}"                         # supports International Domains with Country codes
    email_nospam = _B % r"[\w\-\._\+%]+(?:@|\(at\)|\{at\})(?:[\w-]+\.)+[\w]{2,6}"       # recognizes obfuscated emails

    # HTML/XML tag detector, from: http://gskinner.com/RegExr/?2rj44
    # Supports: tags (name in group 1) with arguments (group 2); closing tags (name in group 4); self-closing tags ('/' in group 3); 
    #           XML-like headers <?...?> (also check group 3); comments <!--...--> (group 5); CDATA sections <![CDATA[...
    # Does NOT allow for spaces after "<".
    # See HTML5 reference: http://www.w3.org/TR/html-markup/syntax.html#syntax-start-tags
    tag = r"""<(?:([a-zA-Z\?][\w:\-]*)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?(\s*[\/\?]?)|\/([a-zA-Z][\w:\-]*)\s*|!--((?:[^\-]|-(?!->))*)--|!\[CDATA\[((?:[^\]]|\](?!\]>))*)\]\])>"""
    tag_re = re.compile(tag, re.IGNORECASE)

    @staticmethod
    def tags(names):
        "Returns a regex pattern matching only the tags with given names, both opening and closing ones; matched tag name is available in 1st (opening) or 4th (closing) group."
        pat = r"""<(?:(%s)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?(\s*[\/\?]?)|\/(%s)\s*)>"""
        if isstring(names) and ' ' in names: names = names.split()
        if islist(names): names = "|".join(names)
        return pat % (names, names)

    @staticmethod
    def tags_except(names, special = True):
        "Returns a regex pattern matching all tags _except_ the given names. If special=True (default), special tags are included: <!-- --> <? ?> <![CDATA"
        pat = r"""<(?:(?!%s)([a-zA-Z\?][\w:\-]*)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?(\s*[\/\?]?)|\/(?!%s)([a-zA-Z][\w:\-]*)\s*"""
        if special: pat += r"|!--((?:[^\-]|-(?!->))*)--|!\[CDATA\[((?:[^\]]|\](?!\]>))*)\]\]"
        pat += r")>"
        if isstring(names) and ' ' in names: names = names.split()
        if islist(names): names = "|".join(names)
        names = r"(?:%s)\b" % names              # must check for word boundary (\b) at the end of a tag name, to avoid prefix matching of other tags
        return pat % (names, names)

    @staticmethod
    def isISSN(s): return re.match(regex.issn + '$', s)


#########################################################################################################################################################

def regexEscape(s):
    """Escape special characters in 's' and encode non-printable characters so that the resulting string 
    can be included in regex pattern as a static string that matches occurences of 's'.
    The output of this function is prettier than produced by re.escape() - here we escape only true special chars,
    while re.escape() escapes ALL non-alphanumeric characters, so its output is long and rather ugly."""
    s = s.encode('unicode_escape')          # encode non-printable characters
    escape = r".^$*+?{}()[]|"               # no backslash \, it's handled in encode() above
    s = ''.join('\\' + c if c in escape else c for c in s)
    return s

def alternative(strings, escape = True, compile = False, flags = 0):                     #@ReservedAssignment
    "Regex (not compiled by default) that matches the logical alternative of given strings. Strings are escaped by default."
    if isstring(strings): strings = strings.split()
    if escape: strings = filter(re.escape, strings)
    pat = "|".join(strings)
    if not compile: return pat
    return re.compile(pat, flags = flags)
        
def extract_regex(regex, text):
    """Extract a list of strings from the given text using the following rules:
    * if the regex contains a named group called "extract" that will be returned
    * if the regex contains multiple numbered groups, all those will be returned (flattened)
    * if the regex doesn't contain any group the entire regex matching is returned
    Adapted from Scrapy code.
    """
    if isinstance(regex, basestring):
        regex = re.compile(regex)
    try:
        strings = [regex.search(text).group('extract')]   # named group
    except:
        strings = regex.findall(text)    # full regex or numbered groups
    return flatten(strings)

def findEmails(text, exclude = set(), pat = re.compile(regex.email_nospam, re.IGNORECASE), patAT = re.compile(r"\(at\)|\{at\}", re.IGNORECASE), unique = True):
    """Extracts all e-mails in 'text', including obfuscated ones, and returns un-obfuscated and uniqified (if unique=True, case-SENsitive uniqueness). 
    Emails present on 'exclude' list/set are removed from the result (but only if unique=True!)."""
    def clean(email): return patAT.sub("@", email)
    emails = pat.findall(text)
    emails = map(clean, emails)
    if not unique: return emails
    emails = set(emails) - set(exclude)
    return list(emails)


#########################################################################################################################################################
###
###  Advanced text processing
###

def extract_text(data, stoplist, trim = -1, sep = '\n'):
    '''Recursively extract all text from a (possibly nested) collection 'data' and concatenate into single string. 
       'data' can be composed of nested dictionaries and lists. Keys in 'stoplist' are excluded. 
       Numbers and other types are ignored. Strings are converted from HTML to plain text before concatenation.
       At the top level, extracted strings are separated by newlines, at lower levels - by tabs. No separator at the end of the last line.  
    '''
    if isinstance(data, (str, unicode)):
        return trim_text(html2text(data, ' '), trim).encode('ascii','ignore')
    
    text = ""
    if isinstance(data, dict):
        for k, v in data.iteritems():
            if k in stoplist: continue
            s = extract_text(v, stoplist, trim, '\t')
            if s: text += s + sep
            
    elif isinstance(data, list):
        for v in data:
            s = extract_text(v, stoplist, trim, '\t')
            if s: text += s + sep
    
    if text: return text[:-1]       # remove extra 'sep' at the end 
    return ""

def substitute_text(text, subst):
    """Given a list of substitutions - pairs (orig,replace) - perform all of them on 'text'"""
    for s1,s2 in subst:
        text = text.replace(s1,s2)
    return text


###  STOPWORDS - lists of common words: short, medium and long one

STOP1 = set("I a about an are as at be by com for from how in is it of on or that the this to was what when where who will with the www".split())

STOP2 = set("a about above after again against all am an and any are aren't as at be because been before being below between both but by " \
            "can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further " \
            "had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's " \
            "i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not " \
            "of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such " \
            "than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too " \
            "under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while " \
            "who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves" \
            .split()) | STOP1

STOP3 = set("a about above above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at " \
           "back be became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by " \
           "call can cannot cant co con could couldnt cry de describe detail do done down due during " \
           "each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except " \
           "few fifteen fifty fill find fire first five for former formerly forty found four from front full further " \
           "get give go had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred " \
           "ie if in inc indeed interest into is it its itself keep last latter latterly least less ltd " \
           "made many may me meanwhile might mill mine more moreover most mostly move much must my myself " \
           "name namely neither never nevertheless next nine no nobody none noone nor not nothing now nowhere " \
           "of off often on once one only onto or other others otherwise our ours ourselves out over own " \
           "part per perhaps please put rather re same see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system " \
           "take ten than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two " \
           "un under until up upon us very via was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would " \
           "yet you your yours yourself yourselves " \
           .split()) | STOP2

STOP = STOP2

def stopwords(words, *stopLists, **params):
    """From a given list of words or a string, filter out: stopwords, numbers (if 'numbers'=True, default; both integers and floats), 
    single letters and characters (if 'singles'=True, default); using all the *stopLists lists/sets combined, or STOP if no list provided.
    Comparison is done in lowercase, but original case is left in the result.
    >>> stopwords("This is an example string.")
    'example string'
    >>> stopwords(u"Echte of neppatiënten")         # unicode characters recognized correctly inside words
    u'Echte neppati\\xc3 nten'
    """
    numbers = params.get('numbers', True)
    singles = params.get('singles', True)
    asstring = params.get('asstring', None)
    
    if not stopLists:
        stop = STOP
    else:
        stop = set()
        for s in stopLists:
            stop |= set(s.split()) if isstring(s) else set(s) if islist(s) else s
    
    if isstring(words):
        if asstring is None: asstring = True
        words = re.split(r'\W+', words, flags = re.UNICODE)
            
    res = []
    for w in words:
        if singles and len(w) < 2: continue
        if numbers and w.isdigit(): continue                    # todo: replace isdigit() with a regex that handles all floats, too
        if w.lower() in stop: continue
        res.append(w)
    
    if asstring: return ' '.join(res)                           # get back to a concatenated text 
    return res

def keyPattern(keys):
    "Regex pattern to discover how many times keywords 'keys' occur in a given text"
    # transform string of 'keys' into regex pattern - alternative "|" of patterns of the form:
    # "\bKEY" - for long keys - written in lowercase (prefix match)
    # "\bKEY\b" - for short keys - written in uppercase (exact match)
    # In the patter, all keywords are inserted in lowercase,
    # uppercase is only an indicator which key to be treated as long or short. 
    keyList = [r"\b" + k.lower() + (r"\b" if k.isupper() else '') for k in keys.split()]
    return '|'.join(keyList)
    

#########################################################################################################################################################
###
###  ALGORITHMS
###

def levenshtein(a, b, casecost = 1, spacecost = 1, totals = False):
    """Calculates the Levenshtein edit distance between strings a and b. 'casecost' is the cost of replacement when only the case is changed, not the actual character.
    If totals=True, returns total character costs of both strings, in addition to the distance value, as a triple (dist, cost_a, cost_b).
    >>> levenshtein("Ala", "OLa")
    2
    >>> levenshtein("Ala", "OLa", 0.5)
    1.5
    >>> round(levenshtein(" a ala", "aala ", 1, 0.1), 5)
    0.3
    >>> levenshtein(" a ala Ola ", "aalaola ", 1, 2)
    7
    """
    #_a, _b = a,b
    reorder = False
    n, m = len(a), len(b)
    if n < m:                                   # ensure that n >= m ('a' is longer), to speed up calculations (short outer loop); but mem usage is O(max(n,m))
        a,b = b,a
        n,m = m,n
        reorder = True
    
    charcost = lambda c: spacecost if c.isspace() else 1
    isint = util.isint(casecost) and util.isint(spacecost)
    typecode = 'l' if isint else 'd'
    zero = array(typecode, [0])
    zeron = zero * n
    
    try:
        alow = a.lower()
        blow = b.lower()
        acost = array(typecode, imap(charcost, a))
        bcost = array(typecode, imap(charcost, b))
        
        #current = range(n+1)                    
        current = zero + acost                  # initially, current[j] is the total cost of letters in a[:j], for j = 0,1,...,n
        for j in range(2,n+1):
            current[j] += current[j-1]          # 'current' must hold cumulative a[:j] costs rather than single-letter costs
        #print current
        
        # loop invariant: current[j] is the cost of transforming a[:j] into b[:i] 
        for i in range(1,m+1):                  # loop over characters of 'b'
            cur_b, cur_bcost = b[i-1], bcost[i-1]
            previous = current
            current = array(typecode, [previous[0] + cur_bcost]) + zeron
            for j in range(1,n+1):              # loop over characters of 'a'
                add = previous[j] + cur_bcost                           # cost of adding extra character in 'b'
                delete = current[j-1] + acost[j-1]                      # cost of deleting extra character from 'a'
                change = previous[j-1]
                if a[j-1] != cur_b:
                    if alow[j-1] == blow[i-1]: change += casecost       # only the case is different?
                    else: change += max(cur_bcost, acost[j-1])
                current[j] = min(add, delete, change)
    
    except UnicodeWarning:    
        print "unicode error in levenshtein(%s, %s)" % (repr(a), repr(b))
        raise
    
    if totals: 
        if reorder: return current[n], sum(bcost), sum(acost)
        else: return current[n], sum(acost), sum(bcost)
    return current[n]

def levendist(a, b, casecost = 0.5, spacecost = 0.5):
    """Like levenshtein(), but normalizes the distance value into [0,1] range: 0.0 iff a==b, 1.0 for total dissimilarity. 
    Warning: distance value can get out of [0,1] range if casecost or spacecost is outside this range!
    >>> levendist("Alama", "ALA")
    0.6
    """
    if a == b: return 0.0
    if not a or not b: return 1.0
    dist, cost_a, cost_b = levenshtein(a, b, casecost, spacecost, totals = True)
    maxcost = max(len(a), len(b)) #max(cost_a, cost_b)
    #print dist, maxcost
    assert 0 <= dist <= maxcost or not (0 <= casecost <= 1) or not (0 <= spacecost <= 1)        # from properties of Levenshtein algorithm
    return dist / float(maxcost)

def levenscore(a, b, casecost = 0.5, spacecost = 0.5):
    """Like levenshtein(), but normalizes the distance value and converts into a score in [0,1]: the more similar the strings, the higher the score, 1.0 iff a==b.
    Warning: score value can get out of [0,1] range if casecost or spacecost is outside this range!
    >>> levenscore("Alama", "ALA")
    0.4
    >>> levenscore("Control of Insect Ve", "Osamu Kanamori: Phil")
    0.025000000000000022
    """
    return 1.0 - levendist(a, b, casecost, spacecost)


def ngrams(text, N = 4):
    '''
    Calculates list of all N-grams (character-wise) of a string or a list of strings 'text'.
    In case of a list, each string is extended with trailing and leading underlines ("___"),
    so that partial n-grams on the margins are also included, padded with "_".
    '''
    if type(text) is list:
        fill = "_" * (N-1)
        text = fill.join(text)
        if text:    # is non-empty -> add fill at the beginning and at the end
            text = fill + text + fill
    ngrams = []

    for i in xrange(len(text) - N + 1):
        ngrams.append(text[i:i+N])

    return ngrams


# TODO: this function should be turned into more general-purpose one
def tokenize(text, stop = "group univ university college school education industry employees area state"):
    '''Turn the document to space-separated string of lowercase words. Remove all common words, punctuactions, numbers and other words 
       that might be meaningless or problematic in semantic analysis. 
    '''
    # normalize (remove) special characters in common abbreviations and multi-part words,
    # so that they're preserved through tokenization and filtering
    abbr = [['ph.d.','phd'], ['r&d','rd'], ['v.p.','vp'], ['start-up','startup'],
            ['bachelor of sciences','bsc'], ['bachelors of science','bsc'], ['bachelor of science','bsc'],
            ['master of sciences','msc'], ['masters of science','msc'], ['master of science','msc']]
    text = substitute_text(text.lower(), abbr)
    
    # glue together word parts connected by dashes
    text = re.sub(r"\b-\b", "", text) 
    
    # tokenize text into words; filter out: stopwords, single letters, numbers
    words = re.split(r'\W+', text)
    words = stopwords(words, stop, STOP3)
    return ' '.join(words)      # get back to concatenated text 


class WordsModel(object):
    '''
    A bag-of-words model with TF-IDF weights for all words in the corpus.
    See Matcher.sameText() for more details.
    '''
    def __init__(self, modelData):
        "'modelData' is the same as returned from WordsModelUnderTraining.getModelData()"
        self.nDocs, self.weights = modelData
        
        # new words, not present in the corpus at all,
        # will get equal weight as words occuring in a single doc
        self.freqMissing = math.log(self.nDocs)
        
    def get(self, word):
        return self.weights.get(word, self.freqMissing)    
    
class WordsModelUnderTraining(WordsModel):
    '''
    A model that's under training now. It returns None as get() values,
    but monitors all incoming data and merges into the model,
    which can be saved later on and then loaded into regular WordsModel.
    '''
    def __init__(self):
        self.nDocs = 0
        self.counts = defaultdict(int)
        
    def get(self, word):
        # weights not yet calculated...
        return None
    
    def addDoc(self, words):
        for w in words:
            self.counts[w] += 1
        self.nDocs += 1
        
    def getModelData(self):
        # transform counts into real-valued weights
        weights = {}
        #delta = self.nDocs / 100.0     # with this delta, weights will range between 1.0 and <100.0
        for word, count in self.counts.iteritems():
            # for efficiency ignore rare terms (lots of them! ~60% in long texts) 
            # - they'll have a default weight assigned anyway
            if count > 1:
                weights[word] = math.log(self.nDocs / float(count))
        return [self.nDocs, weights]


    # TODO: this "method" (without class) should be turned into regular function
    def sameText(self, doc1, doc2, modelName, cleansing = False, scale = 0.1, default = 0.0):
        '''
        Compares two documents (strings) using Bag-of-Words model,
        where each document is represented as a vector of word frequencies.
        Employs TD*IDF metric for weighing frequencies of words
        to assign higher weights to words which are more specific
        (occur less common in the corpus).
        Returns score in [0,1], rescaled so that the 'scale' value is mapped to 1.0.
        'modelName' is the name of model to be used - there can be
        separate models for different parts of profiles.
        Relative length of both docs doesn't matter, because
        frequency vectors are normalized before comparison. 
        
        For more details see:
         - http://en.wikipedia.org/wiki/Vector_space_model
         - http://en.wikipedia.org/wiki/Tf-idf
        '''
        def count(doc):
            freq = {}
            if type(doc) is not list:
                doc = doc.strip().split()
            for w in doc:
                freq[w] = freq.get(w,0) + 1
            return freq
        
        def vectorize(doc, model):
            vec = count(doc)
            if self.isTraining:
                model.addDoc(vec.keys())
                return vec  # no need for further processing of 'vec' when training
            for word in vec.iterkeys():
                vec[word] *= model.get(word)
            return vec
        
        def cosine(v1, v2):
            "Both v1 and v2 are variable-length dictionaries of tf-idf frequencies"
            dot = norm1 = norm2 = 0.0
            for word in set(v1.keys() + v2.keys()):
                f1 = v1.get(word, 0.0)
                f2 = v2.get(word, 0.0)
                dot += f1 * f2
                norm1 += f1 * f1
                norm2 += f2 * f2
            norm = math.sqrt(norm1) * math.sqrt(norm2)
            if norm <= 0.0: return 0.0
            return dot / norm
        
        if doc1 is list: doc1 = " ".join(doc1)
        if doc2 is list: doc2 = " ".join(doc2)
        if cleansing:
            doc1 = tokenize(doc1, "")
            doc2 = tokenize(doc2, "")

        if (not self.isTraining) and not (doc1 and doc2):
            return default
        # empty documents are EXcluded from training and not counted in corpus size;
        # however, vectorization must be performed on non-empty document 
        # even if another document is empty, to allow for training
        model = self.getModel(modelName)
        if doc1: v1 = vectorize(doc1, model)
        if doc2: v2 = vectorize(doc2, model)
        if not (doc1 and doc2): return default
        longEnough = min(len(v1)*2,len(v2),15) / 15.0   # dumps the score if length of vectors - especially v2 - is too small
                                                        # v1 is treated less strict, because this is the user who requests a match!
        
        return bound(cosine(v1, v2) / scale * longEnough)


#########################################################################################################################################################
###
###  DRAFT...
###

"""
Different languages commonly encountered in web applications:
- plain: ASCII or Unicode text, with no internal structure, all characters allowed and interpreted as-is
- HTML - any HTML snippet, with tags (omitted during plain text extraction) and entities (replaced with corresponding characters in plaintext()); must be sanitized before inclusion in HTML page
- safe-HTML - tags (all or only unsafe ones) were stripped out, but all entities are preserved; can be directly embedded in HTML without decoding/encoding
- SQL value - any expression that can be safely used, without additional escaping, in SQL query in any place where a value is expected; SQL-escaped value 
- SQL query
- Solr query, Solr phrase
- wiki text
- URL
- cookie plain - plaintext with limited set of characters: any printable ascii character (! through ~, unicode \u0021 through \u007E) excluding , and ; and excluding whitespace 
- cookie encoded - like "cookie plain", but a special character is introduced to encode other (disallowed) characters
"""

class Text(unicode):
    """A string (unicode, str) that keeps information about language of the text (HTML, SQL, plain text, ...)
    that should be used by the client when interpreting the text.  
    Allows for easy (automatic) conversions - encoding/decoding - between different language representations.
    Conversions can be lossy (!), for example, HTML converted to rawtext loses style information (tags).
    Language of a text is a subjective matter, there are no objective reasons why a given string should be interpreted in this or that language.  
    Can be used for implicit sanitization and output encoding of strings in web applications.
    This class is an (abstract) base class for a number of subclasses, one for each language."""
    
    language = None
    
    def __init__(self, rich = None, raw = None):
        """Exactly one of input texts: 'rich', 'raw' must be provided. 'rich' is already a richtext in encoded form of the target language.
        'raw' will undergo encoding here. Use encode() method for explicit conversion.""" 
        if rich is None: rich = self.encode(raw)
        unicode.__init__(self, rich)
    
    @classmethod
    def encode(cls, rawtext):
        "Convert 'rawtext' into a given language and return as an instance of Text subclass."
        raise NotImplemented()
    
    def asRaw(self):
        "This text converted to a rawtext."
        raise NotImplemented()

    # more custom conversions, to specific target languages, can be implemented in subclasses, for example:
    def asHtml(self): return HTMLText(raw = self.asRaw())
    def asUrl(self): return URLText(raw = self)


class RawText(Text):
    language = 'raw'
    __init__ = unicode.__init__
    # def html(self): encodes special symbols into named entities 
    
class HTMLText(Text):
    language = 'HTML'
    
class SafeHTMLText(HTMLText):
    language = 'safe-HTML'
    
    def __init__(self, html, safetags = [], safeattrs = []):
        pass

    def html(self): return self
    
class InTagText(Text):
    """String written inside an HTML tag, as a variable value, e.g., <meta name="author" content="Luck & Luke"> 
    - inner quotes should be turned into HTML entities: &quot; (&#34;) and &apos; (&#39;)
    """


class URLText(Text): 
    # urllib.quote, urllib.unquote
    pass

class MediaWikiText(Text):
    pass
    # def html(self): converts wiki styles into HTML tags

class CookiePlainText(Text): pass
class CookieEncodedText(Text): pass

class FilenameText(Text): pass


###   PARSERS of simple file formats (DRAFT)

class DictionaryFile(object):
    """Customizable parser of a list of key-value pairs stored in a simple human-readable text format. 
    Each key on a separate line. Key is a single word (identifier). 
    Value is any string (can be empty), stripped of leading and trailing whitespaces. 
    File can include comments ("#") and empty lines.
    Values can include Unicode.
    No multi-line values.
    For example:
    
        # Database settings:
        HOST      localhost
        PORT      4312
        PASSWORD  7 lk 7(*^&$#
    
    Parser can be highly parameterized: characters allowed in keys; multi- or single-line values; comment prefix; 
    inline comments allowed or not; free-text keys (then assignment operator is used to delimit key and value)...
    Default file extension: .dict 
    """
    key = regex.ident
    value = regex.text
    comment = regex.comment
    assignment = ""
    
    inlineComments = True
    multiline = False
    
    def _parseSingle(self):
        "Parses text with single-line values only"
        linePattern = "(%s)\s*(%s)\s*(%s)?\s*(%s)?" % (self.key, self.assignment, self.value, self.comment)
        self.lineR = re.compile(linePattern)
        self.commentR = re.compile(self.comment)
        for line in self.text:
            self._parseLine(line)
    def _parseMulti(self):
        "Parses text with multi-line values"
    def _parseLine(self, line):
        if self.commentR.match(line): return        # full-line comment?
        # split: key - [value] - [comment]
        pass
    
class ConfigFile(DictionaryFile):
    "Special case of a key-value pairs file: only single-line values allowed. File extension: .conf"    
class Snippets():
    """Parser of key-value pairs, for instance language translations for a website. Values can be multi-line.
    """


#########################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()


