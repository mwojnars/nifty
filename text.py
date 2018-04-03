# -*- coding: utf-8 -*-
'''
Text representation, text processing, text mining. Regular expressions.
Basic routines for HTML processing (see also nifty.web module for more).

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import re, math, numpy as np
import HTMLParser
from collections import defaultdict
from array import array
from itertools import imap, izip, groupby
from copy import copy
from numba import jit

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

    _std = basestring           # the standard class (str/unicode) that should be used to access standard string methods; override in subclasses

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
        "Replaces all (or up to 'count') occurences of pattern 'pat' with replacement string 'repl'. 'pat' is a regex pattern (compiled or not)"
        if isinstance(pat, basestring):
            pat = re.compile(pat)
        return self.__class__(pat.sub(repl, self, count))

    def re(self, regex, multi = False):
        """If multi = False, returns the first substring that matches a given regex/group, or empty string.
        If multi = True, returns a list of all matches (can be empty).
        'regex' is either a compiled regular expression or a string pattern.
        See extract_regex() for more details on what is extracted.
        """
        matches = extract_regex(regex, self)
        if multi:           return map(self.__class__, matches)
        elif len(matches):  return self.__class__(matches[0])
        else:               return self.__class__("")

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

# xstr._empty = xstr("")
# xunicode._empty = xunicode("")


#########################################################################################################################################################
###
###  Basic text processing: extraction, cleansing, restructuring etc.
###

# def merge_spaces() -- from 'util' module

def trim_text(text, mlen, pat = re.compile(r"\W+"), ellipsis = ""):
    "Trim 'text' to maximum of 'len' characters, at word boundary (no partial words left). Do nothing if mlen < 0."
    if text is None: return None
    elif mlen >= len(text) or mlen < len(ellipsis): return text
    cut = text[:(mlen-len(ellipsis)+1)]
    splits = pat.findall(cut)
    if not splits: return ellipsis        # only 1 word in whole 'cut'
    p = cut.rfind(splits[-1])      # p = position of the right-most split point
    return cut[:p] + ellipsis

#########################################################################################################################################################
###
###  HTML processing
###

def html_unescape(s, h = HTMLParser.HTMLParser()):
    "Turn HTML entities (&amp; &#169; ...) into characters. 's' string does NOT have to be a correct HTML, any piece of text can be decoded."
    return h.unescape(s)
decode_entities = html_unescape

def html_escape(text):
    """Escape HTML/XML special characters (& < >) in a string that's to be embedded in a text part of an HTML/XML document.
    For escaping attribute values use html_attr_escape() - attributes need a different set of special characters to be escaped.
    """
    return text.replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;')

def html_attr_escape(text):
    """Escape special characters (& ' ") in a string that's to be used as a (quoted!) attribute value inside an HTML/XML tag.
    Don't use for un-quoted values, where escaping should be much more extensive!
    """
    return text.replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;').replace("'", '&#39;').replace('"', '&#34;')

def html2text(html, sub = ''):
    "Simple regex-based converter. Strips out all HTML tags, decodes HTML entities, merges multiple spaces. No HTML parsing is performed, only regexes."
    if not html: return ''
    s = regex.tag_re.sub(sub, html)         # strip html tags, replace with 'sub' (use sub=' ' to avoid concatenation of neighboring words)
    s = html_unescape(s)                    # turn HTML entities (&amp; &#169; ...) into characters
    return merge_spaces(s)                  # merge multiple spaces, remove newlines and tabs

def striptags(html, remove = [], allow = [], replace = '', ignorecase = True):
    r"""Regex-based stripping of HTML tags. Optional 'remove' is a list of tag names to remove.
    Optional 'allow' is a list of tag names to preserve (remove all others).
    At most one of remove/allow parameters can be non-empty. If both are empty, all tags are removed.
    remove/allow are given either as a list of strings or a single string with space-separated tag names.
    By default, tag names are matched case-insensitive.
    Watch out: striptags() removes tags only, not elements - body of the tags being removed is preserved!
    Use stripelem() for simple regex-based removal of arbitrary elements, including their body.

    >>> striptags("<i>one</i> <u>two</u>")
    'one two'
    >>> striptags("  <html><HTML><A><a><a/><a /><img></img><image> <form><script><style><!-- ala --><?xml ?><![CDATA[...]]></body>")    # spaces preserved
    '   '
    >>> striptags("< b></ body>")              # leading spaces inside tags not allowed
    '< b></ body>'
    >>> html = r"<html><A> <a href=\n 'http://xyz.com/'>ala ma <i>kota</i></a><a /><img>\n</img><image><form><!-- \n ala -->"
    >>> striptags(html, allow = 'a i u')
    "<A> <a href=\\n 'http://xyz.com/'>ala ma <i>kota</i></a><a />\\n"
    >>> striptags(html, remove = ['i', 'html', 'form', 'img'])
    "<A> <a href=\\n 'http://xyz.com/'>ala ma kota</a><a />\\n<image><!-- \\n ala -->"
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
    return re.sub(pat, replace, html, flags = re.IGNORECASE if ignorecase else 0)

def stripelem(html, remove = [], replace = '', ignorecase = True):
    r"""Like striptags(), but removes entire elements, including their body: <X>...</X>, not only tags <X> and </X>.
    Elements are detected using simple regex matching, without actual markup parsing!
    This can behave incorrectly in more complex cases: with nested or unclosed elements, HTML comments, script/style elements etc.
    Self-closing or unclosed elements are NOT removed. By default, tag names are matched case-insensitive.

    >>> stripelem("<i>one</i> <u>two</u>")
    ' '
    >>> stripelem(r"  <html></HTML> outside\n <A>inside\n</a> <a/><a /><img src=''></img><image> <form><!--\nala--><?xml ?><![CDATA[...]]></body>")
    '   outside\\n  <a/><a /><image> <form><!--\\nala--><?xml ?><![CDATA[...]]></body>'
    >>> stripelem("< b></ b>")                 # leading spaces inside tags not allowed
    '< b></ b>'
    >>> html = r"<A> <a href=\n 'http://xyz.com/'>ala ma <i>kota</i></a> <img src=''>\n</img> <I>iii</I>"
    >>> stripelem(html, remove = 'a i u')
    " <img src=''>\\n</img> "
    >>> stripelem(html, remove = 'i img')
    "<A> <a href=\\n 'http://xyz.com/'>ala ma </a>  "
    """
    pat = regex.tags_pair(remove)
    flags = re.DOTALL
    if ignorecase: flags |= re.IGNORECASE
    return re.sub(pat, replace, html, flags = flags)


#########################################################################################################################################################
###
###  REGEX-es
###

class regex(object):
    """
    Common regex patterns. See also:
    - http://gskinner.com/RegExr/  -- "Community" tab, thousands of regexes
    - http://www.regular-expressions.info/examples.html
    - http://regexlib.com/

    >>> re.compile(regex.email_nospam, re.IGNORECASE).findall("ala(AT)kot.ac.uk")
    ['ala(AT)kot.ac.uk']
    >>> text = "urls: http://regexlib.com/REDetails.aspx?regexp_id=146, http://regexlib.com/(yes!), https://www.google.com. "
    >>> [m.group() for m in re.compile(regex.url).finditer(text)]
    ['http://regexlib.com/REDetails.aspx?regexp_id=146', 'http://regexlib.com/', 'https://www.google.com']
    """

    _B = r'\b%s\b'              # for patterns bounded by word boundaries (alphanumeric or underscore character preceded/followed by a character from outside this class)

    int   = r'[+-]?\d+'         # can include 0 as the first char (!), sometimes this is recognized as octal; don't include hexadecimal integers   @ReservedAssignment
    float = r'[+-]?((\.\d+)|(\d+(\.\d*)?))([eE][+-]?\d+)?'          # floating-point number in any form recognized by Python, except NaN and Inf   @ReservedAssignment
    escaped_string = r'"(?:[^"\\]|\\.)*"' + r"|'(?:[^'\\]|\\.)*'"   # "string" or 'string', with escaped characters, like \" and \' or other

    ident = r"\b\w+\b"          # identifier, only a-zA-Z0-9_ characters (all alphanumeric, maybe more depending on locale), with word boundaries
    word  = r"\S+"              # word, a sequence of any non-whitespace characters; no boundaries (any length)
    text  = r"\S.+\S|\S"        # free text, possibly with spaces, only must begin and end with a non-whitespace character; no boundaries

    comment      = r"(\A|(?<=\s))#.+$"              # "# ..." until the end of a line; "#" is either the 1st char on the line, or preceded by a whitespace
    html_comment = r"<!--([^\-]|-(?!->))*-->"       # <!-- .. -->, with no '-->' inside

    ISSN = issn = _B % r'\d{4}-\d{3}[\dXx]'
    IP   = ip   = _B % r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'                            # IP number

    email        = _B % r"[\w\-\._\+%]+@(?:[\w-]+\.)+[a-zA-Z]{2,3}"                     # only standard 2-3-letter top-level domains, NO generic domains: .academic .audio. .church ...
    email_nospam = _B % r"[\w\-\._\+%]+(?:@|\(at\)|\{at\})(?:[\w-]+\.)+[a-zA-Z]{2,3}"   # recognizes obfuscated emails: with (at) or {at} instead of @

    # a decent URL with a leading protocol and withOUT trailing dot/comma; after http://regexlib.com/REDetails.aspx?regexp_id=146
    url_protocol = r"http|https|ftp"
    url_domain = r"[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}"
    url_path = r"([a-zA-Z0-9\-\._\?\,\'/\\\+&%\$#\=~:;])*(?<![\.\,])"
    url = r"\b(%s)\://%s(:[a-zA-Z0-9]*)?(/%s)?" % (url_protocol, url_domain, url_path)

    # HTML/XML tag detector, from: http://gskinner.com/RegExr/?2rj44
    # Detects: all opening tags (name in group 1) with arguments (group 2); closing tags (name in group 4); self-closing tags ('/' in group 3);
    #          XML-like headers <?...?> (also check group 3); comments <!--...--> (group 5); CDATA sections <![CDATA[...
    # Does NOT allow for spaces after "<".
    # See HTML5 reference: http://www.w3.org/TR/html-markup/syntax.html#syntax-start-tags
    tag = r"""<(?:([a-zA-Z\?][\w:\-]*)(?:\s+[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*(\s*[\/\?]?)|\/([a-zA-Z][\w:\-]*)\s*|!--((?:[^\-]|-(?!->))*)--|!\[CDATA\[((?:[^\]]|\](?!\]>))*)\]\])>"""
    tag_re = re.compile(tag, re.IGNORECASE)

    @staticmethod
    def tags(names):
        """Returns a regex pattern matching only the tags with given names, both opening and closing ones.
        The matched tag name is available in 1st (opening) or 4th (closing) group.
        """
        pat = r"""<(?:(%s)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?(\s*[\/\?]?)|\/(%s)\s*)>"""
        if isstring(names) and ' ' in names: names = names.split()
        if islist(names): names = "|".join(names)
        return pat % (names, names)

    @staticmethod
    def tags_except(names, special = True):
        """Returns a regex pattern matching all tags _except_ the given names.
        If special=True (default), special tags are included: <!-- --> <? ?> <![CDATA
        """
        pat = r"""<(?:(?!%s)([a-zA-Z\?][\w:\-]*)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?(\s*[\/\?]?)|\/(?!%s)([a-zA-Z][\w:\-]*)\s*"""
        if special: pat += r"|!--((?:[^\-]|-(?!->))*)--|!\[CDATA\[((?:[^\]]|\](?!\]>))*)\]\]"
        pat += r")>"
        if isstring(names): names = names.split()
        if islist(names): names = "|".join(names)
        names = r"(?:%s)\b" % names              # must check for word boundary (\b) at the end of a tag name, to avoid prefix matching of other tags
        return pat % (names, names)

    @staticmethod
    def tags_pair(names = None):
        """Returns a regex pattern matching: (1) an opening tag with a name from 'names', or any name if 'names' is empty/None;
        followed by (2) any number of characters (the "body"), matched lazy (as few characters as possible);
        followed by (3) a closing tag with the same name as the opening tag.
        The matched tag name is available in the 1st group. Self-closing tags <.../> are NOT matched.
        """
        opening = r"""<(%s)(\s(?:\s*[a-zA-Z][\w:\-]*(?:\s*=(?:\s*"(?:\\"|[^"])*"|\s*'(?:\\'|[^'])*'|[^\s>]+))?)*)?\s*>"""
        closing = r"<\/(\1)\s*>"
        body = r".*?"                                   # lazy "match all"
        pat = opening + body + closing

        if not names: names = r"[a-zA-Z\?][\w:\-]*"     # "any tag name"; for XML matching this regex is too strict, as XML allows for other names, too
        else:
            if isstring(names): names = names.split()
            if islist(names): names = "|".join(names)
        return pat % names

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
###  EDIT DISTANCE
###

def levenshtein(a, b, casecost = 1, spacecost = 1, totals = False):
    """
    Calculates the Levenshtein edit distance between strings a and b.
    'casecost' is the cost of replacement when only the case is changed, not the actual character.
    If totals=True, returns total character costs of both strings, in addition to the distance value,
    as a triple (dist, cost_a, cost_b).

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


#########################################################################################################################################################
###
###  STRING ALIGNMENT
###

def align(s1, s2, mismatch = None, GAP = '_', GAP1 = None, GAP2 = None, dtype = 'float32', return_gaps = False, mismatch_pairs = None):
    u"""
    Align two strings, s1 and s2, and compute their Levenshtein distance using Wagner-Fischer algorithm.
    Each of s1/s2 can be a plain character string (str/unicode) or a FuzzyString.
    Return aligned strings and their distance value: (aligned1, aligned2, distance).
    During alignment, GAP character is inserted where gaps are created.
    The cost of each character-level alignment of chars c1 and c2 is evaluated with mismatch(c1,c2) function,
    or a standard 0/1 equality function if mismatch is None.
    The order of characters for mismatch() is always kept the same: c1 from s1 as 1st argument, c2 from s2 as second;
    hence mismatch() can exploit this information in cost calculation, e.g. to weigh differently characters from s1 and s2.

    >>> align('', '', dtype = 'int8')
    ('', '', 0)
    >>> align('align two strings', 'align one string', dtype = 'int8')
    ('align two strings', 'align one string_', 4)
    >>> align('algorithm to align', 'align two', dtype = 'int8')
    ('algorithm t_o align', 'al___ign_ two______', 13)
    >>> align('to align', 'align two', GAP = u'_')
    (u'to align____', u'___align two', 7.0)
    >>> align('to align', 'align two', GAP = u'⫠')
    (u'to align\\u2ae0\\u2ae0\\u2ae0\\u2ae0', u'\\u2ae0\\u2ae0\\u2ae0align two', 7.0)
    >>> charset = Charset(text = 'algorithm to align two_')
    >>> fuzzy1 = FuzzyString('to align', charset, int)
    >>> a1, a2, d = align(fuzzy1, 'align two')
    >>> (a1.discretize(), a2, d)
    ('to align____', '___align two', 7.0)
    >>> fuzzy2 = FuzzyString('align two', charset, int)
    >>> a1, a2, d = align('algorithm to align', fuzzy2)
    >>> (a1, a2.discretize(), d)
    ('algorithm t_o align', 'al___ign_ two______', 13.0)
    >>> a1, a2, d = align(fuzzy1, fuzzy2)
    >>> (a1.discretize(), a2.discretize(), d)
    ('to align____', '___align two', 7.0)
    >>> a1, a2, d = align(FuzzyString(charset = charset, dtype = int), 'to align two')
    >>> (a1.discretize(), a2, d)
    ('____________', 'to align two', 12.0)
    >>> charset = Charset('_abc')
    >>> fuzzy = FuzzyString.merge('abbac', 'abcba', 'bcaaa', charset = charset, dtype = float, norm = True)
    >>> a1, a2, d = align(fuzzy, '')
    >>> (a2, d)
    ('_____', 5.0)
    >>> a1, a2, d = align(fuzzy, 'bba')
    >>> (a2, '%.2f' % d)
    ('bb_a_', '3.33')
    >>> a1, a2, d = align(fuzzy, 'bbaccab')
    >>> (a2, '%.2f' % d)
    ('bbaccab', '5.00')
    """
    from numpy import zeros, array, cumsum

    swap = False

    # set a default mismatch() function, depending on the types of s1/s2 strings (crisp or fuzzy)
    if mismatch is None:
        if isinstance(s1, FuzzyString) or isinstance(s2, FuzzyString):
            if isinstance(s1, basestring):
                s1, s2 = s2, s1                             # make a swap to always ensure that mismatch() has a FuzzyString as its first argument
                swap = True
            mismatch = FuzzyString.mismatch_crisp if isinstance(s2, basestring) else FuzzyString.mismatch
        else:
            def mismatch(c1, c2): return int(c1 != c2)      # crisp 0/1 character comparison for plain strings

    # convert GAP to a FuzzyString if needed, separately for each string (their types can differ: FuzzyString / basestring)
    if GAP1 is None:  GAP1 = s1.convert(GAP) if isinstance(s1, FuzzyString) else GAP
    if GAP2 is None:  GAP2 = s2.convert(GAP) if isinstance(s2, FuzzyString) else GAP

    # memorize char-vs-GAP mismatch costs to avoid repeated calculation of the same values
    mismatch_1_GAP = array([0] + [mismatch(c1, GAP2) for c1 in s1], dtype)
    mismatch_GAP_2 = array([0] + [mismatch(GAP1, c2) for c2 in s2], dtype)

    # initialize 'dist' array: dist[i,j] => distance between s1[:i] and s2[:j]
    n1, n2 = len(s1), len(s2)
    dist = zeros((n1+1, n2+1), dtype = dtype)
    dist[:,0] = cumsum(mismatch_1_GAP)                      # fill out row #0 and column #0
    dist[0,:] = cumsum(mismatch_GAP_2)

    # edit[i,j] = 0/1/2: indicator of the optimal edit operation on (i,j) position when aligning s1[:i] to s2[:j]
    edit = zeros((n1+1, n2+1), dtype = 'int8')
    edit[:,0] = 1

    if mismatch_pairs is None:
        mismatch_pairs = np.zeros((n1, n2), dtype = dtype)
        for i in range(n1):
            c1 = s1[i]
            for j in xrange(n2):
                mismatch_pairs[i,j] = mismatch(c1, s2[j])

    # fill out the rest of 'dist' and 'edit' arrays, in a separate function to allow speed optimization with Numba
    _align_loop(dist, edit, mismatch_1_GAP, mismatch_GAP_2, mismatch_pairs)

    i, j = n1, n2
    a1 = s1.new() if isinstance(s1, FuzzyString) else ''
    a2 = s2.new() if isinstance(s2, FuzzyString) else ''
    gaps1 = []
    gaps2 = []

    # reconstruct aligned strings a1, a2, from the array of optimal edit operations in each step
    while i or j:
        if edit[i,j] == 0:
            if return_gaps: gaps1.append(len(a1))                   # remember position of the gap being inserted
            a1 += GAP1
            a2 += s2[j-1]
            j -= 1
        elif edit[i,j] == 1:
            if return_gaps: gaps2.append(len(a2))                   # remember position of the gap being inserted
            a1 += s1[i-1]
            a2 += GAP2
            i -= 1
        elif edit[i,j] == 2:
            a1 += s1[i-1]
            a2 += s2[j-1]
            i -= 1
            j -= 1

    assert len(a1) == len(a2)
    a1 = a1[::-1]
    a2 = a2[::-1]

    if swap: a1, a2 = a2, a1

    if return_gaps:
        gaps1 = [len(a1)-i-1 for i in reversed(gaps1)]              # reverse the order and values of gap indices, like a1/a2 were reversed
        gaps2 = [len(a2)-j-1 for j in reversed(gaps2)]              # reverse the order and values of gap indices, like a1/a2 were reversed
        # assert all(a1[i] == GAP1 for i in gaps1)
        # assert all(a2[j] == GAP2 for j in gaps2)
        return a1, a2, dist[n1,n2], gaps1, gaps2
    else:
        return a1, a2, dist[n1,n2]

@jit
def _align_loop(dist, edit, mismatch_1_GAP, mismatch_GAP_2, mismatch_pairs):
    """
    The main loop of align() function. Separated out from the main function to allow Numba JIT compilation,
    which gives approx. 6x speedup. Matrices `dist` and `edit` are in-out arguments: they are modified in place
    and serve as return variables.
    """
    n1, n2 = dist.shape
    for i in range(1, n1):
        for j in xrange(1, n2):
            cost_left = dist[i, j-1] + mismatch_GAP_2[j]    #mismatch(GAP1, s2[j-1]) #+ suffix_cost(lastchar1[i,j-1],GAP) + suffix_cost(lastchar2[i,j-1],s2[j-1])
            cost_up   = dist[i-1, j] + mismatch_1_GAP[i]    #mismatch(s1[i-1], GAP2)
            cost_diag = dist[i-1, j-1] + mismatch_pairs[i-1, j-1]  #mismatch_idx(i-1, j-1) #mismatch(s1[i-1], s2[j-1])

            M = dist[i,j] = min(cost_left, cost_up, cost_diag)
            if M == cost_left: step = 0
            elif M == cost_up: step = 1
            else:              step = 2
            edit[i,j] = step


def align_multiple(strings, mismatch = None, GAP = '_', cost_base = 2, cost_case = 1, cost_gap = 3, cost_gap_gap = 0, weights = None,
                   return_consensus = False, verbose = False):
    """
    Multiple Sequence Alignment (MSA) of given strings through the use of incrementally updated FuzzyString consensus.

    >>> align_multiple(['abbac', 'abcbaa', 'bcaa', '  ', 'aaaaaa', 'bbbbbb'], cost_gap = 3)
    ['ab_bac', 'abcbaa', '_b_caa', '_ _ __', 'aaaaaa', 'bbbbbb']
    >>> align_multiple(['aabbcc', 'bbccaa', 'ccaabb'], cost_gap = 2)
    ['aabbcc____', '__bbccaa__', '____ccaabb']
    >>> align_multiple(['aabbcc', 'bbccaa', 'ccaabb'], cost_gap = 3)
    ['aabbcc__', '__bbccaa', 'ccaabb__']
    >>> align_multiple(['aabbcc', 'aadcc', 'aaeecc'], cost_gap = 2)
    ['aabbcc', 'aad_cc', 'aaeecc']
    >>> align_multiple(['aabbcc', 'aadcc', 'aaeecc'], cost_gap = 3)
    ['aabbcc', 'aad_cc', 'aaeecc']
    """
    from numpy import array, dot, ones

    # create charset & cost matrix
    charset = Charset(text = ''.join(strings) + GAP)
    cost_matrix = charset.cost_matrix(GAP, cost_base = cost_base, cost_case = cost_case, cost_gap = cost_gap, cost_gap_gap = cost_gap_gap)  #dtype = 'float32'
    classes = charset.classes
    if verbose: print 'cost_matrix:\n', cost_matrix

    if weights is None: weights = ones(len(strings))

    maxlen = max(map(len, strings))
    consensus = FuzzyString(strings[0], charset = charset, dtype = 'float32', weight = weights[0])      #'float32'... 'int16' if maxlen*cost_base < 10000 else
    if verbose: print '#1: ', '%8s' % strings[0]

    def mismatch(fuzzy, crisp):
        cls = classes[crisp]
        freq = fuzzy.chars[0]
        # assert freq.min() >= 0
        return dot(cost_matrix[cls,:], freq)

    # def get_mismatch_idx(consensus, s):
    #     "Create mismatch_idx(), a partially pre-computed variant of mismatch() function, to speed up the most critical operation in align() calls."
    #     consensus_chars = consensus.chars
    #     classes_s = [classes[c] for c in s]
    #
    #     def mismatch_idx(i, j):
    #         # print 'i,j:', type(i), i, type(j), j
    #         cls = classes_s[j]
    #         freq = consensus_chars[i]
    #         return dot(cost_matrix[cls,:], freq)
    #
    #     return mismatch_idx

    # @jit
    # def _get_mismatch_pairs_loop(consensus_chars, classes_s):
    #     return dot(consensus_chars, cost_matrix[:,classes_s])
    #     # for j, cls in enumerate(classes_s):
    #     #     mismatch_pairs[:,j] = dot(consensus_chars, cost_matrix[:,cls])

    def get_mismatch_pairs(consensus, s, dtype):
        "Create 2D matrix of mismatch() values for all pairs of letters in both strings, to speed up the most critical operation in align() calls."
        n1, n2 = len(consensus), len(s)
        consensus_chars = array(consensus.chars)

        # classes_s = [classes[c] for c in s]
        # return dot(consensus_chars, cost_matrix[:,classes_s])

        # classes_s = array([classes[c] for c in s])
        # return _get_mismatch_pairs_loop(consensus_chars, classes_s)

        mismatch_pairs = np.zeros((n1, n2), dtype)
        for j, c in enumerate(s):
            mismatch_pairs[:,j] = dot(consensus_chars, cost_matrix[:,classes[c]])
            # for i in xrange(n1):
            #     mismatch_pairs[i,j] = dot(cost, consensus_chars[i])

        return mismatch_pairs

    # 1st pass: come up with a stable consensus; strings are accumulated through adding, NO normalization
    for i, s in enumerate(strings[1:]):
        count = i + 1
        weight = sum(weights[:count])

        GAP1 = consensus.convert(GAP)
        GAP1.chars[0] *= weight             # rescale GAP weight to account for increased total weight of accumulated `consensus`

        # all frequency vectors in consensus must sum up to `weight`
        assert all(np.abs(freq.sum() - weight) < 0.0001 for freq in consensus.chars + GAP1.chars)

        dtype = 'float32' #cost_matrix.dtype
        mismatch_pairs = get_mismatch_pairs(consensus, s, dtype = dtype)

        consensus_aligned, string_aligned, dist, c_gaps, s_gaps = \
            align(consensus, s, GAP1 = GAP1, GAP2 = GAP, return_gaps = True, mismatch = mismatch, mismatch_pairs = mismatch_pairs, dtype = dtype)

        consensus = FuzzyString.merge(consensus_aligned, string_aligned, weights = [1,weights[count]], norm = False)

        if verbose:
            print '#%-3d a:' % (i+2), '%8s' % string_aligned
            print '     c: %8s' % consensus_aligned.discretize()
            # print '  gaps:', s_gaps
            # print '       ', c_gaps
            print
    if verbose: print


    consensus = FuzzyString.merge(consensus, norm = True, dtype = 'float32')
    # if verbose: print 'avg:', consensus.chars

    # 2nd pass: align strings once again to a semi-fixed consensus;
    # only gaps can be added to consensus and to previously aligned strings

    dtype = 'float32'
    GAP1 = consensus.convert(GAP)
    def insert_gap(z, pos): return z[:pos] + GAP + z[pos:]

    aligned = []                                        # output list of aligned strings

    for s in strings:

        mismatch_pairs = get_mismatch_pairs(consensus, s, dtype)

        c_aligned, s_aligned, dist, c_gaps, s_gaps = \
            align(consensus, s, GAP1 = GAP1, GAP2 = GAP, return_gaps = True, mismatch = mismatch, mismatch_pairs = mismatch_pairs, dtype = dtype)

        aligned.append(s_aligned)
        if verbose: print s_aligned

        # new gaps have been inserted into consensus (c_aligned)?
        # backpropagate them to already-aligned strings...
        if len(consensus) != len(c_aligned):
            assert len(c_aligned) - len(consensus) == len(c_gaps) > 0
            consensus = c_aligned
            for gap in c_gaps:
                for k in xrange(len(aligned)):
                    aligned[k] = insert_gap(aligned[k], gap)
    if verbose: print

    return (aligned, consensus) if return_consensus else aligned


class Charset(object):
    chars = None            # list of characters in this charset:     chars[0..N-1] -> char
    classes = None          # dictionary of class IDs for all chars:  classes[char] -> 0..N-1

    def __init__(self, chars = None, text = None):
        "'chars': list or string; if a list, it may contain special pseudo-characters in a form of arbitrary string or object."

        if chars is None: chars = []
        if text: chars = list(chars) + sorted(set(text))

        assert len(set(chars)) == len(chars)        # make sure there are no duplicates in 'chars'
        self.chars = chars
        self.classes = {char:cls for cls, char in enumerate(chars)}

    def size(self): return len(self.chars)

    def classOf(self, char):
        "Mapping: char -> 0..N-1 or None if 'char' not in charset."
        return self.classes.get(char)

    __len__ = size
    __getitem__ = classOf

    def encode(self, s, dtype = 'float', weight = 1):
        "Convert a plain string of characters into a list of one-hot numpy vectors encoding class IDs."

        from numpy import zeros
        N = len(self.chars)

        def encode_one(char):
            freq = zeros(N, dtype = dtype)
            hot = self.classes.get(char)
            if hot is None: raise Exception("Charset.encode(): trying to encode a character (%s) that is not in charset." % char)
            freq[hot] = weight
            return freq

        return map(encode_one, s)

    def cost_matrix(self, GAP = '_', cost_base = 2, cost_case = 1, cost_gap = 3, cost_gap_gap = 0, dtype = None):
        "Create a parameterized cost matrix for edit distance."

        costs = np.array([cost_base, cost_case, cost_gap, cost_gap_gap])
        # print "[cost_base, cost_case, cost_gap, cost_gap_gap]:", costs

        # can dtype be int16 to speed up calculations, reduce memory footprint and avoid rounding errors?
        if dtype is None:
            if np.array_equal(costs, costs.astype('int32')):
                dtype = 'int32'
            else:
                dtype = 'float32'

        # misclassficiation cost is normally 'cost_base' everywhere except diagonal, and (-cost_base) on diagonal
        D = cost_base * (1 - np.eye(self.size(), dtype = dtype))

        # GAP vs. other
        cls_gap = self.classes[GAP]
        D[cls_gap,:] = cost_gap
        D[:,cls_gap] = cost_gap
        D[cls_gap,cls_gap] = cost_gap_gap

        # case difference
        if cost_case != cost_base:
            for ch in self.chars:
                if not isinstance(ch, basestring): continue
                lo = self.classes.get(ch.lower())
                up = self.classes.get(ch.upper())
                if lo == up or cls_gap in (lo,up) or None in (lo,up): continue
                D[lo,up] = D[up,lo] = cost_case

        assert 0 <= D.min()

        return D


class FuzzyString(object):
    """
    A string of "fuzzy characters", each being a probability/frequency distribution over a predefined charset.

    >>> charset = Charset('abc')
    >>> fuzzy = FuzzyString('aabccc', charset, dtype = int)
    >>> list(fuzzy.chars[2])
    [0, 1, 0]
    >>> fuzzy += 'aaa'
    >>> list(fuzzy.chars[-1])
    [1, 0, 0]
    >>> fuzzy[0] == 'a' and 'a' == fuzzy[0]
    True
    >>> fuzzy[0] != 'a' or 'a' != fuzzy[0]
    False
    >>> fuzzy[0] == 'b'
    False
    >>> fuzzy[0] != 'b'
    True
    >>> fuzzy.discretize()
    'aabcccaaa'
    >>> fuzzy[::-1].discretize()
    'aaacccbaa'
    >>> fuzzy[::2].discretize()
    'abcaa'
    >>> FuzzyString.merge(fuzzy[::2], 'aacc', norm = False).chars
    [array([2, 0, 0]), array([1, 1, 0]), array([0, 0, 2]), array([1, 0, 1]), array([1, 0, 0])]
    >>> FuzzyString.merge(fuzzy[::2], 'aacc', norm = True, dtype = float).chars
    [array([ 1.,  0.,  0.]), array([ 0.5,  0.5,  0. ]), array([ 0.,  0.,  1.]), array([ 0.5,  0. ,  0.5]), array([ 1.,  0.,  0.])]
    """

    # __isfuzzy__ = True      # flag to replace isinstance() checks with hasattr()

    charset = None          # Charset instance that defines a char-class mapping: char -> 0..N-1
    chars = None            # list of numpy vectors: chars[pos][c] = probability/frequency of character class 'c' on position 'pos' in string
                            # kept as a list, not monolithic 2D array, to enable fast edit operations: character insertion/deletion;
                            # you should treat each array, chars[pos], as IMMUTABLE (!) and make a copy
                            # whenever particular fraquency values need to be modified. The `chars` list itself is mutable (!).
    dtype = None

    def __init__(self, text = '', charset = None, dtype = 'int32', chars = None, weight = 1):
        "Convert a crisp string 'text' to fuzzy."
        assert charset is not None
        self.charset = charset
        self.dtype = dtype
        self.chars = charset.encode(text, dtype, weight) if chars is None else chars

    def copy(self):
        "Shallow copy of self."
        return copy(self)

    def convert(self, text):
        "Create a new FuzzyString, like this one (same charset and dtype), but with a different plain text."
        assert isinstance(text, basestring)
        return FuzzyString(text, self.charset, dtype = self.dtype)

    def new(self):
        "Create a FuzzyString like this one (same charset and dtype), but with empty text."
        return self.convert('')

    def append(self, other):
        "Append a char or a string, crisp or fuzzy, to the end of this string."
        self.chars = self._concat_R(other)

    def discretize(self, minfreq = None, UNKNOWN = None):
        """
        On each position in `chars` pick the first most likely crisp character and return concatenated as a crisp string.
        Optionally, apply minimum frequency threshold, if not satisfied insert GAP.
        """
        if minfreq is None:
            return ''.join(self.charset.chars[freq.argmax()] for freq in self.chars)
        else:
            return ''.join(self.charset.chars[freq.argmax()] if freq.max() >= minfreq else UNKNOWN for freq in self.chars)

    def regexify(self, minfreq = 0.0, maxchars = 3, GAP = None, merge = True, merge_stop = [], _escape = set(r'.[]{}()|?\\^$*+-')):
        """
        Encode this FuzzyString as a regex pattern, where alternative characters (freq > maxfreq) on each position
        are encoded as character sets [ABC], uncertainties (all freq <= minfreq) are replaced with a dot '.',
        gaps are converted to optional markers '?' and repeated code points are merged (if merge=True)
        to repetitions {m,n}. If merge_stop is given, characters (code points) from merge_stop are excluded from merging.
        """
        from nifty.math import np_find
        charset_chars = self.charset.chars
        GAP_cls = self.charset.classOf(GAP)

        def escape(char): return '\\' + char if char in _escape else char

        codes = []
        modes = []

        last_code = None            # recent regex code, pending to be emitted
        last_rep = (0,0)            # (min,max) repetition of the last code

        for freq in self.chars:
            idx = list(np_find(freq > minfreq))
            gap = bool(GAP and GAP_cls in idx)              # gap=True if the current character(s) is optional, or there are no characters (skip)
            if gap: idx.remove(GAP_cls)

            n = len(idx)
            if n == 0 or n > maxchars:
                # if gap: continue                            # no characters, only a gap? skip without emitting any regex code
                code = '.'
            elif n == 1:
                char = charset_chars[idx[0]]
                code = escape(char)
            else:
                chars = [escape(charset_chars[i]) for i in idx]
                code = '[%s]' % ''.join(chars)

            # if code == last_code:
            # if gap: code += '?'

            mode = '?' if gap else ''
            codes.append(code)
            modes.append(mode)

        if not merge:
            return ''.join(c + m for c, m in izip(codes, modes))

        # merge repetitions of the same code
        pos = 0
        codes_final = []

        for code, code_group in groupby(codes):

            code_group = list(code_group)
            k = len(code_group)
            assert k > 0

            mode_group = modes[pos:pos+k]
            pos += k

            # in simple case (low `k`) or character from merge_stop, just copy original <code,mode> pairs to output, no merging
            if k <= 2 or code in merge_stop:
                for c, m in izip(code_group, mode_group):
                    codes_final.append(c + m)
                continue

            # merge modes and convert to {repmin,repmax} pair
            repmin = repmax = 0
            for mode in mode_group:
                repmin += int(mode is not '?')
                repmax += 1

            repmax = k
            repmin = k - len([m for m in mode_group if m is '?'])           # subtract the no. of optional '?' characters

            # output a single `code` token with appropriate `mode` as obtained from merge

            # if repmin == repmax == 1: mode = ''
            # elif repmin == 0 and repmax == 1: mode = '?'
            if repmin == repmax:   mode = '{%s}' % repmax
            elif repmin == 0:      mode = '{,%s}' % repmax
            else:                  mode = '{%s,%s}' % (repmin, repmax)

            codes_final.append(code + mode)

        pattern = ''.join(codes_final)
        return pattern

    @staticmethod
    def merge(*strings, **params):
        """
        On each position, add corresponding char frequencies/probabilities of fuzzy1 and fuzzy2,
        and normalize to unit sums if norm=True. Return as a new FuzzyString.
        Default params: charset=None, dtype=None, norm=False, weights=None.
        """
        weights = params.pop('weights', None)
        charset = params.pop('charset', None)
        dtype = params.pop('dtype', None)
        norm = params.pop('norm', False)

        if weights is not None and len(weights) != len(strings): raise Exception("The number of weights (%s) and strings (%s) differ." % (len(weights), len(strings)))

        # infer charset and dtype
        if dtype is None:
            dtypes = [np.dtype(s.dtype) for s in strings if hasattr(s, 'dtype')]
            dtype = max(dtypes) if dtypes else None

        if charset is None:
            for s in strings:
                if not isinstance(s, basestring):
                    charset = s.charset
                    break

        if charset is None: raise Exception("Cannot infer charset of strings to be combined")
        if dtype is None: raise Exception("Cannot infer dtype of strings to be combined")

        # check compatibility and convert plain strings to fuzzy
        def validate(s):
            if isinstance(s, basestring): return FuzzyString(s, charset = charset, dtype = dtype)
            if not s.charset == charset: raise Exception("Trying to combine FuzzyStrings with different charsets")
            # if not np.dtype(s.dtype) <= dtype: raise Exception("Trying to combine FuzzyStrings with incompatible numeric types: %s, %s" % (s.dtype, dtype))
            return s

        fuzzy = map(validate, strings)

        # combine numpy arrays on each char position
        v = len(charset)
        n = max(len(s) for s in fuzzy)
        chars = [np.zeros(v, dtype) for _ in xrange(n)]

        for i, s in enumerate(fuzzy):
            schars = s.chars
            w = weights[i] if weights is not None else 1
            for j in xrange(len(schars)):
                chars[j] += schars[j] * w

        # normalize?
        if norm: chars = [freq / freq.sum() for freq in chars]

        return FuzzyString(chars = chars, charset = charset, dtype = dtype)

    def dist(self, other, degree = 1):
        "Compute 1-norm or 2-norm distance between frequency vectors of self and 'other', summed up over all characters."
        assert self.charset == other.charset

        chars1 = self.chars
        chars2 = other.chars
        n1, n2 = len(chars1), len(chars2)

        if n1 == n2 == 1:             # most typical case, handled separately for speed
            if degree == 1: return sum(np.abs(chars1[0] - chars2[0]))
            if degree == 2: return sum((chars1[0] - chars2[0]) ** 2) ** 0.5

        if n1 != n2:
            if n2 < n1:
                chars1, chars2 = chars2, chars1
                n1, n2 = n2, n1
            assert len(chars1) < len(chars2)

            chars1 = copy(chars1)
            chars1 += [np.zeros_like(chars2[0])] * (n2 - n1)

        if degree == 1:
            _dist = lambda f1, f2: sum(np.abs(f1 - f2))
        elif degree == 2:
            _dist = lambda f1, f2: sum((f1 - f2) ** 2) ** 0.5
        else:
            raise Exception("Unknown norm type: %s" % degree)

        return sum(_dist(freq1, freq2) for freq1, freq2 in izip(chars1, chars2))

    def mismatch(self, other, degree = 1, is_basestring = None):
        """
        Like dist(), but handles only 1-letter strings, and 'other' can be a plain string.
        If frequency vectors are normalized to unit sum, the distance returned is guaranteed to lie in <0.0,1.0> range.
        """
        assert len(self.chars) == len(other) == 1 and degree in (1,2)
        freq = self.chars[0]

        if is_basestring or (is_basestring is None and isinstance(other, basestring)):
            cls = self.charset.classes[other]
            # diff = freq.copy()
            # diff[cls] -= 1
            if degree == 1: return (sum(freq) - freq[cls] + np.abs(freq[cls]-1)) * 0.5                  # same as: sum(abs(diff)) * 0.5
            if degree == 2: return ((sum(freq**2) - freq[cls]**2 + (freq[cls]-1)**2) * 0.5) ** 0.5      # same as: sum(diff**2 * 0.5) ** 0.5

        diff = freq - other.chars[0]
        if degree == 1: return sum(np.abs(diff)) * 0.5
        if degree == 2: return sum(diff**2 * 0.5) ** 0.5

    def mismatch_crisp(self, other, degree = 1):
        "Like mismatch(), for use when `other` is guaranteed to be a crisp string (basestring)."
        return self.mismatch(other, degree = degree, is_basestring = True)

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, pos):
        "Character #pos returned as a FuzzyString (!), not a numpy array. Can be safely compared using == or != "
        dup = copy(self)
        dup.chars = self.chars[pos] if isinstance(pos, slice) else [self.chars[pos]]
        return dup

    def __eq__(self, other):
        if self is other: return True
        if isinstance(other, basestring):                                       # crisp string?
            other = FuzzyString(other, self.charset, dtype = self.dtype)

        if self.charset is not other.charset: return False
        if len(self.chars) != len(other.chars): return False

        for v1, v2 in izip(self.chars, other.chars):
            if v1 is v2: continue
            if not np.array_equal(v1, v2): return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _concat_R(self, other, inplace = False):
        "Concat raw `chars` of self and `other`."
        if isinstance(other, basestring):                                       # crisp string?
            if inplace:
                self.chars += self.charset.encode(other, self.dtype)
            else:
                return self.chars + self.charset.encode(other, self.dtype)
        else:                                                                   # or FuzzyString?
            # assert isinstance(other, FuzzyString)
            if self.charset is not other.charset: raise Exception("Trying to add two FuzzyStrings with different charsets")
            if inplace:
                self.chars += other.chars
            else:
                return self.chars + other.chars

    def _concat_L(self, other):
        assert isinstance(other, basestring)
        return self.charset.encode(other, self.dtype) + self.chars

    def __add__(self, other):
        return FuzzyString(chars = self._concat_R(other), charset = self.charset, dtype = self.dtype)

    def __radd__(self, other):
        return FuzzyString(chars = self._concat_L(other), charset = self.charset, dtype = self.dtype)

    def __iadd__(self, other):
        self._concat_R(other, inplace = True)
        return self


#########################################################################################################################################################
###
###  LANGUAGE modeling
###

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
    """Turn the document to space-separated string of lowercase words.
       Remove all common words, punctuactions, numbers and other words
       that might be meaningless or problematic in semantic analysis.
    """
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
###  TEXT class for language control
###

class Text(xunicode):
    """
    A string of text (xunicode) with associated information about the language (encoding)
    that should be used for its interpretation, for example:

        HTML, SQL, URL, plain, ...

    Allows languages to be nested in one another, possibly multiple times, which yields *compound* languages:

        HTML/mediawiki, HTML/URL, SQL/HTML/plain, ...

    The Text class keeps track of language embeddings and thus enables safe, transparent and easy
    encoding/decoding/converting - to/from/between different (possibly nested) languages.
    With Text, automatic sanitization of strings can be easily implemented, especially in web applications.
    Nesting/unnesting (vertical transformations, e.g.: plain -> HTML/plain -> mediawiki/HTML/plain) is always lossless.
    Only conversions (horizontal transformations, e.g.: HTML <-> mediawiki) between languages can be lossy.

    Text instances can be manipulated with in a similar way as strings, using all common operators: + * % [].
    All operations and methods that concatenate or combine Texts, or a Text and a string, check whether the language of both operands
    is either the same or None, and propagate the language value to the resulting Text instance.
    WARNING: when using .join() method remember that you have to call it on a Text instance (!), otherwise the resulting
    string will be a plain str/unicode object, not a Text, even when joining Text instances!
    The right way to join a sequence of Text instances is the following:
        Text(sep).join(...)
    - here, the language of resulting Text instance will be derived from the language of sequence items (or None if no item has a language specified);
    or, if you want to enforce a specific language and use it as default when the sequence is empty or the items have no language, use:
        Text(sep, language).join(...)
    To make a Text instance be treated as a regular string, cast it back to unicode or str, like unicode(text).

    Example languages:
    - plain - plain text, without any encoding nor special meaning
    - HTML - rich text expressed in HTML language, a full HTML document; it can't be "decoded", because any decoding would have to be lossy
             (tags would be removed), however it might be converted to another rich-text language (wiki-text, YAML, ...),
             possibly with a loss in style information
    - HTML/plain - plain text escaped for inclusion in HTML; contains entities which must be decoded to get 'plain' text again
    - HyperML/plain - plain text escaped for inclusion in HyperML; contains $* escape strings which must be decoded to get 'plain' text again
    - HyperML/HTML/plain - plain text encoded for HTML and later escaped for HyperML; you first have to decode HyperML, only then HTML,
             only then you will obtain the original string
    - URL/plain - URL-encoded plain text, for inclusion in a URL, typically as a GET parameter
    - URL - full URL of any form
    - SQL
    - value - text representation of a value
    - SQL/value - SQL representation of a value, as an escaped string that can be directly pasted into a query

    The "directory path" notation: ".../.../..." in language naming expresses the concept that when a language is nested in another language,
    it's similar to including a file inside a folder: you first have to visit the parent folder (or decode the outer language)
    in order to access the inner file (the original nested text). Also, this naming convention expresses the fact that a language
    nested in another language forms, in fact, YET ANOTHER language (!) that should have its own name (X/Y), because it has different rules
    for how a plain-text string needs to be encoded in order to form a correct representation in this language.
    For example, JavaScript code embedded in HTML's <script> block is, stricly speaking, no longer a JavaScript code!
    That's because the "</script>" substring - which normally is a valid string in JavaScript - is no longer allowed to appear
    inside this code. Thus, the true language of this piece of code is "HTML/JavaScript", rather than just "JavaScript"!
    Such intricacies are very difficult to track when implementing web applications,
    where strings in many different languages (HTML/JavaScript/CSS/SQL/URL/...) are being passed from one place to another,
    with all different types of conversions done along the way. Securing such an application and performing bullet-proof sanitization
    is close to impossible without a tool like the Text class which automatically tracks the exact language of the text
    and performs automatic conversions whenever necessary.

    >>> t = Text("<a>this is text</a>", "HTML")
    >>> (t).language, (t+t).language
    ('HTML', 'HTML')
    >>> (t + "ala").language, (t + u"ala").language
    ('HTML', 'HTML')
    >>> ((t+" %s") % "ala").language, ((t+" %s") % u"ala").language
    ('HTML', 'HTML')
    >>> s = "'%s'" % t
    >>> type(s), s
    (<type 'unicode'>, u"'<a>this is text</a>'")
    >>> s = u"'%s'" % t
    >>> type(s), s
    (<type 'unicode'>, u"'<a>this is text</a>'")
    >>> ("ala " + t).language, (u"ala " + t).language
    ('HTML', 'HTML')
    >>> (t + t).language, (t * 3).language, (5 * t).language
    ('HTML', 'HTML', 'HTML')
    >>> (Text('<h1>Title</h1>', 'HTML') + Text('ala &amp; ola', 'HTML/plain')).language
    'HTML'
    >>> Text().join([Text('<h1>Title</h1>', 'HTML'), Text('ala &amp; ola', 'HTML/plain')])
    u'<h1>Title</h1>ala &amp; ola'
    >>> Text(' ', 'HTML').join(['<h1>Title</h1>', Text('ala &amp; ola', 'HTML/plain')])
    u'<h1>Title</h1> ala &amp; ola'
    >>> t + Text('ola', "plain")
    Traceback (most recent call last):
        ...
    Exception: Can't combine Text/string instances with incompatible languages: 'HTML' and 'plain'
    >>> unicode(t).language
    Traceback (most recent call last):
        ...
    AttributeError: 'unicode' object has no attribute 'language'

    >>> t = Plain("this is text")
    >>> t.language, (t+t).language, (t+'undefined').language
    ('plain', 'plain', 'plain')
    >>> t = HTML("<a>this is text</a>")
    >>> t.language, (t+t).language, (t+'undefined').language
    ('HTML', 'HTML', 'HTML')
    >>> Plain(Text('<a>this is text</a>', 'HTML')).language
    'plain'
    >>> HTML(Text('this is text', 'plain')).language
    'HTML'
    """

    language = None     # non-empty name of the formal language in which the string is expressed; can be a compound language, like "HTML/plain";
                        # for plain text, we recommend "plain" as a name; None = unspecified language that can be combined with any other language
    settings = None     # (NOT USED) the TextSettings object that contains global configuration for this object: list of converters and conversion settings

    def __new__(cls, text = u'', language = None, settings = None):
        """Wrap up a given string in Text object and mark what language it is. We override __new__ instead of __init__
        because the base class is immutable and overriding __new__ is the only way to modify its initialization.
        """
        self = unicode.__new__(cls, text)
        language = language or (text.language if isinstance(text, Text) else None)
        if language is not None:
            self.language = language
        return self

    @staticmethod
    def combine(lang, text, msg = "Can't combine Text/string instances with incompatible languages: '%s' and '%s'"):
        """
        Check that languages of two texts are compatible with each other and return the language of combined text,
        or raise an exception. Languages are compatible if they're either equal, or one of them is undefined (None),
        or one of them is a '/'-terminating prefix of the other (then the shorter language is the result).
        """
        lang2 = getattr(text, 'language', None)
        if None in (lang, lang2): return lang or lang2          # when one of the languages is None, return the other one
        if lang == lang2: return lang                           # if equal, that's fine

        # not None and not equal? one of the strings ("outer" language) must be a '/'-terminating prefix of the other ("inner" language)
        s1 = lang + '/'
        s2 = lang2 + '/'
        if s1.startswith(s2): return lang2
        if s2.startswith(s1): return lang
        raise Exception(msg % (lang, lang2))


    ### Override all operators & methods to ensure that the 'language' setting is propagated to resulting strings

    def __add__(self, other):
        #if getattr(other, 'language', None) not in (None, self.language):
        #    raise Exception("Can't add Text instances containing incompatible languages: '%s' and '%s'" % (self.language, other.language))
        language = Text.combine(self.language, other)           # first check if languages are compatible
        res = unicode.__add__(self, other)
        return Text(res, language)

    def __radd__(self, other):
        #if getattr(other, 'language', None) not in (None, self.language):
        #    raise Exception("Can't add Text instances containing incompatible languages: '%s' and '%s'" % (other.language, self.language))
        language = Text.combine(self.language, other)
        res = other.__add__(self)
        return Text(res, language)

    def __mul__(self, count):
        res = unicode.__mul__(self, count)
        return Text(res, self.language)
    __rmul__ = __mul__

    def __mod__(self, other):
        res = unicode.__mod__(self, other)
        return Text(res, self.language)

    def __getitem__(self, idx):
        res = unicode.__getitem__(self, idx)
        return Text(res, self.language)

    def __getslice__(self, i, j):
        res = unicode.__getslice__(self, i, j)
        return Text(res, self.language)

    def capitalize(self):
        return Text(unicode.capitalize(self), self.language)
    def center(self, *a, **kw):
        return Text(unicode.center(self, *a, **kw), self.language)

#     def decode(self, *a, **kw):
#         res = unicode.decode(self, *a, **kw)
#         assert isinstance(res, unicode), "Text.decode() can only be used for decoding into <unicode> not <%s>" % type(res)
#         return Text(res, self.language)
#     def encode(self, *a, **kw):
#         res = unicode.encode(self, *a, **kw)
#         assert isinstance(res, unicode), "Text.encode() can only be used for encoding into <unicode> not <%s>" % type(res)
#         return Text(res, self.language)

    def expandtabs(self, *a, **kw):
        return Text(unicode.expandtabs(self, *a, **kw), self.language)
    def format(self, *a, **kw):
        return Text(unicode.format(self, *a, **kw), self.language)

    def join(self, iterable):
        if islist(iterable): items = iterable
        else: items = list(iterable)            # we have to materialize the iterable to check language of each item

        # check that all strings to be joined have compatible languages; calculate the resulting language;
        # the initial self.language can be None - this allows the items set the language
        language = reduce(Text.combine, items, self.language)
        return Text(unicode.join(self, items), language)

    def ljust(self, *a, **kw):
        return Text(unicode.ljust(self, *a, **kw), self.language)
    def lower(self, *a, **kw):
        return Text(unicode.lower(self, *a, **kw), self.language)
    def lstrip(self, *a, **kw):
        return Text(unicode.lstrip(self, *a, **kw), self.language)
    def partition(self, *a, **kw):
        return tuple(Text(s, self.language) for s in unicode.partition(self, *a, **kw))
    def replace(self, *a, **kw):
        return Text(unicode.replace(self, *a, **kw), self.language)
    def rjust(self, *a, **kw):
        return Text(unicode.rjust(self, *a, **kw), self.language)
    def rpartition(self, *a, **kw):
        return tuple(Text(s, self.language) for s in unicode.rpartition(self, *a, **kw))
    def rsplit(self, *a, **kw):
        return [Text(s, self.language) for s in unicode.rsplit(self, *a, **kw)]
    def rstrip(self, *a, **kw):
        return Text(unicode.rstrip(self, *a, **kw), self.language)
    def split(self, *a, **kw):
        return [Text(s, self.language) for s in unicode.split(self, *a, **kw)]
    def splitlines(self, *a, **kw):
        return [Text(s, self.language) for s in unicode.splitlines(self, *a, **kw)]
    def strip(self, *a, **kw):
        return Text(unicode.strip(self, *a, **kw), self.language)
    def swapcase(self, *a, **kw):
        return Text(unicode.swapcase(self, *a, **kw), self.language)
    def title(self, *a, **kw):
        return Text(unicode.title(self, *a, **kw), self.language)
    def translate(self, *a, **kw):
        return Text(unicode.translate(self, *a, **kw), self.language)
    def upper(self, *a, **kw):
        return Text(unicode.upper(self, *a, **kw), self.language)
    def zfill(self, *a, **kw):
        return Text(unicode.zfill(self, *a, **kw), self.language)


# class Plain(Text):
#     language = 'plain'
#
# class HTML(Text):
#     language = 'HTML'

# shorthand for Text(..., "plain")
def Plain(text, settings = None): return Text(text, "plain", settings)

# shorthand for Text(..., "HTML"); in the future may be converted to a subclass with some additional
# HTML-specific functionality or configuration defaults (?)
def HTML(text, settings = None): return Text(text, "HTML", settings)


#########################################################################################################################################################

if __name__ == "__main__":
    import doctest
    print doctest.testmod()

    print
    print "align_multiple..."
    res = align_multiple([u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', 'abcbaabcaa', '  ', u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', u'The following example shows how the Command-Line Interface can be used to compare three different expressions:'])
    for s in res: print s
    print

    from timeit import timeit
    print 'timeit...',
    print timeit("""align_multiple([u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', 'abcbaabcaa', '  ', u'This module provides a simple way to time small bits of Python code. It has both a Command-Line Interface as well as a callable one. It avoids a number of common traps for measuring execution times. See also Tim Peters’ introduction to the “Algorithms” chapter in the Python Cookbook, published by O’Reilly.', u'The following example shows how the Command-Line Interface can be used to compare three different expressions:'])""",
                 "from __main__ import align_multiple",
                 number = 10
                 )

