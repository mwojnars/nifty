# -*- coding: utf-8 -*-
"""
Flexible context-based pattern matching in HTML/XML/markup/plaintext documents.

Redex Pattern is a new type of tool for extracting data from any markup document. 
It defines a new language, "redex" (REgular Document EXpressions, similarity with "regex" intentional),
for concise description of text patterns occuring in documents 
and defining which parts of text should be extracted as data during analysis.

Redex patterns define layout of the document: subsequent portions of text that must be present in given locations, 
and places where substrings - if matched - should be extracted as variables.

Redex combines consistency and compactness of regexes (single pattern matches all document 
and extracts multiple variables at once) with strength and precision of XPaths: 
it is much simpler than regexes and allows patterns to span multiple fragments of the document, 
thus providing precise *context* where each fragment is allowed to match.
Single Pattern can substitute a dozen of XPaths. Redex patterns are also much more readable than 
both regexes and XPath rules, thanks to their literal resemblance to the documents being parsed.
Pattern/redex is tailored to markup languages and is aware of characteristic parts of tagged documents,
making parsing process more reliable. Redex patterns bridge the gap between regexes and XPaths 
as used in web scraping.

Every Pattern instance is built from a string written in redex. 
This string is compiled internally to a regex and subsequently matched against documents.
For example, the following simple redex pattern:

    >>> redex = '<a href="{URL}">the link we are looking for</a>'

is compiled to a much more complicated regex:

    >>> print Pattern(redex).regex.pattern
    (|(?<=>)\s*)<a(?=\s|/|>)([^<>]*?['"\s=]|)(\s*|(['"\s=][^<>]*?['"\s=]))href\s*=\s*("|')(?P<URL>[^<>]*?)("|')(['"\s=][^<>]*?|)>the\s+link\s+we\s+are\s+looking\s+for(|(?<=>)\s*)</a(?=\s|/|>)(['"\s=][^<>]*?|)>

This regex takes care of all intricacies of HTML documents: optional spaces in different places, 
additional surrounding attributes inside tags, different equivalent quoting characters (' or ") 
for tag attribute values, etc. Nobody would be able to write this regex manually.
Redex patterns are designed to do it for you.

You can create a pattern in two ways. By instantiating Pattern class (handy for short patterns):

    >>> pat = Pattern('<a href="{URL}"></a>')
    
or by subclassing Pattern, with redex string put either in the 'pattern' property, or just in pydocs for brevity:

    class UrlPattern(Pattern):
        pattern = '<tr><td><a href="{URL}"> a long and very prominent link with <b>inner <u>elements</u></b>, enclosed in a table row </a></td></tr>'

    class UrlPattern(Pattern):
        '''
        <tr><td>
        <a href="{URL}"> a long and very prominent link with <b>inner <u>elements</u></b>, enclosed in a table row </a>
        </td></tr>
        '''
    pat2 = UrlPattern()

Typically, you match the pattern to extract some data. This can be done with any of Pattern.match*() methods:
match, match1, matchAll (see pydocs for differences between them); 
or match, match1 functions, which compile and match the pattern in one step.
They all return matched values of variables (groups):

    >>> pat.match1('<A href="http://google.com"></A>')
    'http://google.com'
    
    >>> match1('<a href="{URL}"></a>', '<A href="http://google.com"></A>')
    'http://google.com'


REDEX SYNTAX.
    
Static text        -- Regular text (static text) is matched as-is, case-insensitive by default.
                      Spaces between static words match any sequence of 1+ whitespaces.
                      Spaces surrounding non-static expressions on either side 
                      (a variable, optional expression, ...) match 0+ whitespaces (can match empty string).
                      Spaces inside <...> tags match 0+ whitespaces, as well as any sequence of chars 
                      delimited by in-tag separators: ', ", =, or a space.
                      
                      The match of the entire pattern can be preceeded in the document 
                      by any number of whitespaces (they're stripped out before matching),
                      as well as followed by any sequence of characters - not only spaces (!).
                      Effectively, the match starts at the beginning of the non-whitespace part of the document,
                      and can terminate anywhere in the doc, not necessarily at the end 
                      (doesn't have to consume all characters).
                      
                      >>> match('Alice in Wonderland', '  Alice in Wonderland -- Alice was beginning to get')
                      'Alice in Wonderland'

Apples and ~       -- Tilde (~) matches any word: a sequence of 1+ alphanumeric characters 
                      (letters, digits, underscore "_" and tilde "~"), without spaces,
                      as many characters as possible (greedy match).
                      
                      >>> match('Apples and ~', 'Apples and oranges and pears')
                      'Apples and oranges'

Version .          -- Dot (.) matches any continuous sequence of 0+ non-space characters except tags (no '>' or '<'),
                      but as few characters as possible (lazy match); can also be used inside tags.
                      Put "{.}" in the pattern to match literal dot (exactly one) instead of a sequence.
                      >>> match1('Python {VER .} on .', 'Python 2.7 on Linux')
                      '2.7'
                      >>> match1('<a href="./{USER ~}">', '<a href="http://twitter.com/wojnarski">')
                      'wojnarski'

Address: ..        -- Two dots (..) match any sequence of 0+ characters except tags (no '<' or '>'), lazy match; 
                      can be used inside tags, although usually a single dot '.' suffices.
                      >>> match1('Address: {ADDR ..} {.}', 'Address: 12-345 Warsaw, Poland.')
                      '12-345 Warsaw, Poland'

<p>...</p>         -- Three or more dots (...) match any sequence of 0+ characters, including tags 
                      ('<' and '>' allowed). Cannot be used inside tags: between < and >.
                      The dots . .. ... and the tilde ~ are called jointly the *fillers*.
                      
                      >>> match1('<p>{TEXT ...}</p>', '<p><u>Apples</u> and <i>oranges</i></p>')
                      '<u>Apples</u> and <i>oranges</i>'

<a>link</a>        -- Attributes of a tag are matched implicitly: <tag> is equivalent to <tag ..>, 
                      so there's no need to put ".." everywhere inside tags.
                      
                      >>> match1('<div>{MSG}</div>', '<div style="width:500px;border:1;" id="message">Success</div>')
                      'Success'

                      Moreover, implicit 0+ spaces are matched between neighboring tags even if no explicit
                      space is present in the pattern:
                      
                      >>> match('</td><td>', '</td>   <td>')
                      '</td>   <td>'

<div message>...   -- Each word on attribute list: attribute name, or a value surrounded by quotes, spaces or =, 
                      is treated as a separate item. Surrounding items (and entire attributes) are matched implicitly, 
                      so you can specify one item without worrying about the rest. No need to use surrounding ".." 
                      or to worry about unseen attributes and values that might be added to the tag in the future.
                      Compare this to the weirdness of the XPath expression that's needed to match just one class 
                      in a multi-class element, like "title" in class="long wide title", 
                      when the ordering of names is unknown:
                          xpath = "//div[contains(concat(' ', normalize-space(@class), ' '), ' title ')]"
                      Using redex patterns is a lot easier:
                      
                      >>> match1('<div title>{TITLE}</div>', 
                      ...        '<div style="width:800px" class="long title wide">About this website</div>')
                      'About this website'

<. price>          -- Dot (.) after '<' matches any tag name. If you don't know whether the element of interest 
                      is a <div>, or a <p>, or a <table>, or a <span>, ... or you just want to abstract from 
                      the particular name, use "<." to match any tag.
                      Note that here, the dot behaves slightly differently than the general-purpose dot used in 
                      other places: when replacing a tag name, it matches at least 1 character (no empty string), 
                      the match is greedy and it must terminate on a word boundary (space or end of tag).
                       
                      >>> match1('<. price>{PRICE}</.>', '<span class="price">$9.99</span>')
                      '$9.99'

{VAR ...}          -- Braces define a *variable*: a named subexpression, equivalent to a *group* in a regex pattern.
                      After matching, the string matched by the expression is returned under the name of the variable.
                      Definition of a variable has a form of {NAME expression}, where NAME is the variable name
                      (case-sensitive, typically in all-caps to distinguish from static parts of the pattern);
                      and 'expression' is any redex pattern, possibly containing other nested variables.
                      The 'expression' can be omitted ({NAME}), in such case the default pattern ".." is assumed.
                      By default, Pattern.match() returns an ObjDict of all variables, which is a regular dict
                      extended with object-like access to values:

                      >>> items = match('<a href="{URL ./photos/{USER ~}/{ID}/}">', 
                      ...               '<a href="http://www.flickr.com/photos/atelier/738724/">')
                      >>> items
                      {'URL': 'http://www.flickr.com/photos/atelier/738724/', 'USER': 'atelier', 'ID': '738724'}
                      >>> items['USER']
                      'atelier'
                      >>> items.USER
                      'atelier'
                      
                      If you want standard 'dict' to be returned instead, set dicttype=dict inside your pattern class.
                      If you wish that names of variables in the result are all changed to lowercase
                      (convenient if you want to use them as object properties later on), 
                      set tolower=True in your pattern:
                      
                      >>> pat = Pattern('{FRUIT1 ~} and {FRUIT2 ~}')
                      >>> pat.tolower = True
                      >>> items = pat.match('apples and oranges')
                      >>> items.fruit1, items.fruit2
                      ('apples', 'oranges')
                      
                      By default, all variables which do NOT contain 3-dots '...' and thus match only regular text
                      without tags, undergo HTML cleansing after extraction: striping of leading/trailing spaces,
                      merging of multiple whitespaces (incl. newlines and tabs) into a single regular space ' ',
                      and HTML entity decoding.
                      To switch this behavior off, set self.html=False in your pattern,
                      either permanently in the subclass definition or before calling Pattern.match().
                      Compare the two calls below, the 1st one executed with the default setting of html=True,
                      and the 2nd one with 'html' changed to False:
                      
                      >>> pat = Pattern('<td>{CELL}</td>')
                      >>> pat.match1('<td>  apples \\n &amp; oranges  </td>')
                      u'apples & oranges'
                      >>> pat.html = False
                      >>> pat.match1('<td>  apples \\n &amp; oranges  </td>')
                      '  apples \\n &amp; oranges  '
                      
                      However, if the variable contains 3-dots '...', no cleansing is performed 
                      regardless of 'html' setting:
                      
                      >>> match1('<tr> {ROW ...} </tr>', '<tr><td>  apples \\n &amp; oranges  </td></tr>')
                      '<td>  apples \\n &amp; oranges  </td>'

{VAR~regex}        -- If the variable name is followed immediately by a tilde (~), the remaining part 
                      of variable definition is treated as a raw regex - not redex - expression.
                      This is useful for matching arbitrary regex sub-patterns, 
                      to handle complex cases not covered by basic redex syntax.
                      Watch out: any spaces after '~' and before '}' are treated as part of the regex.
                      The regex can't contain '}' char itself, to avoid ambiguity with terminating '}'
                      (use {~regex~} syntax instead).

                      >>> match1('<a ./{ID~\d+}>', '<a href="http://www.flickr.com/photos/atelier/738724">')
                      '738724'

{~regex~}          -- Braces with inner tildes and an expression inside denote an unnamed regex pattern:
                      the regex is matched just like a {VAR~regex} variable, but the matched string is not extracted.
                      Enables inclusion of arbitrary regex sub-patterns in a redex pattern.
                      In contrary to {VAR~regex} syntax, here the right brace '}' is allowed inside 'regex',
                      and only a tilde-brace pair '~}' is disallowed.
                      This enables the use of {min,max} construct inside the regular expression.
                      If you want to use a regex containing '~}', split it into two separate regexes.

                      >>> match1('<a ./{ID {~\d{6}~}}>', '<a href="http://www.flickr.com/photos/atelier/738724">')
                      '738724'

{* expr} {+ expr}  -- Zero-or-more and one-or-more repetitions of a given redex expression 'expr'.
                      The expression can contain nested variables {* ..{X}..}, and/or hold a name itself: {*ITEM ...}.
                      In both cases, the returned value of each variable is a *list* of strings 
                      extracted by all repetitions rather than a single string.
                      
                      >>> match('{* ...<td>{ARTICLE}</td><td>{PRICE}</td>}', 
                      ...       '<tr><td>Pen</td><td>$3.50</td></tr> '
                      ...       '<tr><td>Pencil</td><td>$2.00</td></tr> '
                      ...       '<tr><td>Eraser</td><td>$1.50</td></tr> ')
                      {'ARTICLE': ['Pen', 'Pencil', 'Eraser'], 'PRICE': ['$3.50', '$2.00', '$1.50']}

{*+ ...} {++ ...}  -- You can use possessive quantifiers and atomic grouping to achieve finer control over 
{> ...}               regex backtracking and optimize the speed of regex matching. For more information, see:
                          www.regular-expressions.info/possessive.html
                          www.regular-expressions.info/atomic.html

[optional]         -- Square brackets [expr] enclose an optional expression: matched if possible,
                      but skipped if a matching - starting at the current position, where the preceeding match ended
                      - can't be found. If skipped optional expression contained variables, their values will be None.
                      Optional expressions and variables can be nested in each other.
                      
                      >>> pat = '<li> {ENTRY [<img src="{AVATAR}">] {NAME}} </li>'
                      >>> match(pat, '<li><img src="http://domain.com/john.png"> John Smith </li>')
                      {'ENTRY': '<img src="http://domain.com/john.png"> John Smith', 'AVATAR': 'http://domain.com/john.png', 'NAME': 'John Smith'}
                      >>> match(pat, '<li> Ken Edwards </li>')
                      {'ENTRY': 'Ken Edwards', 'AVATAR': None, 'NAME': 'Ken Edwards'}
                      
                      Be careful when using optional expressions. They may often exhibit unexpected behavior,
                      caused by the complex structure of backtracking during the regex matching process.
                      The possibily of finding a match depends on what preceeding string (prefix)
                      has been matched currently. Moreover, the necessity to match actual data instead of 
                      an empty string depends on what next expression follows after the optional block
                      and whether it enforces the optional to find a long match.
                      
                      Let's see an example.
                      
                      >>> match('. [tail]', "head tail")
                      ''
                      
                      Here, one might expect that the pattern would match entire "head tail" string.
                      This is not the case because the dot '.' performs a *lazy* match 
                      (matches as few characters as possible), so it first tries to match an empty string,
                      which succeeds, than matches the space ' ' to an empty string, too,
                      and only than tries to match '[tail]' to "head tail", which also succeeds (!),
                      by (unexpectedly) ignoring the optional expression in its entirety.
                      
                      To solve this problem, you can either replace the dot with a tilde, 
                      which performs a more constraint match - not only that it matches exactly one non-empty word,
                      but also the match is greedy and tries to match as many characters as possible:
                      
                      >>> match('~ [tail]', "head tail")
                      'head tail'
                      
                      Alternatively, you can move the dot and put it inside the optional expression:
                      
                      >>> match('[. tail]', "head tail")
                      'head tail'
                      
                      It is a good general rule to:
                      - use the most constrained expression that's possible in a given place, 
                        when choosing between '~', '.', '..' and '...'
                      - never put dots before optional or repeated expressions: [...], {*...},
                        but rather include them inside the expression.
                      

{.} {{} {}}        -- To match an occurence of a redex-special character, enclose it in braces;
{~} {[} {]}           a brace itself can also be enclosed:

                      >>> print match('{.} {{} {}} {~} {[} {]}', '. { } ~ [ ]')
                      . { } ~ [ ]
                      
                          
VALUE CONVERSIONS.

All extracted variables are strings by default. You can request that they are automatically casted onto other data types,
and/or their values are converted, by defining 'convert' property of the Pattern, which is a dictionary of variables names
and corresponding types/converter functions that shall be applied to extracted values before returning them to the caller.

    >>> class Pat(Pattern):
    ...     pattern = "<a {URL /comment/{USER}/{ID}}> {DATE} </a>"
    ...     convert = {'URL': url, 'USER': url_unquote, 'ID': pint, 'DATE': pdate}
    ...
    >>> Pat().match('<a href="/comment/billy%20the%20kid/34"> July 25, 2015 </a>')
    {'URL': '/comment/billy%20the%20kid/34', 'DATE': datetime.date(2015, 7, 25), 'USER': 'billy the kid', 'ID': 34}

The 'url' converter is special, because it accepts an additional parameter upon conversion: 
the URL of the page being scraped, 'baseurl', that is used for converting relative URLs in the page to absolute ones. 
When calling Pattern.match(), you can specify a 'baseurl' and it will be forwarded to all 'url' converters:

    >>> items = Pat().match('<a href="/comment/billy%20the%20kid/34"> July 25, 2015 </a>', 
    ...                     baseurl = "http://www.domain.com/home")
    >>> items['URL']
    'http://www.domain.com/comment/billy%20the%20kid/34'

None values (from optional expressions) are NOT passed through converters but returned as-is.
If a variable appears inside repeated expression, like {+ ..{NAME}..}, and multiple strings are extracted,
the converter is applied to EACH string independently rather than to a list.

Keys in 'convert' dictionary can comprise multiple names, for brevity (multi-keys). For example:
    {"PRICE CHANGE VOLUME": pdecimal}
Additionaly, keys can contain '*' wildcard to match all names of a given form, like:
    {"URL_*": url}

Standard converters provided along with Pattern include:
    url, url_unquote, pdate, pdatetime, pint, pfloat, pdecimal, percent.

You can use these ones and/or define your own. In the latter case, each converter should be a function that takes 
an extracted string and casts/converts it to a destination type/value. In 'convert' dict, you can also specify a class
instead of a function, which will instantiate objects of this class upon conversion, passing the raw value 
to this class'es __init__().

Moreover, you can use patterns as converters themselves. The inner pattern (converter) will be applied to the extracted 
string passed as an input document, and a whole dictionary (or object) of extracted values will be returned 
as a value of the variable of the outer pattern. Effectively, this mechanism allows patterns to be nested in each other.

Last but not least, you can implement custom post-processing of all extracted values at once
by overloading Pattern's 'epilog' method. This method takes a dict of extracted items after all conversions,
together with the original document, and either returns a new dict of items, or modifies the input dict in-place.
Epilog can not only change existing values, but also remove variables or add new ones to the result,
for instance, by implementing additional custom extraction based on the provided document object.
See Pattern.epilog() for details.


EMBEDDED UNIT TESTS.

Web pages evolve. Sometimes their structure changes only slightly, due to minor fixes and updates in their design,
layout or contents. Another time, the structure may get redesigned entirely from scratch.
In any case, the corresponding pattern must be updated accordingly to handle new structure.
The critical thing to watch out is whether the updated pattern still handles all special cases of 
the page's appearance - all those special cases that were carefully analysed and implemented
in the first version of the pattern, and must not get lost now, when the pattern is being updated.

In order to enable easy future verification of the pattern's behavior,
you can embed *unit tests* inside its class. 
Each test consists of a pair of class attributes named 'textX' and 'goalX',
with X being an integer (1,2,... for consecutive independent tests),
'testX' being a snippet of text that shall be used as an input document for pattern matching,
and 'goalX' being the expected output: a dictionary of extracted variables and their values.
When specifying the goal, you can use a shorthand for string-valued variables and write, for instance:
    goal1 = "NAME1 value1", "NAME2 value2", {...(remaining non-string values)...}
instead of:
    goal1 = {"NAME1": "value1", "NAME2": "value2", ...}

Whenever Python comes across a definition of a new Pattern subclass, it automatically invokes all tests 
present inside. This behavior is implemented in Pattern's custom metaclass, __Pattern__,
and is launched just after a new class is created (the class itself, not an instance!).
Upon execution, all failing tests will print detailed log to stdout. Successful tests will stay quiet.

If a given test doesn't have a goal defined (the attribute 'goalX' is missing),
the extracted dictionary of values is printed to stdout instead of being passed to verification.
Moreover, it is printed in a form suitable for direct inclusion in corresponding goalX.
Thus, when writing a new test, you can first: write testN without goalN, to see what output is produced 
by the pattern; verify manually that the output is correct; and only then copy-paste it as goalN = ...,
to keep a record for future reference when the pattern needs to be updated.
   
If you want to switch off the tests, set autotest=False inside a given Pattern subclass,
or Pattern.autotest=False to change global behavior.


PRACTICAL GUIDELINES.

When implementing redex patterns, it is helpful to understand what functional types of expressions can occur in them:
 * STATIC EXPRESSION - Any expression that behaves similarly to static text, in that it obligatory matches 
                       a non-empty string. Static expression may contain variables and tildes ~, 
                       but not optional blocks [], repetitions {* } {+ } or lazy fillers (dots, explicit or implicit).
                       Static expressions serve as pillars that support correct specification of all parts of the pattern
                       and improve speed of pattern matching.
 * REFERENCE POINTS  - Static expressions located outside optional blocks, in different places of the pattern. 
                       They help position optional subpatterns relative to other parts of the document and ensure 
                       global consistency of the pattern.
 * GUARDS            - Static expressions that surround a variable, like "<td>" and "</td>" in "<td>{DATA}</td>". 
                       They position the variable precisely in relation to neighboring text and protect 
                       against *over-matching*.
 * CLOSING ANCHOR    - Static expression inside optional block, located at its end, like "</p>" in: "[... </p>]".
                       Enforces maximum match of the inner pattern of optional block and protects against *under-matching* 
                       of variables and fillers contained inside [].

When devising a redex pattern, it is best to start with a real snippet of HTML code extracted from the actual document
to be parsed, and then convert it step by step to a redex pattern, by removing unnecessary parts, 
inserting fillers and optionality markers, reducing repeated subexpressions to {* } or {+ } blocks,
and replacing meaningful parts with variables.
Usually, pattern matching is applied to a specific node in HTML syntax tree of the document,
thus it is easiest to extract the snippet with tools like FireBug, by visually selecting the page block to be processed,
and then copy-pasting all its raw HTML contents.

A typical procedure for converting HTML snippet to a redex may look like this:
 - In the snippet, replace all occurences of values to be extracted with {NAME} or {NAME subpattern} or {*NAME ...}.
   Put explicit 3-dots '...' inside {} wherever text containing tags must be matched 
   (2-dots '..' are put by default when no sub-pattern is given).
 - Remove unnecessary parts of text by replacing them with ~ (word) or . (non-space sequence) 
   or .. (non-tag sequence) or ... (any sequence):
   - tagged parts of the document replace with ... (matches all chars, including tags)
   - untagged text between tags replace with .. (matches all chars except <>)
   - sequence of non-space chars replace with . (matches all except <> and spaces)
   - regular words comprising only alpha-numeric characters replace with ~ 
   - tag attributes list inside <> replace with a space, or ., or .., or nothing (the latter works best in many cases)
   - tag name inside <> replace with a dot '.'
   Ensure that every variable {} is still surrounded by enough amount of static text (GUARDS / REFERENCE POINTS) 
   to uniquely identify its location in the document.
 - Insert a space wherever 1+ spaces may occur in the document between remaining static parts.
   Neighboring tags don't need a space: whitespaces between > and < are matched implicitly.
 - Mark non-obligatory parts of the pattern by surrounding them with []. If the part starts with . or .. or ..., 
   include them inside [] if only possible (do this even if the optional block is the 1st sub-expression of the pattern).
   Remember to put a CLOSING ANCHOR at the end of [], to avoid under-matching of the optional block.
 - If pattern matching works too slow - which may happen particularly on negative examples, due to more laborious 
   regex backtracking needed to check all possible matchings paths - add more REFERENCE POINTS.
   If the problem persists (rare case), try to manually limit backtracking by applying atomic grouping {>...}
   or possessive quantifiers {*+ }, {++ } - but note that these constructs change semantics of the pattern,
   so you should use them with care, only when you are certain that they don't break the pattern.
 
Tips & tricks:
 - Most of the time, pattern matching is LAZY: it matches as few characters as possible, 
   and quite often an empty string turns out to be a correct match.
   This happens not only with entire pattern, but also with its sub-expressions, esp. when [...] is used.
   Thus, you should always put GUARDS at the end of (sub)expressions: static pieces of text 
   or other obligatory expressions that would force the preceding expression to match as much as possible,
   until the guard is matched, too. Without guards, you may encounter unexpected behavior: 
   the pattern can miss parts of the document despite they ARE present there.
 - When extracting URLs or their portions, ALWAYS use url() or url_unquote() converters, 
   to ensure proper unquoting of extracted strings and convert relative URLs to absolute ones.
 - When implementing a new pattern, you can set verbose=True inside the class to request that debug information
   is printed out when the pattern gets defined and compiled. The print out includes list of all variables detected 
   and the regex string produced from compilation.
 - Troubleshooting. When the pattern doesn't work, try removing a part of it, say the 2nd half,
   and check if it works in the shorter form. If not, cut it again and again. In this way, you can easily track down
   the place which causes problems.
 - If you need to test regexes, for debugging purposes, check online interactive sites like http://gskinner.com/RegExr/


---
Dependencies: waxeye 0.8.0, regex 2013-03-11.
This file uses an improved and extended version of 're' module, the 'regex' - see http://pypi.python.org/pypi/regex

---
This file is part of Nifty python package. Copyright (c) 2009-2014 by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
"""

import re, regex as re2 #@UnresolvedImport
import copy, urllib
from collections import namedtuple
from datetime import datetime

# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages;
# only when executed as a standalone file, for unit tests, do an absolute import
if __name__ != "__main__":
    from .. import util
    from ..util import isstring, islist, isdict, istuple, issubclass, subdict, prefix, ObjDict, lowerkeys, classname
    from ..text import merge_spaces, decode_entities, regexEscape, html2text
    from ..web import urljoin, xdoc
    from ..parsing import parsing
else:
    import nifty.util as util
    from nifty.util import isstring, islist, isdict, istuple, issubclass, subdict, prefix, ObjDict, lowerkeys, classname
    from nifty.text import merge_spaces, decode_entities, regexEscape, html2text
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
        """Global semantic data collected by a Context object when passing through the tree. 
        Kept in a separate class so that context.copy() still references the same data as 'context'."""
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

    parser  = pattern_parser.PatternParser()
    Context = Context

    class node(BaseTree.node):
        def analyse(self, ctx):
            if len(self.children) == 0: return ctx
            if len(self.children) == 1: return self.children[0].analyse(ctx)
            longfill = False
            for c in self.children: 
                ctx2 = c.analyse(ctx.copy())
                longfill |= ctx2.longfill
            ctx.longfill = longfill
            return ctx

    static = BaseTree.static
    const = BaseTree.const
    expr = node                                     # "Generic expression - a sequence of subnodes. Consecutive expressions can be flatten: merged into one."

#    class xwordfill(node):
#        def compile(self): return r'\w+' #r'[^<>\s/\\]*'                                            #@ReservedAssignment
#    class xshortfill(node):
#        def compile(self): return r'[^<>]*?'                                                        #@ReservedAssignment
#    class xlongfill(node):
#        display = "..."
#        def compile(self): return r'.*?'                                                            #@ReservedAssignment
#        def analyse(self, ctx):
#            ctx.longfill = True
#            return ctx
#    class xjustadot(static):
#        def compile(self): return r'\.'                                                             #@ReservedAssignment
    
    class xtilde(const): value = r'[\w~]+' #r'[^<>\s]*'
    class xdot1(const): value = r'[^<>\s]*?'
    class xdot2(const): value = r'[^<>]*?'
    class xdot3(const):
        value = r'.*?'
        def analyse(self, ctx):
            ctx.longfill = True
            return ctx
    class xspecial(static):
        def compile(self): return regexEscape(self.value)                                           #@ReservedAssignment

    class space(node):
        display = " "
    class xspace0(space):
        "Maybe-space. Can match some spaces, but not obligatory"
        def compile(self): return r'\s*'                                                            #@ReservedAssignment
    class xspace1(space):
        "Must-space. Matches 1 or more spaces."
        def compile(self): return r'\s+'                                                            #@ReservedAssignment
    class xspaceX(space):
        """In-tag filler (space on steroids): like xdot2 matches any sequence of chars except <>, 
        if only the 1st and last char are in-tag separators: one of ['"\s=].
        Alternatively, matches a regular sequence of 0+ spaces, like xspace0.
        """
        def compile(self): return r'''(\s*|(['"\s=][^<>]*?['"\s=]))'''                              #@ReservedAssignment

    class xword(static):
        def compile(self): return regexEscape(self.value)                                           #@ReservedAssignment
    class xstatic(expr): pass
    class xwordB(expr):
        """Like 'word', but modifies output regex to allow spaces around '=' and substitution of " with ' or the other way round."""
        def compile(self):                                                                          #@ReservedAssignment
            r = super(Tree.xwordB, self).compile()
            r = r.replace('=', r'\s*=\s*')
            r = r.replace('"', "'")
            r = r.replace("'", r'''("|')''')
            return r

    class xexpr(expr): pass
    xexprA = xexprB = xexpr
    
    class xtagspecial(static):
        def compile(self): return regexEscape(self.value)                                           #@ReservedAssignment
    class xtagname(node): pass
    class xnoname(node):
        "Match any word as a tag name, with optional leading '/'; 'tag' node below will ensure that the name matched here is followed only by space or end of tag."
        display = "."
        def compile(self): return r'/?\w+'                                                              #@ReservedAssignment
    class xtag(node):
        name = expr = closing = None
        def init(self, tree, waxnode):
            self.tagspecial, self.name, self.expr, self.closing = self.children[:4]
            
        def __str__(self): return '<%s%s%s%s' % (self.tagspecial, self.name, prefix(' ', self.expr), self.closing)
        def compile(self):                                                                              #@ReservedAssignment
            def comp(node): return node.compile() if node else ''
            lead = r'(|(?<=>)\s*)'              # backward-check for a preceeding tag, then match 0+ spaces between both tags
            gap  = r'(?=\s|/|>)'                # forward-check for a separator between the tag name and attribute list, or a tag end if no sep 
            end  = r'''['"\s=]'''
            spaceL = r'([^<>]*?%s|)' % end      # match attributes on the left of 'expr', or nothing
            spaceR = r'(%s[^<>]*?|)' % end      # match attributes on the right of 'expr', or nothing
            expr = comp(self.expr)
            expr = spaceL + expr if expr else ''
            return lead + '<' + comp(self.tagspecial) + comp(self.name) + gap + expr + spaceR + comp(self.closing)
    
    class xrepeat(static): pass
    class xregex(static): pass
    class xvregex(static): pass
    class xvarname(static): pass
    class xvar(node):
        "A {xxx} element - named group and/or raw regex. If doesn't contain any expression, '..' is used; put '...' inside {} to get a longfill."
        repeat = name = regex = expr = None
        
        def init(self, tree, waxnode):
            for c in self.children:
                if c.type == 'repeat': self.repeat = c
                elif c.type == 'varname': self.name = c
                elif c.type == 'vregex': self.regex = c
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
    
    class xoptional(node):
        "A [xxx] element. Resolves into a *greedy* optional match of 'xxx' pattern."
        def init(self, tree, waxnode):  self.expr = self.children[0]
        def __str__(self):              return '[%s]' % self.expr
        def compile(self):              return r'(%s)?' % self.expr.compile()                       #@ReservedAssignment
            
    xoptionalA = xoptionalB = xoptional

    class xatomic(node):
        "A {> xxx} element. Resolves into atomic grouping (?>...) that limits backtracking during regex matching, see: www.regular-expressions.info/atomic.html."
        def init(self, tree, waxnode):  self.expr = self.children[0]
        def __str__(self):              return '{> %s}' % self.expr
        def compile(self):              return r'(?>%s)' % self.expr.compile()                      #@ReservedAssignment


########################################################################################################################################################
###
###  PATTERN
###

class MetaPattern(type):
    "Implements execution of Pattern's unit tests upon definition of subclasses, as well as combining Pattern classes by '+' operator."
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
    Redex pattern.
    
    >>> p = Pattern("  {* ... {A}}{B}  ")
    >>> v = p.semantics.variables; A, B = v['A'], v['B']
    >>> A.longfill, A.repeated, B.longfill, B.repeated
    (False, True, False, False)
    >>> print p.regex.pattern
    (.*?(?P<A>[^<>]*?))*(?P<B>[^<>]*?)\s*
    
    >>> p2 = Pattern("{> [ala]} ala")
    >>> print p2.regex.pattern
    (?>(ala)?)\\s*ala
    >>> print p2.match1("ala")
    None
    
    >>> p = Pattern('<a href="{URL}"></a>')
    >>> print p.match1('<A href="http://google.com"></A>')
    http://google.com
    """
    __metaclass__ = MetaPattern     # responsible for executing unit tests defined at class level in subclasses
    MISSING = object()              # in internal unit tests, a token that indicates that true test outcome (goal) is undefined
    
    # input parameters
    pattern   = None    # source redex code of the patter, compiled into regex during __init__ and then matched against documents in match*()
    path      = None    # optional XPath string; if present, pattern will be matched only against node(s) selected by this path - document must be an xdoc or will be parsed as HTML
    convert   = {}      # dict of converters or types that the extracted values shall be casted onto, ex: {'DATE': pdatetime, 'PRICE': pfloat}
    extract   = {}      # stand-alone extractors: functions that take an entire document and return extracted value or object for a given item; dict
    case      = False   # shall regex matching be case-sensitive (True)? INsensitive (False) by default
    tolower   = False   # shall all item names be converted to lowercase at the end of parsing?
    html      = True    # shall match() perform HTML entity decoding and normalization of spaces in extracted items? done before extractors/converters. May produce <unicode>!
    mapping   = {}      # mapping of item names, for easier integration with other parts of the application (currently unused!)
    strtype   = unicode # what type of string to cast the document onto before parsing; this determines also the type of returned result strings
    dicttype  = ObjDict # what type of dictionary to return; ObjDict allows .xxx access to values in addition to standard ['xxx']
    model     = None    # class to be used as a wrapper for the dictionary of matched fields passed in kwargs: __init__(**items) 
    verbose   = False   # if True, Pattern.__init__ will print out debug information
    autotest  = True    # if True, unit tests (see below) will be executed automatically upon class declaration

    # output variables
    tree      = None    # syntax tree of the 'pattern', before compilation to regex
    regex     = None    # regex object compiled from 'pattern' with the enhanced 'regex' module; check regex.pattern to see what regex expression was produced from a given redex
    semantics = None    # Context.Data object with global information about the pattern tree, collected during semantic analysis
    variables = None    # list of names of variables present in the pattern, extracted from 'semantics'
    
    # optional unit tests defined in subclasses, named 'testN' and 'goalN', or just 'testN' (no ground truth, only print the output)
    # ... 

    # TODO: implement error detection: binary search for the smallest element of the pattern that causes it to break on a given input string

    def __init__(self, pattern = None, **kwargs):
        if self.verbose: print classname(self)
        
        if pattern is not None: self.pattern = pattern
        if self.pattern is None: self.pattern = self.__class__.__doc__          # subclasses can define patterns in pydocs, for convenience
        params = subdict(kwargs, "extract convert case tolower html".split())
        self.__dict__.update(params)

        self._compile()                                                         # compile self.pattern to self.regex

        # decode compact notation of keys in 'convert': split multi-name keys, resolve wildcard keys
        self.convert = util.splitkeys(self.convert)
        self.variables = self.semantics.variables.keys()
        for name, conv in self.convert.items():
            if issubclass(conv, Pattern): raise Exception("A Pattern class used as a converter: %s. Use an instance instead: %s()." % ((conv.__name__,)*2))
            if '*' not in name: continue
            pat = name.replace('*', '.*') + "$"
            for name2 in self.variables:
                if re.match(pat, name2): self.convert[name2] = conv
            del self.convert[name]
        
        if self.verbose: 
            print " variables:", self.variables
            print " regex:", self.regex.pattern
        
    def _compile(self):                                                                                      #@ReservedAssignment
        "Translate the original reDex self.pattern into a regex pattern and compile to regex object."
        self.tree = Tree(self.pattern)
        regex, self.semantics = self.tree.compile()
        try:
            #regex = re.compile(regex, re.DOTALL)       # DOTALL: dot will match newline, too
            # (!) correct pattern which causes error in standard 're' module: re.compile("((?P<TIME>[^<>]*?))?")
            
            # consume leading whitespace if necessary; remember the match under the __lead__ name to cut it off from the final result
            #space = r"\s*?"  #r"(?P<__lead__>\s*?)"
            #if not regex.startswith(space): regex = space + regex
            flags = re2.DOTALL | re2.VERSION1 | (re2.IGNORECASE if not self.case else 0)
            self.regex = re2.compile(regex, flags)
        except:
            print "exception raised when compiling regex:"
            print regex
            raise
        
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
                else: var,val = split[0], ""
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
        
        print "%s.test%s," % (classname(self), testID),
        
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
        "Postprocessing of variables extracted from 'doc': entity decoding, merging spaces, type casting, converting to object. Can modify 'items'."
        
        def rawtext(val, isstring = None):
            "Convert HTML text to raw text: merge spaces, decode HTML entities. Outputs <unicode> (!) because entities may represent Unicode characters."
            if not isstring and islist(val): return [rawtext(s,True) for s in val]
            return decode_entities(merge_spaces(val))
            #return self.strtype(decode_entities(merge_spaces(val)))
            
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
            
            # convert extracted HTML text to raw text (merge spaces, decode HTML entities), but only for variables 
            # which don't contain longfills "..." in their pattern: longfills match HTML tags, therefore simple cleaning
            # can be incorrect, because entity decoding should be accompanied by tag stripping  
            if self.html:
                var = self.semantics.variables
                for name, val in items.iteritems():
                    if val and not var[name].longfill: items[name] = rawtext(val)
            
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
        """Matches the pattern against document 'doc'. On successful match returns self.dicttype (ObjDict by default) 
        with extracted and converted values, possibly wrapped up in model() if not-None 
        ('model' can be given here as function argument, or as a property of the object or class); 
        or the entire matched string if the pattern doesn't contain any variables. None on failure.
        Typically 'doc' is a string, but it can also be any other type of object convertable into <str> or <unicode>
        - this is useful if custom extractors are to be used that require another type of object. 
        If 'baseurl' is given, all extracted URLs will be turned into absolute URLs based at 'baseurl'.
        'testing': True in unit tests, indicates that final processing (item names renaming, class wrapping) 
        should be skipped and 'path' shall not be used.
        """
        path = path or self.path                            # select XPath nodes if requested to do so
        if path and not testing: 
            if isstring(doc): doc = xdoc(doc)
            doc = doc.node(path)
        
        if not isinstance(doc, basestring):                 # convert the doc to str/unicode from an object
            doc = self.strtype(doc)
        doc = doc.lstrip()                                  # remove leading whitespace
        #doc = self.strtype(doc).lstrip()                   # convert the doc to a string and remove leading whitespace
        
        if self.variables:                                  # extract variables?
            items, _ = self._matchRaw(doc)
            return self._convert(items, doc, model, baseurl, testing)
        else:                                               # extract the entire matched part of the document
            match = self.regex.match(doc)
            if match is None: return None
            return match.captures()[0]
    
    def match1(self, *args, **kwargs):
        """Shorthand for patterns that extract exactly 1 variable: returns *value* of this variable, 
        as an atomic value rather than a dictionary. None when no matching found. 
        If no variables are present, returns the string matched by entire pattern, 
        like if the pattern were enclosed in {VAR ...}.
        """
        vals = self.match(*args, **kwargs)
        if isstring(vals) or vals is None: return vals
        if len(vals) != 1: raise Exception("Pattern.match1: pattern contains too many variables (%d), should only contain one" % len(vals), self.variables, vals)
        return vals.values()[0]
        
    
    def matchAll(self, doc, path = None, **kwargs):
        """Like match(), but finds all non-overlapping matches of the pattern in 'doc', 
        not only one and not necessarily anchored at the beginning of the doc. 
        Returns a list of result sets (dicts/objs), empty if no match was found.
        If path or self.path is present, matches the pattern to all nodes selected by this path 
        (nodes can be nested in each other if the path selects so), ignoring the unmatching nodes.
        """
        path = path or self.path
        if path: return self._matchAllXPath(doc, path, **kwargs)
        return self._matchAllRegex(doc, **kwargs)
        
    def _matchAllRegex(self, doc, **kwargs):
        data = []; pos = 0
        udoc = self.strtype(doc) if not isinstance(doc, basestring) else doc                # convert the doc to str/unicode from an object        
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

def match(pat, doc, **kwargs):
    """For easy construction of 1-liners that match short custom pattern (string 'pat') to a given text. 
    Watch out: the pattern is compiled from scratch on every call, which can be very inefficient
    if the pattern is used many times."""
    return Pattern(pat).match(doc, **kwargs)

def match1(pat, doc, **kwargs):
    "Like match(), but calls Pattern.match1 instead of Pattern.match, to extract a single variable, without its name."
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
###  Standard converters for use in Pattern.convert
###

html2text = html2text               # use html2text in Pattern.convert if you want extracted HTML code to be converted into raw text
from decimal import Decimal         # for parsing floating-point numbers without rounding errors

def url(s, baseurl):
    """Turn the (relative) URL 's' into an absolute URL anchored at 'baseurl'. Do NOT unquote! 
    (If you need unquoting, do it manually afterwards or use url_unquote() instead). 
    When used in Pattern.convert, 'baseurl' will be supplied at parsing time by the match() method itself."""
    if baseurl is None: return s
    return urljoin(baseurl, s)

def url_unquote(s, baseurl = None):
    """Unquotes the URL 's'. Optionally can also perform absolutization like url(), 
    but NOT when used in Pattern.convert (match() calls url_unquote with baseurl=None, unlike url()).
    Use ALWAYS when extracting portions of text (IDs, names) from href anchors - 
    the portions which won't be used as URLs themselves and MUST be properly unquoted,
    which should be done on entire URL, before the portion is extracted."""
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
    if ' ' in s: 
        if s[0].isdigit(): return check("%d %B %Y", "%d %b %Y")            # 25 November 2010; 25 Nov 2010
        elif ',' in s: return check("%B %d, %Y", "%b %d, %Y")              # November 25, 2010; Nov 25, 2010
        else: return check("%B %Y", "%b %Y")                               # November 2010; Nov 2010
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
    
