# -*- coding: utf-8 -*-

"""
Functions for:
1) generating character n-grams from Wikipedia pages, to find most frequent n-grams later on;
2) matching most frequent n-grams in a text and finding coverage [%], as an indicator of text quality.

Usage from a folder where "nifty" package is visible:

1)  python -m nifty.algo.ngrammer wikipedia pl 10 10000000 > ngrams_pl_10.txt
    cat ngrams_pl_10.txt | env LC_ALL=C sort | uniq -c | sort -n
    cat ngrams_it_10.txt | env LC_ALL=C sort | uniq -c | sort -nr | head -n300000 > freq_ngrams_it_10.txt

2)  python -m nifty.algo.ngrammer score document.txt ...


This code is compatible with Python 2 & 3.

@author:  Marcin Wojnarski
@contact: mwojnars@ns.onet.pl
"""


from __future__ import print_function
import os, sys, re, random, unicodedata, chardet, click
from itertools import islice
from collections import namedtuple
from six.moves import urllib
from glob import glob

if __name__ != "__main__":
    from ..text import merge_spaces, html2text_smart
else:
    from nifty.text import merge_spaces, html2text_smart


PY3  = (sys.version_info.major >= 3)

PATH = os.path.dirname(__file__) + '/'

RE   = re.compile


#####################################################################################################################################################
#####
#####  WIKIPEDIA SCRAPING & NGRAMS EXTRACTION
#####

class Wikipedia(object):
    """
    Scraper of Wikipedia pages.
    """
    
    @classmethod
    def get_random_page(cls, lang = 'pl'):
        
        resp = urllib.request.urlopen('http://%s.wikipedia.org/wiki/Special:Random' % lang)
        status = getattr(resp, 'status', None) or getattr(resp, 'code')
        if status == 200:
            return resp.read()
        else:
            raise Exception("Failed to open page, status code: %s" % resp.status)
    
    @classmethod
    def stream_pages(cls, lang = 'pl'):
        
        while True:
            try:
                page = cls.get_random_page(lang)
                enc  = chardet.detect(page)['encoding']
                page = page.decode(enc)
                yield page
            
            except Exception as ex:
                print(ex)
                
            
    @classmethod
    def stream_plaintext(cls, lang = 'pl', truncate = True,
                           _frame  = RE(r'(?uis)^.*</head>|</html>.*$'),
                           _script = RE(r'(?uis)<script(?:\s+[^>]*)?>.*?</\s*script>'),
                           _style  = RE(r'(?uis)<style(?:\s+[^>]*)?>.*?</\s*style>'),
                           _mw_cat = RE(r'(?uis)<div[^>]*\bcatlinks\b.*'),                      # MediaWiki categories box
                           _mw_nav = RE(r'(?uis)<div[^>]*mw-data-after-content.*'),             # MediaWiki navigation box (left panel) and footer
                           _block  = RE(r'(?uis)<h\d\b.*?</h\d>|<p\b.*?</p>|<dl\b.*?</dl>|<li\b.*?</li>'),
                           _latin  = RE(r'(?uis)[a-zA-Z\d\{\}/\\\&|\.,;*\-' u'–°' r']+|\(\s*\)'),
                           _brackets = RE(r'(?uis)\[.*?\]|\(\s*\)'),
                           ):
        
        def as_plaintext(block):
            text = html2text_smart(block)
            text = _brackets.sub(' ', text)
            
            if lang in ('zh','jp'):
                text = _latin.sub('', text)
            
            text = merge_spaces(text)

            return text
        
        for page in cls.stream_pages(lang):
            
            #print(page)
            #print('---' * 30)
            
            if truncate:
                page = _frame .sub(' ', page)
                page = _script.sub(' ', page)
                page = _style. sub(' ', page)
                page = _mw_cat.sub(' ', page)
                page = _mw_nav.sub(' ', page)
            
            texts = map(as_plaintext, _block.findall(page))
            texts = filter((lambda t: len(t) > 1), texts)
            
            yield '\n'.join(texts)


    @classmethod
    def stream_blocks(cls, lang = 'pl', min_len_block = 100, drop_biblio = True,
                      _biblio = RE(r'(?uis)^\s*' u'\u2191' r'.*|^\s*\^.*'),              # ↑ ^ ..... (bibliography entries)
                      ):
        
        for text in cls.stream_plaintext(lang):
            
            for block in text.split('\n'):
                if None != min_len_block > len(block): continue
                if drop_biblio and _biblio.match(block): continue
                yield block
                

    @classmethod
    def stream_ngrams(cls, lang, length, limit = None, *args, **kwargs):
        
        length = int(length)
        if limit: limit = int(limit)
        
        count = 0
        for block in cls.stream_blocks(lang, *args, **kwargs):
            block = block.lower()
            for i in range(len(block)-length+1):
                yield block[i:i+length]
                count += 1
                if count % 1000 == 0: print(count, file = sys.stderr)
                if None != limit <= count: return
                      

#####################################################################################################################################################
#####
#####  TEXT SCORING
#####

class Scorer(object):
    """
    In method score(), Scorer takes a text and finds in it all overlapping matches of frequent n-grams (loaded in __init__),
    to calculate: (1) the fraction of text matched (after lowercase + spaces merged); (2) no. of different unique n-grams matched.
    """
    
    verbose = True
    
    def __init__(self, minfreq = 50, lang = '*', length = '*', path = PATH + "ngrams/"):
        
        # filtering out strings like:
        #   athbf {x}      \mathbf {x
        #   \color {gr     r {gray}{0    &\color {g    {gray}{0}&
        #   {\displays     playstyle     aystyle {\     style \mat
        #   \operatorn     0 0 0 0 0     x • xi • x
        re_invalid = RE(r"\\mat|\\col|\\disp|\\oper|\{|\}|\&\\|displayst|aystyle|0 0 0|" u"x • x|codec can't decode")
        
        def invalid(s):
            return re_invalid.search(s)
        
        if self.verbose: print("Loading frequent n-grams (phrases)...")
        
        filenames = glob(path + "freq_ngrams_%s_%s.txt" % (lang, length))
        if not filenames: raise Exception("No file with ngrams found")
        
        phrases = []
        for fn in filenames:
            
            target_len = int(fn.rsplit('.',1)[0].rsplit('_',1)[1])
            f = open(fn, 'rt')
            
            for line in f:
                if not PY3: line = line.decode('utf-8')
                line = line.lstrip()
                if line[-1] == '\n': line = line[:-1]
                if not line: continue
                assert ' ' in line
                
                split = line.index(' ')
                freq  = int(line[:split])
                ngram = line[split+1:]
                if len(ngram) != target_len:
                    if "codec can't decode" not in ngram:
                        print("INCORRECT LENGTH (%s): [%s]" % (len(ngram), ngram))
                else: 
                    assert len(ngram) == target_len
                
                if freq < minfreq: continue
                if invalid(ngram): continue
                
                phrases.append(ngram)
                #print(freq, '[%s]' % ngram)
                
        if self.verbose: print("Phrases (with duplicates): ", len(phrases))
        phrases = set(phrases)
        if self.verbose: print("Phrases (no duplicates):   ", len(phrases))

        if self.verbose: print("Building regex pattern...")
        phrases = map(re.escape, phrases)
        self.re_phrases = RE("(?iu)(?=(%s))" % "|".join(phrases))           # standard re.compile() works 2-3x faster than regex.compile() that allows overlapped=True in finditer()
        #print(self.re_phrases.pattern)

    def score(self, text):
        
        text = text.lower()
        text = merge_spaces(text)
        total_matched = 0
        unique = set()
        
        last = 0
        for match in self.re_phrases.finditer(text):
            phrase = match.group(1)
            start  = match.start()              # we look for OVERLAPPING matches! for this reason, we use lookahead matches in the regex and the actual match returned is always 0-length
            stop   = start + len(phrase)        # we need to take group(1) - from inside the lookahead match - to retrieve the length of the matched substring
            unique.add(phrase)
            if start < last: start = last
            assert start < stop
            
            if self.verbose:
                click.secho(text[last:start], nl = False)
                click.secho(text[start:stop], nl = False, bg = 'blue')
            
            total_matched += stop - start
            last = stop
        
        if self.verbose: click.secho(text[last:])
        
        #matches = self.re_phrases.findall(text)
        #total_matched = sum(map(len, matches))
        total_len = len(text)
        
        if self.verbose: print("Matched characters: %s of %s" % (total_matched, total_len))
        return float(total_matched) / total_len, len(unique)


#####################################################################################################################################################
#####
#####  MAIN
#####

if __name__ == '__main__':

    cmd = sys.argv[1]
    
    if cmd == 'wikipedia':
        
        for ngram in Wikipedia.stream_ngrams(*sys.argv[2:]):
            print(ngram if PY3 else ngram.encode('utf-8'))
        
    elif cmd == 'score':
        
        scorer = Scorer()
        for fname in sys.argv[2:]:
            text = open(fname).read()
            if not PY3: text = text.decode('utf-8')
            print("Matching...")
            score, unique = scorer.score(text)
            print("%.2f%% matched (%d unique frequent n-grams)" % (score * 100, unique), '-', fname)
    
    else:
        print("Unknown command:", cmd)
        
    
