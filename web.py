# -*- coding: utf-8 -*-
'''
Routines for web access, web scraping and HTML/XML processing.
External dependencies: Scrapy 0.16.4 (for HTML/XML)

TODO: possibly might replace urllib2 with Requests (http://docs.python-requests.org/en/latest/)

---
This file is part of Nifty python package. Copyright (c) by Marcin Wojnarski.

Nifty is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Nifty is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Nifty. If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import absolute_import
import os, sys, subprocess, threading
#os.environ['http_proxy'] = ''                       # to fix urllib2 problem:  urllib2.URLError: <urlopen error [Errno -2] Name or service not known> 

import urllib2, urlparse, random, time, socket, json, re
from collections import namedtuple, deque
from copy import deepcopy
from datetime import datetime
from urllib2 import HTTPError, URLError
from socket import timeout as Timeout
#from lxml.html.clean import Cleaner        -- might be good for HTML sanitization (no scritps, styles, frames, ...), but not for general HTML tag filering 

if __name__ != "__main__":
    from .util import islinux, isint, islist, isnumber, isstring, JsonDict, mnoise, unique, classname, noLogger, defaultLogger, Object
    from .text import regex, xbasestring
    from . import util
else:
    from nifty.util import islinux, isint, islist, isnumber, isstring, JsonDict, mnoise, unique, classname, noLogger, defaultLogger, Object
    from nifty.text import regex, xbasestring
    from nifty import util
    
now = time.time                 # shorthand for calling now() function, for process-local time measurement


########################################################################################################################################################################
###
###  UTILITIES
###

###  URLs  ###

def fix_url(url):
    """Add 'http' at the beginning of a URL if doesn't exist.
    Can be extended to provide character encoding, see werkzeug and: http://stackoverflow.com/a/121017/1202674
    """
    if "://" not in url[:12]:
        return "http://" + url
    return url

def urljoin(base, url, allow_fragments=True, empty=False):
    """" Extended and slightly modified version of original urlparse.urljoin, in that: 
    (1) url can be a list of URL fragments; then all of them are appended independently to the same base and a list of results is returned;
    (2) if empty=False (default!), every empty or None fragment yields None as a result instead of 'base'; incompatible with HTML standard, but convenient in crawling
    """
    if islist(url):
        if empty: return [urlparse.urljoin(base, u, allow_fragments) for u in url]
        else:     return [urlparse.urljoin(base, u, allow_fragments) if u else None for u in url]
    if empty: return urlparse.urljoin(base, url, allow_fragments)
    else:     return urlparse.urljoin(base, url, allow_fragments) if url else None
    
class ShortURL(object):
    """Encodes integers (IDs of objects in DB) as short strings: something like base-XX encoding of a number, with XX ~= 60.
    Code derived from stackoverflow: http://stackoverflow.com/questions/1119722/base-62-conversion-in-python"""
    
    BASE_LIST = "123456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"      # confusing characters left out: 0OolI
    BASE_DICT = dict((c, i) for i, c in enumerate(BASE_LIST))
    
    @staticmethod
    def encode(integer, base = BASE_LIST, degree = len(BASE_LIST)):
        ret = ''
        while integer != 0:
            ret = base[integer % degree] + ret
            integer /= degree
        return ret
    
    @staticmethod
    def decode(string, reverse_base = BASE_DICT, degree = len(BASE_DICT)):
        ret = 0
        for i, c in enumerate(string[::-1]):
            ret += (degree ** i) * reverse_base[c]
        return ret

###  Errors  ###

# HTTPError, URLError, Timeout -- standard exception classes that can all be imported from this module

def failedToConnect(ex):
    "True if exception 'ex' indicates initial connection (not server) error: no internet connection, service unknown etc."
    # recognized two types of URLError exceptions: <urlopen error [Errno -2] Name or service not known> and <urlopen error timed out> 
    return isinstance(ex, URLError) and hasattr(ex, 'reason') and (ex.reason[0] == -2 or str(ex.reason) == "timed out")


###  HTML  ###

def noscript(html, pat1 = re.compile(r"<script", re.IGNORECASE), pat2 = re.compile(r"</script>", re.IGNORECASE)):
    "Comment out all <script.../script> blocks in a given HTML text. In rare cases may break consistency, e.g., when '<script' or '/script>' text occurs inside a comment or string"
    html = pat1.sub(r'<!--<script', html)
    html = pat2.sub(r'</script>-->', html)
    return html


def striptags(html, norm = True):
    """Parses HTML snippet with libxml2 and extracts text contents using XPath. Decodes entities. 
    If norm=True, strips and normalizes spaces. HTML comments ignored, <script> <style> contents included.
    >>> striptags("<i>one</i><u>two</u><p>three</p><div>four</div>")
    u'onetwothreefour'
    """
    return xdoc(html).text(norm = norm)


###  Other  ###

def readsocket(sock):
    """Reads ALL contents from the socket. Workaround for the known problem of library sockets (also in urllib2): 
    that read() may sometimes return only a part of the contents and it must be called again and again, until empty result, to read everything. 
    Should always be used in place of .read(). Closes the socket at the end."""
    content = []
    while True:
        cont = sock.read()
        if cont: content.append(cont)
        else: 
            sock.close()
            return ''.join(content)
        

# list from: http://techblog.willshouse.com/2012/01/03/most-common-user-agents/
common_user_agents = \
"""
Mozilla/5.0 (Windows NT 6.1; WOW64; rv:13.0) Gecko/20100101 Firefox/13.0.1
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.47 Safari/536.11
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11
Mozilla/5.0 (Windows NT 5.1; rv:13.0) Gecko/20100101 Firefox/13.0.1
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.56 Safari/536.5
Mozilla/5.0 (Windows NT 6.1; rv:13.0) Gecko/20100101 Firefox/13.0.1
Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)
Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.47 Safari/536.11
Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.47 Safari/536.11
Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11
Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.56 Safari/536.5
Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11
Mozilla/4.0 (compatible; MSIE 6.0; MSIE 5.5; Windows NT 5.0) Opera 7.02 Bork-edition [en]
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2) Gecko/20100115 Firefox/3.6
Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; FunWebProducts; .NET CLR 1.1.4322; PeoplePal 6.2)
Mozilla/5.0 (Windows NT 6.1; WOW64; rv:14.0) Gecko/20100101 Firefox/14.0.1
Mozilla/5.0 (Windows NT 6.1; WOW64; rv:5.0) Gecko/20100101 Firefox/5.0
Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.56 Safari/536.5
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; .NET CLR 1.1.4322)
Mozilla/5.0 (Windows NT 5.1; rv:5.0.1) Gecko/20100101 Firefox/5.0.1
Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)
Mozilla/5.0 (Windows NT 6.1; rv:5.0) Gecko/20100101 Firefox/5.02
Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ; .NET CLR 3.5.30729)
Mozilla/5.0 (Windows NT 6.0) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.112 Safari/535.1
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.112 Safari/535.1
Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:13.0) Gecko/20100101 Firefox/13.0.1
Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre
Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)
Mozilla/5.0 (Windows NT 6.1; WOW64; rv:12.0) Gecko/20100101 Firefox/12.0
Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:13.0) Gecko/20100101 Firefox/13.0.1
Mozilla/5.0 (Windows NT 6.0; rv:13.0) Gecko/20100101 Firefox/13.0.1
""".strip().split("\n")


def webpage_simple(url):
    "Get HTML page from the web (simple variant)"
    url = fix_url(url)
    return readsocket(urllib2.urlopen(url))

def webpage(url, timeout = None, identity = None, agent = 0, referer = None, header = None, opener = urllib2.build_opener()): #mechanize.Browser()):
    """
    Get HTML page from the web. Headers are set to enable robust scraping.
    Follows redirects, but there is no way for the client to detect this.
    """
    url = fix_url(url)
    
    # create header; overwrite values in 'header' if more specific settings are given
    if not header:
        header = { 'Accept':'*/*' }
    if identity:
        header.update(identity.header())
    elif not 'user-agent' in [k.lower() for k in header.keys()]:
        if agent is None:
            agent = random.choice(common_user_agents)
        elif isinstance(agent, int):
            agent = common_user_agents[agent]
        header['User-Agent'] = agent
    if referer is not None:
        header['Referer'] = referer
    
    # download page
    req = urllib2.Request(url, None, header)
    if timeout:
        stream = opener.open(req, timeout = timeout)
    else:
        stream = opener.open(req)
    page = readsocket(stream)
    stream.close()
    if identity:
        identity.update(url, page)
    return page
    
def _webpage(url, timeout = None, identity = None, agent = 0, referer = None, client = None):
    "Get HTML page from the web. Headers are set to enable robust scraping."
    client = client or WebClient()
    return client().open(url)

def checkMyIP(web = None, test = 1):
    """Check what external IP of mine will be visible for servers when I connect via a given web client. 
    Disables cache beforehand if needed (re-enables after check).
    See also test=2 and proxy detection:
       > print web.open("http://www.iprivacytools.com/proxy-checker-anonymity-test/")
    """
    web = web or WebClient(tor=True)
    def test1():
        "Fast and simple, based on public high-load API. See: http://www.exip.org/api"
        return web.open("http://api-ams01.exip.org/?call=ip")
    def test2():
        "Tries to detect proxies, too; returns a *list* of all IPs that can be detected"
        page = web.open("http://www.cloakfish.com/?tab=proxy-analysis")
        ips = xbasestring(page).re(regex.ip, True)
        return unique(ips)
    
    cache = web._cache.enabled
    if cache: web._cache.disable()
    tests = {1:test1, 2:test2}
    ip = tests[test]()
    if cache: web._cache.enable()
    return ip


################################################################################################################################################
###
###  REQUEST, RESPONSE, WEBHANDLER - base classes
###

class Request(urllib2.Request):
    """ When setting headers (self.headers from base class), all keys are capitalized by urllib2 (!) to avoid duplicates.
    To assign individual items in the header, use add_header() instead of manual modification of self.headers!
    """
    def __init__(self, url, data = None, headers = {}, timeout = None):
        urllib2.Request.__init__(self, url = url, data = data, headers = headers)
        self.url = url
        self.timeout = timeout

class Response():

    redirect = url = request = info = headers = status = time = None
    content = None                                      # string with all contents of the page, loaded in a lazy way: on explicit client's request
    fromCache = False
    
    def __init__(self, resp = None, url = None, read = True):
        "resp: open file (socket) returned by urllib2 (type: urllib2.addinfourl) or None. url: optionally the original URL of the request (before any redirection)"
        if not resp: return
        self.resp      = resp                           # keep original urllib response
        self.redirect  = resp.geturl()                  # if redirect happened, contains final URL (after all redirections) of the contents; None if no redirect
        self.url       = self.redirect or url           # final URL of the contents: either the redirected URL or the original one if no redirection happened
        self.request   = url                            # original URL of the request, before redirections
        self.info      = resp.info()
        self.headers   = dict(self.info)                # HTTP headers as a plain dictionary, all keys lower-case, e.g.: content-type, content-length, last-modified, server, date, ...
        self.status    = self.code = resp.getcode()     # HTTP response status code; self.code is deprecated, use self.status instead
        self.time      = datetime.now()                 # timestamp when the page was originally retrieved; can be overriden, e.g., by cache handler when an older version is loaded from disk 

        if read: self.read() 
        
    def __deepcopy__(self, memo):
        "Custom implementation of deepcopy(). Makes shallow copy of self.resp and deep copy of all other properties."
        dup = Response()
        for key, val in self.__dict__.iteritems():
            if key == 'resp': dup.resp = val
            else: setattr(dup, key, deepcopy(val, memo))
        return dup    
    
    def read(self):
        if self.content is None and self.resp: 
            self.content = readsocket(self.resp)        # the socket is closed afterwards, by readsocket()
        return self.content
            
class WebHandler(Object):
    """ Base class for handlers of web requests & responses, which handle different atomic aspects of web access.
        Handlers can be chained together to provide flexible and configurable behavior when accessing the web.
        The concept is similar to urllib2's BaseHandler and OpenerDirector, only done much better, 
        with wider range of tasks that can be handled by WebHandlers (e.g., page caching can't be implemented in urllib2's framework).
    """
    
    # fall-back properties for reading, in case if __init__ wasn't invoked in the subclass
    next = None
    enabled = True 
    log = noLogger                      # logger to be used by handlers for printing messages and errors
    
    __shared__ = 'log'
    
    def __init__(self, nextHandler = None):
        self.next = nextHandler
        self.enabled = True
    
    @classmethod
    def chain(cls, listOfHandlers):
        "Connects given handlers into a chain using their .next fields. Automatically filters out None items. The list can contain nested sublists (will be flattened). Returns head of the chain"
        listOfHandlers = filter(None, util.flatten(listOfHandlers))
        if not listOfHandlers: return None
        for i in range(len(listOfHandlers) - 1):
            prev, next = listOfHandlers[i:i+2]
            prev.next = next
        return listOfHandlers[0]
    
    @classmethod
    def unchain(cls, firstHandler):
        "Disconnects the chain and returns all handlers as a plain list"
        h = firstHandler
        l = []
        while h:
            l.append(h)
            h = h.next
            h.next = None
        return l
    
    def list(self):
        "List of all handlers of the chain that starts at 'self' as a head."
        l = []; cur = self
        while cur:  #and isinstance(cur, WebHandler):
            l.append(cur)
            cur = cur.next
        return l
        
    def handle(self, req):
        """Handles Request 'req' using the chain of handlers headed by self. Should return Response, or exception in case of error."""
        raise Exception("Method WebHandler.handle() is abstract")

    def disable(self):
        "Substitutes 'handle' method of 'self' with a mock-up that passes all requests down the chain unmodified. Call enable() to recover original handler"
        self.handle = self.mockup
        self.enabled = False
    def enable(self):
        "Reverses the effect of disable()"
        del self.handle
        self.enabled = True
    def mockup(self, req):
        return self.next.handle(req)


################################################################################################################################################
###
###  WEB HANDLERS - concrete classes for different tasks
###

class StandardClient(WebHandler):
    "Returns a web page using standard urllib2 access. Custom urllib2 handlers can be added upon initialization"
    def __init__(self, addHandlers = []):
        self.opener = urllib2.build_opener(*addHandlers)
        self.added = [h.__class__.__name__ for h in addHandlers]
    def handle(self, req):
        assert isinstance(req, Request)
        #self.log.info("StandardClient, downloading page. Request & handlers: " + jsondump([req, self.added]))
        self.log.info("StandardClient, downloading", req.url)
        try:
            if req.timeout:
                stream = self.opener.open(req, timeout = req.timeout)
            else:
                stream = self.opener.open(req)
        except HTTPError, e:
            e.msg += ", " + req.url
            raise
        return Response(stream, req.url)
    
class FixURL(WebHandler):
    def handle(self, req):
        req.url = fix_url(req.url)
        return self.next.handle(req)

class Delay(WebHandler):
    "Delays web requests so that they are separated by at least 'delay' seconds (but possibly no more than this); 'delay' is slightly randomly disturbed each time"
    def __init__(self, delay = 1.5):
        self.last = now() - delay
        self.delay = delay
    def handle(self, req):
        delay = self.delay * (random.random()/5 + 0.9)
        t = delay - (now() - self.last)
        if t > 0: time.sleep(t)
        self.last = now()
        return self.next.handle(req)

class Timeout(WebHandler):
    "Add timeout value to every request"
    def __init__(self, timeout = 10):
        self.timeout = timeout
    def handle(self, req):
        req.timeout = self.timeout
        return self.next.handle(req)
    
class RetryOnError(WebHandler):
    """In case of an exception of a given class retries the request a given number of times, only then forwards to the caller.
    Default exception class: Exception. Default excludes: 'timeout', HTTPError 403 (Forbidden), HTTPError 404 (Not Found)"""
    def __init__(self, attempts = 3, delay = 5, exception = Exception, exclude = [Timeout, 403, 404]):
        self.attempts = attempts
        self.delay = delay
        self.exception = exception
        self.exclude = [cls for cls in exclude if not isint(cls)]
        self.excludeHTTP = [code for code in exclude if isint(code)]
    def handle(self, req):
        for i in range(self.attempts + 1):
            try:
                _req = deepcopy(req)                    # we may need original 'req' again in the future, thus copying
                return self.next.handle(_req)
            except self.exception, e:
                for x in self.exclude:
                    if isinstance(e,x): raise
                if isinstance(e, HTTPError):
                    if e.getcode() in self.excludeHTTP: raise
                self.log.warning("%s, attempt #%d, %s trying again... Caught '%s'" % (classname(self,False), i+1, req.url, e))
                time.sleep(self.delay * mnoise(1.1))
        return self.next.handle(req)

class RetryOnTimeout(RetryOnError):
    """In case of timeout error, retry the request a given number of times, only then forward Timeout exception to the caller. 
    Only for response timeout (!), NOT for connection opening timeout (that's a different class: URLError 'timed out' not Timeout)."""
    def __init__(self, attempts = 3, delay = 5):
        handlers.RetryOnError.__init__(self, attempts, delay, exception = Timeout, exclude = [])

class RetryCustom(WebHandler):
    "Uses client-provided function 'test' for analyzing errors (exceptions) and deciding whether to retry (return False if not), and with what delay (return >0)"
    def __init__(self, test):
        "'test' is a function of 2 arguments: exception and the no. of attempts done so far, returning new delay or None for stop. See exampleTest() below."
        self.test = test
        
        def exampleTest(ex, attempt):
            "attempt: no. of attempts done so far, always >= 1"
            if isinstance(ex, Timeout): return 5.0 if attempt < 3 else False
            if isinstance(ex, HTTPError):
                status = ex.getcode()
                if status != 404: return 1.0
            return False            # forward other exceptions
        
    def handle(self, req):
        attempt = 0
        while True:
            try:
                attempt += 1
                _req = deepcopy(req)                    # we may need original 'req' again in the future, thus copying
                return self.next.handle(_req)
            except Exception, e:
                delay = self.test(e, attempt)
                if not delay: raise
                delay *= mnoise(1.1)
                self.log.warning("RetryCustom, attempt #%d, trying again after %d seconds... Caught %s" % (attempt, delay, e))
                time.sleep(delay)
        return self.next.handle(req)

class UserAgent(WebHandler):
    def __init__(self, agent = None, change = None):
        """agent: predefined User-Agent (string) to be used; or None to pick User-Agent randomly from a list of most common ones.
           change: time (in minutes) how often UA should be randomly changed; or None if no changes should be done.
        """
        if agent:
            self.agent = agent
        else:
            self.agent = random.choice(common_user_agents)
        self.change = change * 60 if change else None                 # convert minutes to seconds               
        self.lastChange = now()
        
    def handle(self, req):
        req.add_header('User-Agent', self.agent)
        if self.change and (now() - self.lastChange > self.change):
            self.agent = random.choice(common_user_agents)            
            self.lastChange = now()
        return self.next.handle(req)

    
class History(WebHandler):
    Event = namedtuple('Event', 'req resp')
    def __init__(self, maxlen = None):
        "maxlen: must be >= 1, or None (no limit)"
        self.events = []            # a list of "back" and "forward" events, as (request,response) pairs
        self.current = 0            # no. of "back" events in self.events (remaining events are "forward")
        if maxlen and (not isnumber(maxlen) or maxlen < 1):
            maxlen = 1
        self.maxlen = maxlen
    def handle(self, req):
        _req = deepcopy(req)
        resp = self.next.handle(req)
        self.events = self.events[:self.current]                            # we're moving forward, so forget all "forward" events, if present
        M = self.maxlen
        if M and len(self.events) >= M:
            self.events = self.events[-(M-1):] if M > 1 else []             # create space for new event
        self.events.append(self.Event(_req, deepcopy(resp)))                # must perform deepcopies because req/resp objects are modified down and up the handlers chain
        self.current = len(self.events)
        return resp
    def last(self):
        "Return last (request,response) if present; otherwise None. Don't move history pointer"
        if self.current > 0:
            return self.events[self.current - 1]
        return None
    def back(self):
        "If possible, move history pointer 1 step back and return that response object again; otherwise None"
        if self.current > 1:
            self.current -= 1
            return self.last()
        return None
    def forward(self):
        "If possible, move history pointer 1 step forward and return that response object again; otherwise None"
        if self.current < len(self.events):
            self.current += 1
            return self.last()
        return None
    def reset(self):
        "Clear history entirely"
        self.events = []
        self.current = 0
    
class Referer(WebHandler):
    def __init__(self, history):
        "history: the History handler instance which will be used to get info about last webpage visited"
        self.history = history
    def handle(self, req):
        last = self.history.last()
        if last:
            lasturl = last.resp.url or last.req.url       # better to take url from response, but if missing we must use url from request 
            if lasturl:
                prefix = os.path.commonprefix([lasturl, req.url])
                suffix = req.url[len(prefix):-1]
                #print repr(suffix)
                #print repr(last.resp.content)
                if str(suffix) in last.resp.content:        # suffix - simple heuristic to check if the new URL really occured in the previous page; str() to handle URL being unicode object
                    req.add_header('Referer', lasturl) 
        return self.next.handle(req)

class Cache(WebHandler):
    """Web caching: enables repeated access to the same www page without its reloading.
    Cache is located on disk, in a folder given as parameter; pages stored in separate files named after their URLs.
    When redirection occurs, a special type of file (*.redirect) is created pointing to the new URL, so that the returned response
    can have final URL set correctly.
    
    https://pypi.python.org/pypi/pyxattr/0.5.2 - module for Extended File Attributes (might be needed)
    """
    DEFAULT_PATH = ".webcache/"            # default folder where cached pages are stored (will be created if doesn't exist)
    STATE_FILE   = ".state.json"
    
    def __init__(self, path = DEFAULT_PATH, refresh = 1.0, retain = 30):
        """refresh: how often pages in cache should be refreshed, in days; default: 1 day
           retain: for how long pages should be kept in cache even after refresh period (for safety); default: 30 days; 
                   not less than 'refresh' (increased up to 'refresh' if necessary)
        """
        if not isstring(path): path = self.DEFAULT_PATH
        if path[-1] != '/': path += '/' 
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        if not refresh: refresh = 1.0
        self.refresh = refresh * 24*60*60                       # refresh copies after this time, in seconds
        self.retain = max(retain, refresh) * 24*60*60           # keep copies in cache for this long, in seconds
        self.clean = self.refresh / 10.0                        # how often to clean the cache: on every startup + 50 times over 'refresh' period
        self.clean = max(self.clean, 60*60)                     # ...but not more often than every hour
        
        self.state = JsonDict(path + self.STATE_FILE, indent = 4)
        self.state.setdefault('lastClean')
    
    def _clean_cache(self):
        if self.state['lastClean'] and (now() - self.state['lastClean']) < self.clean: return
        self.state['lastClean'] = now()
        self.state.sync()
        self.log.warn("Cache, cleaning of the cache started in a separate thread...")
        
        if islinux():                                           # on Linux, use faster shell command (find) to find and remove old files, in one step
            retain = self.retain / (24*60*60) + 1               # retension time in days, for 'find' command
            subprocess.call("find '%s' -maxdepth 1 -type f -mtime +%d -exec rm '{}' \;" % (self.path, retain), shell=True)
        else:
            MAX_CLEAN = 10000                                   # for performance reasons, if there are many files in cache check only a random subset of MAX_CLEAN ones for removal
            _now = now()
            files = os.listdir(self.path)
            self.log.info("Cache, cleaning, got file list...")
            if len(files) > MAX_CLEAN: files = random.sample(files, MAX_CLEAN)
            for f in files:
                f = self.path + f
                created = os.path.getmtime(f)
                if (_now - created) > self.retain:
                    os.remove(f)
        self.log.info("Cache, cleaning completed.")
    
    def _url2file_old(self, url, ext = "html"):  
        # Deprecated
        safeurl = url.replace('/', '\\')
        filename = safeurl + " " + str(hash(url))
        return self.path + filename + "." + ext
    
    def _url2file(self, url, ext = "html", pat = re.compile(r"""[/"'!?\\&=:]"""), maxlen = 60):
        "Encode URL to obtain a correct file name, preceeded by cache path"
        safeurl = pat.sub('_', url.replace('://', '_'))[:maxlen]
        filename = safeurl + "_" + str(hash(url))
        return self.path + filename + "." + ext
    
    def _cachedFile(self, url, ext = "html"):
        "if possible, return cached copy and its file modification time, otherwise (None,None)"
        filename = self._url2file(url, ext)
        if not os.path.exists(filename):
            filename = self._url2file_old(url, ext)
            if not os.path.exists(filename):
                return None, None

        created = os.path.getmtime(filename)
        if now() - created > self.refresh: return None, None        # we have a copy, but time to refresh (don't delete instantly for safety, if web access fails)
        with open(filename) as f:
            time = util.filedatetime(filename)
            return f.read(), time
        
    
    def _cachedResponse(self, req):
        # is there a .redirect file?
        url = req.url
        content, time1 = self._cachedFile(url, 'redirect')
        if content: url = content                                       # .redirect file contains just the target URL in plain text form
        
        # now check the actual .html file
        content, time2 = self._cachedFile(url)
        if content == None: return None
        
        # found in cache; return a Response() object
        resp = Response()
        resp.content = content
        resp.fromCache = True
        resp.url = url
        resp.time = min(time1 or time2, time2 or time1)
        self.log.info("Cache, loaded from cache: " + req.url + (" -> " + url if url != req.url else ""))
        return resp
    
    def handle(self, req):
        # page in cache?
        resp = self._cachedResponse(req)
        if resp != None: return resp
        
        # download page and save in cache under final URL 
        resp = self.next.handle(req)
        url = resp.url
        filename = self._url2file(url)
        with open(filename, 'wt') as f:
            f.write(resp.content)
        
        # redirection occured? create a .redirect file under original URL to indicate this fact
        if url != req.url:
            filename = self._url2file(req.url, 'redirect')
            with open(filename, 'wt') as f:
                f.write(url)                                                    # .redirect file contains only the target URL in plain text form
        
        self.log.info("Cache, downloaded from web: " + req.url + (" -> " + url if url != req.url else ""))
        
        lastClean = self.state['lastClean']
        if not lastClean or (now() - lastClean) > self.clean:                   # remove old files from the cache before proceeding
            threading.Thread(target = self._clean_cache).start()
            # we'll not join this thread, but application will not terminate until this thread ends (!); set .deamon=True otherwise
            
        return resp

class CustomTransform(WebHandler):
    "Base class for any handler that performs simple 1-1 transformation of either the request or/and the response object."
    
    def handle(self, req):
        req = self.preprocess(req)
        resp = self.next.handle(req)
        resp = self.postprocess(resp)
        return resp
    
    def preprocess(self, req):
        "Override in subclasses"
        return req
    
    def postprocess(self, resp):
        "Override in subclasses"
        return resp
        
class Callback(WebHandler):
    """Call predefined external functions on forward and backward passes, with Request or Request+Response objects as arguments.
       Typically, the functions should only perform monitoring and reporting,
       but it's also technically possible that they modify the internals of Request/Response objects.
    """
    def __init__(self, onRequest = None, onResponse = None):
        self.onRequest = onRequest
        self.onResponse = onResponse
        
    def handle(self, req):
        if self.onRequest is not None:
            self.onRequest(req)
        resp = self.next.handle(req)
        if self.onResponse is not None:
            self.onResponse(req, resp)
        return resp
    

class handlers(object):
    "Legacy."
    # TODO: remove!
    StandardClient = StandardClient
    FixURL = FixURL
    Delay = Delay
    Timeout = Timeout
    RetryOnError = RetryOnError
    RetryOnTimeout = RetryOnTimeout
    RetryCustom = RetryCustom
    UserAgent = UserAgent
    History = History
    Referer = Referer
    Cache = Cache
    

##########################################################################################################################################
###
###  WEB CLIENT
###

class WebClient(Object):
    """
    >>> w1 = WebClient()
    >>> w2 = w1.copy()
    >>> id(w1.logger) == id(w2.logger)
    True
    >>> id(w1.handlers) <> id(w2.handlers) and id(w1.handlers.log) == id(w2.handlers.log)
    True
    """
    __shared__  = 'logger'
    
    # atomic handlers that comprise the 'handlers' chain, in the same order;
    # _head and _tail are lists of custom handlers that go at the beginning or at the end of all handlers list
    _history = _head = _cache = _useragent = _referer = _timeout = _retryCustom = _retryOnError = _retryOnTimeout = _delay = _tail = _client = None
    _tor = False            # self._tor is a read-only attr., changing it does NOT influence whether Tor is used or not, this is decided in __init__ and can't be changed

    handlers = None         # head (only!) of the chain of handlers
    logger   = None         # the logger that was passed down to all handlers in setLogger()
    
    
    def __init__(self, timeout = None, identity = True, referer = True, cache = None, cacheRefresh = None, tor = False, history = 5, delay = None, 
                 retryOnTimeout = None, retryOnError = None, retryCustom = None, head = [], tail = [], logger = None):
        """
        :param identity: how to set User-Agent. Can be either: 
            None/False (no custom identity); 
            or True (identity will be selected randomly once and never changed);
            or <str> (string to be used as User-Agent);
            or <number> X (identity will be picked randomly and changed to another random one after every 'X' minutes) 
        :param history: if number, maximum num of extract to be kept in web history; if True, history with no limit; otherwise (None, <1), limit=1
        :param cacheRefresh: either None, or a number (refresh == retain), or a pair (refresh, retain); typically refresh <= retain
        """
        H = handlers
        urllib2hand = []
        self.logger = logger
        
        if isnumber(history): histLimit = max(history, 1)           # always keep at least 1 history item
        elif history is True: histLimit = None
        else: histLimit = 1
        
        self._history = H.History(histLimit)
        if timeout:     self._timeout = H.Timeout(timeout)
        if identity:    self._useragent = H.UserAgent(identity if isstring(identity) else None, identity if isnumber(identity) else None)
        if referer:     self._referer = H.Referer(self._history)
        if cache:       self.setCache(cache, cacheRefresh)
        if delay:       self._delay = H.Delay(delay)
        if retryOnError:   self._retryOnError = H.RetryOnError(retryOnError)
        if retryOnTimeout: self._retryOnTimeout = H.RetryOnTimeout(retryOnTimeout)
        if retryCustom:    self.setRetryCustom(retryCustom)
        if tor:         self._tor = True; urllib2hand.append(urllib2.ProxyHandler({'http': '127.0.0.1:8118'}))
        self._head = head if islist(head) else [head]
        self._tail = tail if islist(tail) else [tail]
        self._client = H.StandardClient(urllib2hand)
        self._rebuild()                                             # connect all the handlers into a chain
        
        self.url_now = None                 # URL being processed now (started but not finished); for debugging purposes, when exception occurs inside open()

    def copy(self):
        return deepcopy(self)

    def setCache(self, path, refresh = None, retain = None):
        "Default retain period = 1 year. 'refresh' can hold a pair: (refresh, retain), then 'retain' is not used."
        if islist(refresh) and len(refresh) >= 2:
            refresh, retain = refresh[:2]
        if not retain: retain = refresh
        self._cache = handlers.Cache(path, refresh, retain)
        
    def setRetryCustom(self, retryCustom):
        self._retryCustom = handlers.RetryCustom(retryCustom)
    
    def setLogger(self, logger):
        if logger is True: logger = defaultLogger
        elif not logger: logger = noLogger
        self.logger = logger
        if not self.handlers: return
        for h in self.handlers.list():
            h.log = self.logger
            
    def addHandler(self, handler, location = 'head'):
        """Add custom handler, either at the beginning of _head if location='head' (default); or at the end of _tail, 
        as the deepest handler that will directly connect to the actual client (StandardClient), if location='tail'."""
        if location == 'head':
            self._head = [handler] + self._head
        else:
            self._tail.append(handler)
        self._rebuild()
        return self                                         # chaining the calls is possible: return client.addHandler(...).addHandler(...)
        
    def removeHandler(self, handler):
        """Remove a handler that was added with addHandler(), either to the head or tail of the handlers list.
           If the same handler has been added multiple times, its first occurence is removed.
           ValueError is raised if the handler is not present in the list.
        """
        if handler in self._head:
            self._head.remove(handler)
        elif handler in self._tail:
            self._tail.remove(handler)
        else:
            raise ValueError("WebClient.removeHandler: handler (%s) not in list" % handler)
        self._rebuild()
        return self                                         # chaining the calls is possible: return client.removeHandler(...).removeHandler(...)
        
    def _rebuild(self):
        "Rearrange handlers into a chain once again."
        self.handlers = WebHandler.chain([self._history, self._head, self._cache, self._useragent, self._referer, self._timeout, 
                                          self._retryCustom, self._retryOnError, self._retryOnTimeout, self._delay, self._tail, self._client])
        self.setLogger(self.logger)
    
    def response(self, url = None, data = None, headers = {}):
        """Return current (last) response object if url=None, or make a new request like open() and return full response object. 
        The method is aware of movements along history: back(), forward(), ..."""
        if not url:
            last = self._history.last()
            return last.resp if last else None
        # new request...
        self.url_now = url
        url = fix_url(url)
        req = Request(url = url, data = data, headers = headers)
        resp = self.handlers.handle(req)
        self.url_now = None
        return resp                         # implicitly, the 'resp' object is remembered in browsing history, too
    
    open = response                         #@ReservedAssignment

    def get(self, url = None):
        """Main method for downloading pages. Calls response() and returns all contents of the page as string (without metadata). 
        If url=None, loads and returns the contents of the last accessed URL - which typically was only opened with open() or response(), but not fully loaded."""
        return self.response(url).read()
    #open = get                              # TODO: change open() API to only initiate the connection but not read the data
    
    def download(self, filename, url = None):
        "Download a page and save in file. The file will be overriden if exists. If url=None, the last accessed page is downloaded (or just saved if already retrieved)."
        # TODO: transform to stream not batch download, to handle pages of arbitrary size
        page = self.get(url)
        with open(filename, 'wt') as f:
            f.write(page)
    
    def url(self):
        "Return requested URL of the last web access. (Use response() to get last response object.)"
        last = self._history.last()
        return last.req.url if last else None
    def final(self):
        "Return final URL of the last web access, after all redirections."
        last = self._history.last()
        return last.resp.url if last else None
    def redirect(self):
        "Return final URL of the last web access, but only if redirection happened. None otherwise."
        last = self._history.last()
        return last.resp.redirect if last else None
    
    def back(self):
        "Move 1 step back in history"
        return self._history.back()
    def forward(self):
        "Move 1 step forward in history"
        return self._history.forward()
    def reset(self):
        "Clear history"
        self._history.reset()

    def publicIP(self, api = "http://icanhazip.com"):
        """Connects with a public API that returns our current public IP number. Returns this number in text form, 
        as received from the API, only stripped of spaces. Skips all handlers, uses only the last one: self._client."""
        req = Request(fix_url(api))
        resp = self._client.handle(req)
        return resp.read().strip()
        

########################################################################################################################################################################
###
###  Crawler (draft)
###

class Crawler(object):
    
    client = WebClient(timeout = 60, retryOnTimeout = 2, history = 1)
    
    start = []                      # list of start URLs
    domains = None                  # list of domain names to crawl (others will be ignored), case insensitive, implicitly includes all subdomains; None if all domains to be included 
    url_include = None              # if not-None, every visited URL must match this pattern or function
    url_exclude = None              # if not-None, every visited URL must NOT match this pattern or function
    pages_limit = None              # max. number of pages to visit
    links_limit = None              # max. no. of URLs extracted from a single page; if more links are present, only the first 'limit_page' are used
    random = False                  # if True, URLs will be visited in random order and not strictly breadth-first, rather than in their order on page 
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.urls = deque(self.start)                       # queue of pending URLs (pages not downloaded yet)
        self.visited = set()
        
        # combine filtering patterns into one regex
        self.url_filter 
        
    def pages(self):
        """Generator that yields consecutive URLs and pages visited, as pairs (url, page_content, response_object), starting URLs included;
        'url' and 'page' are strings, 'response' is an http Response object, with fields like status code, headers etc.
        At the end (when all urls processed or terminated by client) the final state of crawling process is still present in self.urls and self.visited.
        Invoking the crawler again will start from the point where previous call has finished! """
        def nexturl(self):
            if not self.urls: return None
            if random: 
                i = random.randint(0, len(self.urls)-1)
                url = self.urls[i]
                self.urls[i] = self.urls.popright()
                return url
            return self.urls.popleft()
        def skip(url):
            if not self.allowed(url) or url in self.visited: return True
            return False        
    
        while True:
            url = nexturl()
            if url is None: break       # no more pages to visit
            if skip(url): continue
            self.visited.add(url)
            try: page = self.client.get(url)
            except: continue            # failed to open the page? ignore
            yield url, page
            self.urls += self.process(page, url)            

    def allowed(self, url):
        "Check if this url is allowed to visit."
        

    @staticmethod            
    def extractUrls(page, parsehtml = False):
        #if self.nofollow(url): continue         # filter out non-HTML contents
        return []
    
    def process(self, page, url):
        "Called in crawler loop. Can be overriden in subclasses to provide custom processing of pages. extraction of URLs and/or custom data collection from visited pages."
        return self.extractUrls(page, url)
    

########################################################################################################################################################################
###
###  XML/HTML processing, XDoc
###
###  Classes from Scrapy for XPath selections: HtmlXPathSelector, ...  - monkey-patched to add several useful methods.
###  Scrapy's XPath is just a wrapper for libxml2, perhaps slightly easier to use than raw libxml2, 
###  but should be replaced in the future with reference to underlying basic implementation to remove dependency on Scrapy.
###  Scrapy home: http://scrapy.org/ 
###
###  Example use:
###      page = xdoc(URL)
###      node = page['xpath...']
###      print node.text()
###
###  Methods (some of them from Scrapy):
###      css, css1, xpath, node, nodes, html, text, texts, anchor
###      nodeWithID, nodeOfClass, nodesOfClass, nodeAfter, textAfter
###      [], ... in
###

try:                                                                            # newer versions of Scrapy
    from scrapy.selector.unified import SelectorList as XPathSelectorList
    try: from scrapy.selector import Selector   # newer
    except: from scrapy import Selector         # older
    HtmlXPathSelector = XmlXPathSelector = XPathSelector   =   Selector
    OLD_SCRAPY = False
except ImportError:                                                             # older versions of Scrapy; TODO: drop entirely
    from scrapy.selector.list import XPathSelectorList
    from scrapy.selector import HtmlXPathSelector, XmlXPathSelector, XPathSelector
    OLD_SCRAPY = True


def xpath_escape(s):
    """Utility function that works around XPath's lack of escape character and converts given string (if necessary) into concat(...) expression, 
    so that all characters can be used. Returns either a quoted string or concat(..) expression (as a string)"""
    if "'" not in s: return "'%s'" % s
    if '"' not in s: return '"%s"' % s
    return "concat('%s')" % s.replace("'", "',\"'\",'")

def xdoc(text, mode = "html"):
    "Wrap up the 'text' in a XPathSelector object of appropriate type (xml/html). If 'text' is already an X, return unchanged."
    if isinstance(text, XPathSelector): return text
    return XmlXPathSelector(text=text) if mode.lower() == 'xml' else HtmlXPathSelector(text=text)

def isxdoc(obj):
    "In the future we might use custom XDoc class instead of XPathSelector. Use this function rather than directly isinstance(XPathSelector)"
    return isinstance(obj, XPathSelector)


# XNone: analog of None returned by default from all node*() operators instead of None (to get None use none=True parameter);
# like None evaluates to False, but at the same time is a regular X node and thus has all XPath methods defined (they return empty strings or XNone's).
# Useful when more operations are to be executed on the result node, to prevent exceptions of access to non-existing methods.  
class XNoneType(HtmlXPathSelector):
    """
    >>> bool(XNone)
    False
    """
    def __init__(self): HtmlXPathSelector.__init__(self, text = "<_XNone_></_XNone_>")
    def __bool__(self): return False
    __nonzero__ = __bool__

    @staticmethod
    def text(*a, **kw): return ''
        
XNone = XNoneType()


class XPathSelectorPatch(object):
    "All the methods and properties below will be copied subsequently to XPathSelector (monkey patching). @staticmethod is necessary for this."
    
    nodes = xpath = XPathSelector.select if OLD_SCRAPY else Selector.xpath          # nodes() and xpath() will be aliases for select/xpath()
    html = __unicode__ = XPathSelector.extract
    
    @staticmethod
    def css1(self, css, none = False):
        "Similar to css() but returns always 1 node rather than a list of nodes. None or XNone if no node has been found"
        l = self.css(css)
        if l: return l[0]
        return None if none else XNone
    @staticmethod
    def node(self, xpath = None, css = None, none = False):
        "Similar to nodes() but returns always 1 node rather than a list of nodes. None or XNone if no node has been found"
        l = self.css(css) if css != None else self.xpath(xpath)
        if l: return l[0]
        return None if none else XNone
    @staticmethod
    def text(self, xpath = ".", norm = True):
        """ Returns all text contained in the 1st node selected by 'xpath', as a list of x-strings (xbasestring) 
        with tags stripped out and entities decoded. Empty string if 'xpath' doesn't select any node.
        If norm=True, whitespaces are normalized: multiple spaces merged, leading/trailing spaces stripped out.
        WARNING: doesn't work for an HTML text (untagged) node, use extract() instead.
        """
        xpath = "string(" + xpath + ")"
        if norm: xpath = "normalize-space(" + xpath + ")"
        l = self.nodes(xpath)
        return xbasestring(l[0].extract() if l else '')
    @staticmethod
    def texts(self, xpath, norm = True):
        """ Returns all texts selected by given 'xpath', as a list of x-strings (xbasestring),
        with tags stripped out and entities decoded. Empty list if 'xpath' doesn't select any node.
        If norm=True, whitespaces are normalized: multiple spaces merged, leading/trailing spaces stripped out.
        WARNING: doesn't work for HTML text (untagged) nodes, use extract() instead.
        """
        nodes = self.nodes(xpath)
        return [n.text(".", norm) for n in nodes]
    
    _path_anchor       = "(self::node()[name()='a']|.//a)/@href"
    _path_class        = ".//%s[contains(concat(' ', @class, ' '), ' %s ')]"
    _path_id           = ".//%s[@id='%s']"
    _path_after        = "(.//th|.//td)[contains(.,%s)]/following-sibling::td[1]"
    _path_after_exact  = "(.//th|.//td)[.=%s]/following-sibling::td[1]"
    
    @staticmethod
    def anchor(self, xpath = None, none = False):
        "Extracts href attribute from self (if <a>) or from the 1st <a> descendant. If 'xpath' can't be found and none=False (default), returns '', else None"
        node = self.node(xpath) if xpath else self
        if not node: return None if none else ''
        return node.text(self._path_anchor)
    @staticmethod
    def nodeWithID(self, cls, tag = "*", none=False):
        return self.node(self._path_id % (tag,cls), none)
    @staticmethod
    def nodeOfClass(self, cls, tag = "*", none=False):
        "Checks for inclusion of 'cls' in the class list, rather than strict equality."
        return self.node(self._path_class % (tag,cls), none)
    @staticmethod
    def nodesOfClass(self, cls, tag = "*"):
        "Checks for inclusion of 'cls' in the class list, rather than strict equality."
        return self.nodes(self._path_class % (tag,cls))
    @staticmethod
    def nodeAfter(self, title = None, exact = None, none=False):
        """Returns first (only 1) <TD> element that follows <TD>...[title]...</TD> or <TH>. If 'title' is None, 'exact' is matched instead, 
        using equality match rather than contains(). Usually used to extract contents of a table cell given title of the row"""
        if title is not None:
            return self.node(self._path_after % xpath_escape(title), none)
        return self.node(self._path_after_exact % xpath_escape(exact), none)
    @staticmethod
    def textAfter(self, title = None, exact = None, norm = True):
        "Like nodeAfter(), but returns text of the node"
        if title is not None:
            return self.text(self._path_after % xpath_escape(title), norm)
        return self.text(self._path_after_exact % xpath_escape(exact), norm)
    
    @staticmethod
    def reText(self, regex, multi = False, norm = True):
        "Call text() followed by xbasestring.re(regex, multi). Space normalization performed by default."
        return self.text(norm = norm).re(regex, multi = multi)
    @staticmethod
    def reHtml(self, regex, multi = False):
        "Call html() followed by xbasestring.re(regex, multi)."
        return self.html().re(regex, multi = multi)

    @staticmethod
    def __getitem__(self, path):
        """For convenient [...] selection of subnodes and attributes: node['subnode'] or node['@attr'] or node['any_XPath']. 
        If the path contains '@' character anywhere, text() is returned, as for an attribute. A node() otherwise."""
        if '@' in path: return self.text(path)
        return self.node(path)
    
#     @staticmethod
#     def html(self):
#         "'print xnode' will print FULL original html/xml code of the node"
#         return self.extract()
#     @staticmethod
#     def __unicode__(self):
#         "'print xnode' will print FULL original html/xml code of the node"
#         return self.extract()
    @staticmethod
    def __str__(self):
        return self.extract().encode('utf-8')
    @staticmethod
    def __contains__(self, s):
        "Checks for occurence of a given plain text in the document (tags stripped out). Shorthand for 's in x.text()'."
        return s in self.text()
    

def monkeyPatch():
    # copy methods and properties to XPathSelector (monkey patching):
    methods = filter(lambda k: not k.startswith('__'), XPathSelectorPatch.__dict__.keys())          # all properties with standard names (no __*__)
    methods += "__unicode__ __str__ __contains__ __getitem__".split()                               # additionally these special names
    for name in methods:
        method = getattr(XPathSelectorPatch, name)          # here, MUST use getattr not __dict__[name] - they give different results!!! <unbound method> vs <function>!
        setattr(XPathSelector, name, method) 

    # additional patches...
    def XPathSelectorList_text(self, xpath = ".", norm = True):
        "Runs text() method on all selectors contained in this list"
        return [x.text(xpath, norm) for x in self]
    XPathSelectorList.text     = XPathSelectorList_text

monkeyPatch()


#class XDoc(object):
#    "Currently just a wrapper for scrapy's HtmlXPathSelector/XmlXPathSelector"
#    def __init__(self, text, mode = "html"):
#        "'text' is either raw text (string/unicode), or XPathSelector. 'mode' can be either 'html' or 'xml'"
#        if isinstance(text, XPathSelector):
#            self.selector = text
#        else:
#            self.selector = xdoc(text)
#    def select(self, xpath):
#        return self.selector.select(xpath)
#    def text(self, xpath='.', norm=True):
#        return self.selector.text(xpath, norm)
#    def nodeAfter(self, title):
#        return self.selector.nodeAfter(title)
#    def textAfter(self, title, norm=True):
#        return self.selector.textAfter(title, norm)
#    def re(self, regex):
#        return self.selector.re(regex)
#    def __unicode__(self):
#        return unicode(self.selector)
#    def __str__(self):
#        return str(self.selector)
    

# if False:   # just a draft
#     class XDoc(object):
#         """XML or HTML document represented as a tree; or a part of it (subtree, node)."""
#         def __init__(self):
#             self.items = []         # XString's and XTag's that constitute content (children) of this XDoc
#         
#         def text(self):
#             "Raw text of this XDoc, with all tags stripped and entities decoded"
#             return ""
#             
#         def xml(self, compact = False):
#             "This XDoc in text form: XMl or HTML, with all tags preserved"
#     
#         def html(self):
#             "Alias for xml()" 
#             return self.xml()
#     
#     
#     class XTag(XDoc):
#         "XML/HTML element: a tag with XDoc inside"
#     
#         def __init__(self):
#             self.tag = None             # [str] name of the tag that encloses entire content of this XDoc
#             self.attr = []              # list of attributes: (attribute,value) pairs
#             self.orig_opening = ""      # opening tag in its original form, as occured in source text
#             self.orig_closing = ""      # closing --- "" ---
#         
#     class XString(XDoc):
#         "XML/HTML string node (raw text between or inside tags)"
#     class XComment(XDoc):
#         "XML/HTML comment node (<!-- ... -->)"
        

"""
DRAFT...

from lxml import etree
html = "<html>ala < a attr='cos' >ma< /a > kota</i>  <I><script>   shittttt</script> &amp; <!-- <style> css </style> --></html>"
root = etree.fromstring(html, etree.HTMLParser())

tree = lxml.html.fromstring(html)
tree.text_content
for e in tree.iter() :
  print e.tag

# see: http://shallowsky.com/blog/programming/parsing-html-python.html
# for using lxml.html


# converting Element to DOM:
from xml.dom.pulldom import SAX2DOM
from lxml.sax import saxify
handler = SAX2DOM()
saxify(tree, handler)
dom = handler.document

dom.childNodes[0].childNodes[0].childNodes[0].childNodes
>>> [<DOM Text node "'ala ma kot'...">, <DOM Element: i at 0x97c6aac>]

dom.toxml()

# converting from DOM back to Element, via parsing (watch for appended enclosing <?xml> and <p> tags):
tree = lxml.html.fromstring(dom.toxml())

"""

if __name__ == "__main__":
    import doctest
    print doctest.testmod()
    
