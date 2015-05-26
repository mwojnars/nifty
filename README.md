Nifty
=====

*Nifty* is a library of utility functions and classes that simplify various common tasks in Python programming. A handy add-on to standard libraries that makes Python even easier to use. 
Contains also a number of advanced tools for specific tasks of *web scraping*, *data processing* and *data mining*.

Brought to you by **Marcin Wojnarski** (see my [blog](http://wojnarski.wordpress.com/marcin-wojnarski), [Twitter](http://twitter.com/mwojnarski), [LinkedIn](http://www.linkedin.com/in/marcinwojnarski)). Licensed on GPL.

Contents
--------

Basic utilities in [nifty.util](https://github.com/mwojnars/nifty/blob/master/util.py), including 100 one-liners for common tasks:

- **is...()** dynamic type checking: *isstring*, *isint*, *isnumber*, *islist*, *istuple*, *isdict*, *istype*, *isfunction*, *isiterable*, *isgenerator*, ...
- **classes** and **types** inspection: *classname*, *issubclass*, *baseclasses*, *subclasses*, *types*
- **objects**, generic types with extended interface: *Object*, *NoneObject*, *ObjDict*
- **collections**: *unique*, *flatten*, *list2str*, *obj2dict*, *dict2obj*, *subdict*, *splitkeys*, *lowerkeys*, *getattrs*, *setattrs*, *copyattrs*, *setdefaults*, *Heap*
- **strings** and **text**: *merge_spaces*, *ascii*, *prefix*, *indent*
- **JSON** encoding & serialization of arbitrary objects: *JsonObjEncoder*, *dumpjson*, *printjson*, *JsonDict*
- **numbers**: *minmax*, *percent*, *bound*, *divup*, *noise*, *mnoise*, *parseint*
- **date & time**: *Timer*, *now*, *nowString*, *utcnow*, *timestamp*, *asdatetime*, *convertTimezone*, *secondsBetween* (minutes, hours, ...), *secondsSince* (minutes, hours, ...)
- **files**: *fileexists*, *normpath*, *filesize*, *filetime*, *filectime*, *filedatetime*, *readfile*, *writefile*, *Tee*
- **file folders**: *normdir*, *listdir*, *listdirs*, *listfiles*, *findfiles*, *ifindfiles*
- **concurrency**: *Lock*, *NoneLock*

Text processing routines in [nifty.text](https://github.com/mwojnars/nifty/blob/master/text.py):

- **Levenshtein distance**: *levenshtein*, *levendist*, *levenscore*
- **Bag-of-words** model with **TF-IDF** weights: *WordsModel*
- **N-grams**: *ngrams*

Web scraping tools in [nifty.redex.pattern](https://github.com/mwojnars/nifty/blob/master/redex/pattern.py):
- **Pattern** class - a brand new type of tool for extracting data from any markup document. 
  Bridges the gap between regexes and XPaths as used in web scraping.
  Combines consistency and compactness of regexes (single pattern matches all document and extracts multiple variables at once)
  with strength and precision of XPaths: the pattern is defined in a language much simpler than regexes 
  and can span multiple fragments of the document, providing precise *context* where each fragment is allowed to match.
- **parsing** of basic data types from human-readable formats used in web pages: *pdate*, *pdatetime*, *pint*, *pfloat*, *pdecimal*, *percent*
- **url** absolutization & unquoting: *url*, *url_unquote*

**Data Pipes**. Architecture for scalable pipeline-oriented processing of Big Data, in [nifty.data.pipes](https://github.com/mwojnars/nifty/blob/master/data/pipes.py).

Data storage and object serialization with a new **DAST** format, in [nifty.data.dast](https://github.com/mwojnars/nifty/blob/master/data/dast.py).

For more information, check pydocs and comments in the source code. Other modules to be documented in the near future.

Nifty includes code of [Waxeye](http://waxeye.org/), a PEG parser generator (MIT license) used to generate parser for [Patterns](https://github.com/mwojnars/nifty/blob/master/redex/pattern.py).
