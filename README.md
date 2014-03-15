Nifty
=====

*Nifty* is a library of utility functions and classes that simplify various common tasks in Python programming. A handy add-on to standard libraries that makes Python even easier to use. 
Contains also a number of advanced tools for specific tasks of *web scraping*, *data processing* and *data mining*.

Brought to you by [Marcin Wojnarski](http://wojnarski.wordpress.com/marcin-wojnarski) (follow me on [Twitter](http://twitter.com/mwojnarski)). Licensed on GPL.

Contents
--------

In [nifty.util](https://github.com/mwojnars/nifty/blob/master/util.py):

- **is...()** functions for dynamic type checking: *isstring*, *isint*, *isnumber*, *islist*, *istuple*, *isdict*, *istype*, *isfunction*, *isiterable*, *isgenerator*, ...
- **classes** and **types** inspection: *classname*, *issubclass*, *baseclasses*, *subclasses*, *types*
- **objects**, generic types with extended interface: *Object*, *NoneObject*, *ObjDict*
- **collections**: *unique*, *flatten*, *list2str*, *obj2dict*, *dict2obj*, *subdict*, *splitkeys*, *lowerkeys*, *getattrs*, *setattrs*, *copyattrs*, *setdefaults*, *Heap*
- **strings** and **text**: *merge_spaces*, *ascii*, *prefix*, *indent*
- **JSON** encoding of arbitrary objects, serialization: *JsonObjEncoder*, *dumpjson*, *printjson*, *JsonDict*
- **numbers**: *minmax*, *percent*, *bound*, *divup*, *noise*, *mnoise*, *parseint*
- **date & time**: *Timer*, *now*, *nowString*, *utcnow*, *timestamp*, *asdatetime*, *convertTimezone*, *secondsBetween* (minutes, hours, ...), *secondsSince* (minutes, hours, ...)
- **files**: *fileexists*, *normpath*, *filesize*, *filetime*, *filectime*, *filedatetime*, *readfile*, *writefile*, *Tee*
- **file folders**: *normdir*, *listdir*, *listdirs*, *listfiles*, *findfiles*, *ifindfiles*
- **concurrency**: *Lock*, *NoneLock*

In [nifty.text](https://github.com/mwojnars/nifty/blob/master/text.py):

- **Levenshtein distance**: *levenshtein*, *levendist*, *levenscore*
- **Bag-of-words** model with **TF-IDF** weights: *WordsModel*
- **N-grams**: *ngrams*

Other modules to be documented in the near future. Check the source code for details.

Nifty includes code of [Waxeye](http://waxeye.org/), a PEG parser generator (MIT license). It is used to generate parser for [Pattern](https://github.com/mwojnars/nifty/blob/master/pattern/pattern.py) class.
