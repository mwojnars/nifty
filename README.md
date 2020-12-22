Nifty
=====

*Nifty* is a library of utility functions and classes that simplify various common tasks in Python programming; a handy add-on to standard libraries that makes Python even easier and more fun to use. In addition to simple utilities, Nifty contains a number of advanced tools for *web scraping*, *data processing* and *data mining*. 

Authored by **Marcin Wojnarski** ([LinkedIn](http://www.linkedin.com/in/marcinwojnarski), [Twitter](http://twitter.com/mwojnarski)). Licensed on **GPL**.

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

Multiple Sequence Alignment (MSA) of any Unicode strings for advanced text processing and NLP, in [nifty.algo.alignment](https://github.com/mwojnars/nifty/blob/master/algo/alignment.py): class **FuzzyString**, function **align_multiple**(). These are fast routines based on my custom algorithm of iteratively improving a *fuzzy consensus* string, where every character is a *probability distribution* over a full charset rather than a single crisp char. This MSA routine can be used to detect fuzzy matchings of a search phrase in a larger text; and statistically analyse deviations between multiple fuzzy matches.

Math classes in [nifty.math](https://github.com/mwojnars/nifty/blob/master/math.py):

- **namedarray**: a subclass of *numpy.ndarray* that implements *named columns* for 2D numpy arrays - something similar to Pandas, but fully compatible with numpy API (unlike Pandas) and providing fast processing, approx. *7x faster* than Pandas' DataFrame.
- **Stack** class: a wrapper around any numpy array that allows incremental addition of items (values, rows, subarrays, ...) and provides automatic reallocation when the contents grows larger than the underlying array.
- **Distribution** and its subclasses (Interval, Range, Choice, Switch, ...): a framework for defining custom composite probability distributions in a hierarchical way, and sampling from such distributions.

Machine Learning classes in [nifty.learn](https://github.com/mwojnars/nifty/blob/master/learn.py), compatible with Scikit-Learn API, e.g., **OneHotFrame** for trainable auto-detection and transformation of columns in a data set that need one-hot encoding for Scikit estimators - much easier to use than standard Scikit tools for this purpose, handles encoding of unknown symbols and can ignore infrequent ones.

**Data Pipes**. Architecture for scalable pipeline-oriented processing of unbounded data streams, in [nifty.data.pipes](https://github.com/mwojnars/nifty/blob/master/data/pipes.py). A "data pipeline" comprises an arbitrary number of "data cells" connected linearly and/or with branching; each cell is autonomous in pulling data from source(s) and performing any type of stream processing: data access, generation, buffering, filtering, pre-processing, post-processing, monitoring, reporting, model training etc. Data Pipes provide support for hyper-parameterization ("knobs"); creating higher-level meta-cells (e.g., for meta-optimization); and for multi-threading across parallel branches of a processing network.

Data storage and object serialization with a new **DAST** format, in [nifty.data.dast](https://github.com/mwojnars/nifty/blob/master/data/dast.py).

Web scraping tools in [nifty.redex](https://github.com/mwojnars/nifty/blob/master/redex/redex.py):
- **Redex** patterns - a new language for extracting data from any markup document. Similar in spirit and structure to regular expressions (regex), but better suited to searching in large tagged documents. Bridges the gap between regex and XPaths as used in web scraping.
  Combines consistency and compactness of regexes (single pattern matches all document and extracts multiple variables at once)
  with strength and precision of XPaths: redex pattern is defined in a form much simpler than regexes 
  and can span multiple fragments of the document, providing precise *context* where each fragment is allowed to match.
- **parsing** of basic data types from human-readable formats used in web pages: *pdate*, *pdatetime*, *pint*, *pfloat*, *pdecimal*, *percent*
- **url** absolutization & unquoting: *url*, *url_unquote*

For more information, check pydocs and comments in the code, or post questions in [Discussions](https://github.com/mwojnars/nifty/discussions).


Nifty includes code of [Waxeye](http://waxeye.org/), a PEG parser generator (MIT license) used in [Redex](https://github.com/mwojnars/nifty/blob/master/redex/redex.py).

Use cases
---------

Projects that use Nifty:
- [Paperity](http://paperity.org/), an aggregator of scholarly literature

