# For backward compatibility
from holodata.util import *  # noqa (API import)

import sys, warnings, operator
import json
import time
import types
import inspect
import itertools
import datetime as dt

from collections import defaultdict
from contextlib import contextmanager
from distutils.version import LooseVersion as _LooseVersion
from functools import partial
from threading import Thread, Event

import numpy as np
import param

# Python3 compatibility
from holodata.util import wrap_tuple, is_number, is_nan, \
    group_sanitizer, label_sanitizer

if sys.version_info.major >= 3:
    import builtins as builtins   # noqa (compatibility)

    if sys.version_info.minor > 3:
        from collections.abc import Iterable # noqa (compatibility)
    else:
        from collections import Iterable # noqa (compatibility)

    basestring = str
    unicode = str
    long = int
    cmp = lambda a, b: (a>b)-(a<b)
    generator_types = (zip, range, types.GeneratorType)
    RecursionError = RecursionError if sys.version_info.minor > 4 else RuntimeError # noqa
    _getargspec = inspect.getfullargspec
    get_keywords = operator.attrgetter('varkw')
    LooseVersion = _LooseVersion
else:
    import __builtin__ as builtins # noqa (compatibility)
    from collections import Iterable # noqa (compatibility)

    basestring = basestring
    unicode = unicode
    from itertools import izip
    generator_types = (izip, xrange, types.GeneratorType) # noqa
    RecursionError = RuntimeError
    _getargspec = inspect.getargspec
    get_keywords = operator.attrgetter('keywords')

    class LooseVersion(_LooseVersion):
        """
        Subclassed to avoid unicode issues in python2
        """

        def __init__ (self, vstring=None):
            if isinstance(vstring, unicode):
                vstring = str(vstring)
            self.parse(vstring)

        def __cmp__(self, other):
            if isinstance(other, unicode):
                other = str(other)
            if isinstance(other, basestring):
                other = LooseVersion(other)
            return cmp(self.version, other.version)

numpy_version = LooseVersion(np.__version__)
param_version = LooseVersion(param.__version__)

datetime_types = (np.datetime64, dt.datetime, dt.date, dt.time)
timedelta_types = (np.timedelta64, dt.timedelta,)
arraylike_types = (np.ndarray,)

try:
    import pandas as pd
except ImportError:
    pd = None

if pd:
    pandas_version = LooseVersion(pd.__version__)
    try:
        if pandas_version >= '0.24.0':
            from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtypeType
            from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
        elif pandas_version > '0.20.0':
            from pandas.core.dtypes.dtypes import DatetimeTZDtypeType
            from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
        else:
            from pandas.types.dtypes import DatetimeTZDtypeType
            from pandas.types.dtypes.generic import ABCSeries, ABCIndexClass
        pandas_datetime_types = (pd.Timestamp, DatetimeTZDtypeType, pd.Period)
        pandas_timedelta_types = (pd.Timedelta,)
        datetime_types = datetime_types + pandas_datetime_types
        timedelta_types = timedelta_types + pandas_timedelta_types
        arraylike_types = arraylike_types + (ABCSeries, ABCIndexClass)
        if pandas_version > '0.23.0':
            from pandas.core.dtypes.generic import ABCExtensionArray
            arraylike_types = arraylike_types + (ABCExtensionArray,)
    except Exception as e:
        param.main.warning('pandas could not register all extension types '
                           'imports failed with the following error: %s' % e)

try:
    import cftime
    cftime_types = (cftime.datetime,)
    datetime_types += cftime_types
except:
    cftime_types = ()
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])


class VersionError(Exception):
    "Raised when there is a library version mismatch."
    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super(VersionError, self).__init__(msg, **kwargs)


class Config(param.ParameterizedFunction):
    """
    Set of boolean configuration values to change HoloViews' global
    behavior. Typically used to control warnings relating to
    deprecations or set global parameter such as style 'themes'.
    """

    future_deprecations = param.Boolean(default=False, doc="""
       Whether to warn about future deprecations""")

    image_rtol = param.Number(default=10e-4, doc="""
      The tolerance used to enforce regular sampling for regular,
      gridded data where regular sampling is expected. Expressed as the
      maximal allowable sampling difference between sample
      locations.""")

    no_padding = param.Boolean(default=False, doc="""
       Disable default padding (introduced in 1.13.0).""")

    warn_options_call = param.Boolean(default=True, doc="""
       Whether to warn when the deprecated __call__ options syntax is
       used (the opts method should now be used instead). It is
       recommended that users switch this on to update any uses of
       __call__ as it will be deprecated in future.""")

    default_cmap = param.String(default='kbc_r', doc="""
       Global default colormap. Prior to HoloViews 1.14.0, the default
       value was 'fire' which can be set for backwards compatibility.""")

    default_gridded_cmap = param.String(default='kbc_r', doc="""
       Global default colormap for gridded elements (i.e. Image, Raster
       and QuadMesh). Can be set to 'fire' to match raster defaults
       prior to HoloViews 1.14.0 while allowing the default_cmap to be
       the value of 'kbc_r' used in HoloViews >= 1.14.0""")

    default_heatmap_cmap = param.String(default='kbc_r', doc="""
       Global default colormap for HeatMap elements. Prior to HoloViews
       1.14.0, the default value was the 'RdYlBu_r' colormap.""")

    def __call__(self, **params):
        self.param.set_param(**params)
        return self

config = Config()

class HashableJSON(json.JSONEncoder):
    """
    Extends JSONEncoder to generate a hashable string for as many types
    of object as possible including nested objects and objects that are
    not normally hashable. The purpose of this class is to generate
    unique strings that once hashed are suitable for use in memoization
    and other cases where deep equality must be tested without storing
    the entire object.

    By default JSONEncoder supports booleans, numbers, strings, lists,
    tuples and dictionaries. In order to support other types such as
    sets, datetime objects and mutable objects such as pandas Dataframes
    or numpy arrays, HashableJSON has to convert these types to
    datastructures that can normally be represented as JSON.

    Support for other object types may need to be introduced in
    future. By default, unrecognized object types are represented by
    their id.

    One limitation of this approach is that dictionaries with composite
    keys (e.g. tuples) are not supported due to the JSON spec.
    """
    string_hashable = (dt.datetime,)
    repr_hashable = ()

    def default(self, obj):
        if isinstance(obj, set):
            return hash(frozenset(obj))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd and isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_csv(header=True).encode('utf-8')
        elif isinstance(obj, self.string_hashable):
            return str(obj)
        elif isinstance(obj, self.repr_hashable):
            return repr(obj)
        try:
            return hash(obj)
        except:
            return id(obj)


def merge_option_dicts(old_opts, new_opts):
    """
    Update the old_opts option dictionary with the options defined in
    new_opts. Instead of a shallow update as would be performed by calling
    old_opts.update(new_opts), this updates the dictionaries of all option
    types separately.

    Given two dictionaries
        old_opts = {'a': {'x': 'old', 'y': 'old'}}
    and
        new_opts = {'a': {'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    this returns a dictionary
        {'a': {'x': 'old', 'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    """
    merged = dict(old_opts)

    for option_type, options in new_opts.items():
        if option_type not in merged:
            merged[option_type] = {}

        merged[option_type].update(options)

    return merged


def merge_options_to_dict(options):
    """
    Given a collection of Option objects or partial option dictionaries,
    merge everything to a single dictionary.
    """
    merged_options = {}
    for obj in options:
        if isinstance(obj,dict):
            new_opts = obj
        else:
            new_opts = {obj.key: obj.kwargs}

        merged_options = merge_option_dicts(merged_options, new_opts)
    return merged_options


def deprecated_opts_signature(args, kwargs):
    """
    Utility to help with the deprecation of the old .opts method signature

    Returns whether opts.apply_groups should be used (as a bool) and the
    corresponding options.
    """
    from .options import Options
    groups = set(Options._option_groups)
    opts = {kw for kw in kwargs if kw != 'clone'}
    apply_groups = False
    options = None
    new_kwargs = {}
    if len(args) > 0 and isinstance(args[0], dict):
        apply_groups = True
        if (not set(args[0]).issubset(groups) and
            all(isinstance(v, dict) and not set(v).issubset(groups)
                for v in args[0].values())):
            apply_groups = False
        elif set(args[0].keys()) <= groups:
            new_kwargs = args[0]
        else:
            options = args[0]
    elif opts and opts.issubset(set(groups)):
        apply_groups = True
    elif kwargs.get('options', None) is not None:
        apply_groups = True
    elif not args and not kwargs:
        apply_groups = True

    return apply_groups, options, new_kwargs


class periodic(Thread):
    """
    Run a callback count times with a given period without blocking.

    If count is None, will run till timeout (which may be forever if None).
    """

    def __init__(self, period, count, callback, timeout=None, block=False):

        if isinstance(count, int):
            if count < 0: raise ValueError('Count value must be positive')
        elif not type(count) is type(None):
            raise ValueError('Count value must be a positive integer or None')

        if block is False and count is None and timeout is None:
            raise ValueError('When using a non-blocking thread, please specify '
                             'either a count or a timeout')

        super(periodic, self).__init__()
        self.period = period
        self.callback = callback
        self.count = count
        self.counter = 0
        self.block = block
        self.timeout = timeout
        self._completed = Event()
        self._start_time = None

    @property
    def completed(self):
        return self._completed.is_set()

    def start(self):
        self._start_time = time.time()
        if self.block is False:
            super(periodic,self).start()
        else:
            self.run()

    def stop(self):
        self.timeout = None
        self._completed.set()

    def __repr__(self):
        return 'periodic(%s, %s, %s)' % (self.period,
                                         self.count,
                                         callable_name(self.callback))
    def __str__(self):
        return repr(self)

    def run(self):
        while not self.completed:
            if self.block:
                time.sleep(self.period)
            else:
                self._completed.wait(self.period)
            self.counter += 1
            try:
                self.callback(self.counter)
            except Exception:
                self.stop()

            if self.timeout is not None:
                dt = (time.time() - self._start_time)
                if dt > self.timeout:
                    self.stop()
            if self.counter == self.count:
                self.stop()



def deephash(obj):
    """
    Given an object, return a hash using HashableJSON. This hash is not
    architecture, Python version or platform independent.
    """
    try:
        return hash(json.dumps(obj, cls=HashableJSON, sort_keys=True))
    except:
        return None


def tree_attribute(identifier):
    """
    Predicate that returns True for custom attributes added to AttrTrees
    that are not methods, properties or internal attributes.

    These custom attributes start with a capitalized character when
    applicable (not applicable to underscore or certain unicode characters)
    """
    if identifier[0].upper().isupper() is False and identifier[0] != '_':
        return True
    else:
        return identifier[0].isupper()

def argspec(callable_obj):
    """
    Returns an ArgSpec object for functions, staticmethods, instance
    methods, classmethods and partials.

    Note that the args list for instance and class methods are those as
    seen by the user. In other words, the first argument which is
    conventionally called 'self' or 'cls' is omitted in these cases.
    """
    if (isinstance(callable_obj, type)
        and issubclass(callable_obj, param.ParameterizedFunction)):
        # Parameterized function.__call__ considered function in py3 but not py2
        spec = _getargspec(callable_obj.__call__)
        args = spec.args[1:]
    elif inspect.isfunction(callable_obj):    # functions and staticmethods
        spec = _getargspec(callable_obj)
        args = spec.args
    elif isinstance(callable_obj, partial): # partials
        arglen = len(callable_obj.args)
        spec =  _getargspec(callable_obj.func)
        args = [arg for arg in spec.args[arglen:] if arg not in callable_obj.keywords]
    elif inspect.ismethod(callable_obj):    # instance and class methods
        spec = _getargspec(callable_obj)
        args = spec.args[1:]
    else:                                   # callable objects
        return argspec(callable_obj.__call__)

    return inspect.ArgSpec(args=args,
                           varargs=spec.varargs,
                           keywords=get_keywords(spec),
                           defaults=spec.defaults)



def validate_dynamic_argspec(callback, kdims, streams):
    """
    Utility used by DynamicMap to ensure the supplied callback has an
    appropriate signature.

    If validation succeeds, returns a list of strings to be zipped with
    the positional arguments, i.e. kdim values. The zipped values can then
    be merged with the stream values to pass everything to the Callable
    as keywords.

    If the callbacks use *args, None is returned to indicate that kdim
    values must be passed to the Callable by position. In this
    situation, Callable passes *args and **kwargs directly to the
    callback.

    If the callback doesn't use **kwargs, the accepted keywords are
    validated against the stream parameter names.
    """
    argspec = callback.argspec
    name = callback.name
    kdims = [kdim.name for kdim in kdims]
    stream_params = stream_parameters(streams)
    defaults = argspec.defaults if argspec.defaults else []
    all_posargs = argspec.args[:-len(defaults)] if defaults else argspec.args
    # Filter out any posargs for streams
    posargs = [arg for arg in all_posargs if arg not in stream_params]
    kwargs = argspec.args[-len(defaults):]

    if argspec.keywords is None:
        unassigned_streams = set(stream_params) - set(argspec.args)
        if unassigned_streams:
            unassigned = ','.join(unassigned_streams)
            raise KeyError('Callable {name!r} missing keywords to '
                           'accept stream parameters: {unassigned}'.format(name=name,
                                                                    unassigned=unassigned))


    if len(posargs) > len(kdims) + len(stream_params):
        raise KeyError('Callable {name!r} accepts more positional arguments than '
                       'there are kdims and stream parameters'.format(name=name))
    if kdims == []:                  # Can be no posargs, stream kwargs already validated
        return []
    if set(kdims) == set(posargs):   # Posargs match exactly, can all be passed as kwargs
        return kdims
    elif len(posargs) == len(kdims): # Posargs match kdims length, supplying names
        if argspec.args[:len(kdims)] != posargs:
            raise KeyError('Unmatched positional kdim arguments only allowed at '
                           'the start of the signature of {name!r}'.format(name=name))

        return posargs
    elif argspec.varargs:            # Posargs missing, passed to Callable directly
        return None
    elif set(posargs) - set(kdims):
        raise KeyError('Callable {name!r} accepts more positional arguments {posargs} '
                       'than there are key dimensions {kdims}'.format(name=name,
                                                                      posargs=posargs,
                                                                      kdims=kdims))
    elif set(kdims).issubset(set(kwargs)): # Key dims can be supplied by keyword
        return kdims
    elif set(kdims).issubset(set(posargs+kwargs)):
        return kdims
    elif argspec.keywords:
        return kdims
    else:
        raise KeyError('Callback {name!r} signature over {names} does not accommodate '
                       'required kdims {kdims}'.format(name=name,
                                                       names=list(set(posargs+kwargs)),
                                                       kdims=kdims))


def callable_name(callable_obj):
    """
    Attempt to return a meaningful name identifying a callable or generator
    """
    try:
        if (isinstance(callable_obj, type)
            and issubclass(callable_obj, param.ParameterizedFunction)):
            return callable_obj.__name__
        elif (isinstance(callable_obj, param.Parameterized)
              and 'operation' in callable_obj.param):
            return callable_obj.operation.__name__
        elif isinstance(callable_obj, partial):
            return str(callable_obj)
        elif inspect.isfunction(callable_obj):  # functions and staticmethods
            return callable_obj.__name__
        elif inspect.ismethod(callable_obj):    # instance and class methods
            meth = callable_obj
            if sys.version_info < (3,0):
                owner =  meth.im_class if meth.im_self is None else meth.im_self
                if meth.__name__ == '__call__':
                    return type(owner).__name__
                return '.'.join([owner.__name__, meth.__name__])
            else:
                return meth.__func__.__qualname__.replace('.__call__', '')
        elif isinstance(callable_obj, types.GeneratorType):
            return callable_obj.__name__
        else:
            return type(callable_obj).__name__
    except Exception:
        return str(callable_obj)


def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.nanmin(a1, b1), np.nanmax(a2, b2)). Used to calculate
    min and max values of a number of items.
    """
    try:
        limzip = zip(list(lims), list(olims), [np.nanmin, np.nanmax])
        limits = tuple([float(fn([l, ol])) for l, ol, fn in limzip])
    except:
        limits = (np.NaN, np.NaN)
    return limits


def max_extents(extents, zrange=False):
    """
    Computes the maximal extent in 2D and 3D space from
    list of 4-tuples or 6-tuples. If zrange is enabled
    all extents are converted to 6-tuples to compute
    x-, y- and z-limits.
    """
    if zrange:
        num = 6
        inds = [(0, 3), (1, 4), (2, 5)]
        extents = [e if len(e) == 6 else (e[0], e[1], None,
                                          e[2], e[3], None)
                   for e in extents]
    else:
        num = 4
        inds = [(0, 2), (1, 3)]
    arr = list(zip(*extents)) if extents else []
    extents = [np.NaN] * num
    if len(arr) == 0:
        return extents
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        for lidx, uidx in inds:
            lower = [v for v in arr[lidx] if v is not None and not is_nan(v)]
            upper = [v for v in arr[uidx] if v is not None and not is_nan(v)]
            if lower and isinstance(lower[0], datetime_types):
                extents[lidx] = np.min(lower)
            elif any(isinstance(l, basestring) for l in lower):
                extents[lidx] = np.sort(lower)[0]
            elif lower:
                extents[lidx] = np.nanmin(lower)
            if upper and isinstance(upper[0], datetime_types):
                extents[uidx] = np.max(upper)
            elif any(isinstance(u, basestring) for u in upper):
                extents[uidx] = np.sort(upper)[-1]
            elif upper:
                extents[uidx] = np.nanmax(upper)
    return tuple(extents)


def int_to_alpha(n, upper=True):
    "Generates alphanumeric labels of form A-Z, AA-ZZ etc."
    casenum = 65 if upper else 97
    label = ''
    count= 0
    if n == 0: return str(chr(n + casenum))
    while n >= 0:
        mod, div = n % 26, n
        for _ in range(count):
            div //= 26
        div %= 26
        if count == 0:
            val = mod
        else:
            val = div
        label += str(chr(val + casenum))
        count += 1
        n -= 26**count
    return label[::-1]


def int_to_roman(input):
   if type(input) != type(1):
      raise TypeError("expected integer, got %s" % type(input))
   if not 0 < input < 4000:
      raise ValueError("Argument must be between 1 and 3999")
   ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
   nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   result = ""
   for i in range(len(ints)):
      count = int(input / ints[i])
      result += nums[i] * count
      input -= ints[i] * count
   return result


# Copied from param should make param version public


class ProgressIndicator(param.Parameterized):
    """
    Baseclass for any ProgressIndicator that indicates progress
    as a completion percentage.
    """

    percent_range = param.NumericTuple(default=(0.0, 100.0), doc="""
        The total percentage spanned by the progress bar when called
        with a value between 0% and 100%. This allows an overall
        completion in percent to be broken down into smaller sub-tasks
        that individually complete to 100 percent.""")

    label = param.String(default='Progress', allow_None=True, doc="""
        The label of the current progress bar.""")

    def __call__(self, completion):
        raise NotImplementedError


def sort_topologically(graph):
    """
    Stackless topological sorting.

    graph = {
        3: [1],
        5: [3],
        4: [2],
        6: [4],
    }

    sort_topologically(graph)
    [[1, 2], [3, 4], [5, 6]]
    """
    levels_by_name = {}
    names_by_level = defaultdict(list)

    def add_level_to_name(name, level):
        levels_by_name[name] = level
        names_by_level[level].append(name)


    def walk_depth_first(name):
        stack = [name]
        while(stack):
            name = stack.pop()
            if name in levels_by_name:
                continue

            if name not in graph or not graph[name]:
                level = 0
                add_level_to_name(name, level)
                continue

            children = graph[name]

            children_not_calculated = [child for child in children if child not in levels_by_name]
            if children_not_calculated:
                stack.append(name)
                stack.extend(children_not_calculated)
                continue

            level = 1 + max(levels_by_name[lname] for lname in children)
            add_level_to_name(name, level)

    for name in graph:
        walk_depth_first(name)

    return list(itertools.takewhile(lambda x: x is not None,
                                    (names_by_level.get(i, None)
                                     for i in itertools.count())))


def is_cyclic(graph):
    """
    Return True if the directed graph g has a cycle. The directed graph
    should be represented as a dictionary mapping of edges for each node.
    """
    path = set()

    def visit(vertex):
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in graph)


def one_to_one(graph, nodes):
    """
    Return True if graph contains only one to one mappings. The
    directed graph should be represented as a dictionary mapping of
    edges for each node. Nodes should be passed a simple list.
    """
    edges = itertools.chain.from_iterable(graph.values())
    return len(graph) == len(nodes) and len(set(edges)) == len(nodes)


def get_overlay_spec(o, k, v):
    """
    Gets the type.group.label + key spec from an Element in an Overlay.
    """
    k = wrap_tuple(k)
    return ((type(v).__name__, v.group, v.label) + k if len(o.kdims) else
            (type(v).__name__,) + k)


def layer_sort(hmap):
   """
   Find a global ordering for layers in a HoloMap of CompositeOverlay
   types.
   """
   orderings = {}
   for o in hmap:
      okeys = [get_overlay_spec(o, k, v) for k, v in o.data.items()]
      if len(okeys) == 1 and not okeys[0] in orderings:
         orderings[okeys[0]] = []
      else:
         orderings.update({k: [] if k == v else [v] for k, v in zip(okeys[1:], okeys)})
   return [i for g in sort_topologically(orderings) for i in sorted(g)]


def layer_groups(ordering, length=2):
   """
   Splits a global ordering of Layers into groups based on a slice of
   the spec.  The grouping behavior can be modified by changing the
   length of spec the entries are grouped by.
   """
   group_orderings = defaultdict(list)
   for el in ordering:
      group_orderings[el[:length]].append(el)
   return group_orderings


def get_spec(obj):
   """
   Gets the spec from any labeled data object.
   """
   return (obj.__class__.__name__,
           obj.group, obj.label)


@contextmanager
def disable_constant(parameterized):
    """
    Temporarily set parameters on Parameterized object to
    constant=False.
    """
    params = parameterized.param.objects('existing').values()
    constants = [p.constant for p in params]
    for p in params:
        p.constant = False
    try:
        yield
    except:
        raise
    finally:
        for (p, const) in zip(params, constants):
            p.constant = const


def stream_name_mapping(stream, exclude_params=['name'], reverse=False):
    """
    Return a complete dictionary mapping between stream parameter names
    to their applicable renames, excluding parameters listed in
    exclude_params.

    If reverse is True, the mapping is from the renamed strings to the
    original stream parameter names.
    """
    filtered = [k for k in stream.param if k not in exclude_params]
    mapping = {k:stream._rename.get(k,k) for k in filtered}
    if reverse:
        return {v:k for k,v in mapping.items()}
    else:
        return mapping

def rename_stream_kwargs(stream, kwargs, reverse=False):
    """
    Given a stream and a kwargs dictionary of parameter values, map to
    the corresponding dictionary where the keys are substituted with the
    appropriately renamed string.

    If reverse, the output will be a dictionary using the original
    parameter names given a dictionary using the renamed equivalents.
    """
    mapped_kwargs = {}
    mapping = stream_name_mapping(stream, reverse=reverse)
    for k,v in kwargs.items():
        if k not in mapping:
            msg = 'Could not map key {key} {direction} renamed equivalent'
            direction = 'from' if reverse else 'to'
            raise KeyError(msg.format(key=repr(k), direction=direction))
        mapped_kwargs[mapping[k]] = v
    return mapped_kwargs


def stream_parameters(streams, no_duplicates=True, exclude=['name', '_memoize_key']):
    """
    Given a list of streams, return a flat list of parameter name,
    excluding those listed in the exclude list.

    If no_duplicates is enabled, a KeyError will be raised if there are
    parameter name clashes across the streams.
    """
    from ..streams import Params
    param_groups = {}
    for s in streams:
        if not s.contents and isinstance(s.hashkey, dict):
            param_groups[s] = list(s.hashkey)
        else:
            param_groups[s] = list(s.contents)

    if no_duplicates:
        seen, clashes = {}, []
        clash_streams = []
        for s in streams:
            if isinstance(s, Params):
                continue
            for c in param_groups[s]:
                if c in seen:
                    clashes.append(c)
                    if seen[c] not in clash_streams:
                        clash_streams.append(seen[c])
                    clash_streams.append(s)
                else:
                    seen[c] = s
        clashes = sorted(clashes)
        if clashes:
            clashing = ', '.join([repr(c) for c in clash_streams[:-1]])
            raise Exception('The supplied stream objects %s and %s '
                            'clash on the following parameters: %r'
                            % (clashing, clash_streams[-1], clashes))
    return [name for group in param_groups.values() for name in group
            if name not in exclude]


def dimensionless_contents(streams, kdims, no_duplicates=True):
    """
    Return a list of stream parameters that have not been associated
    with any of the key dimensions.
    """
    names = stream_parameters(streams, no_duplicates)
    return [name for name in names if name not in kdims]


def unbound_dimensions(streams, kdims, no_duplicates=True):
    """
    Return a list of dimensions that have not been associated with
    any streams.
    """
    params = stream_parameters(streams, no_duplicates)
    return [d for d in kdims if d not in params]


def wrap_tuple_streams(unwrapped, kdims, streams):
    """
    Fills in tuple keys with dimensioned stream values as appropriate.
    """
    param_groups = [(s.contents.keys(), s) for s in streams]
    pairs = [(name,s)  for (group, s) in param_groups for name in group]
    substituted = []
    for pos,el in enumerate(wrap_tuple(unwrapped)):
        if el is None and pos < len(kdims):
            matches = [(name,s) for (name,s) in pairs if name==kdims[pos].name]
            if len(matches) == 1:
                (name, stream) = matches[0]
                el = stream.contents[name]
        substituted.append(el)
    return tuple(substituted)


def drop_streams(streams, kdims, keys):
    """
    Drop any dimensioned streams from the keys and kdims.
    """
    stream_params = stream_parameters(streams)
    inds, dims = zip(*[(ind, kdim) for ind, kdim in enumerate(kdims)
                       if kdim not in stream_params])
    get = operator.itemgetter(*inds) # itemgetter used for performance
    keys = (get(k) for k in keys)
    return dims, ([wrap_tuple(k) for k in keys] if len(inds) == 1 else list(keys))


def iterkeys(obj):
    "Get key iterator from dictionary for Python 2 and 3"
    return iter(obj.keys()) if sys.version_info.major == 3 else obj.iterkeys()


def capitalize(string):
    """
    Capitalizes the first letter of a string.
    """
    return string[0].upper() + string[1:]


def get_path(item):
    """
    Gets a path from an Labelled object or from a tuple of an existing
    path and a labelled object. The path strings are sanitized and
    capitalized.
    """
    sanitizers = [group_sanitizer, label_sanitizer]
    if isinstance(item, tuple):
        path, item = item
        if item.label:
            if len(path) > 1 and item.label == path[1]:
                path = path[:2]
            else:
                path = path[:1] + (item.label,)
        else:
            path = path[:1]
    else:
        path = (item.group, item.label) if item.label else (item.group,)
    return tuple(capitalize(fn(p)) for (p, fn) in zip(path, sanitizers))


def make_path_unique(path, counts, new):
    """
    Given a path, a list of existing paths and counts for each of the
    existing paths.
    """
    added = False
    while any(path == c[:i] for c in counts for i in range(1, len(c)+1)):
        count = counts[path]
        counts[path] += 1
        if (not new and len(path) > 1) or added:
            path = path[:-1]
        else:
            added = True
        path = path + (int_to_roman(count),)
    if len(path) == 1:
        path = path + (int_to_roman(counts.get(path, 1)),)
    if path not in counts:
        counts[path] = 1
    return path


def dimensioned_streams(dmap):
    """
    Given a DynamicMap return all streams that have any dimensioned
    parameters, i.e. parameters also listed in the key dimensions.
    """
    dimensioned = []
    for stream in dmap.streams:
        stream_params = stream_parameters([stream])
        if set([str(k) for k in dmap.kdims]) & set(stream_params):
            dimensioned.append(stream)
    return dimensioned


def search_indices(values, source):
    """
    Given a set of values returns the indices of each of those values
    in the source array.
    """
    orig_indices = source.argsort()
    return orig_indices[np.searchsorted(source[orig_indices], values)]


def compute_edges(edges):
    """
    Computes edges as midpoints of the bin centers.  The first and
    last boundaries are equidistant from the first and last midpoints
    respectively.
    """
    edges = np.asarray(edges)
    if edges.dtype.kind == 'i':
        edges = edges.astype('f')
    midpoints = (edges[:-1] + edges[1:])/2.0
    boundaries = (2*edges[0] - midpoints[0], 2*edges[-1] - midpoints[-1])
    return np.concatenate([boundaries[:1], midpoints, boundaries[-1:]])


def mimebundle_to_html(bundle):
    """
    Converts a MIME bundle into HTML.
    """
    if isinstance(bundle, tuple):
        data, metadata = bundle
    else:
        data = bundle
    html = data.get('text/html', '')
    if 'application/javascript' in data:
        js = data['application/javascript']
        html += '\n<script type="application/javascript">{js}</script>'.format(js=js)
    return html


def closest_match(match, specs, depth=0):
    """
    Recursively iterates over type, group, label and overlay key,
    finding the closest matching spec.
    """
    new_specs = []
    match_lengths = []
    for i, spec in specs:
        if spec[0] == match[0]:
            new_specs.append((i, spec[1:]))
        else:
            if all(isinstance(s[0], basestring) for s in [spec, match]):
                match_length = max(i for i in range(len(match[0]))
                                   if match[0].startswith(spec[0][:i]))
            elif is_number(match[0]) and is_number(spec[0]):
                m = bool(match[0]) if isinstance(match[0], np.bool_) else match[0]
                s = bool(spec[0]) if isinstance(spec[0], np.bool_) else spec[0]
                match_length = -abs(m-s)
            else:
                match_length = 0
            match_lengths.append((i, match_length, spec[0]))

    if len(new_specs) == 1:
        return new_specs[0][0]
    elif new_specs:
        depth = depth+1
        return closest_match(match[1:], new_specs, depth)
    else:
        if depth == 0 or not match_lengths:
            return None
        else:
            return sorted(match_lengths, key=lambda x: -x[1])[0][0]
