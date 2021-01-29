import datetime as dt
import inspect
import itertools
import numbers
import operator
import string
import sys
import time
import unicodedata
import warnings
from collections import defaultdict, OrderedDict
from distutils.version import LooseVersion
from functools import partial
from types import FunctionType

import numpy as np
import param


# Old Python 2 compatibility stuff. To remove

basestring = str
unicode = str
long = int

datetime_types = (np.datetime64, dt.datetime, dt.date, dt.time)
timedelta_types = (np.timedelta64, dt.timedelta,)
arraylike_types = (np.ndarray,)

try:
    import pandas as pd
except ImportError:
    pd = None

if pd:
    from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtypeType
    from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
    from pandas.core.dtypes.generic import ABCExtensionArray

    pandas_datetime_types = (pd.Timestamp, DatetimeTZDtypeType, pd.Period)
    pandas_timedelta_types = (pd.Timedelta,)
    datetime_types = datetime_types + pandas_datetime_types
    timedelta_types = timedelta_types + pandas_timedelta_types
    arraylike_types = arraylike_types + (ABCSeries, ABCIndexClass)
    arraylike_types = arraylike_types + (ABCExtensionArray,)
    pandas_version = LooseVersion(pd.__version__)


numpy_version = LooseVersion(np.__version__)
param_version = LooseVersion(param.__version__)

try:
    import cftime
    cftime_types = (cftime.datetime,)
    datetime_types += cftime_types
except:
    cftime_types = ()
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])


def is_labelled_data(data):
    from .label import LabelledData
    if isinstance(data, LabelledData):
        return True
    try:
        from holoviews.core import LabelledData as _LabelledData
        return isinstance(data, _LabelledData)
    except:
        pass
    return False


def is_dimensioned(data):
    from .dimension import Dimensioned
    if isinstance(data, Dimensioned):
        return True
    try:
        from holoviews.core import Dimensioned as _Dimensioned
        return isinstance(data, _Dimensioned)
    except:
        pass
    return False


def is_multi_dimensional_mapping(data):
    from .ndmapping import MultiDimensionalMapping
    if isinstance(data, MultiDimensionalMapping):
        return True

    try:
        from holoviews.core.ndmapping import \
            MultiDimensionalMapping as _MultiDimensionalMapping
        return isinstance(data, _MultiDimensionalMapping)
    except:
        pass
    return False


def is_nd_mapping(data):
    from .ndmapping import NdMapping
    if isinstance(data, NdMapping):
        return True

    try:
        from holoviews.core.ndmapping import \
            NdMapping as _NdMapping
        return isinstance(data, _NdMapping)
    except:
        pass
    return False


def is_uniform_nd_mapping(data):
    from .ndmapping import UniformNdMapping
    if isinstance(data, UniformNdMapping):
        return True

    try:
        from holoviews.core.ndmapping import \
            UniformNdMapping as _UniformNdMapping
        return isinstance(data, _UniformNdMapping)
    except:
        pass
    return False


def is_dataframe(data):
    """
    Checks whether the supplied data is of DataFrame type.
    """
    dd = None
    if 'dask.dataframe' in sys.modules and 'pandas' in sys.modules:
        import dask.dataframe as dd
    return((pd is not None and isinstance(data, pd.DataFrame)) or
          (dd is not None and isinstance(data, dd.DataFrame)))


def is_series(data):
    """
    Checks whether the supplied data is of Series type.
    """
    dd = None
    if 'dask.dataframe' in sys.modules:
        import dask.dataframe as dd
    return((pd is not None and isinstance(data, pd.Series)) or
          (dd is not None and isinstance(data, dd.Series)))


def is_dask_array(data):
    da = None
    if 'dask.array' in sys.modules:
        import dask.array as da
    return (da is not None and isinstance(data, da.Array))


def is_cupy_array(data):
    if 'cupy' in sys.modules:
        import cupy
        return isinstance(data, cupy.ndarray)
    return False


def is_ibis_expr(data):
    if 'ibis' in sys.modules:
        import ibis
        return isinstance(data, ibis.expr.types.ColumnExpr)
    return False


def wrap_tuple(unwrapped):
    """ Wraps any non-tuple types in a tuple """
    return (unwrapped if isinstance(unwrapped, tuple) else (unwrapped,))


def unique_iterator(seq):
    """
    Returns an iterator containing all non-duplicate elements
    in the input sequence.
    """
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item


def lzip(*args):
    """
    zip function that returns a list.
    """
    return list(zip(*args))


def unique_zip(*args):
    """
    Returns a unique list of zipped values.
    """
    return list(unique_iterator(zip(*args)))


def unique_array(arr):
    """
    Returns an array of unique values in the input order.

    Args:
       arr (np.ndarray or list): The array to compute unique values on

    Returns:
       A new array of unique values
    """
    if not len(arr):
        return np.asarray(arr)
    elif pd:
        if isinstance(arr, np.ndarray) and arr.dtype.kind not in 'MO':
            # Avoid expensive unpacking if not potentially datetime
            return pd.unique(arr)

        values = []
        for v in arr:
            if (isinstance(v, datetime_types) and
                not isinstance(v, cftime_types)):
                v = pd.Timestamp(v).to_datetime64()
            values.append(v)
        return pd.unique(values)
    else:
        arr = np.asarray(arr)
        _, uniq_inds = np.unique(arr, return_index=True)
        return arr[np.sort(uniq_inds)]


def match_spec(element, specification):
    """
    Matches the group.label specification of the supplied
    element against the supplied specification dictionary
    returning the value of the best match.
    """
    match_tuple = ()
    match = specification.get((), {})
    for spec in [type(element).__name__,
                 group_sanitizer(element.group, escape=False),
                 label_sanitizer(element.label, escape=False)]:
        match_tuple += (spec,)
        if match_tuple in specification:
            match = specification[match_tuple]
    return match


def python2sort(x,key=None):
    if len(x) == 0: return x
    it = iter(x)
    groups = [[next(it)]]
    for item in it:
        for group in groups:
            try:
                item_precedence = item if key is None else key(item)
                group_precedence = group[0] if key is None else key(group[0])
                item_precedence < group_precedence  # exception if not comparable
                group.append(item)
                break
            except TypeError:
                continue
        else:  # did not break, make new group
            groups.append([item])
    return itertools.chain.from_iterable(sorted(group, key=key) for group in groups)


def merge_dimensions(dimensions_list):
    """
    Merges lists of fully or partially overlapping dimensions by
    merging their values.

    >>> from holodata.dimension import Dimension
    >>> dim_list = [[Dimension('A', values=[1, 2, 3]), Dimension('B')],
    ...             [Dimension('A', values=[2, 3, 4])]]
    >>> dimensions = merge_dimensions(dim_list)
    >>> dimensions
    [Dimension('A'), Dimension('B')]
    >>> dimensions[0].values
    [1, 2, 3, 4]
    """
    dvalues = defaultdict(list)
    dimensions = []
    for dims in dimensions_list:
        for d in dims:
            dvalues[d.name].append(d.values)
            if d not in dimensions:
                dimensions.append(d)
    dvalues = {k: list(unique_iterator(itertools.chain(*vals)))
               for k, vals in dvalues.items()}
    return [d.clone(values=dvalues.get(d.name, [])) for d in dimensions]


def dimension_sort(odict, kdims, vdims, key_index):
    """
    Sorts data by key using usual Python tuple sorting semantics
    or sorts in categorical order for any categorical Dimensions.
    """
    sortkws = {}
    ndims = len(kdims)
    dimensions = kdims+vdims
    indexes = [(dimensions[i], int(i not in range(ndims)),
                    i if i in range(ndims) else i-ndims)
                for i in key_index]
    cached_values = {d.name: [None]+list(d.values) for d in dimensions}

    if len(set(key_index)) != len(key_index):
        raise ValueError("Cannot sort on duplicated dimensions")
    else:
       sortkws['key'] = lambda x: tuple(cached_values[dim.name].index(x[t][d])
                                        if dim.values else x[t][d]
                                        for i, (dim, t, d) in enumerate(indexes))
    if sys.version_info.major == 3:
        return python2sort(odict.items(), **sortkws)
    else:
        return sorted(odict.items(), **sortkws)


def is_number(obj):
    if isinstance(obj, numbers.Number): return True
    elif isinstance(obj, (np.str_, np.unicode_)): return False
    # The extra check is for classes that behave like numbers, such as those
    # found in numpy, gmpy, etc.
    elif (hasattr(obj, '__int__') and hasattr(obj, '__add__')): return True
    # This is for older versions of gmpy
    elif hasattr(obj, 'qdiv'): return True
    else: return False


def is_int(obj, int_like=False):
    """
    Checks for int types including the native Python type and NumPy-like objects

    Args:
        obj: Object to check for integer type
        int_like (boolean): Check for float types with integer value

    Returns:
        Boolean indicating whether the supplied value is of integer type.
    """
    real_int = isinstance(obj, int) or getattr(getattr(obj, 'dtype', None), 'kind', 'o') in 'ui'
    if real_int or (int_like and hasattr(obj, 'is_integer') and obj.is_integer()):
        return True
    return False


def isdatetime(value):
    """
    Whether the array or scalar is recognized datetime type.
    """
    if isinstance(value, np.ndarray):
        return (value.dtype.kind == "M" or
                (value.dtype.kind == "O" and len(value) and
                 isinstance(value[0], datetime_types)))
    else:
        return isinstance(value, datetime_types)


def date_range(start, end, length, time_unit='us'):
    """
    Computes a date range given a start date, end date and the number
    of samples.
    """
    step = (1./compute_density(start, end, length, time_unit))
    if pd and isinstance(start, pd.Timestamp):
        start = start.to_datetime64()
    step = np.timedelta64(int(round(step)), time_unit)
    return start+step/2.+np.arange(length)*step


def parse_datetime(date):
    """
    Parses dates specified as string or integer or pandas Timestamp
    """
    if pd is None:
        raise ImportError('Parsing dates from strings requires pandas')
    return pd.to_datetime(date).to_datetime64()


def parse_datetime_selection(sel):
    """
    Parses string selection specs as datetimes.
    """
    if isinstance(sel, basestring) or isdatetime(sel):
        sel = parse_datetime(sel)
    if isinstance(sel, slice):
        if isinstance(sel.start, basestring) or isdatetime(sel.start):
            sel = slice(parse_datetime(sel.start), sel.stop)
        if isinstance(sel.stop, basestring) or isdatetime(sel.stop):
            sel = slice(sel.start, parse_datetime(sel.stop))
    if isinstance(sel, (set, list)):
        sel = [parse_datetime(v) if isinstance(v, basestring) else v for v in sel]
    return sel


def dt_to_int(value, time_unit='us'):
    """
    Converts a datetime type to an integer with the supplied time unit.
    """
    if pd:
        if isinstance(value, pd.Period):
            value = value.to_timestamp()
        if isinstance(value, pd.Timestamp):
            try:
                value = value.to_datetime64()
            except:
                value = np.datetime64(value.to_pydatetime())
    elif isinstance(value, cftime_types):
        return cftime_to_timestamp(value, time_unit)

    if isinstance(value, dt.date):
        value = dt.datetime(*value.timetuple()[:6])

    # Handle datetime64 separately
    if isinstance(value, np.datetime64):
        try:
            value = np.datetime64(value, 'ns')
            tscale = (np.timedelta64(1, time_unit)/np.timedelta64(1, 'ns'))
            return value.tolist()/tscale
        except:
            # If it can't handle ns precision fall back to datetime
            value = value.tolist()

    if time_unit == 'ns':
        tscale = 1e9
    else:
        tscale = 1./np.timedelta64(1, time_unit).tolist().total_seconds()

    try:
        # Handle python3
        return int(value.timestamp() * tscale)
    except:
        # Handle python2
        return (time.mktime(value.timetuple()) + value.microsecond / 1e6) * tscale


def cftime_to_timestamp(date, time_unit='us'):
    """Converts cftime to timestamp since epoch in milliseconds

    Non-standard calendars (e.g. Julian or no leap calendars)
    are converted to standard Gregorian calendar. This can cause
    extra space to be added for dates that don't exist in the original
    calendar. In order to handle these dates correctly a custom bokeh
    model with support for other calendars would have to be defined.

    Args:
        date: cftime datetime object (or array)

    Returns:
        time_unit since 1970-01-01 00:00:00
    """
    import cftime
    utime = cftime.utime('microseconds since 1970-01-01 00:00:00')
    if time_unit == 'us':
        tscale = 1
    else:
        tscale = (np.timedelta64(1, 'us')/np.timedelta64(1, time_unit))
    return utime.date2num(date)*tscale


def numpy_scalar_to_python(scalar):
    """
    Converts a NumPy scalar to a regular python type.
    """
    scalar_type = type(scalar)
    if np.issubclass_(scalar_type, np.float_):
        return float(scalar)
    elif np.issubclass_(scalar_type, np.int_):
        return int(scalar)
    return scalar


def isscalar(val):
    """
    Value is scalar or None
    """
    return val is None or np.isscalar(val) or isinstance(val, datetime_types)


def isnumeric(val):
    if isinstance(val, (basestring, bool, np.bool_)):
        return False
    try:
        float(val)
        return True
    except:
        return False


def asarray(arraylike, strict=True):
    """
    Converts arraylike objects to NumPy ndarray types. Errors if
    object is not arraylike and strict option is enabled.
    """
    if isinstance(arraylike, np.ndarray):
        return arraylike
    elif isinstance(arraylike, list):
        return np.asarray(arraylike, dtype=object)
    elif not isinstance(arraylike, np.ndarray) and isinstance(arraylike, arraylike_types):
        return arraylike.values
    elif hasattr(arraylike, '__array__'):
        return np.asarray(arraylike)
    elif strict:
        raise ValueError('Could not convert %s type to array' % type(arraylike))
    return arraylike


def isnat(val):
    """
    Checks if the value is a NaT. Should only be called on datetimelike objects.
    """
    if (isinstance(val, (np.datetime64, np.timedelta64)) or
        (isinstance(val, np.ndarray) and val.dtype.kind == 'M')):
        if numpy_version >= '1.13':
            return np.isnat(val)
        else:
            return val.view('i8') == nat_as_integer
    elif pd and val is pd.NaT:
        return True
    elif pd and isinstance(val, pandas_datetime_types+pandas_timedelta_types):
        return pd.isna(val)
    else:
        return False


def isfinite(val):
    """
    Helper function to determine if scalar or array value is finite extending
    np.isfinite with support for None, string, datetime types.
    """
    is_dask = is_dask_array(val)
    if not np.isscalar(val) and not is_dask:
        val = asarray(val, strict=False)

    if val is None:
        return False
    elif is_dask:
        import dask.array as da
        return da.isfinite(val)
    elif isinstance(val, np.ndarray):
        if val.dtype.kind == 'M':
            return ~isnat(val)
        elif val.dtype.kind == 'O':
            return np.array([isfinite(v) for v in val], dtype=bool)
        elif val.dtype.kind in 'US':
            return ~pd.isna(val) if pd else np.ones_like(val, dtype=bool)
        finite = np.isfinite(val)
        if pd and pandas_version >= '1.0.0':
            finite &= ~pd.isna(val)
        return finite
    elif isinstance(val, datetime_types+timedelta_types):
        return not isnat(val)
    elif isinstance(val, (basestring, bytes)):
        return True
    finite = np.isfinite(val)
    if pd and pandas_version >= '1.0.0':
        if finite is pd.NA:
            return False
        return finite & (~pd.isna(val))
    return finite


nat_as_integer = np.datetime64('NAT').view('i8')


def cartesian_product(arrays, flat=True, copy=False):
    """
    Efficient cartesian product of a list of 1D arrays returning the
    expanded array views for each dimensions. By default arrays are
    flattened, which may be controlled with the flat flag. The array
    views can be turned into regular arrays with the copy flag.
    """
    arrays = np.broadcast_arrays(*np.ix_(*arrays))
    if flat:
        return tuple(arr.flatten() if copy else arr.flat for arr in arrays)
    return tuple(arr.copy() if copy else arr for arr in arrays)


def cross_index(values, index):
    """
    Allows efficiently indexing into a cartesian product without
    expanding it. The values should be defined as a list of iterables
    making up the cartesian product and a linear index, returning
    the cross product of the values at the supplied index.
    """
    lengths = [len(v) for v in values]
    length = np.product(lengths)
    if index >= length:
        raise IndexError('Index %d out of bounds for cross-product of size %d'
                         % (index, length))
    indexes = []
    for i in range(1, len(values))[::-1]:
        p = np.product(lengths[-i:])
        indexes.append(index//p)
        index -= indexes[-1] * p
    indexes.append(index)
    return tuple(v[i] for v, i in zip(values, indexes))


def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray['f%s' % i] = array
    return recarray.argsort()


def expand_grid_coords(dataset, dim):
    """
    Expand the coordinates along a dimension of the gridded
    dataset into an ND-array matching the dimensionality of
    the dataset.
    """
    irregular = [d.name for d in dataset.kdims
                 if d is not dim and dataset.interface.irregular(dataset, d)]
    if irregular:
        array = dataset.interface.coords(dataset, dim, True)
        example = dataset.interface.values(dataset, irregular[0], True, False)
        return array * np.ones_like(example)
    else:
        arrays = [dataset.interface.coords(dataset, d.name, True)
                  for d in dataset.kdims]
        idx = dataset.get_dimension_index(dim)
        return cartesian_product(arrays, flat=False)[idx].T


def dt64_to_dt(dt64):
    """
    Safely converts NumPy datetime64 to a datetime object.
    """
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=ts)


def is_nan(x):
    """
    Checks whether value is NaN on arbitrary types
    """
    try:
        return np.isnan(x)
    except:
        return False


def validate_regular_sampling(values, rtol=10e-6):
    """
    Validates regular sampling of a 1D array ensuring that the difference
    in sampling steps is at most rtol times the smallest sampling step.
    Returns a boolean indicating whether the sampling is regular.
    """
    diffs = np.diff(values)
    return (len(diffs) < 1) or abs(diffs.min()-diffs.max()) < abs(diffs.min()*rtol)


def bound_range(vals, density, time_unit='us'):
    """
    Computes a bounding range and density from a number of samples
    assumed to be evenly spaced. Density is rounded to machine precision
    using significant digits reported by sys.float_info.dig.
    """
    if not len(vals):
        return(np.nan, np.nan, density, False)
    low, high = vals.min(), vals.max()
    invert = False
    if len(vals) > 1 and vals[0] > vals[1]:
        invert = True
    if not density:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in double_scalars')
            full_precision_density = compute_density(low, high, len(vals)-1)
            density = round(full_precision_density, sys.float_info.dig)
        if density == 0:
            density = full_precision_density
    if density == 0:
        raise ValueError('Could not determine Image density, ensure it has a non-zero range.')
    halfd = 0.5/density
    if isinstance(low, datetime_types):
        halfd = np.timedelta64(int(round(halfd)), time_unit)
    return low-halfd, high+halfd, density, invert


def max_range(ranges, combined=True):
    """
    Computes the maximal lower and upper bounds from a list bounds.

    Args:
       ranges (list of tuples): A list of range tuples
       combined (boolean, optional): Whether to combine bounds
          Whether range should be computed on lower and upper bound
          independently or both at once

    Returns:
       The maximum range as a single tuple
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            values = [tuple(np.NaN if v is None else v for v in r) for r in ranges]
            if pd and any(isinstance(v, datetime_types) and not isinstance(v, cftime_types+(dt.time,))
                          for r in values for v in r):
                converted = []
                for l, h in values:
                    if isinstance(l, datetime_types) and isinstance(h, datetime_types):
                        l, h = (pd.Timestamp(l).to_datetime64(),
                                pd.Timestamp(h).to_datetime64())
                    converted.append((l, h))
                values = converted

            arr = np.array(values)
            if not len(arr):
                return np.NaN, np.NaN
            elif arr.dtype.kind in 'OSU':
                arr = list(python2sort([
                    v for r in values for v in r
                    if not is_nan(v) and v is not None]))
                return arr[0], arr[-1]
            elif arr.dtype.kind in 'M':
                drange = ((arr.min(), arr.max()) if combined else
                          (arr[:, 0].min(), arr[:, 1].max()))
                return drange

            if combined:
                return (np.nanmin(arr), np.nanmax(arr))
            else:
                return (np.nanmin(arr[:, 0]), np.nanmax(arr[:, 1]))
    except:
        return (np.NaN, np.NaN)


def bytes_to_unicode(value):
    """
    Safely casts bytestring to unicode
    """
    if isinstance(value, bytes):
        return unicode(value.decode('utf-8'))
    return value


def find_range(values, soft_range=[]):
    """
    Safely finds either the numerical min and max of
    a set of values, falling back to the first and
    the last value in the sorted list of values.
    """
    try:
        values = np.array(values)
        values = np.squeeze(values) if len(values.shape) > 1 else values
        if len(soft_range):
            values = np.concatenate([values, soft_range])
        if values.dtype.kind == 'M':
            return values.min(), values.max()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            return np.nanmin(values), np.nanmax(values)
    except:
        try:
            values = sorted(values)
            return (values[0], values[-1])
        except:
            return (None, None)


def dimension_range(lower, upper, hard_range, soft_range, padding=None, log=False):
    """
    Computes the range along a dimension by combining the data range
    with the Dimension soft_range and range.
    """
    plower, pupper = range_pad(lower, upper, padding, log)
    if isfinite(soft_range[0]) and soft_range[0] <= lower:
        lower = soft_range[0]
    else:
        lower = max_range([(plower, None), (soft_range[0], None)])[0]
    if isfinite(soft_range[1]) and soft_range[1] >= upper:
        upper = soft_range[1]
    else:
        upper = max_range([(None, pupper), (None, soft_range[1])])[1]
    dmin, dmax = hard_range
    lower = lower if dmin is None or not isfinite(dmin) else dmin
    upper = upper if dmax is None or not isfinite(dmax) else dmax
    return lower, upper


def capitalize_unicode_name(s):
    """
    Turns a string such as 'capital delta' into the shortened,
    capitalized version, in this case simply 'Delta'. Used as a
    transform in sanitize_identifier.
    """
    index = s.find('capital')
    if index == -1: return s
    tail = s[index:].replace('capital', '').strip()
    tail = tail[0].upper() + tail[1:]
    return s[:index] + tail


class sanitize_identifier_fn(param.ParameterizedFunction):
    """
    Sanitizes group/label values for use in AttrTree attribute
    access. Depending on the version parameter, either sanitization
    appropriate for Python 2 (no unicode gn identifiers allowed) or
    Python 3 (some unicode allowed) is used.

    Note that if you are using Python 3, you can switch to version 2
    for compatibility but you cannot enable relaxed sanitization if
    you are using Python 2.

    Special characters are sanitized using their (lowercase) unicode
    name using the unicodedata module. For instance:

    >>> unicodedata.name(u'$').lower()
    'dollar sign'

    As these names are often very long, this parameterized function
    allows filtered, substitutions and transforms to help shorten these
    names appropriately.
    """

    version = param.ObjectSelector(sys.version_info.major, objects=[2,3], doc="""
        The sanitization version. If set to 2, more aggressive
        sanitization appropriate for Python 2 is applied. Otherwise,
        if set to 3, more relaxed, Python 3 sanitization is used.""")

    capitalize = param.Boolean(default=True, doc="""
       Whether the first letter should be converted to
       uppercase. Note, this will only be applied to ASCII characters
       in order to make sure paths aren't confused with method
       names.""")

    eliminations = param.List(['extended', 'accent', 'small', 'letter', 'sign', 'digit',
                               'latin', 'greek', 'arabic-indic', 'with', 'dollar'], doc="""
       Lowercase strings to be eliminated from the unicode names in
       order to shorten the sanitized name ( lowercase). Redundant
       strings should be removed but too much elimination could cause
       two unique strings to map to the same sanitized output.""")

    substitutions = param.Dict(default={'circumflex':'power',
                                        'asterisk':'times',
                                        'solidus':'over'}, doc="""
       Lowercase substitutions of substrings in unicode names. For
       instance the ^ character has the name 'circumflex accent' even
       though it is more typically used for exponentiation. Note that
       substitutions occur after filtering and that there should be no
       ordering dependence between substitutions.""")

    transforms = param.List(default=[capitalize_unicode_name], doc="""
       List of string transformation functions to apply after
       filtering and substitution in order to further compress the
       unicode name. For instance, the default capitalize_unicode_name
       function will turn the string "capital delta" into "Delta".""")

    disallowed = param.List(default=['trait_names', '_ipython_display_',
                                     '_getAttributeNames'], doc="""
       An explicit list of name that should not be allowed as
       attribute names on Tree objects.

       By default, prevents IPython from creating an entry called
       Trait_names due to an inconvenient getattr check (during
       tab-completion).""")

    disable_leading_underscore = param.Boolean(default=False, doc="""
       Whether leading underscores should be allowed to be sanitized
       with the leading prefix.""")

    aliases = param.Dict(default={}, doc="""
       A dictionary of aliases mapping long strings to their short,
       sanitized equivalents""")

    prefix = 'A_'

    _lookup_table = param.Dict(default={}, doc="""
       Cache of previously computed sanitizations""")


    @param.parameterized.bothmethod
    def add_aliases(self_or_cls, **kwargs):
        """
        Conveniently add new aliases as keyword arguments. For instance
        you can add a new alias with add_aliases(short='Longer string')
        """
        self_or_cls.aliases.update({v:k for k,v in kwargs.items()})

    @param.parameterized.bothmethod
    def remove_aliases(self_or_cls, aliases):
        """
        Remove a list of aliases.
        """
        for k,v in self_or_cls.aliases.items():
            if v in aliases:
                self_or_cls.aliases.pop(k)

    @param.parameterized.bothmethod
    def allowable(self_or_cls, name, disable_leading_underscore=None):
       disabled_reprs = ['javascript', 'jpeg', 'json', 'latex',
                         'latex', 'pdf', 'png', 'svg', 'markdown']
       disabled_ = (self_or_cls.disable_leading_underscore
                    if disable_leading_underscore is None
                    else disable_leading_underscore)
       if disabled_ and name.startswith('_'):
          return False
       isrepr = any(('_repr_%s_' % el) == name for el in disabled_reprs)
       return (name not in self_or_cls.disallowed) and not isrepr

    @param.parameterized.bothmethod
    def prefixed(self, identifier, version):
        """
        Whether or not the identifier will be prefixed.
        Strings that require the prefix are generally not recommended.
        """
        invalid_starting = ['Mn', 'Mc', 'Nd', 'Pc']
        if identifier.startswith('_'):  return True
        return((identifier[0] in string.digits) if version==2
               else (unicodedata.category(identifier[0]) in invalid_starting))

    @param.parameterized.bothmethod
    def remove_diacritics(self_or_cls, identifier):
        """
        Remove diacritics and accents from the input leaving other
        unicode characters alone."""
        chars = ''
        for c in identifier:
            replacement = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
            if replacement != '':
                chars += bytes_to_unicode(replacement)
            else:
                chars += c
        return chars

    @param.parameterized.bothmethod
    def shortened_character_name(self_or_cls, c, eliminations=[], substitutions={}, transforms=[]):
        """
        Given a unicode character c, return the shortened unicode name
        (as a list of tokens) by applying the eliminations,
        substitutions and transforms.
        """
        name = unicodedata.name(c).lower()
        # Filtering
        for elim in eliminations:
            name = name.replace(elim, '')
        # Substitution
        for i,o in substitutions.items():
            name = name.replace(i, o)
        for transform in transforms:
            name = transform(name)
        return ' '.join(name.strip().split()).replace(' ','_').replace('-','_')


    def __call__(self, name, escape=True, version=None):
        if name in [None, '']:
           return name
        elif name in self.aliases:
            return self.aliases[name]
        elif name in self._lookup_table:
           return self._lookup_table[name]
        name = bytes_to_unicode(name)
        version = self.version if version is None else version
        if not self.allowable(name):
            raise AttributeError("String %r is in the disallowed list of attribute names: %r" % (name, self.disallowed))

        if version == 2:
            name = self.remove_diacritics(name)
        if self.capitalize and name and name[0] in string.ascii_lowercase:
            name = name[0].upper()+name[1:]

        sanitized = (self.sanitize_py2(name) if version==2 else self.sanitize_py3(name))
        if self.prefixed(name, version):
           sanitized = self.prefix + sanitized
        self._lookup_table[name] = sanitized
        return sanitized


    def _process_underscores(self, tokens):
        "Strip underscores to make sure the number is correct after join"
        groups = [[str(''.join(el))] if b else list(el)
                  for (b,el) in itertools.groupby(tokens, lambda k: k=='_')]
        flattened = [el for group in groups for el in group]
        processed = []
        for token in flattened:
            if token == '_':  continue
            if token.startswith('_'):
                token = str(token[1:])
            if token.endswith('_'):
                token = str(token[:-1])
            processed.append(token)
        return processed

    def sanitize_py2(self, name):
        # This fix works but masks an issue in self.sanitize (py2)
        prefix = '_' if name.startswith('_') else ''
        valid_chars = string.ascii_letters+string.digits+'_'
        return prefix + str('_'.join(self.sanitize(name, lambda c: c in valid_chars)))


    def sanitize_py3(self, name):
        if not name.isidentifier():
            return '_'.join(self.sanitize(name, lambda c: ('_'+c).isidentifier()))
        else:
            return name

    def sanitize(self, name, valid_fn):
        "Accumulate blocks of hex and separate blocks by underscores"
        invalid = {'\a':'a','\b':'b', '\v':'v','\f':'f','\r':'r'}
        for cc in filter(lambda el: el in name, invalid.keys()):
            raise Exception("Please use a raw string or escape control code '\%s'"
                            % invalid[cc])
        sanitized, chars = [], ''
        for split in name.split():
            for c in split:
                if valid_fn(c): chars += str(c) if c=='_' else c
                else:
                    short = self.shortened_character_name(c, self.eliminations,
                                                         self.substitutions,
                                                         self.transforms)
                    sanitized.extend([chars] if chars else [])
                    if short != '':
                       sanitized.append(short)
                    chars = ''
            if chars:
                sanitized.extend([chars])
                chars=''
        return self._process_underscores(sanitized + ([chars] if chars else []))


sanitize_identifier = sanitize_identifier_fn.instance()
group_sanitizer = sanitize_identifier_fn.instance()
label_sanitizer = sanitize_identifier_fn.instance()
dimension_sanitizer = sanitize_identifier_fn.instance(capitalize=False)


def get_param_values(data):
    params = dict(kdims=data.kdims, vdims=data.vdims,
                  label=data.label)
    if (data.group != data.param.objects(False)['group'].default and not
        isinstance(type(data).group, property)):
        params['group'] = data.group
    return params


class ndmapping_groupby(param.ParameterizedFunction):
    """
    Apply a groupby operation to an NdMapping, using pandas to improve
    performance (if available).
    """

    sort = param.Boolean(default=False, doc='Whether to apply a sorted groupby')

    def __call__(self, ndmapping, dimensions, container_type,
                 group_type, sort=False, **kwargs):
        try:
            import pandas # noqa (optional import)
            groupby = self.groupby_pandas
        except:
            groupby = self.groupby_python
        return groupby(ndmapping, dimensions, container_type,
                       group_type, sort=sort, **kwargs)

    @param.parameterized.bothmethod
    def groupby_pandas(self_or_cls, ndmapping, dimensions, container_type,
                       group_type, sort=False, **kwargs):
        if 'kdims' in kwargs:
            idims = [ndmapping.get_dimension(d) for d in kwargs['kdims']]
        else:
            idims = [dim for dim in ndmapping.kdims if dim not in dimensions]

        all_dims = [d.name for d in ndmapping.kdims]
        inds = [ndmapping.get_dimension_index(dim) for dim in idims]
        getter = operator.itemgetter(*inds) if inds else lambda x: tuple()

        multi_index = pd.MultiIndex.from_tuples(ndmapping.keys(), names=all_dims)
        df = pd.DataFrame(list(map(wrap_tuple, ndmapping.values())), index=multi_index)

        # TODO: Look at sort here
        kwargs = dict(dict(get_param_values(ndmapping), kdims=idims), sort=sort, **kwargs)
        groups = ((wrap_tuple(k), group_type(OrderedDict(unpack_group(group, getter)), **kwargs))
                   for k, group in df.groupby(level=[d.name for d in dimensions], sort=sort))

        if sort:
            selects = list(get_unique_keys(ndmapping, dimensions))
            groups = sorted(groups, key=lambda x: selects.index(x[0]))

        return container_type(groups, kdims=dimensions, sort=sort)

    @param.parameterized.bothmethod
    def groupby_python(self_or_cls, ndmapping, dimensions, container_type,
                       group_type, sort=False, **kwargs):
        idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
        dim_names = [dim.name for dim in dimensions]
        selects = get_unique_keys(ndmapping, dimensions)
        selects = group_select(list(selects))
        groups = [(k, group_type((v.reindex(idims) if hasattr(v, 'kdims')
                                  else [((), v)]), **kwargs))
                  for k, v in iterative_select(ndmapping, dim_names, selects)]
        return container_type(groups, kdims=dimensions)


def process_ellipses(obj, key, vdim_selection=False):
    """
    Helper function to pad a __getitem__ key with the right number of
    empty slices (i.e. :) when the key contains an Ellipsis (...).

    If the vdim_selection flag is true, check if the end of the key
    contains strings or Dimension objects in obj. If so, extra padding
    will not be applied for the value dimensions (i.e. the resulting key
    will be exactly one longer than the number of kdims). Note: this
    flag should not be used for composite types.
    """
    if getattr(getattr(key, 'dtype', None), 'kind', None) == 'b':
        return key
    wrapped_key = wrap_tuple(key)
    ellipse_count = sum(1 for k in wrapped_key if k is Ellipsis)
    if ellipse_count == 0:
        return key
    elif ellipse_count != 1:
        raise Exception("Only one ellipsis allowed at a time.")
    dim_count = len(obj.dimensions())
    index = wrapped_key.index(Ellipsis)
    head = wrapped_key[:index]
    tail = wrapped_key[index+1:]

    padlen = dim_count - (len(head) + len(tail))
    if vdim_selection:
        # If the end of the key (i.e. the tail) is in vdims, pad to len(kdims)+1
        if wrapped_key[-1] in obj.vdims:
            padlen = (len(obj.kdims) +1 ) - len(head+tail)
    return head + ((slice(None),) * padlen) + tail


def get_ndmapping_label(ndmapping, attr):
    """
    Function to get the first non-auxiliary object
    label attribute from an NdMapping.
    """
    label = None
    els = itervalues(ndmapping.data)
    while label is None:
        try:
            el = next(els)
        except StopIteration:
            return None
        if not getattr(el, '_auxiliary_component', True):
            label = getattr(el, attr)
    if attr == 'group':
        tp = type(el).__name__
        if tp == label:
            return None
    return label


def compute_density(start, end, length, time_unit='us'):
    """
    Computes a grid density given the edges and number of samples.
    Handles datetime grids correctly by computing timedeltas and
    computing a density for the given time_unit.
    """
    if isinstance(start, int): start = float(start)
    if isinstance(end, int): end = float(end)
    diff = end-start
    if isinstance(diff, timedelta_types):
        if isinstance(diff, np.timedelta64):
            diff = np.timedelta64(diff, time_unit).tolist()
        tscale = 1./np.timedelta64(1, time_unit).tolist().total_seconds()
        return (length/(diff.total_seconds()*tscale))
    else:
        return length/diff

def range_pad(lower, upper, padding=None, log=False):
    """
    Pads the range by a fraction of the interval
    """
    if padding is not None and not isinstance(padding, tuple):
        padding = (padding, padding)
    if is_number(lower) and is_number(upper) and padding is not None:
        if not isinstance(lower, datetime_types) and log and lower > 0 and upper > 0:
            log_min = np.log(lower) / np.log(10)
            log_max = np.log(upper) / np.log(10)
            lspan = (log_max-log_min)*(1+padding[0]*2)
            uspan = (log_max-log_min)*(1+padding[1]*2)
            center = (log_min+log_max) / 2.0
            start, end = np.power(10, center-lspan/2.), np.power(10, center+uspan/2.)
        else:
            if isinstance(lower, datetime_types) and not isinstance(lower, cftime_types):
                # Ensure timedelta can be safely divided
                lower, upper = np.datetime64(lower), np.datetime64(upper)
                span = (upper-lower).astype('>m8[ns]')
            else:
                span = (upper-lower)
            lpad = span*(padding[0])
            upad = span*(padding[1])
            start, end = lower-lpad, upper+upad
    else:
        start, end = lower, upper

    return start, end


def group_select(selects, length=None, depth=None):
    """
    Given a list of key tuples to select, groups them into sensible
    chunks to avoid duplicating indexing operations.
    """
    if length == None and depth == None:
        length = depth = len(selects[0])
    getter = operator.itemgetter(depth-length)
    if length > 1:
        selects = sorted(selects, key=getter)
        grouped_selects = defaultdict(dict)
        for k, v in itertools.groupby(selects, getter):
            grouped_selects[k] = group_select(list(v), length-1, depth)
        return grouped_selects
    else:
        return list(selects)


def iterative_select(obj, dimensions, selects, depth=None):
    """
    Takes the output of group_select selecting subgroups iteratively,
    avoiding duplicating select operations.
    """
    ndims = len(dimensions)
    depth = depth if depth is not None else ndims
    items = []
    if isinstance(selects, dict):
        for k, v in selects.items():
            items += iterative_select(obj.select(**{dimensions[ndims-depth]: k}),
                                      dimensions, v, depth-1)
    else:
        for s in selects:
            items.append((s, obj.select(**{dimensions[-1]: s[-1]})))
    return items


def itervalues(obj):
    "Get value iterator from dictionary for Python 2 and 3"
    return iter(obj.values()) if sys.version_info.major == 3 else obj.itervalues()


def get_unique_keys(ndmapping, dimensions):
    inds = [ndmapping.get_dimension_index(dim) for dim in dimensions]
    getter = operator.itemgetter(*inds)
    return unique_iterator(getter(key) if len(inds) > 1 else (key[inds[0]],)
                           for key in ndmapping.data.keys())


def unpack_group(group, getter):
    for k, v in group.iterrows():
        obj = v.values[0]
        key = getter(k)
        if hasattr(obj, 'kdims'):
            yield (key, obj)
        else:
            yield (wrap_tuple(key), obj)


def resolve_dependent_value(value):
    """Resolves parameter dependencies on the supplied value

    Resolves parameter values, Parameterized instance methods and
    parameterized functions with dependencies on the supplied value.

    Args:
       value: A value which will be resolved

    Returns:
       A new dictionary where any parameter dependencies have been
       resolved.
    """
    range_widget = False
    if 'panel' in sys.modules:
        from panel.widgets import RangeSlider, Widget
        range_widget = isinstance(value, RangeSlider)
        try:
            from panel.depends import param_value_if_widget
            value = param_value_if_widget(value)
        except Exception:
            if isinstance(value, Widget):
                value = value.param.value
    if is_param_method(value, has_deps=True):
        value = value()
    elif isinstance(value, param.Parameter) and isinstance(value.owner, param.Parameterized):
        value = getattr(value.owner, value.name)
    elif isinstance(value, FunctionType) and hasattr(value, '_dinfo'):
        deps = value._dinfo
        args = (getattr(p.owner, p.name) for p in deps.get('dependencies', []))
        kwargs = {k: getattr(p.owner, p.name) for k, p in deps.get('kw', {}).items()}
        value = value(*args, **kwargs)
    if isinstance(value, tuple) and range_widget:
        value = slice(*value)
    return value


def resolve_dependent_kwargs(kwargs):
    """Resolves parameter dependencies in the supplied dictionary

    Resolves parameter values, Parameterized instance methods and
    parameterized functions with dependencies in the supplied
    dictionary.

    Args:
       kwargs (dict): A dictionary of keyword arguments

    Returns:
       A new dictionary with where any parameter dependencies have been
       resolved.
    """
    return {k: resolve_dependent_value(v) for k, v in kwargs.items()}


def is_param_method(obj, has_deps=False):
    """Whether the object is a method on a parameterized object.

    Args:
       obj: Object to check
       has_deps (boolean, optional): Check for dependencies
          Whether to also check whether the method has been annotated
          with param.depends

    Returns:
       A boolean value indicating whether the object is a method
       on a Parameterized object and if enabled whether it has any
       dependencies
    """
    parameterized = (inspect.ismethod(obj) and
                     isinstance(get_method_owner(obj), param.Parameterized))
    if parameterized and has_deps:
        return getattr(obj, "_dinfo", {}).get('dependencies')
    return parameterized


def get_method_owner(method):
    """
    Gets the instance that owns the supplied method
    """
    if isinstance(method, partial):
        method = method.func
    return method.__self__ if sys.version_info.major >= 3 else method.im_self
