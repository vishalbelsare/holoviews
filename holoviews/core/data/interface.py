from __future__ import absolute_import

import sys
import warnings

import six
import param
import numpy as np

from .. import util
from ..element import Element
from ..ndmapping import NdMapping


def get_array_types():
    array_types = (np.ndarray,)
    da = dask_array_module()
    if da is not None:
        array_types += (da.Array,)
    return array_types

def dask_array_module():
    try:
        import dask.array as da
        return da
    except:
        return None

def is_dask(array):
    da = dask_array_module()
    if da is None:
        return False
    return da and isinstance(array, da.Array)

def cached(method):
    """
    Decorates an Interface method and using a cached version
    """
    def cached(*args, **kwargs):
        cache = getattr(args[1], '_cached')
        if cache is None:
            return method(*args, **kwargs)
        else:
            args = (cache,)+args[2:]
            return getattr(cache.interface, method.__name__)(*args, **kwargs)
    return cached


class DataError(ValueError):
    "DataError is raised when the data cannot be interpreted"

    def __init__(self, msg, interface=None):
        if interface is not None:
            msg = '\n\n'.join([msg, interface.error()])
        super(DataError, self).__init__(msg)


class Accessor(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        from ..data import Dataset
        from ...operation.element import method
        in_method = self.dataset._in_method
        if not in_method:
            self.dataset._in_method = True
        try:
            res = self._perform_getitem(self.dataset, index)
            if not in_method and isinstance(res, Dataset):
                getitem_op = method.instance(
                    input_type=type(self),
                    output_type=type(self.dataset),
                    method_name='_perform_getitem',
                    args=[index],
                )
                res._pipeline = self.dataset.pipeline.instance(
                    operations=self.dataset.pipeline.operations + [getitem_op],
                    output_type=type(self.dataset)
                )
        finally:
            if not in_method:
                self.dataset._in_method = False
        return res

    @classmethod
    def _perform_getitem(cls, dataset, index):
        raise NotImplementedError()


class iloc(Accessor):
    """
    iloc is small wrapper object that allows row, column based
    indexing into a Dataset using the ``.iloc`` property.  It supports
    the usual numpy and pandas iloc indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.iloc`` property docstring.
    """

    @classmethod
    def _perform_getitem(cls, dataset, index):
        index = util.wrap_tuple(index)
        if len(index) == 1:
            index = (index[0], slice(None))
        elif len(index) > 2:
            raise IndexError('Tabular index not understood, index '
                             'must be at most length 2.')

        rows, cols = index
        if rows is Ellipsis:
            rows = slice(None)

        data = dataset.interface.iloc(dataset, (rows, cols))
        kdims = dataset.kdims
        vdims = dataset.vdims
        if util.isscalar(data):
            return data
        elif cols == slice(None):
            pass
        else:
            if isinstance(cols, slice):
                dims = dataset.dimensions()[index[1]]
            elif np.isscalar(cols):
                dims = [dataset.get_dimension(cols)]
            else:
                dims = [dataset.get_dimension(d) for d in cols]
            kdims = [d for d in dims if d in kdims]
            vdims = [d for d in dims if d in vdims]

        datatypes = util.unique_iterator([dataset.interface.datatype]+dataset.datatype)
        datatype = [dt for dt in datatypes if dt in Driver.interfaces and
                    not Driver.interfaces[dt].gridded]
        if not datatype: datatype = ['dataframe', 'dictionary']
        return dataset.clone(data, kdims=kdims, vdims=vdims, datatype=datatype)


class ndloc(Accessor):
    """
    ndloc is a small wrapper object that allows ndarray-like indexing
    for gridded Datasets using the ``.ndloc`` property. It supports
    the standard NumPy ndarray indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.ndloc`` property docstring.
    """
    @classmethod
    def _perform_getitem(cls, dataset, indices):
        ds = dataset
        indices = util.wrap_tuple(indices)
        if not ds.interface.gridded:
            raise IndexError('Cannot use ndloc on non nd-dimensional datastructure')
        selected = dataset.interface.ndloc(ds, indices)
        if np.isscalar(selected):
            return selected
        params = {}
        if hasattr(ds, 'bounds'):
            params['bounds'] = None
        return dataset.clone(selected, datatype=[ds.interface.datatype]+ds.datatype, **params)


class Driver(param.Parameterized):

    interfaces = {}

    datatype = None

    types = ()

    # Denotes whether the interface expects gridded data
    gridded = False

    # Denotes whether the interface expects ragged data
    multi = False

    # Whether the interface stores the names of the underlying dimensions
    named = True

    @classmethod
    def loaded(cls):
        """
        Indicates whether the required dependencies are loaded.
        """
        return True

    @classmethod
    def applies(cls, obj):
        """
        Indicates whether the interface is designed specifically to
        handle the supplied object's type. By default simply checks
        if the object is one of the types declared on the class,
        however if the type is expensive to import at load time the
        method may be overridden.
        """
        return type(obj) in cls.types

    @classmethod
    def register(cls, interface):
        cls.interfaces[interface.datatype] = interface

    @classmethod
    def cast(cls, datasets, datatype=None, cast_type=None):
        """
        Given a list of Dataset objects, cast them to the specified
        datatype (by default the format matching the current interface)
        with the given cast_type (if specified).
        """
        datatype = datatype or cls.datatype
        cast = []
        for ds in datasets:
            if cast_type is not None or ds.interface.datatype != datatype:
                ds = ds.clone(ds, datatype=[datatype], new_type=cast_type)
            cast.append(ds)
        return cast

    @classmethod
    def error(cls):
        info = dict(interface=cls.__name__)
        url = "http://holoviews.org/user_guide/%s_Datasets.html"
        if cls.multi:
            datatype = 'a list of tabular'
            info['url'] = url % 'Tabular'
        else:
            if cls.gridded:
                datatype = 'gridded'
            else:
                datatype = 'tabular'
            info['url'] = url % datatype.capitalize()
        info['datatype'] = datatype
        return ("{interface} expects {datatype} data, for more information "
                "on supported datatypes see {url}".format(**info))


    @classmethod
    def initialize(cls, eltype, data, kdims, vdims, datatype=None):
        # Process params and dimensions
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kdims = pvals.get('kdims') if kdims is None else kdims
            vdims = pvals.get('vdims') if vdims is None else vdims

        # Process Element data
        if (hasattr(data, 'interface') and isinstance(data.interface, Interface)):
            if datatype is None:
                datatype = [dt for dt in data.datatype if dt in eltype.datatype]
                if not datatype:
                    datatype = eltype.datatype

            interface = data.interface
            if interface.datatype in datatype and interface.datatype in eltype.datatype and interface.named:
                data = data.data
            elif interface.multi and any(cls.interfaces[dt].multi for dt in datatype if dt in cls.interfaces):
                data = [d for d in data.interface.split(data, None, None, 'columns')]
            elif interface.gridded and any(cls.interfaces[dt].gridded for dt in datatype):
                new_data = []
                for kd in data.kdims:
                    irregular = interface.irregular(data, kd)
                    coords = data.dimension_values(kd.name, expanded=irregular,
                                                   flat=not irregular)
                    new_data.append(coords)
                for vd in data.vdims:
                    new_data.append(interface.values(data, vd, flat=False, compute=False))
                data = tuple(new_data)
            elif 'dataframe' in datatype and util.pd:
                data = data.dframe()
            else:
                data = tuple(data.columns().values())
        elif isinstance(data, Element):
            data = tuple(data.dimension_values(d) for d in kdims+vdims)
        elif isinstance(data, util.generator_types):
            data = list(data)

        if datatype is None:
            datatype = eltype.datatype

        # Set interface priority order
        prioritized = [cls.interfaces[p] for p in datatype
                       if p in cls.interfaces]
        head = [intfc for intfc in prioritized if intfc.applies(data)]
        if head:
            # Prioritize interfaces which have matching types
            prioritized = head + [el for el in prioritized if el != head[0]]

        # Iterate over interfaces until one can interpret the input
        priority_errors = []
        for interface in prioritized:
            if not interface.loaded() and len(datatype) != 1:
                # Skip interface if it is not loaded and was not explicitly requested
                continue
            try:
                (data, dims, extra_kws) = interface.init(eltype, data, kdims, vdims)
                break
            except DataError:
                raise
            except Exception as e:
                if interface in head or len(prioritized) == 1:
                    priority_errors.append((interface, e, True))
        else:
            error = ("None of the available storage backends were able "
                     "to support the supplied data format.")
            if priority_errors:
                intfc, e, _ = priority_errors[0]
                priority_error = ("%s raised following error:\n\n %s"
                                  % (intfc.__name__, e))
                error = ' '.join([error, priority_error])
                raise six.reraise(DataError, DataError(error, intfc), sys.exc_info()[2])
            raise DataError(error)

        return data, interface, dims, extra_kws


    @classmethod
    def validate(cls, dataset, vdims=True):
        dims = 'all' if vdims else 'key'
        not_found = [d for d in dataset.dimensions(dims, label='name')
                     if d not in dataset.data]
        if not_found:
            raise DataError("Supplied data does not contain specified "
                            "dimensions, the following dimensions were "
                            "not found: %s" % repr(not_found), cls)

    @classmethod
    def persist(cls, dataset):
        """
        Should return a persisted version of the Dataset.
        """
        return dataset

    @classmethod
    def compute(cls, dataset):
        """
        Should return a computed version of the Dataset.
        """
        return dataset

    @classmethod
    def expanded(cls, arrays):
        return not any(array.shape not in [arrays[0].shape, (1,)] for array in arrays[1:])

    @classmethod
    def isscalar(cls, dataset, dim):
        return len(cls.values(dataset, dim, expanded=False)) == 1

    @classmethod
    def isunique(cls, dataset, dim, per_geom=False):
        """
        Compatibility method introduced for v1.13.0 to smooth
        over addition of per_geom kwarg for isscalar method.
        """
        try:
            return cls.isscalar(dataset, dim, per_geom)
        except TypeError:
            return cls.isscalar(dataset, dim)

    @classmethod
    def dtype(cls, dataset, dimension):
        name = dataset.get_dimension(dimension, strict=True).name
        data = dataset.data[name]
        if util.isscalar(data):
            return np.array([data]).dtype
        else:
            return data.dtype

    @classmethod
    def replace_value(cls, data, nodata):
        """
        Replace `nodata` value in data with NaN
        """
        data = data.astype('float64')
        mask = data != nodata
        if hasattr(data, 'where'):
            return data.where(mask, np.NaN)
        return np.where(mask, data, np.NaN)

    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e. tuple ranges, slices, sets, lists, or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        mask = np.ones(len(dataset), dtype=np.bool)
        for dim, sel in selection.items():
            if isinstance(sel, tuple):
                sel = slice(*sel)
            arr = cls.values(dataset, dim)
            if util.isdatetime(arr) and util.pd:
                try:
                    sel = util.parse_datetime_selection(sel)
                except:
                    pass
            if isinstance(sel, slice):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered')
                    if sel.start is not None:
                        mask &= sel.start <= arr
                    if sel.stop is not None:
                        mask &= arr < sel.stop
            elif isinstance(sel, (set, list)):
                iter_slcs = []
                for ik in sel:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered')
                        iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            elif callable(sel):
                mask &= sel(arr)
            else:
                index_mask = arr == sel
                if dataset.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - sel))
                    mask = np.zeros(len(dataset), dtype=np.bool)
                    mask[data_index] = True
                else:
                    mask &= index_mask
        return mask


    @classmethod
    def indexed(cls, dataset, selection):
        """
        Given a Dataset object and selection to be applied returns
        boolean to indicate whether a scalar value has been indexed.
        """
        selected = list(selection.keys())
        all_scalar = all((not isinstance(sel, (tuple, slice, set, list))
                          and not callable(sel)) for sel in selection.values())
        all_kdims = all(d in selected for d in dataset.kdims)
        return all_scalar and all_kdims


    @classmethod
    def range(cls, dataset, dimension):
        column = dataset.dimension_values(dimension)
        if column.dtype.kind == 'M':
            return column.min(), column.max()
        elif len(column) == 0:
            return np.NaN, np.NaN
        else:
            try:
                assert column.dtype.kind not in 'SUO'
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                    return (np.nanmin(column), np.nanmax(column))
            except (AssertionError, TypeError):
                column = [v for v in util.python2sort(column) if v is not None]
                if not len(column):
                    return np.NaN, np.NaN
                return column[0], column[-1]

    @classmethod
    def concatenate(cls, datasets, datatype=None, new_type=None):
        """
        Utility function to concatenate an NdMapping of Dataset objects.
        """
        from . import Dataset, default_datatype
        new_type = new_type or Dataset
        if isinstance(datasets, NdMapping):
            dimensions = datasets.kdims
            keys, datasets = zip(*datasets.data.items())
        elif isinstance(datasets, list) and all(not isinstance(v, tuple) for v in datasets):
            # Allow concatenating list of datasets (by declaring no dimensions and keys)
            dimensions, keys = [], [()]*len(datasets)
        else:
            raise DataError('Concatenation only supported for NdMappings '
                            'and lists of Datasets, found %s.' % type(datasets).__name__)

        template = datasets[0]
        datatype = datatype or template.interface.datatype

        # Handle non-general datatypes by casting to general type
        if datatype == 'array':
            datatype = default_datatype
        elif datatype == 'image':
            datatype = 'grid'

        if len(datasets) > 1 and not dimensions and cls.interfaces[datatype].gridded:
            raise DataError('Datasets with %s datatype cannot be concatenated '
                            'without defining the dimensions to concatenate along. '
                            'Ensure you pass in a NdMapping (e.g. a HoloMap) '
                            'of Dataset types, not a list.' % datatype)

        datasets = template.interface.cast(datasets, datatype)
        template = datasets[0]
        data = list(zip(keys, datasets)) if keys else datasets
        concat_data = template.interface.concat(data, dimensions, vdims=template.vdims)
        return template.clone(concat_data, kdims=dimensions+template.kdims, new_type=new_type)

    @classmethod
    def histogram(cls, array, bins, density=True, weights=None):
        if util.is_dask_array(array):
            import dask.array as da
            histogram = da.histogram
        elif util.is_cupy_array(array):
            import cupy
            histogram = cupy.histogram
        else:
            histogram = np.histogram
        hist, edges = histogram(array, bins=bins, density=density, weights=weights)
        if util.is_cupy_array(hist):
            edges = cupy.asnumpy(edges)
            hist = cupy.asnumpy(hist)
        return hist, edges

    @classmethod
    def reduce(cls, dataset, reduce_dims, function, **kwargs):
        kdims = [kdim for kdim in dataset.kdims if kdim not in reduce_dims]
        return cls.aggregate(dataset, kdims, function, **kwargs)

    @classmethod
    def array(cls, dataset, dimensions):
        return Element.array(dataset, dimensions)

    @classmethod
    def dframe(cls, dataset, dimensions):
        return Element.dframe(dataset, dimensions)

    @classmethod
    def columns(cls, dataset, dimensions):
        return Element.columns(dataset, dimensions)

    @classmethod
    def shape(cls, dataset):
        return dataset.data.shape

    @classmethod
    def length(cls, dataset):
        return len(dataset.data)

    @classmethod
    def nonzero(cls, dataset):
        return bool(cls.length(dataset))

    @classmethod
    def redim(cls, dataset, dimensions):
        return dataset.data

    @classmethod
    def has_holes(cls, dataset):
        return False

    @classmethod
    def holes(cls, dataset):
        coords = cls.values(dataset, dataset.kdims[0])
        splits = np.where(np.isnan(coords.astype('float')))[0]
        return [[[]]*(len(splits)+1)]

    @classmethod
    def as_dframe(cls, dataset):
        """
        Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.
        """
        return dataset.dframe()


class Interface(param.Parameterized):
    drivers_by_kind = {}
    drivers_by_datatype = {}
    datatypes_by_kind = {}
    kind = None

    def __init__(self, driver: Driver, **params):
        super(Interface, self).__init__(**params)
        self.driver = driver

    @classmethod
    def get_datatypes_for_kinds(cls, kinds):
        return [
            datatype
            for kind in kinds
            for datatype in cls.datatypes_by_kind.get(kind, [])
        ]

    @classmethod
    def get_driver(cls, datatype):
        return cls.drivers_by_datatype[datatype][1]

    @property
    def gridded(self):
        return self.driver.gridded

    @property
    def datatype(self):
        return self.driver.datatype

    @property
    def multi(self):
        return self.driver.multi

    @property
    def named(self):
        return self.driver.named

    @classmethod
    def register_driver(cls, driver):
        cls.drivers_by_kind.setdefault(cls.kind, []).append((cls, driver))
        cls.datatypes_by_kind.setdefault(cls.kind, []).append(driver.datatype)
        cls.drivers_by_datatype[driver.datatype] = (cls, driver)

    @classmethod
    def initialize(cls, eltype, data, kdims, vdims, datatype=None, kind=None):

        if datatype is None and kind is None:
            datatype = eltype.datatype
            driver_pairs = [
                cls.drivers_by_datatype.get(dt)
                for dt in datatype if dt in cls.drivers_by_datatype
            ]
        elif kind is not None:
            driver_pairs = [pair for k in kind for pair in cls.drivers_by_kind.get(k, [])]
        else:  # datatype is not None
            driver_pairs = [
                cls.drivers_by_datatype.get(dt)
                for dt in datatype if dt in cls.drivers_by_datatype
            ]

        # Build list of drivers to try
        head = []
        tail = []

        for interface_cls, driver_cls in driver_pairs:
            if driver_cls.applies(data):
                head.append((interface_cls, driver_cls))
            else:
                tail.append((interface_cls, driver_cls))

        prioritized_pairs = head + tail
        for interface_cls, driver_cls in prioritized_pairs:
            try:
                data, driver, dims, extra_kws = \
                    driver_cls.initialize(
                        eltype, data, kdims, vdims, datatype=[driver_cls.datatype]
                    )
                interface = interface_cls(driver_cls)
                return data, interface, dims, extra_kws
            except DataError:
                pass
            except:
                raise

        raise DataError("No compatible driver")

    def cast(self, datasets, datatype=None, cast_type=None):
        """
        Given a list of Dataset objects, cast them to the specified
        datatype (by default the format matching the current interface)
        with the given cast_type (if specified).
        """
        return self.driver.cast(datasets, datatype, cast_type)

    def validate(self, dataset, vdims=True):
        return self.driver.validate(dataset, vdims)

    def persist(self, dataset):
        """
        Should return a persisted version of the Dataset.
        """
        return self.driver.persist(dataset)

    def compute(self, dataset):
        """
        Should return a computed version of the Dataset.
        """
        return self.driver.compute(dataset)

    def expanded(self, arrays):
        return self.driver.expanded(arrays)

    def isscalar(self, dataset, dim):
        return self.driver.isscalar(dataset, dim)

    def isunique(self, dataset, dim, per_geom=False):
        """
        Compatibility method introduced for v1.13.0 to smooth
        over addition of per_geom kwarg for isscalar method.
        """
        return self.driver.isunique(dataset, dim, per_geom)

    def dtype(self, dataset, dimension):
        return self.driver.dtype(dataset, dimension)

    def replace_value(self, data, nodata):
        """
        Replace `nodata` value in data with NaN
        """
        return self.driver.replace_value(data, nodata)

    def select_mask(self, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e. tuple ranges, slices, sets, lists, or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        return self.driver.select_mask(dataset, selection)

    def indexed(self, dataset, selection):
        """
        Given a Dataset object and selection to be applied returns
        boolean to indicate whether a scalar value has been indexed.
        """
        return self.driver.indexed(dataset, selection)

    def range(self, dataset, dimension):
        return self.driver.range(dataset, dimension)

    def concatenate(self, datasets, datatype=None, new_type=None):
        """
        Utility function to concatenate an NdMapping of Dataset objects.
        """
        return self.driver.concatenate(datasets, datatype, new_type)

    def histogram(self, array, bins, density=True, weights=None):
        return self.driver.histogram(array, bins, density, weights)

    def reduce(self, dataset, reduce_dims, function, **kwargs):
        return self.driver.reduce(dataset, reduce_dims, function, **kwargs)

    def array(self, dataset, dimensions):
        return self.driver.array(dataset, dimensions)

    def dframe(self, dataset, dimensions):
        return self.driver.dframe(dataset, dimensions)

    def columns(self, dataset, dimensions):
        return self.driver.columns(dataset, dimensions)

    def shape(self, dataset, **kwargs):
        return self.driver.shape(dataset, **kwargs)

    def length(self, dataset):
        return self.driver.length(dataset)

    def nonzero(self, dataset):
        return self.driver.nonzero(dataset)

    def redim(self, dataset, dimensions):
        return self.driver.redim(dataset, dimensions)

    def as_dframe(self, dataset):
        """
        Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.
        """
        return self.driver.as_dframe(dataset)

    def values(self, *args, **kwargs):
        return self.driver.values(*args, **kwargs)


# Interface should get a reference to a driver class
class TabularInterface(Interface):
    kind = "tabular"

    def aggregate(self, *args, **kwargs):
        return self.driver.aggregate(*args, **kwargs)

    def unpack_scalar(self, *args, **kwargs):
        return self.driver.unpack_scalar(*args, **kwargs)

    def add_dimension(self, *args, **kwargs):
        return self.driver.add_dimension(*args, **kwargs)

    def select(self, *args, **kwargs):
        return self.driver.select(*args, **kwargs)

    def groupby(self, *args, **kwargs):
        return self.driver.groupby(*args, **kwargs)

    def reindex(self, *args, **kwargs):
        return self.driver.reindex(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.driver.sample(*args, **kwargs)

    def sort(self, *args, **kwargs):
        return self.driver.sort(*args, **kwargs)

    def assign(self, *args, **kwargs):
        return self.driver.assign(*args, **kwargs)

    def iloc(self, *args, **kwargs):
        return self.driver.iloc(*args, **kwargs)


class GriddedInterface(Interface):
    kind = "gridded"

    def aggregate(self, *args, **kwargs):
        return self.driver.aggregate(*args, **kwargs)

    def unpack_scalar(self, *args, **kwargs):
        return self.driver.unpack_scalar(*args, **kwargs)

    def add_dimension(self, *args, **kwargs):
        return self.driver.add_dimension(*args, **kwargs)

    def coords(self, *args, **kwargs):
        return self.driver.coords(*args, **kwargs)

    def irregular(self, *args, **kwargs):
        return self.driver.irregular(*args, **kwargs)

    def select(self, *args, **kwargs):
        return self.driver.select(*args, **kwargs)

    def groupby(self, *args, **kwargs):
        return self.driver.groupby(*args, **kwargs)

    def reindex(self, *args, **kwargs):
        return self.driver.reindex(*args, **kwargs)

    def ndloc(self, *args, **kwargs):
        return self.driver.ndloc(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.driver.sample(*args, **kwargs)

    def sort(self, *args, **kwargs):
        return self.driver.sort(*args, **kwargs)

    def assign(self, *args, **kwargs):
        return self.driver.assign(*args, **kwargs)

    def iloc(self, *args, **kwargs):
        return self.driver.iloc(*args, **kwargs)

    def mask(self, *args, **kwargs):
        return self.driver.mask(*args, **kwargs)

    def concat(self, *args, **kwargs):
        return self.driver.concat(*args, **kwargs)

class ImageInterface(GriddedInterface):
    kind = "image"


class GeometryInterface(Interface):
    kind = "geometry"

    def has_holes(self, dataset):
        return self.driver.has_holes(dataset)

    def holes(self, dataset):
        return self.driver.holes(dataset)
