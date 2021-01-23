from __future__ import absolute_import

import sys
import warnings
from collections import OrderedDict

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
    def validate(cls, dataset, vdims=True, **kwargs):
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
    def isscalar(cls, dataset, dim, **kwargs):
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
    def shape(cls, dataset, **kwargs):
        return dataset.data.shape

    @classmethod
    def length(cls, dataset, **kwargs):
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
    auto_indexable_1d = param.Boolean(default=False, constant=True, doc="""
        In the 1D case the interfaces should not automatically add x-values
        to supplied data""")

    binned = param.Boolean(default=False, constant=True, doc="""
        Whether the key dimensions are specified as bins""")

    name = param.String(default=None)

    kdims_spec = param.Dict()
    vdims_spec = param.Dict()

    drivers_by_kind = {}
    drivers_by_datatype = {}
    datatypes_by_kind = {}
    kind = None

    def __init__(self, driver: Driver, **params):
        super(Interface, self).__init__(
            **{p: v for p, v in params.items() if p in self.param}
        )
        self.driver = driver

    @property
    def interface_opts(self):
        return {p: getattr(self, p) for p in self.param.objects()}

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
    def initialize(
            cls, data, kdims_spec, vdims_spec, datatype=None, kind=None,
            **interface_opts
    ):

        if datatype is None and kind is None:
            raise ValueError("Either datatype or kind must be provided")
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
        # TODO: move driver_cls.initialize logic here
        priority_errors = []
        for interface_cls, driver_cls in prioritized_pairs:
            try:
                (data, dims, extra_kws) = driver_cls.init(
                    data, kdims_spec, vdims_spec, **interface_opts
                )
                interface = interface_cls(
                    driver_cls, kdims_spec=kdims_spec, vdims_spec=vdims_spec,
                    **interface_opts
                )
                return data, interface, dims, extra_kws
            except DataError:
                pass
            except Exception as e:
                priority_errors.append((driver_cls, e, True))

        error = ("None of the available storage backends were able "
                 "to support the supplied data format.")
        if priority_errors:
            intfc, e, _ = priority_errors[0]
            priority_error = ("%s raised following error:\n\n %s"
                              % (intfc.__name__, e))
            error = ' '.join([error, priority_error])
            raise six.reraise(DataError, DataError(error, intfc), sys.exc_info()[2])
        raise DataError(error)

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

    def shape(self, dataset):
        return self.driver.shape(dataset)

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

    def groupby(self, dataset, dimensions, kdims=None):
        grouped_list = self.driver.groupby(dataset, dimensions, kdims=kdims)
        return OrderedDict(grouped_list)

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

    def holes(self, *args, **kwargs):
        # TODO: Remove after separating dictionary fro geodictionary
        return self.driver.holes(*args, **kwargs)

    def split(self, *args, **kwargs):
        # TODO: Remove after separating dictionary fro geodictionary
        return self.driver.split(*args, **kwargs)

    def geom_dims(self, *args, **kwargs):
        # TODO: Remove after separating dictionary fro geodictionary
        return self.driver.geom_dims(*args, **kwargs)


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

    def groupby(self, dataset, dimensions, kdims=None):
        grouped_list = self.driver.groupby(dataset, dimensions, kdims=kdims)
        return OrderedDict(grouped_list)

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

    def shape(self, dataset, gridded=False):
        return self.driver.shape(dataset, gridded=gridded)

class ImageInterface(GriddedInterface):

    rtol = param.Number(default=10e-4, constant=True, doc="""
        The tolerance used to enforce regular sampling for regular, gridded
        data where regular sampling is expected. Expressed as the maximal
        allowable sampling difference between sample locations.""")

    time_unit = param.String(default="us", constant=True, doc="""
        Determines the unit of time densities are defined relative to
        when one or both axes are datetime types""")

    kind = "image"


class GeometryInterface(Interface):
    geom_type = param.Selector(
        objects=["Point", "Line", "Polygon"], default=None, doc="""
            Geometry type interpretation for data structures that do not contain
            their own geometry type metadata""")

    hole_key = param.String(default=None)

    kind = "geometry"

    def validate(self, dataset, vdims=True, geom_type=None):
        return self.driver.validate(dataset, vdims, geom_type=geom_type)

    def has_holes(self, dataset):
        return self.driver.has_holes(dataset)

    def holes(self, dataset):
        return self.driver.holes(dataset)

    def values(self, dataset, dimension, expanded=True, flat=True, compute=True, keep_index=False):
        return self.driver.values(
            dataset, dimension, expanded=expanded, flat=flat, compute=compute,
            keep_index=keep_index, geom_type=self.geom_type
        )

    def shape(self, dataset):
        return self.driver.shape(dataset, geom_type=self.geom_type)

    def length(self, dataset):
        return self.driver.length(dataset, geom_type=self.geom_type)

    def split(self, dataset, start, end, datatype, **kwargs):
        return self.driver.split(
            dataset, start, end, datatype,
            geom_type=self.geom_type, hole_key=self.hole_key, **kwargs
        )

    def add_dimension(self, dataset, *args, **kwargs):
        return self.driver.add_dimension(dataset, *args, **kwargs)

    def groupby(self, dataset, dimensions, kdims=None):
        grouped_list = self.driver.groupby(dataset, dimensions, kdims=kdims)
        return OrderedDict(grouped_list)

    def iloc(self, dataset, index):
        return self.driver.iloc(dataset, index, geom_type=self.geom_type)

    def isscalar(self, dataset, dim, per_geom=False):
        return self.driver.isscalar(
            dataset, dim, per_geom=per_geom, geom_type=self.geom_type
        )

    def select(self, dataset, selection_mask=None, **selection):
        return self.driver.select(
            dataset, selection_mask=selection_mask, hole_key=self.hole_key, **selection
        )

    def sort(self, *args, **kwargs):
        return self.driver.sort(*args, **kwargs)

    def geom_dims(self, *args, **kwargs):
        return self.driver.geom_dims(*args, **kwargs)
