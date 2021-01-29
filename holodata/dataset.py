import itertools
from collections import OrderedDict

import numpy as np
import param

from holodata.dimension import Dimensioned, Dimension, dimension_name
from holodata.interface import Interface
from holodata import util
from holodata.ndmapping import UniformNdMapping, NdMapping, item_check, sorted_context


class Dataset(Dimensioned):
    def compute(self):
        """
        Computes the data to a data format that stores the daata in
        memory, e.g. a Dask dataframe or array is converted to a
        Pandas DataFrame or NumPy array.

        Returns:
            Dataset with the data stored in in-memory format
        """
        return self.interface.compute(self)

    def persist(self):
        """
        Persists the results of a lazy data interface to memory to
        speed up data manipulation and visualization. If the
        particular data backend already holds the data in memory
        this is a no-op. Unlike the compute method this maintains
        the same data type.

        Returns:
            Dataset with the data persisted to memory
        """
        persisted = self.interface.persist(self)
        return persisted

    def range(self, dim, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        dim = self.get_dimension(dim)

        if dim is None or (not data_range and not dimension_range):
            return (None, None)
        elif all(util.isfinite(v) for v in dim.range) and dimension_range:
            return dim.range
        elif dim in self.dimensions() and data_range and bool(self):
            lower, upper = self.interface.range(self, dim)
        else:
            lower, upper = (np.NaN, np.NaN)
        if not dimension_range:
            return lower, upper
        return util.dimension_range(lower, upper, dim.range, dim.soft_range)

    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        """Adds a dimension and its values to the Dataset

        Requires the dimension name or object, the desired position in
        the key dimensions and a key value scalar or array of values,
        matching the length or shape of the Dataset.

        Args:
            dimension: Dimension or dimension spec to add
            dim_pos (int): Integer index to insert dimension at
            dim_val (scalar or ndarray): Dimension value(s) to add
            vdim: Disabled, this type does not have value dimensions
            **kwargs: Keyword arguments passed to the cloned element
        Returns:
            Cloned object containing the new dimension
        """
        if isinstance(dimension, (util.basestring, tuple)):
            dimension = Dimension(dimension)

        if dimension.name in self.kdims:
            raise Exception('{dim} dimension already defined'.format(dim=dimension.name))

        if vdim:
            dims = self.vdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(vdims=dims)
            dim_pos += self.ndims
        else:
            dims = self.kdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(kdims=dims)

        # TODO: what was this for in holoviews.Dataset
        # if issubclass(self.interface.driver, ArrayDriver) and np.asarray(dim_val).dtype != self.data.dtype:
        #     element = self.clone(datatype=[default_datatype])
        #     data = element.interface.add_dimension(element, dimension, dim_pos, dim_val, vdim)
        # else:
        #     data = self.interface.add_dimension(self, dimension, dim_pos, dim_val, vdim)

        data = self.interface.add_dimension(self, dimension, dim_pos, dim_val, vdim)
        return self.clone(data, **dimensions)


class TabularDataset(Dataset):
    group = param.String(default='TabularDataset', constant=True)

    def __init__(self, data, kdims=None, vdims=None, **params):
        validate_vdims = params.pop('_validate_vdims', True)

        kdims_spec = self.kdims_spec(kdims)
        vdims_spec = self.vdims_spec(vdims)

        # Determine which drivers to try
        datatype = params.pop("datatype", None)
        if datatype:
            # Use tabular drivers included in datatype
            datatype = [
                dt for dt in datatype
                if dt in Interface.datatypes_by_kind.get("tabular", [])
            ]
        else:
            datatype = Interface.datatypes_by_kind.get("tabular", [])

        initialized = Interface.initialize(
            data, kdims_spec=kdims_spec, vdims_spec=vdims_spec, datatype=datatype
        )
        (data, self.interface, dims, extra_kws) = initialized
        super(TabularDataset, self).__init__(data, kdims, vdims, **params)
        self.interface.validate(self, validate_vdims)

    def unpack_scalar(self, *args, **kwargs):
        return self.interface.unpack_scalar(self, *args, **kwargs)

    def sort(self, *args, **kwargs):
        return self.interface.sort(self, *args, **kwargs)

    def assign(self, *args, **kwargs):
        return self.interface.assign(self, *args, **kwargs)

    def iloc(self, *args, **kwargs):
        return self.interface.iloc(self, *args, **kwargs)

    def __len__(self):
        "Number of values in the Dataset."
        return self.interface.length(self)

    # Copied from Dataset to get things working
    def array(self, dimensions=None):
        """Convert dimension values to columnar array.

        Args:
            dimensions: List of dimensions to return

        Returns:
            Array of columns corresponding to each dimension
        """
        if dimensions is None:
            dims = [d for d in self.kdims + self.vdims]
        else:
            dims = [self.get_dimension(d, strict=True) for d in dimensions]

        columns, types = [], []
        for dim in dims:
            column = self.dimension_values(dim)
            columns.append(column)
            types.append(column.dtype.kind)
        if len(set(types)) > 1:
            columns = [c.astype('object') for c in columns]
        return np.column_stack(columns)

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
                Whether to return the expanded values, behavior depends
                on the type of data:
                  * Columnar: If false returns unique values
                  * Geometry: If false returns scalar values per geometry
                  * Gridded: If false returns 1D coordinates
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        import sys
        dim = self.get_dimension(dimension, strict=True)
        values = self.interface.values(self, dim, expanded, flat)
        if dim.nodata is not None:
            # Ensure nodata applies to boolean data in py2
            if sys.version_info.major == 2 and values.dtype.kind == 'b':
                values = values.astype('int')
            values = np.where(values==dim.nodata, np.NaN, values)
        return values

    def select(self, selection_expr=None, selection_specs=None, **selection):
        """Applies selection by dimension name

        Applies a selection along the dimensions of the object using
        keyword arguments. The selection may be narrowed to certain
        objects using selection_specs. For container objects the
        selection will be applied to all children as well.

        Selections may select a specific value, slice or set of values:

        * value: Scalar values will select rows along with an exact
                 match, e.g.:

            ds.select(x=3)

        * slice: Slices may be declared as tuples of the upper and
                 lower bound, e.g.:

            ds.select(x=(0, 3))

        * values: A list of values may be selected using a list or
                  set, e.g.:

            ds.select(x=[0, 1, 2])

        * predicate expression: A holoviews.dim expression, e.g.:

            from holoviews import dim
            ds.select(selection_expr=dim('x') % 2 == 0)

        Args:
            selection_expr: holoviews.dim predicate expression
                specifying selection.
            selection_specs: List of specs to match on
                A list of types, functions, or type[.group][.label]
                strings specifying which objects to apply the
                selection on.
            **selection: Dictionary declaring selections by dimension
                Selections can be scalar values, tuple ranges, lists
                of discrete values and boolean arrays

        Returns:
            Returns an Dimensioned object containing the selected data
            or a scalar if a single value was selected
        """
        from holodata.transform import dim
        if selection_expr is not None and not isinstance(selection_expr, dim):
            raise ValueError("""\
The first positional argument to the Dataset.select method is expected to be a
holoviews.util.transform.dim expression. Use the selection_specs keyword
argument to specify a selection specification""")

        if selection_specs is not None and not isinstance(selection_specs,
                                                          (list, tuple)):
            selection_specs = [selection_specs]
        selection = {dim_name: sel for dim_name, sel in selection.items()
                     if dim_name in self.dimensions() + ['selection_mask']}
        if (selection_specs and not any(self.matches(sp) for sp in selection_specs)
                or (not selection and not selection_expr)):
            return self

        # Handle selection dim expression
        if selection_expr is not None:
            mask = selection_expr.apply(self, compute=False, keep_index=True)
            selection = {'selection_mask': mask}

        # Handle selection kwargs
        if selection:
            data = self.interface.select(self, **selection)
        else:
            data = self.data

        if np.isscalar(data):
            return data
        else:
            return self.clone(data)

    def reindex(self, kdims=None, vdims=None):
        """Reindexes Dataset dropping static or supplied kdims

        Creates a new object with a reordered or reduced set of key
        dimensions. By default drops all non-varying key dimensions.x

        Args:
            kdims (optional): New list of key dimensionsx
            vdims (optional): New list of value dimensions

        Returns:
            Reindexed object
        """

        scalars = []
        if kdims is None:
            # If no key dimensions are defined and interface is gridded
            # drop all scalar key dimensions
            key_dims = [d for d in self.kdims if (not vdims or d not in vdims)
                        and not d in scalars]
        elif not isinstance(kdims, list):
            key_dims = [self.get_dimension(kdims, strict=True)]
        else:
            key_dims = [self.get_dimension(k, strict=True) for k in kdims]

        if vdims is None:
            val_dims = [d for d in self.vdims if not kdims or d not in kdims]
        else:
            val_dims = [self.get_dimension(v, strict=True) for v in vdims]

        data = self.interface.reindex(self, key_dims, val_dims)

        return self.clone(data, kdims=key_dims, vdims=val_dims)

    def __getitem__(self, slices):
        """
        Allows slicing and selecting values in the Dataset object.
        Supports multiple indexing modes:

           (1) Slicing and indexing along the values of each dimension
               in the columns object using either scalars, slices or
               sets of values.
           (2) Supplying the name of a dimension as the first argument
               will return the values along that dimension as a numpy
               array.
           (3) Slicing of all key dimensions and selecting a single
               value dimension by name.
           (4) A boolean array index matching the length of the Dataset
               object.
        """
        slices = util.process_ellipses(self, slices, vdim_selection=True)
        if getattr(getattr(slices, 'dtype', None), 'kind', None) == 'b':
            if not len(slices) == len(self):
                raise IndexError("Boolean index must match length of sliced object")
            return self.clone(self.select(selection_mask=slices))
        elif (isinstance(slices, ()) and len(slices) == 1) or slices is Ellipsis:
            return self
        if not isinstance(slices, tuple): slices = (slices,)
        value_select = None
        if len(slices) == 1 and slices[0] in self.dimensions():
            return self.dimension_values(slices[0])
        elif len(slices) == self.ndims+1 and slices[self.ndims] in self.dimensions():
            selection = dict(zip(self.dimensions('key', label=True), slices))
            value_select = slices[self.ndims]
        elif len(slices) == self.ndims+1 and isinstance(slices[self.ndims],
                                                        (Dimension,str)):
            raise IndexError("%r is not an available value dimension" % slices[self.ndims])
        else:
            selection = dict(zip(self.dimensions(label=True), slices))
        data = self.select(**selection)
        if value_select:
            if data.shape[0] == 1:
                return data[value_select][0]
            else:
                return data.reindex(vdims=[value_select])
        return data

    def _reduce_map(self, dimensions, function, reduce_map):
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument "
                            "or as part of the kwargs not both.")
        if len(set(reduce_map.values())) > 1:
            raise Exception("Cannot define reduce operations with more than "
                            "one function at a time.")
        if reduce_map:
            reduce_map = reduce_map.items()
        if dimensions:
            reduce_map = [(d, function) for d in dimensions]
        elif not reduce_map:
            reduce_map = [(d, function) for d in self.kdims]
        reduced = [(self.get_dimension(d, strict=True).name, fn)
                   for d, fn in reduce_map]
        grouped = [(fn, [dim for dim, _ in grp]) for fn, grp in itertools.groupby(reduced, lambda x: x[1])]
        return grouped[0]

    def reduce(self, dimensions=[], function=None, spreadfn=None, **reductions):
        """Applies reduction along the specified dimension(s).

        Allows reducing the values along one or more key dimension
        with the supplied function. Supports two signatures:

        Reducing with a list of dimensions, e.g.:

            ds.reduce(['x'], np.mean)

        Defining a reduction using keywords, e.g.:

            ds.reduce(x=np.mean)

        Args:
            dimensions: Dimension(s) to apply reduction on
                Defaults to all key dimensions
            function: Reduction operation to apply, e.g. numpy.mean
            spreadfn: Secondary reduction to compute value spread
                Useful for computing a confidence interval, spread, or
                standard deviation.
            **reductions: Keyword argument defining reduction
                Allows reduction to be defined as keyword pair of
                dimension and function

        Returns:
            The Dataset after reductions have been applied.
        """
        if any(dim in self.vdims for dim in dimensions):
            raise Exception("Reduce cannot be applied to value dimensions")
        function, dims = self._reduce_map(dimensions, function, reductions)
        dims = [d for d in self.kdims if d not in dims]
        return self.aggregate(dims, function, spreadfn)

    def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """Aggregates data on the supplied dimensions.

        Aggregates over the supplied key dimensions with the defined
        function or dim_transform specified as a tuple of the transformed
        dimension name and dim transform.

        Args:
            dimensions: Dimension(s) to aggregate on
                Default to all key dimensions
            function: Aggregation function or transform to apply
                Supports both simple functions and dimension transforms
            spreadfn: Secondary reduction to compute value spread
                Useful for computing a confidence interval, spread, or
                standard deviation.
            **kwargs: Keyword arguments either passed to the aggregation function
                or to create new names for the transformed variables

        Returns:
            Returns the aggregated Dataset
        """
        from holodata.transform import dim
        if dimensions is None: dimensions = self.kdims
        elif not isinstance(dimensions, list): dimensions = [dimensions]
        if isinstance(function, tuple) or any(isinstance(v, dim) for v in kwargs.values()):
            dataset = self.clone(new_type=Dataset)
            if dimensions:
                dataset = dataset.groupby(dimensions)
            args = () if function is None else (function,)
            transformed = dataset.apply.transform(*args, drop=True, **kwargs)
            if not isinstance(transformed, Dataset):
                transformed = transformed.collapse()
            return transformed.clone(new_type=type(self))

        # Handle functions
        kdims = [self.get_dimension(d, strict=True) for d in dimensions]
        if not self:
            if spreadfn:
                spread_name = spreadfn.__name__
                vdims = [d for vd in self.vdims for d in [vd, vd.clone('_'.join([vd.name, spread_name]))]]
            else:
                vdims = self.vdims
            return self.clone([], kdims=kdims, vdims=vdims)

        vdims = self.vdims
        aggregated, dropped = self.interface.aggregate(self, kdims, function, **kwargs)
        aggregated = self.interface.unpack_scalar(self, aggregated)
        vdims = [vd for vd in vdims if vd not in dropped]

        ndims = len(dimensions)
        min_d, max_d = self.param.objects('existing')['kdims'].bounds
        generic_type = (min_d is not None and ndims < min_d) or (max_d is not None and ndims > max_d)

        if spreadfn:
            error, _ = self.interface.aggregate(self, dimensions, spreadfn)
            spread_name = spreadfn.__name__
            ndims = len(vdims)
            error = self.clone(error, kdims=kdims, new_type=Dataset)
            combined = self.clone(aggregated, kdims=kdims, new_type=Dataset)
            for i, d in enumerate(vdims):
                dim = d.clone('_'.join([d.name, spread_name]))
                dvals = error.dimension_values(d, flat=False)
                idx = vdims.index(d)
                combined = combined.add_dimension(dim, idx+1, dvals, True)
                vdims = combined.vdims
            return combined.clone(new_type=Dataset if generic_type else type(self))

        if np.isscalar(aggregated):
            return aggregated
        else:
            try:
                # Should be checking the dimensions declared on the element are compatible
                return self.clone(aggregated, kdims=kdims, vdims=vdims)
            except:
                datatype = self.param.objects('existing')['datatype'].default
                return self.clone(aggregated, kdims=kdims, vdims=vdims,
                                  new_type=Dataset if generic_type else None,
                                  datatype=datatype)

    def groupby(self, dimensions=[], container_type=UniformNdMapping, group_type=None,
                **kwargs):
        """Groups object by one or more dimensions

        Applies groupby operation over the specified dimensions
        returning an object of type container_type (expected to be
        dictionary-like) containing the groups.

        Args:
            dimensions: Dimension(s) to group by
            container_type: Type to cast group container to
            group_type: Type to cast each group to
            **kwargs: Keyword arguments to pass to each group

        Returns:
            Returns object of supplied container_type containing the
            groups. If dynamic=True returns a DynamicMap instead.
        """
        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        if not len(dimensions):
            dimensions = self.dimensions('key', True)
        if group_type is None:
            group_type = type(self)

        dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        dim_names = [d.name for d in dimensions]

        grouped_data = self.interface.groupby(
            self, dim_names, kdims=kwargs.get("kdims", None)
        )

        # Get group
        group_kwargs = {}
        group_kwargs.update(kwargs)

        # Replace raw group data with group_type objects
        if group_type != 'raw':
            for group in grouped_data:
                group_data = grouped_data[group]
                if issubclass(group_type, dict):
                    group_data = {
                        d.name: group_data[:, i]
                        for i, d in enumerate(self.kdims+self.vdims)
                    }
                else:
                    group_data = group_type(group_data, **group_kwargs)

                grouped_data[group] = group_data

        # Wrap in container type
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(grouped_data, kdims=dimensions)
        else:
            return container_type(grouped_data)

    def transform(self, *args, **kwargs):
        """Transforms the Dataset according to a dimension transform.

        Transforms may be supplied as tuples consisting of the
        dimension(s) and the dim transform to apply or keyword
        arguments mapping from dimension(s) to dim transforms. If the
        arg or kwarg declares multiple dimensions the dim transform
        should return a tuple of values for each.

        A transform may override an existing dimension or add a new
        one in which case it will be added as an additional value
        dimension.

        Args:
            args: Specify the output arguments and transforms as a
                  tuple of dimension specs and dim transforms
            drop (bool): Whether to drop all variables not part of the transform
            keep_index (bool): Whether to keep indexes
                  Whether to apply transform on datastructure with
                  index, e.g. pandas.Series or xarray.DataArray,
                  (important for dask datastructures where index may
                  be required to align datasets).
            kwargs: Specify new dimensions in the form new_dim=dim_transform

        Returns:
            Transformed dataset with new dimensions
        """
        drop = kwargs.pop('drop', False)
        keep_index = kwargs.pop('keep_index', True)
        transforms = OrderedDict()
        for s, transform in list(args)+list(kwargs.items()):
            transforms[util.wrap_tuple(s)] = transform

        new_data = OrderedDict()
        for signature, transform in transforms.items():
            applied = transform.apply(
                self, compute=False, keep_index=keep_index
            )
            if len(signature) == 1:
                new_data[signature[0]] = applied
            else:
                for s, vals in zip(signature, applied):
                    new_data[s] = vals

        new_dims = []
        for d in new_data:
            if self.get_dimension(d) is None:
                new_dims.append(d)

        ds = self
        if ds.interface.datatype == 'array':
            # Use a different tabular interface than array
            ds = ds.clone(datatype=[dt for dt in ds.datatype if dt != ds.interface.datatype])

        if drop:
            kdims = [ds.get_dimension(d) for d in new_data if d in ds.kdims]
            vdims = [ds.get_dimension(d) or d for d in new_data if d not in ds.kdims]
            data = OrderedDict([(dimension_name(d), values) for d, values in new_data.items()])
            return ds.clone(data, kdims=kdims, vdims=vdims)
        else:
            new_data = OrderedDict([(dimension_name(d), values) for d, values in new_data.items()])
            data = ds.interface.assign(ds, new_data)
            data, drop = data if isinstance(data, tuple) else (data, [])
            kdims = [kd for kd in self.kdims if kd.name not in drop]
            return ds.clone(data, kdims=kdims, vdims=ds.vdims+new_dims)


class GriddedDataset(Dataset):
    group = param.String(default='GriddedDataset', constant=True)

    def __init__(self, data, kdims=None, vdims=None, **params):
        validate_vdims = params.pop('_validate_vdims', True)

        kdims_spec = self.kdims_spec(kdims)
        vdims_spec = self.vdims_spec(vdims)

        # Determine which drivers to try
        datatype = params.pop("datatype", None)
        if datatype:
            # Use tabular drivers included in datatype
            datatype = [
                dt for dt in datatype
                if dt in Interface.datatypes_by_kind.get("gridded", [])
            ]
        else:
            datatype = Interface.datatypes_by_kind.get("gridded", [])

        initialized = Interface.initialize(
            data, kdims_spec=kdims_spec, vdims_spec=vdims_spec, datatype=datatype
        )
        (data, self.interface, dims, extra_kws) = initialized
        super(GriddedDataset, self).__init__(data, kdims, vdims, **params)
        self.interface.validate(self, validate_vdims)
