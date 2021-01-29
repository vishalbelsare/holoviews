import numpy as np

from holodata.dimension import Dimensioned
from holodata.interface import Interface


class TabularDataset(Dimensioned):
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

    def aggregate(self, *args, **kwargs):
        return self.interface.aggregate(self, *args, **kwargs)

    def unpack_scalar(self, *args, **kwargs):
        return self.interface.unpack_scalar(self, *args, **kwargs)

    def add_dimension(self, *args, **kwargs):
        return self.interface.add_dimension(self, *args, **kwargs)

    def select(self, *args, **kwargs):
        return self.clone(
            data=self.interface.select(self, *args, **kwargs)
        )

    def groupby(self, dimensions, kdims=None):
        return self.interface.groupby(self, dimensions, kdims=kdims)

    def reindex(self, *args, **kwargs):
        return self.interface.reindex(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.interface.sample(self, *args, **kwargs)

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
