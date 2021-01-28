from holodata.dimension import Dimensioned
from holodata.interface import Interface


class TabularDataset(Dimensioned):
    def __init__(self, data, kdims=None, vdims=None, **params):
        validate_vdims = params.pop('_validate_vdims', True)

        kdims_spec = self.kdims_spec(kdims)
        vdims_spec = self.vdims_spec(vdims)
        initialized = Interface.initialize(
            data, kdims_spec=kdims_spec, vdims_spec=vdims_spec, kind="tabular"
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
