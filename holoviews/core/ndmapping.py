"""
Supplies MultiDimensionalMapping and NdMapping which are multi-dimensional
map types. The former class only allows indexing whereas the latter
also enables slicing over multiple dimension ranges.
"""
import holodata.util
from holodata.ndmapping import item_check, UniformNdMapping
from .pprint import PrettyPrinter
from .dimension import ViewableMixin
from .util import (unicode)


class ViewableUniformNdMapping(UniformNdMapping, ViewableMixin):
    def __init__(
            self, initial_items=None, kdims=None, group=None, label=None,
            id=None, plot_id=None, **params
    ):
        super(ViewableUniformNdMapping, self).__init__(initial_items, kdims, group, label, **params)
        ViewableMixin.__init__(self, id=id, plot_id=plot_id)

    def __repr__(self):
        return PrettyPrinter.pprint(self)

    def __str__(self):
        return repr(self)

    def __unicode__(self):
        return unicode(PrettyPrinter.pprint(self))

    def clone(self, data=None, shared_data=True, new_type=None, link=True,
              *args, **overrides):
        """Clones the object, overriding data and parameters.

        Args:
            data: New data replacing the existing data
            shared_data (bool, optional): Whether to use existing data
            new_type (optional): Type to cast object to
            link (bool, optional): Whether clone should be linked
                Determines whether Streams and Links attached to
                original object will be inherited.
            *args: Additional arguments to pass to constructor
            **overrides: New keyword arguments to pass to constructor

        Returns:
            Cloned object
        """
        settings = dict(self.param.get_param_values())
        if settings.get('group', None) != self._group:
            settings.pop('group')
        if settings.get('label', None) != self._label:
            settings.pop('label')
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.param.objects()
            settings = {k: v for k, v in settings.items()
                      if k in new_params}

        if data is not None:
            shared_data = False
        else:
            if shared_data:
                data = self.data
            else:
                raise NotImplementedError("clone data not implemented")

        settings = dict(settings, **overrides)

        self.update_plot_id_settings(settings, shared_data=shared_data, link=link)

        # Apply name mangling for __ attribute
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        with item_check(not shared_data and self._check_items):
            return clone_type(data, *args, **{k:v for k,v in settings.items()
                                              if k not in pos_args})
