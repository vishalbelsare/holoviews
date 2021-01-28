

# class Apply(object):
#     """
#     Utility to apply a function or operation to all viewable elements
#     inside the object.
#     """
#
#     def __init__(self, obj, mode=None):
#         self._obj = obj
#
#     def __call__(self, apply_function, streams=[], link_inputs=True,
#                  link_dataset=True, dynamic=None, per_element=False, **kwargs):
#         """Applies a function to all (Nd)Overlay or Element objects.
#
#         Any keyword arguments are passed through to the function. If
#         keyword arguments are instance parameters, or streams are
#         supplied the returned object will dynamically update in
#         response to changes in those objects.
#
#         Args:
#             apply_function: A callable function
#                 The function will be passed the return value of the
#                 DynamicMap as the first argument and any supplied
#                 stream values or keywords as additional keyword
#                 arguments.
#             streams (list, optional): A list of Stream objects
#                 The Stream objects can dynamically supply values which
#                 will be passed to the function as keywords.
#             link_inputs (bool, optional): Whether to link the inputs
#                 Determines whether Streams and Links attached to
#                 original object will be inherited.
#             link_dataset (bool, optional): Whether to link the dataset
#                 Determines whether the dataset will be inherited.
#             dynamic (bool, optional): Whether to make object dynamic
#                 By default object is made dynamic if streams are
#                 supplied, an instance parameter is supplied as a
#                 keyword argument, or the supplied function is a
#                 parameterized method.
#             per_element (bool, optional): Whether to apply per element
#                 By default apply works on the leaf nodes, which
#                 includes both elements and overlays. If set it will
#                 apply directly to elements.
#             kwargs (dict, optional): Additional keyword arguments
#                 Keyword arguments which will be supplied to the
#                 function.
#
#         Returns:
#             A new object where the function was applied to all
#             contained (Nd)Overlay or Element objects.
#         """
#         from .data import Dataset
#         from .dimension import ViewableElement
#         from .element import Element
#         from .spaces import HoloMap, DynamicMap
#         from ..util import Dynamic
#
#         if isinstance(self._obj, DynamicMap) and dynamic == False:
#             samples = tuple(d.values for d in self._obj.kdims)
#             if not all(samples):
#                 raise ValueError('Applying a function to a DynamicMap '
#                                  'and setting dynamic=False is only '
#                                  'possible if key dimensions define '
#                                  'a discrete parameter space.')
#             if not len(samples):
#                 return self._obj[samples]
#             return HoloMap(self._obj[samples]).apply(
#                 apply_function, streams, link_inputs, link_dataset,
#                 dynamic, per_element, **kwargs
#             )
#
#         if isinstance(apply_function, util.basestring):
#             args = kwargs.pop('_method_args', ())
#             method_name = apply_function
#             def apply_function(object, **kwargs):
#                 method = getattr(object, method_name, None)
#                 if method is None:
#                     raise AttributeError('Applied method %s does not exist.'
#                                          'When declaring a method to apply '
#                                          'as a string ensure a corresponding '
#                                          'method exists on the object.' %
#                                          method_name)
#                 return method(*args, **kwargs)
#
#         if 'panel' in sys.modules:
#             from panel.widgets.base import Widget
#             kwargs = {k: v.param.value if isinstance(v, Widget) else v
#                       for k, v in kwargs.items()}
#
#         spec = Element if per_element else ViewableElement
#         applies = isinstance(self._obj, spec)
#         params = {p: val for p, val in kwargs.items()
#                   if isinstance(val, param.Parameter)
#                   and isinstance(val.owner, param.Parameterized)}
#
#         dependent_kws = any(
#             (isinstance(val, FunctionType) and hasattr(val, '_dinfo')) or
#             util.is_param_method(val, has_deps=True) for val in kwargs.values()
#         )
#
#         if dynamic is None:
#             is_dynamic = (bool(streams) or isinstance(self._obj, DynamicMap) or
#                           util.is_param_method(apply_function, has_deps=True) or
#                           params or dependent_kws)
#         else:
#             is_dynamic = dynamic
#
#         if (applies or isinstance(self._obj, HoloMap)) and is_dynamic:
#             return Dynamic(self._obj, operation=apply_function, streams=streams,
#                            kwargs=kwargs, link_inputs=link_inputs,
#                            link_dataset=link_dataset)
#         elif applies:
#             inner_kwargs = util.resolve_dependent_kwargs(kwargs)
#             if hasattr(apply_function, 'dynamic'):
#                 inner_kwargs['dynamic'] = False
#             new_obj = apply_function(self._obj, **inner_kwargs)
#             if (link_dataset and isinstance(self._obj, Dataset) and
#                 isinstance(new_obj, Dataset) and new_obj._dataset is None):
#                 new_obj._dataset = self._obj.dataset
#             return new_obj
#         elif self._obj._deep_indexable:
#             mapped = []
#             for k, v in self._obj.data.items():
#                 new_val = v.apply(apply_function, dynamic=dynamic, streams=streams,
#                                   link_inputs=link_inputs, link_dataset=link_dataset,
#                                   **kwargs)
#                 if new_val is not None:
#                     mapped.append((k, new_val))
#             return self._obj.clone(mapped, link=link_inputs)
#
#     def aggregate(self, dimensions=None, function=None, spreadfn=None, **kwargs):
#         """Applies a aggregate function to all ViewableElements.
#
#         See :py:meth:`Dimensioned.aggregate` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         kwargs['_method_args'] = (dimensions, function, spreadfn)
#         kwargs['per_element'] = True
#         return self.__call__('aggregate', **kwargs)
#
#     def opts(self, *args, **kwargs):
#         """Applies options to all ViewableElement objects.
#
#         See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         from ..util.transform import dim
#         from ..streams import Params
#         params = {}
#         for arg in kwargs.values():
#             if isinstance(arg, dim):
#                 params.update(arg.params)
#         streams = Params.from_params(params, watch_only=True)
#         kwargs['streams'] = kwargs.get('streams', []) + streams
#         kwargs['_method_args'] = args
#         return self.__call__('opts', **kwargs)
#
#     def reduce(self, dimensions=[], function=None, spreadfn=None, **kwargs):
#         """Applies a reduce function to all ViewableElement objects.
#
#         See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         kwargs['_method_args'] = (dimensions, function, spreadfn)
#         kwargs['per_element'] = True
#         return self.__call__('reduce', **kwargs)
#
#     def sample(self, samples=[], bounds=None, **kwargs):
#         """Samples element values at supplied coordinates.
#
#         See :py:meth:`Dataset.sample` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         kwargs['_method_args'] = (samples, bounds)
#         kwargs['per_element'] = True
#         return self.__call__('sample', **kwargs)
#
#     def select(self, **kwargs):
#         """Applies a selection to all ViewableElement objects.
#
#         See :py:meth:`Dimensioned.opts` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         return self.__call__('select', **kwargs)
#
#     def transform(self, *args, **kwargs):
#         """Applies transforms to all Datasets.
#
#         See :py:meth:`Dataset.transform` and :py:meth:`Apply.__call__`
#         for more information.
#         """
#         from ..util.transform import dim
#         from ..streams import Params
#         params = {}
#         for _, arg in list(args)+list(kwargs.items()):
#             if isinstance(arg, dim):
#                 params.update(arg.params)
#         streams = Params.from_params(params, watch_only=True)
#         kwargs['streams'] = kwargs.get('streams', []) + streams
#         kwargs['_method_args'] = args
#         kwargs['per_element'] = True
#         return self.__call__('transform', **kwargs)


class Redim(object):
    """
    Utility that supports re-dimensioning any HoloViews object via the
    redim method.
    """

    def __init__(self, obj):
        self._obj = obj

    def __str__(self):
        return "<holoviews.core.dimension.redim method>"

    @classmethod
    def replace_dimensions(cls, dimensions, overrides):
        """Replaces dimensions in list with dictionary of overrides.

        Args:
            dimensions: List of dimensions
            overrides: Dictionary of dimension specs indexed by name

        Returns:
            list: List of dimensions with replacements applied
        """
        from holodata.dimension import Dimension
        from holodata import util

        replaced = []
        for d in dimensions:
            if d.name in overrides:
                override = overrides[d.name]
            elif d.label in overrides:
                override = overrides[d.label]
            else:
                override = None

            if override is None:
                replaced.append(d)
            elif isinstance(override, (util.basestring, tuple)):
                replaced.append(d.clone(override))
            elif isinstance(override, Dimension):
                replaced.append(override)
            elif isinstance(override, dict):
                replaced.append(d.clone(override.get('name',None),
                                        **{k:v for k,v in override.items() if k != 'name'}))
            else:
                raise ValueError('Dimension can only be overridden '
                                 'with another dimension or a dictionary '
                                 'of attributes')
        return replaced


    def _filter_cache(self, dmap, kdims):
        """
        Returns a filtered version of the DynamicMap cache leaving only
        keys consistently with the newly specified values
        """
        filtered = []
        for key, value in dmap.data.items():
            if not any(kd.values and v not in kd.values for kd, v in zip(kdims, key)):
                filtered.append((key, value))
        return filtered

    def _transform_dimension(self, kdims, vdims, dimension):
        if dimension in kdims:
            idx = kdims.index(dimension)
            dimension = self._obj.kdims[idx]
        elif dimension in vdims:
            idx = vdims.index(dimension)
            dimension = self._obj.vdims[idx]
        return dimension

    def _create_expression_transform(self, kdims, vdims, exclude=[]):
        from holodata.dimension import dimension_name
        from holodata.transform import dim

        def _transform_expression(expression):
            if dimension_name(expression.dimension) in exclude:
                dimension = expression.dimension
            else:
                dimension = self._transform_dimension(
                    kdims, vdims, expression.dimension
                )
            expression = expression.clone(dimension)
            ops = []
            for op in expression.ops:
                new_op = dict(op)
                new_args = []
                for arg in op['args']:
                    if isinstance(arg, dim):
                        arg = _transform_expression(arg)
                    new_args.append(arg)
                new_op['args'] = tuple(new_args)
                new_kwargs = {}
                for kw, kwarg in op['kwargs'].items():
                    if isinstance(kwarg, dim):
                        kwarg = _transform_expression(kwarg)
                    new_kwargs[kw] = kwarg
                new_op['kwargs'] = new_kwargs
                ops.append(new_op)
            expression.ops = ops
            return expression
        return _transform_expression

    def __call__(self, specs=None, **dimensions):
        """
        Replace dimensions on the dataset and allows renaming
        dimensions in the dataset. Dimension mapping should map
        between the old dimension name and a dictionary of the new
        attributes, a completely new dimension or a new string name.
        """
        obj = self._obj
        redimmed = obj

        if specs is not None:
            if not isinstance(specs, list):
                specs = [specs]
            matches = any(obj.matches(spec) for spec in specs)
            if not matches:
                return redimmed

        kdims = self.replace_dimensions(obj.kdims, dimensions)
        vdims = self.replace_dimensions(obj.vdims, dimensions)
        zipped_dims = zip(obj.kdims+obj.vdims, kdims+vdims)
        renames = {pk.name: nk for pk, nk in zipped_dims if pk.name != nk.name}

        data = obj.data
        if renames:
            data = obj.interface.redim(obj, renames)
        transform = self._create_expression_transform(kdims, vdims, list(renames.values()))
        transforms = obj._transforms + [transform]
        clone = obj.clone(data, kdims=kdims, vdims=vdims, transforms=transforms)
        if self._obj.dimensions(label='name') == clone.dimensions(label='name'):
            # Ensure that plot_id is inherited as long as dimension
            # name does not change
            clone._plot_id = self._obj._plot_id
        return clone

    def _redim(self, name, specs, **dims):
        dimensions = {k:{name:v} for k,v in dims.items()}
        return self(specs, **dimensions)

    def cyclic(self, specs=None, **values):
        return self._redim('cyclic', specs, **values)

    def value_format(self, specs=None, **values):
        return self._redim('value_format', specs, **values)

    def range(self, specs=None, **values):
        return self._redim('range', specs, **values)

    def label(self, specs=None, **values):
        for k, v in values.items():
            dim = self._obj.get_dimension(k)
            if dim and dim.name != dim.label and dim.label != v:
                raise ValueError('Cannot override an existing Dimension label')
        return self._redim('label', specs, **values)

    def soft_range(self, specs=None, **values):
        return self._redim('soft_range', specs, **values)

    def type(self, specs=None, **values):
        return self._redim('type', specs, **values)

    def nodata(self, specs=None, **values):
        return self._redim('nodata', specs, **values)

    def step(self, specs=None, **values):
        return self._redim('step', specs, **values)

    def default(self, specs=None, **values):
        return self._redim('default', specs, **values)

    def unit(self, specs=None, **values):
        return self._redim('unit', specs, **values)

    def values(self, specs=None, **ranges):
        return self._redim('values', specs, **ranges)
