import datetime as dt
import re
from collections import OrderedDict

import numpy as np
import param

import holodata.util
from holodata.label import LabelledData
from holodata.pprint import SimplePrettyPrinter
from holodata.util import unicode, basestring, dimension_sanitizer
from holodata import util

ALIASES = {'key_dimensions': 'kdims', 'value_dimensions': 'vdims',
           'constant_dimensions': 'cdims'}
title_format = "{name}: {val}{unit}"


def asdim(dimension):
    """Convert the input to a Dimension.

    Args:
        dimension: tuple, dict or string type to convert to Dimension

    Returns:
        A Dimension object constructed from the dimension spec. No
        copy is performed if the input is already a Dimension.
    """
    if isinstance(dimension, Dimension):
        return dimension
    elif isinstance(dimension, (tuple, dict, basestring)):
        return Dimension(dimension)
    else:
        raise ValueError('%s type could not be interpreted as Dimension. '
                         'Dimensions must be declared as a string, tuple, '
                         'dictionary or Dimension type.')


def dimension_name(dimension):
    """Return the Dimension.name for a dimension-like object.

    Args:
        dimension: Dimension or dimension string, tuple or dict

    Returns:
        The name of the Dimension or what would be the name if the
        input as converted to a Dimension.
    """
    if isinstance(dimension, Dimension):
        return dimension.name
    elif isinstance(dimension, basestring):
        return dimension
    elif isinstance(dimension, tuple):
        return dimension[0]
    elif isinstance(dimension, dict):
        return dimension['name']
    elif dimension is None:
        return None
    else:
        raise ValueError('%s type could not be interpreted as Dimension. '
                         'Dimensions must be declared as a string, tuple, '
                         'dictionary or Dimension type.'
                         % type(dimension).__name__)


def process_dimensions(kdims, vdims):
    """Converts kdims and vdims to Dimension objects.

    Args:
        kdims: List or single key dimension(s) specified as strings,
            tuples dicts or Dimension objects.
        vdims: List or single value dimension(s) specified as strings,
            tuples dicts or Dimension objects.

    Returns:
        Dictionary containing kdims and vdims converted to Dimension
        objects:

        {'kdims': [Dimension('x')], 'vdims': [Dimension('y')]
    """
    dimensions = {}
    for group, dims in [('kdims', kdims), ('vdims', vdims)]:
        if dims is None:
            continue
        elif isinstance(dims, (tuple, basestring, Dimension, dict)):
            dims = [dims]
        elif not isinstance(dims, list):
            raise ValueError("%s argument expects a Dimension or list of dimensions, "
                             "specified as tuples, strings, dictionaries or Dimension "
                             "instances, not a %s type. Ensure you passed the data as the "
                             "first argument." % (group, type(dims).__name__))
        for dim in dims:
            if not isinstance(dim, (tuple, basestring, Dimension, dict)):
                raise ValueError('Dimensions must be defined as a tuple, '
                                 'string, dictionary or Dimension instance, '
                                 'found a %s type.' % type(dim).__name__)
        dimensions[group] = [asdim(d) for d in dims]
    return dimensions


class Dimension(param.Parameterized):
    """
    Dimension objects are used to specify some important general
    features that may be associated with a collection of values.

    For instance, a Dimension may specify that a set of numeric values
    actually correspond to 'Height' (dimension name), in units of
    meters, with a descriptive label 'Height of adult males'.

    All dimensions object have a name that identifies them and a label
    containing a suitable description. If the label is not explicitly
    specified it matches the name.

    These two parameters define the core identity of the dimension
    object and must match if two dimension objects are to be considered
    equivalent. All other parameters are considered optional metadata
    and are not used when testing for equality.

    Unlike all the other parameters, these core parameters can be used
    to construct a Dimension object from a tuple. This format is
    sufficient to define an identical Dimension:

    Dimension('a', label='Dimension A') == Dimension(('a', 'Dimension A'))

    Everything else about a dimension is considered to reflect
    non-semantic preferences. Examples include the default value (which
    may be used in a visualization to set an initial slider position),
    how the value is to rendered as text (which may be used to specify
    the printed floating point precision) or a suitable range of values
    to consider for a particular analysis.

    Units
    -----

    Full unit support with automated conversions are on the HoloViews
    roadmap. Once rich unit objects are supported, the unit (or more
    specifically the type of unit) will be part of the core dimension
    specification used to establish equality.

    Until this feature is implemented, there are two auxiliary
    parameters that hold some partial information about the unit: the
    name of the unit and whether or not it is cyclic. The name of the
    unit is used as part of the pretty-printed representation and
    knowing whether it is cyclic is important for certain operations.
    """

    name = param.String(doc="""
       Short name associated with the Dimension, such as 'height' or
       'weight'. Valid Python identifiers make good names, because they
       can be used conveniently as a keyword in many contexts.""")

    label = param.String(default=None, doc="""
        Unrestricted label used to describe the dimension. A label
        should succinctly describe the dimension and may contain any
        characters, including Unicode and LaTeX expression.""")

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic such that the
        maximum allowed value (defined by the range parameter) is
        continuous with the minimum allowed value.""")

    default = param.Parameter(default=None, doc="""
        Default value of the Dimension which may be useful for widget
        or other situations that require an initial or default value.""")

    nodata = param.Integer(default=None, doc="""
        Optional missing-data value for integer data.
        If non-None, data with this value will be replaced with NaN.""")

    range = param.Tuple(default=(None, None), doc="""
        Specifies the minimum and maximum allowed values for a
        Dimension. None is used to represent an unlimited bound.""")

    soft_range = param.Tuple(default=(None, None), doc="""
        Specifies a minimum and maximum reference value, which
        may be overridden by the data.""")

    step = param.Number(default=None, doc="""
        Optional floating point step specifying how frequently the
        underlying space should be sampled. May be used to define a
        discrete sampling over the range.""")

    type = param.Parameter(default=None, doc="""
        Optional type associated with the Dimension values. The type
        may be an inbuilt constructor (such as int, str, float) or a
        custom class object.""")

    unit = param.String(default=None, allow_None=True, doc="""
        Optional unit string associated with the Dimension. For
        instance, the string 'm' may be used represent units of meters
        and 's' to represent units of seconds.""")

    value_format = param.Callable(default=None, doc="""
        Formatting function applied to each value before display.""")

    values = param.List(default=[], doc="""
        Optional specification of the allowed value set for the
        dimension that may also be used to retain a categorical
        ordering.""")

    # Defines default formatting by type
    type_formatters = {}
    unit_format = ' ({unit})'
    presets = {} # A dictionary-like mapping name, (name,) or
                 # (name, unit) to a preset Dimension object

    def __init__(self, spec, **params):
        """
        Initializes the Dimension object with the given name.
        """
        if 'name' in params:
            raise KeyError('Dimension name must only be passed as the positional argument')

        if isinstance(spec, Dimension):
            existing_params = dict(self.param.get_param_values())
        elif (spec, params.get('unit', None)) in self.presets.keys():
            preset = self.presets[(str(spec), str(params['unit']))]
            existing_params = dict(self.param.get_param_values())
        elif isinstance(spec, dict):
            existing_params = spec
        elif spec in self.presets:
            existing_params = dict(self.param.get_param_values())
        elif (spec,) in self.presets:
            existing_params = dict(self.param.get_param_values())
        else:
            existing_params = {}

        all_params = dict(existing_params, **params)
        if isinstance(spec, tuple):
            if not all(isinstance(s, basestring) for s in spec) or len(spec) != 2:
                raise ValueError("Dimensions specified as a tuple must be a tuple "
                                 "consisting of the name and label not: %s" % str(spec))
            name, label = spec
            all_params['name'] = name
            all_params['label'] = label
            if 'label' in params and (label != params['label']):
                if params['label'] != label:
                    self.param.warning(
                        'Using label as supplied by keyword ({!r}), ignoring '
                        'tuple value {!r}'.format(params['label'], label))
                all_params['label'] = params['label']
        elif isinstance(spec, basestring):
            all_params['name'] = spec
            all_params['label'] = params.get('label', spec)

        if all_params['name'] == '':
            raise ValueError('Dimension name cannot be the empty string')
        if all_params['label'] in ['', None]:
            raise ValueError('Dimension label cannot be None or the empty string')

        values = params.get('values', [])
        if isinstance(values, basestring) and values == 'initial':
            self.param.warning("The 'initial' string for dimension values "
                               "is no longer supported.")
            values = []

        all_params['values'] = list(util.unique_array(values))
        super(Dimension, self).__init__(**all_params)
        if self.default is not None:
            if self.values and self.default not in values:
                raise ValueError('%r default %s not found in declared values: %s' %
                                 (self, self.default, self.values))
            elif (self.range != (None, None) and
                  ((self.range[0] is not None and self.default < self.range[0]) or
                   (self.range[0] is not None and self.default > self.range[1]))):
                raise ValueError('%r default %s not in declared range: %s' %
                                 (self, self.default, self.range))


    @property
    def spec(self):
        """"Returns the Dimensions tuple specification

        Returns:
            tuple: Dimension tuple specification
        """
        return (self.name, self.label)


    def __call__(self, spec=None, **overrides):
        self.param.warning('Dimension.__call__ method has been deprecated, '
                           'use the clone method instead.')
        return self.clone(spec=spec, **overrides)


    def clone(self, spec=None, **overrides):
        """Clones the Dimension with new parameters

        Derive a new Dimension that inherits existing parameters
        except for the supplied, explicit overrides

        Args:
            spec (tuple, optional): Dimension tuple specification
            **overrides: Dimension parameter overrides

        Returns:
            Cloned Dimension object
        """
        settings = dict(self.param.get_param_values(), **overrides)

        if spec is None:
            spec = (self.name, overrides.get('label', self.label))
        if 'label' in overrides and isinstance(spec, basestring) :
            spec = (spec, overrides['label'])
        elif 'label' in overrides and isinstance(spec, tuple) :
            if overrides['label'] != spec[1]:
                self.param.warning(
                    'Using label as supplied by keyword ({!r}), ignoring '
                    'tuple value {!r}'.format(overrides['label'], spec[1]))
            spec = (spec[0],  overrides['label'])
        return self.__class__(spec, **{k:v for k,v in settings.items()
                                       if k not in ['name', 'label']})

    def __hash__(self):
        """Hashes object on Dimension spec, i.e. (name, label).
        """
        return hash(self.spec)

    def __setstate__(self, d):
        """
        Compatibility for pickles before alias attribute was introduced.
        """
        super(Dimension, self).__setstate__(d)
        self.label = self.name

    def __eq__(self, other):
        "Implements equals operator including sanitized comparison."

        if isinstance(other, Dimension):
            return self.spec == other.spec

        # For comparison to strings. Name may be sanitized.
        return other in [self.name, self.label, dimension_sanitizer(self.name)]

    def __ne__(self, other):
        "Implements not equal operator including sanitized comparison."
        return not self.__eq__(other)

    def __lt__(self, other):
        "Dimensions are sorted alphanumerically by name"
        return self.name < other.name if isinstance(other, Dimension) else self.name < other

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.pprint()

    @property
    def pprint_label(self):
        "The pretty-printed label string for the Dimension"
        unit = ('' if self.unit is None
                else type(self.unit)(self.unit_format).format(unit=self.unit))
        return util.bytes_to_unicode(self.label) + util.bytes_to_unicode(unit)

    def pprint(self):
        changed = dict(self.param.get_param_values(onlychanged=True))
        if len(set([changed.get(k, k) for k in ['name','label']])) == 1:
            return 'Dimension({spec})'.format(spec=repr(self.name))

        params = self.param.objects('existing')
        ordering = sorted(
            sorted(changed.keys()), key=lambda k: (
                -float('inf') if params[k].precedence is None
                else params[k].precedence))
        kws = ", ".join('%s=%r' % (k, changed[k]) for k in ordering if k != 'name')
        return 'Dimension({spec}, {kws})'.format(spec=repr(self.name), kws=kws)


    def pprint_value(self, value, print_unit=False):
        """Applies the applicable formatter to the value.

        Args:
            value: Dimension value to format

        Returns:
            Formatted dimension value
        """
        own_type = type(value) if self.type is None else self.type
        formatter = (self.value_format if self.value_format
                     else self.type_formatters.get(own_type))
        if formatter:
            if callable(formatter):
                formatted_value = formatter(value)
            elif isinstance(formatter, basestring):
                if isinstance(value, (dt.datetime, dt.date)):
                    formatted_value = value.strftime(formatter)
                elif isinstance(value, np.datetime64):
                    formatted_value = util.dt64_to_dt(value).strftime(formatter)
                elif re.findall(r"\{(\w+)\}", formatter):
                    formatted_value = formatter.format(value)
                else:
                    formatted_value = formatter % value
        else:
            formatted_value = unicode(util.bytes_to_unicode(value))

        if print_unit and self.unit is not None:
            formatted_value = formatted_value + ' ' + util.bytes_to_unicode(self.unit)
        return formatted_value

    def pprint_value_string(self, value):
        """Pretty print the dimension value and unit with title_format

        Args:
            value: Dimension value to format

        Returns:
            Formatted dimension value string with unit
        """
        unit = '' if self.unit is None else ' ' + util.bytes_to_unicode(self.unit)
        value = self.pprint_value(value)
        return title_format.format(name=util.bytes_to_unicode(self.label), val=value, unit=unit)


class Dimensioned(LabelledData):
    """
    Dimensioned is a base class that allows the data contents of a
    class to be associated with dimensions. The contents associated
    with dimensions may be partitioned into one of three types

    * key dimensions: These are the dimensions that can be indexed via
                      the __getitem__ method. Dimension objects
                      supporting key dimensions must support indexing
                      over these dimensions and may also support
                      slicing. This list ordering of dimensions
                      describes the positional components of each
                      multi-dimensional indexing operation.

                      For instance, if the key dimension names are
                      'weight' followed by 'height' for Dimensioned
                      object 'obj', then obj[80,175] indexes a weight
                      of 80 and height of 175.

                      Accessed using either kdims.

    * value dimensions: These dimensions correspond to any data held
                        on the Dimensioned object not in the key
                        dimensions. Indexing by value dimension is
                        supported by dimension name (when there are
                        multiple possible value dimensions); no
                        slicing semantics is supported and all the
                        data associated with that dimension will be
                        returned at once. Note that it is not possible
                        to mix value dimensions and deep dimensions.

                        Accessed using either vdims.

    * deep dimensions: These are dynamically computed dimensions that
                       belong to other Dimensioned objects that are
                       nested in the data. Objects that support this
                       should enable the _deep_indexable flag. Note
                       that it is not possible to mix value dimensions
                       and deep dimensions.

                       Accessed using either ddims.

    Dimensioned class support generalized methods for finding the
    range and type of values along a particular Dimension. The range
    method relies on the appropriate implementation of the
    dimension_values methods on subclasses.

    The index of an arbitrary dimension is its positional index in the
    list of all dimensions, starting with the key dimensions, followed
    by the value dimensions and ending with the deep dimensions.
    """

    cdims = param.Dict(default=OrderedDict(), doc="""
       The constant dimensions defined as a dictionary of Dimension:value
       pairs providing additional dimension information about the object.

       Aliased with constant_dimensions.""")

    kdims = param.List(bounds=(0, None), constant=True, doc="""
       The key dimensions defined as list of dimensions that may be
       used in indexing (and potential slicing) semantics. The order
       of the dimensions listed here determines the semantics of each
       component of a multi-dimensional indexing operation.

       Aliased with key_dimensions.""")

    vdims = param.List(bounds=(0, None), constant=True, doc="""
       The value dimensions defined as the list of dimensions used to
       describe the components of the data. If multiple value
       dimensions are supplied, a particular value dimension may be
       indexed by name after the key dimensions.

       Aliased with value_dimensions.""")

    group = param.String(default='Dimensioned', constant=True, doc="""
       A string describing the data wrapped by the object.""")

    __abstract = True
    _dim_groups = ['kdims', 'vdims', 'cdims', 'ddims']
    _dim_aliases = dict(key_dimensions='kdims', value_dimensions='vdims',
                        constant_dimensions='cdims', deep_dimensions='ddims')

    @classmethod
    def kdims_spec(cls, kdims):
        return dict(
            value=kdims, bounds=cls.param.kdims.bounds, default=cls.param.kdims.default
        )

    @classmethod
    def vdims_spec(cls, vdims):
        return dict(
            value=vdims, bounds=cls.param.vdims.bounds, default=cls.param.vdims.default
        )

    def __init__(self, data, kdims=None, vdims=None, **params):
        params.update(process_dimensions(kdims, vdims))
        if 'cdims' in params:
            params['cdims'] = {d if isinstance(d, Dimension) else Dimension(d): val
                               for d, val in params['cdims'].items()}
        super(Dimensioned, self).__init__(data, **params)
        self.ndims = len(self.kdims)
        cdims = [(d.name, val) for d, val in self.cdims.items()]
        self._cached_constants = OrderedDict(cdims)
        self._settings = None

        # Instantiate accessors

    def __repr__(self):
        return SimplePrettyPrinter.pprint(self)

    def __str__(self):
        return repr(self)

    def __unicode__(self):
        return unicode(SimplePrettyPrinter.pprint(self))

    # TODO: need versions without pipeline metaclass
    @property
    def apply(self):
        from holoviews.core.accessors import Apply
        return Apply(self)

    @property
    def redim(self):
        from holoviews.core import Redim
        return Redim(self)

    def _valid_dimensions(self, dimensions):
        """Validates key dimension input

        Returns kdims if no dimensions are specified"""
        if dimensions is None:
            dimensions = self.kdims
        elif not isinstance(dimensions, list):
            dimensions = [dimensions]

        valid_dimensions = []
        for dim in dimensions:
            if isinstance(dim, Dimension): dim = dim.name
            if dim not in self.kdims:
                raise Exception("Supplied dimensions %s not found." % dim)
            valid_dimensions.append(dim)
        return valid_dimensions


    @property
    def ddims(self):
        "The list of deep dimensions"
        if self._deep_indexable and self:
            return self.values()[0].dimensions()
        else:
            return []


    def dimensions(self, selection='all', label=False):
        """Lists the available dimensions on the object

        Provides convenient access to Dimensions on nested Dimensioned
        objects. Dimensions can be selected by their type, i.e. 'key'
        or 'value' dimensions. By default 'all' dimensions are
        returned.

        Args:
            selection: Type of dimensions to return
                The type of dimension, i.e. one of 'key', 'value',
                'constant' or 'all'.
            label: Whether to return the name, label or Dimension
                Whether to return the Dimension objects (False),
                the Dimension names (True/'name') or labels ('label').

        Returns:
            List of Dimension objects or their names or labels
        """
        if label in ['name', True]:
            label = 'short'
        elif label == 'label':
            label = 'long'
        elif label:
            raise ValueError("label needs to be one of True, False, 'name' or 'label'")

        lambdas = {'k': (lambda x: x.kdims, {'full_breadth': False}),
                   'v': (lambda x: x.vdims, {}),
                   'c': (lambda x: x.cdims, {})}
        aliases = {'key': 'k', 'value': 'v', 'constant': 'c'}
        if selection in ['all', 'ranges']:
            groups = [d for d in self._dim_groups if d != 'cdims']
            dims = [dim for group in groups
                    for dim in getattr(self, group)]
        elif isinstance(selection, list):
            dims =  [dim for group in selection
                     for dim in getattr(self, '%sdims' % aliases.get(group))]
        elif aliases.get(selection) in lambdas:
            selection = aliases.get(selection, selection)
            lmbd, kwargs = lambdas[selection]
            key_traversal = self.traverse(lmbd, **kwargs)
            dims = [dim for keydims in key_traversal for dim in keydims]
        else:
            raise KeyError("Invalid selection %r, valid selections include"
                           "'all', 'value' and 'key' dimensions" % repr(selection))
        return [(dim.label if label == 'long' else dim.name)
                if label else dim for dim in dims]


    def get_dimension(self, dimension, default=None, strict=False):
        """Get a Dimension object by name or index.

        Args:
            dimension: Dimension to look up by name or integer index
            default (optional): Value returned if Dimension not found
            strict (bool, optional): Raise a KeyError if not found

        Returns:
            Dimension object for the requested dimension or default
        """
        if dimension is not None and not isinstance(dimension, (int, basestring, Dimension)):
            raise TypeError('Dimension lookup supports int, string, '
                            'and Dimension instances, cannot lookup '
                            'Dimensions using %s type.' % type(dimension).__name__)
        all_dims = self.dimensions()
        if isinstance(dimension, int):
            if 0 <= dimension < len(all_dims):
                return all_dims[dimension]
            elif strict:
                raise KeyError("Dimension %r not found" % dimension)
            else:
                return default

        if isinstance(dimension, Dimension):
            dims = [d for d in all_dims if dimension == d]
            if strict and not dims:
                raise KeyError("%r not found." % dimension)
            elif dims:
                return dims[0]
            else:
                return None
        else:
            dimension = dimension_name(dimension)
            name_map = {dim.spec: dim for dim in all_dims}
            name_map.update({dim.name: dim for dim in all_dims})
            name_map.update({dim.label: dim for dim in all_dims})
            name_map.update({dimension_sanitizer(dim.name): dim for dim in all_dims})
            if strict and dimension not in name_map:
                raise KeyError("Dimension %r not found." % dimension)
            else:
                return name_map.get(dimension, default)


    def get_dimension_index(self, dimension):
        """Get the index of the requested dimension.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Integer index of the requested dimension
        """
        if isinstance(dimension, int):
            if (dimension < (self.ndims + len(self.vdims)) or
                dimension < len(self.dimensions())):
                return dimension
            else:
                return IndexError('Dimension index out of bounds')
        dim = dimension_name(dimension)
        try:
            dimensions = self.kdims+self.vdims
            return [i for i, d in enumerate(dimensions) if d == dim][0]
        except IndexError:
            raise Exception("Dimension %s not found in %s." %
                            (dim, self.__class__.__name__))


    def get_dimension_type(self, dim):
        """Get the type of the requested dimension.

        Type is determined by Dimension.type attribute or common
        type of the dimension values, otherwise None.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Declared type of values along the dimension
        """
        dim_obj = self.get_dimension(dim)
        if dim_obj and dim_obj.type is not None:
            return dim_obj.type
        dim_vals = [type(v) for v in self.dimension_values(dim)]
        if len(set(dim_vals)) == 1:
            return dim_vals[0]
        else:
            return None


    def __getitem__(self, key):
        """
        Multi-dimensional indexing semantics is determined by the list
        of key dimensions. For instance, the first indexing component
        will index the first key dimension.

        After the key dimensions are given, *either* a value dimension
        name may follow (if there are multiple value dimensions) *or*
        deep dimensions may then be listed (for applicable deep
        dimensions).
        """
        return self


    def select(self, selection_specs=None, **kwargs):
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

        Args:
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
        if selection_specs is not None and not isinstance(selection_specs, (list, tuple)):
            selection_specs = [selection_specs]

        # Apply all indexes applying on this object
        vdims = self.vdims+['value'] if self.vdims else []
        kdims = self.kdims
        local_kwargs = {k: v for k, v in kwargs.items()
                        if k in kdims+vdims}

        # Check selection_spec applies
        if selection_specs is not None:
            if not isinstance(selection_specs, (list, tuple)):
                selection_specs = [selection_specs]
            matches = any(self.matches(spec)
                          for spec in selection_specs)
        else:
            matches = True

        # Apply selection to self
        if local_kwargs and matches:
            ndims = self.ndims
            if any(d in self.vdims for d in kwargs):
                ndims = len(self.kdims+self.vdims)
            select = [slice(None) for _ in range(ndims)]
            for dim, val in local_kwargs.items():
                if dim == 'value':
                    select += [val]
                else:
                    if isinstance(val, tuple): val = slice(*val)
                    select[self.get_dimension_index(dim)] = val
            if self._deep_indexable:
                selection = self.get(tuple(select), None)
                if selection is None:
                    selection = self.clone(shared_data=False)
            else:
                selection = self[tuple(select)]
        else:
            selection = self

        if not isinstance(selection, Dimensioned):
            return selection
        elif type(selection) is not type(self) and isinstance(selection, Dimensioned):
            # Apply the selection on the selected object of a different type
            dimensions = selection.dimensions() + ['value']
            if any(kw in dimensions for kw in kwargs):
                selection = selection.select(selection_specs=selection_specs, **kwargs)
        elif isinstance(selection, Dimensioned) and selection._deep_indexable:
            # Apply the deep selection on each item in local selection
            items = []
            for k, v in selection.items():
                dimensions = v.dimensions() + ['value']
                if any(kw in dimensions for kw in kwargs):
                    items.append((k, v.select(selection_specs=selection_specs, **kwargs)))
                else:
                    items.append((k, v))
            selection = selection.clone(items)
        return selection


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
        val = self._cached_constants.get(dimension, None)
        if val:
            return np.array([val])
        else:
            raise Exception("Dimension %s not found in %s." %
                            (dimension, self.__class__.__name__))


    def range(self, dimension, data_range=True, dimension_range=True):
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
        dimension = self.get_dimension(dimension)
        if dimension is None or (not data_range and not dimension_range):
            return (None, None)
        elif all(util.isfinite(v) for v in dimension.range) and dimension_range:
            return dimension.range
        elif data_range:
            if dimension in self.kdims+self.vdims:
                dim_vals = self.dimension_values(dimension.name)
                lower, upper = util.find_range(dim_vals)
            else:
                dname = dimension.name
                match_fn = lambda x: dname in x.kdims + x.vdims
                range_fn = lambda x: x.range(dname)
                ranges = self.traverse(range_fn, [match_fn])
                lower, upper = util.max_range(ranges)
        else:
            lower, upper = (np.NaN, np.NaN)
        if not dimension_range:
            return lower, upper
        return util.dimension_range(lower, upper, dimension.range, dimension.soft_range)
