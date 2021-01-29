"""
Provides Dimension objects for tracking the properties of a value,
axis or map dimension. Also supplies the Dimensioned abstract
baseclass for classes that accept Dimension values.
"""
from __future__ import unicode_literals

import weakref

from collections import defaultdict, Counter
from itertools import chain
from functools import partial

import param
import numpy as np

import holodata.util
from holodata.dimension import Dimensioned, ALIASES
from . import util
from .accessors import Opts
from .options import Store, Options, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import basestring, unicode


def param_aliases(d):
    """
    Called from __setstate__ in LabelledData in order to load
    old pickles with outdated parameter names.

    Warning: We want to keep pickle hacking to a minimum!
    """
    for old, new in ALIASES.items():
        old_param = '_%s_param_value' % old
        new_param = '_%s_param_value' % new
        if old_param in d:
            d[new_param] = d.pop(old_param)
    return d


class ViewableMixin(object):
    def __init__(self, id=None, plot_id=None):
        self._id = None
        self.id = id
        self._plot_id = plot_id or util.builtins.id(self)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, opts_id):
        """Handles tracking and cleanup of custom ids."""
        old_id = self._id
        self._id = opts_id
        if old_id is not None:
            cleanup_custom_options(old_id)
        if opts_id is not None and opts_id != old_id:
            if opts_id not in Store._weakrefs:
                Store._weakrefs[opts_id] = []
            ref = weakref.ref(self, partial(cleanup_custom_options, opts_id))
            Store._weakrefs[opts_id].append(ref)

    def update_plot_id_settings(self, settings, shared_data=True, link=True):
        if 'id' not in settings:
            settings['id'] = self.id

        if shared_data:
            if link:
                settings['plot_id'] = self._plot_id

        return settings

    @property
    def opts(self):
        return Opts(self)

    def __call__(self, options=None, **kwargs):
        self.param.warning(
            'Use of __call__ to set options will be deprecated '	
            'in the next major release (1.14.0). Use the equivalent .opts '
            'method instead.')

        if not kwargs and options is None:
            return self.opts.clear()

        return self.opts(options, **kwargs)

    def options(self, *args, **kwargs):
        """Applies simplified option definition returning a new object.

        Applies options on an object or nested group of objects in a
        flat format returning a new object with the options
        applied. If the options are to be set directly on the object a
        simple format may be used, e.g.:

            obj.options(cmap='viridis', show_title=False)

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.options('Image', cmap='viridis', show_title=False)

        or using:

            obj.options({'Image': dict(cmap='viridis', show_title=False)})

        Identical to the .opts method but returns a clone of the object
        by default.

        Args:
            *args: Sets of options to apply to object
                Supports a number of formats including lists of Options
                objects, a type[.group][.label] followed by a set of
                keyword options to apply and a dictionary indexed by
                type[.group][.label] specs.
            backend (optional): Backend to apply options to
                Defaults to current selected backend
            clone (bool, optional): Whether to clone object
                Options can be applied inplace with clone=False
            **kwargs: Keywords of options
                Set of options to apply to the object

        Returns:
            Returns the cloned object with the options applied
        """
        backend = kwargs.get('backend', None)
        clone = kwargs.pop('clone', True)

        if len(args) == 0 and len(kwargs)==0:
            options = None
        elif args and isinstance(args[0], basestring):
            options = {args[0]: kwargs}
        elif args and isinstance(args[0], list):
            if kwargs:
                raise ValueError('Please specify a list of option objects, or kwargs, but not both')
            options = args[0]
        elif args and [k for k in kwargs.keys() if k != 'backend']:
            raise ValueError("Options must be defined in one of two formats. "
                             "Either supply keywords defining the options for "
                             "the current object, e.g. obj.options(cmap='viridis'), "
                             "or explicitly define the type, e.g. "
                             "obj.options({'Image': {'cmap': 'viridis'}}). "
                             "Supplying both formats is not supported.")
        elif args and all(isinstance(el, dict) for el in args):
            if len(args) > 1:
                self.param.warning('Only a single dictionary can be passed '
                                   'as a positional argument. Only processing '
                                   'the first dictionary')
            options = [Options(spec, **kws) for spec,kws in args[0].items()]
        elif args:
            options = list(args)
        elif kwargs:
            options = {type(self).__name__: kwargs}

        from ..util import opts
        if options is None:
            expanded_backends = [(backend, {})]
        elif isinstance(options, list): # assuming a flat list of Options objects
            expanded_backends = opts._expand_by_backend(options, backend)
        else:
            expanded_backends = [(backend, opts._expand_options(options, backend))]

        obj = self
        for backend, expanded in expanded_backends:
            obj = obj.opts._dispatch_opts(expanded, backend=backend, clone=clone)
        return obj

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        Resolves the class hierarchy for the class rendering the
        object using any display hooks registered on Store.display
        hooks.  The output of all registered display_hooks is then
        combined and returned.
        """
        return Store.render(self)


class ViewableElement(Dimensioned, ViewableMixin):
    """
    A ViewableElement is a dimensioned datastructure that may be
    associated with a corresponding atomic visualization. An atomic
    visualization will display the data on a single set of axes
    (i.e. excludes multiple subplots that are displayed at once). The
    only new parameter introduced by ViewableElement is the title
    associated with the object for display.
    """

    __abstract = True
    _auxiliary_component = False

    group = param.String(default='ViewableElement', constant=True)

    def __init__(self, data, kdims=None, vdims=None, id=None, plot_id=None, **params):
        Dimensioned.__init__(self, data, kdims, vdims, **params)
        ViewableMixin.__init__(self, id=id, plot_id=plot_id)

    def __repr__(self):
        return PrettyPrinter.pprint(self)

    def __str__(self):
        return repr(self)

    def __unicode__(self):
        return unicode(PrettyPrinter.pprint(self))

    def __getstate__(self):
        "Ensures pickles save options applied to this objects."
        obj_dict = self.__dict__.copy()
        try:
            if Store.save_option_state and (obj_dict.get('_id', None) is not None):
                custom_key = '_custom_option_%d' % obj_dict['_id']
                if custom_key not in obj_dict:
                    obj_dict[custom_key] = {backend:s[obj_dict['_id']]
                                            for backend,s in Store._custom_options.items()
                                            if obj_dict['_id'] in s}
            else:
                obj_dict['_id'] = None
        except:
            self.param.warning("Could not pickle custom style information.")
        return obj_dict

    def __setstate__(self, d):
        "Restores options applied to this object."
        d = param_aliases(d)

        # Backwards compatibility for objects before id was made a property
        opts_id = d['_id'] if '_id' in d else d.pop('id', None)
        try:
            load_options = Store.load_counter_offset is not None
            if load_options:
                matches = [k for k in d if k.startswith('_custom_option')]
                for match in matches:
                    custom_id = int(match.split('_')[-1])+Store.load_counter_offset
                    if not isinstance(d[match], dict):
                        # Backward compatibility before multiple backends
                        backend_info = {'matplotlib':d[match]}
                    else:
                        backend_info = d[match]
                    for backend, info in  backend_info.items():
                        if backend not in Store._custom_options:
                            Store._custom_options[backend] = {}
                        Store._custom_options[backend][custom_id] = info
                    if backend_info:
                        if custom_id not in Store._weakrefs:
                            Store._weakrefs[custom_id] = []
                        ref = weakref.ref(self, partial(cleanup_custom_options, custom_id))
                        Store._weakrefs[opts_id].append(ref)
                    d.pop(match)

                if opts_id is not None:
                    opts_id += Store.load_counter_offset
        except:
            self.param.warning("Could not unpickle custom style information.")
        d['_id'] = opts_id
        self.__dict__.update(d)
        super(ViewableElement, self).__setstate__({})

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
        params = dict(self.param.get_param_values())
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.param.objects('existing')
            params = {k: v for k, v in params.items()
                      if k in new_params}
            if params.get('group') == self.param.objects('existing')['group'].default:
                params.pop('group')

        if data is not None:
            shared_data = False
        else:
            if shared_data:
                data = self.data

        settings = dict(params, **overrides)

        self.update_plot_id_settings(settings, shared_data=shared_data, link=link)

        # Apply name mangling for __ attribute
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        return clone_type(data, *args, **{k:v for k,v in settings.items()
                                          if k not in pos_args})


class ViewableTree(AttrTree, ViewableElement):
    """
    A ViewableTree is an AttrTree with Viewable objects as its leaf
    nodes. It combines the tree like data structure of a tree while
    extending it with the deep indexable properties of Dimensioned
    and LabelledData objects.
    """

    group = param.String(default='ViewableTree', constant=True)

    _deep_indexable = True

    def __init__(
            self, items=None, identifier=None, parent=None,
            id=None, plot_id=None, **kwargs
    ):
        if items and all(isinstance(item, Dimensioned) for item in items):
            items = self._process_items(items)
        params = {p: kwargs.pop(p) for p in list(self.param)+['id', 'plot_id'] if p in kwargs}

        AttrTree.__init__(self, items, identifier, parent, **kwargs)
        ViewableElement.__init__(self, self.data, **params)

    @classmethod
    def from_values(cls, vals):
        "Deprecated method to construct tree from list of objects"
        name = cls.__name__
        param.main.param.warning("%s.from_values is deprecated, the %s "
                                 "constructor may now be used directly."
                                 % (name, name))
        return cls(items=cls._process_items(vals))


    @classmethod
    def _process_items(cls, vals):
        "Processes list of items assigning unique paths to each."
        if type(vals) is cls:
            return vals.data
        elif not isinstance(vals, (list, tuple)):
            vals = [vals]
        items = []
        counts = defaultdict(lambda: 1)
        cls._unpack_paths(vals, items, counts)
        items = cls._deduplicate_items(items)
        return items


    def __setstate__(self, d):
        """
        Ensure that object does not try to reference its parent during
        unpickling.
        """
        parent = d.pop('parent', None)
        d['parent'] = None
        super(AttrTree, self).__setstate__(d)
        self.__dict__['parent'] = parent


    @classmethod
    def _deduplicate_items(cls, items):
        "Deduplicates assigned paths by incrementing numbering"
        counter = Counter([path[:i] for path, _ in items for i in range(1, len(path)+1)])
        if sum(counter.values()) == len(counter):
            return items

        new_items = []
        counts = defaultdict(lambda: 0)
        for i, (path, item) in enumerate(items):
            if counter[path] > 1:
                path = path + (util.int_to_roman(counts[path]+1),)
            else:
                inc = 1
                while counts[path]:
                    path = path[:-1] + (util.int_to_roman(counts[path]+inc),)
                    inc += 1
            new_items.append((path, item))
            counts[path] += 1
        return new_items


    @classmethod
    def _unpack_paths(cls, objs, items, counts):
        """
        Recursively unpacks lists and ViewableTree-like objects, accumulating
        into the supplied list of items.
        """
        if type(objs) is cls:
            objs = objs.items()
        for item in objs:
            path, obj = item if isinstance(item, tuple) else (None, item)
            if type(obj) is cls:
                cls._unpack_paths(obj, items, counts)
                continue
            new = path is None or len(path) == 1
            path = util.get_path(item) if new else path
            new_path = util.make_path_unique(path, counts, new)
            items.append((new_path, obj))


    @property
    def uniform(self):
        "Whether items in tree have uniform dimensions"
        from .traversal import uniform
        return uniform(self)


    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Concatenates values on all nodes with requested dimension.

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
        dimension = self.get_dimension(dimension, strict=True).name
        all_dims = self.traverse(lambda x: [d.name for d in x.dimensions()])
        if dimension in chain.from_iterable(all_dims):
            values = [el.dimension_values(dimension) for el in self
                      if dimension in el.dimensions(label=True)]
            vals = np.concatenate(values)
            return vals if expanded else holodata.util.unique_array(vals)
        else:
            return super(ViewableTree, self).dimension_values(
                dimension, expanded, flat)

    def __len__(self):
        return len(self.data)
