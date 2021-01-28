import sys
import textwrap

import param
from param.parameterized import bothmethod


class BasePrettyPrinter(param.Parameterized):
    """
    The PrettyPrinter used to print all HoloView objects via the
    pprint method.
    """
    tab = '   '

    type_formatter = ':{type}'

    @bothmethod
    def pprint(cls_or_slf, node):
        reprval = cls_or_slf.serialize(cls_or_slf.recurse(node))
        if sys.version_info.major == 2:
            return str(reprval.encode("utf8"))
        else:
            return str(reprval)

    @bothmethod
    def serialize(cls_or_slf, lines):
        accumulator = []
        for level, line in lines:
            accumulator.append((level * cls_or_slf.tab) + line)
        return "\n".join(accumulator)

    @bothmethod
    def shift(cls_or_slf, lines, shift=0):
        return [(lvl + shift, line) for (lvl, line) in lines]

    @bothmethod
    def padding(cls_or_slf, items):
        return max(len(p) for p in items) if len(items) > 1 else len(items[0])

    @bothmethod
    def component_type(cls_or_slf, node):
        "Return the type.group.label dotted information"
        if node is None: return ''
        return cls_or_slf.type_formatter.format(type=str(type(node).__name__))

    @bothmethod
    def recurse(cls_or_slf, node, attrpath=None, attrpaths=[], siblings=[], level=0,
                value_dims=True):
        """
        Recursive function that builds up an ASCII tree given an
        AttrTree node.
        """
        level, lines = cls_or_slf.node_info(node, attrpath, attrpaths, siblings, level,
                                            value_dims)
        attrpaths = ['.'.join(k) for k in node.keys()] if hasattr(node,
                                                                  'children') else []
        siblings = [node.get(child) for child in attrpaths]
        for attrpath in attrpaths:
            lines += cls_or_slf.recurse(node.get(attrpath), attrpath,
                                        attrpaths=attrpaths,
                                        siblings=siblings, level=level + 1,
                                        value_dims=value_dims)
        return lines

    @bothmethod
    def node_info(cls_or_slf, node, attrpath, attrpaths, siblings, level, value_dims):
        """
        Given a node, return relevant information.
        """
        raise NotImplementedError

    @bothmethod
    def format_options(cls_or_slf, opts, wrap_count=100):
        opt_repr = str(opts)
        cls_name = type(opts).__name__
        indent = ' ' * (len(cls_name) + 1)
        wrapper = textwrap.TextWrapper(width=wrap_count, subsequent_indent=indent)
        return [' | ' + l for l in wrapper.wrap(opt_repr)]


class SimplePrettyPrinter(BasePrettyPrinter):
    """
    For HoloData, supports elements and ndmappings without options
    """
    @bothmethod
    def node_info(cls_or_slf, node, attrpath, attrpaths, siblings, level, value_dims):
        """
        Given a node, return relevant information.
        """
        if getattr(node, '_deep_indexable', False):
            (lvl, lines) = cls_or_slf.ndmapping_info(node, siblings, level, value_dims)
        elif hasattr(node, 'unit_format'):
            (lvl, lines) = level, [(level, repr(node))]
        else:
            (lvl, lines) = cls_or_slf.element_info(node, siblings, level, value_dims)

        # The attribute indexing path acts as a prefix (if applicable)
        if attrpath is not None:
            padding = cls_or_slf.padding(attrpaths)
            (fst_lvl, fst_line) = lines[0]
            line = '.'+attrpath.ljust(padding) +' ' + fst_line
            lines[0] = (fst_lvl, line)

        return (lvl, lines)

    @bothmethod
    def element_info(cls_or_slf, node, siblings, level, value_dims):
        """
        Return the information summary for an Element. This consists
        of the dotted name followed by an value dimension names.
        """
        info = cls_or_slf.component_type(node)
        if len(node.kdims) >= 1:
            info += cls_or_slf.tab + '[%s]' % ','.join(d.name for d in node.kdims)
        if value_dims and len(node.vdims) >= 1:
            info += cls_or_slf.tab + '(%s)' % ','.join(d.name for d in node.vdims)
        return level, [(level, info)]

    @bothmethod
    def ndmapping_info(cls_or_slf, node, siblings, level, value_dims):
        key_dim_info = '[%s]' % ','.join(d.name for d in node.kdims)
        first_line = cls_or_slf.component_type(node) + cls_or_slf.tab + key_dim_info
        lines = [(level, first_line)]

        # opts = cls_or_slf.option_info(node)
        # if cls_or_slf.show_options and opts and opts.kwargs:
        #     lines += [(level, l) for l in cls_or_slf.format_options(opts)]

        if len(node.data) == 0:
            return level, lines
        # .last has different semantics for GridSpace
        last = list(node.data.values())[-1]
        if last is not None and getattr(last, '_deep_indexable'):
            level, additional_lines = cls_or_slf.ndmapping_info(last, [], level, value_dims)
        else:
            additional_lines = cls_or_slf.recurse(last, level=level, value_dims=value_dims)
        lines += cls_or_slf.shift(additional_lines, 1)
        return level, lines
