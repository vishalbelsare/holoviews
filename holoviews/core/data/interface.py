import param

from holodata.interface import GriddedInterface


class ImageInterface(GriddedInterface):

    rtol = param.Number(default=10e-4, constant=True, doc="""
        The tolerance used to enforce regular sampling for regular, gridded
        data where regular sampling is expected. Expressed as the maximal
        allowable sampling difference between sample locations.""")

    time_unit = param.String(default="us", constant=True, doc="""
        Determines the unit of time densities are defined relative to
        when one or both axes are datetime types""")

    kind = "image"
