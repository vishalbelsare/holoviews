from holodata.interface import Interface, Driver

default_datatype = "dataframe"


def concat(datasets, datatype=None):
    """Concatenates collection of datasets along NdMapping dimensions.

    Concatenates multiple datasets wrapped in an NdMapping type along
    all of its dimensions. Before concatenation all datasets are cast
    to the same datatype, which may be explicitly defined or
    implicitly derived from the first datatype that is
    encountered. For columnar data concatenation adds the columns for
    the dimensions being concatenated along and then concatenates all
    the old and new columns. For gridded data a new axis is created
    for each dimension being concatenated along and then
    hierarchically concatenates along each dimension.

    Args:
        datasets: NdMapping of Datasets to concatenate
        datatype: Datatype to cast data to before concatenation

    Returns:
        Concatenated dataset
    """
    return Driver.concatenate(datasets, datatype)
