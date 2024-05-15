from .data_access import data_accessor
from .dataset import DataKeys, NCT2013Dataset
import typing as tp


def get_other_metadata_by_core_ids(core_ids, metadata_keys: tp.List[DataKeys]):
    """
    Given the iterable of core_ids, return the metadata values for the specified metadata_keys.

    Args:
        core_ids: Iterable of core_ids.
        metadata_keys: List of metadata keys to return.

    Returns:
        List of arrays, where each array corresponds to the metadata values for the specified metadata_keys.
    """

    metadata_keys = [DataKeys(key) for key in metadata_keys]
    assert all(key.is_metadata() for key in metadata_keys)

    table = data_accessor.get_metadata_table()
    table = table[table.core_id.isin(core_ids)]

    outs = []
    for key in metadata_keys:
        column = table[key]
        outs.append(column.values)
    
    return outs