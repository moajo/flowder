from flowder.abstracts import Field, SourceBase
from flowder.sources import TextFileSource, ZipSource, Dataset, DirectorySource, CollectSource


def file(path):
    return TextFileSource(path)


def directory(path):
    return DirectorySource(path)


def zip_source(*sources):
    return ZipSource(*sources)


def create_dataset(size, *fields, return_as_tuple=False):
    assert isinstance(size, int)
    assert all(isinstance(f, Field) for f in fields)
    return Dataset(
        fields,
        size,
        return_as_tuple=return_as_tuple
    )


def collect(base_source: SourceBase, target_key_source: SourceBase, target_source: SourceBase):
    """

    :param base_source: 元となるソース
    :param target_key_source: base_sourceの値の並び替えたもの。重複した値は持てない。メモリ上にロードされる。
    :param target_source: base_sourceと等しい値を持つindexをtarget_key_sourceから探し、そのインデックスでアクセスされるソース
    :return:
    """

    key_index_map = {}
    for i, key in enumerate(target_key_source):
        assert key not in key_index_map
        key_index_map[key] = i
    return CollectSource(base_source, key_index_map, target_source)