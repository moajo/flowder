from .abstracts import Field
from .source.file import TextFileSource, DirectorySource
from .source.base import ZipSource, Dataset


def file(path):
    return TextFileSource(path)


def directory(path):
    return DirectorySource(path)


def zip_source(*sources):
    if len(sources) == 1 and isinstance(sources[0], dict):
        d = sources[0]
        keys = list(d)
        ss = [d[k] for k in keys]

        def wrapper(args):
            return {
                k: v
                for k, v in zip(keys, args)
            }

        return ZipSource(*ss).map(wrapper)

    return ZipSource(*sources)


def create_dataset(*fields, size: int = None, return_as_tuple=False):
    assert all(isinstance(f, Field) for f in fields), "fields must be instance of Field. is argument correct?"
    if size is None:
        size_list = [len(f.target_source) for f in fields if f.target_source.has_length]
        assert len(size_list) > 0, "every fields has no size. least one field must has size."
        assert all(size_list[0] == it for it in size_list), "all size of field must be equals"
        size = size_list[0]
    assert isinstance(size, int)
    return Dataset(
        fields,
        size,
        return_as_tuple=return_as_tuple
    )

# def collect(base_source: SourceBase, target_key_source: SourceBase, target_source: SourceBase):
#     """
#
#     :param base_source: 元となるソース
#     :param target_key_source: base_sourceの値の並び替えたもの。重複した値は持てない。メモリ上にロードされる。
#     :param target_source: base_sourceと等しい値を持つindexをtarget_key_sourceから探し、そのインデックスでアクセスされるソース
#     :return:
#     """
#
#     key_index_map = {}
#     for i, key in enumerate(target_key_source):
#         assert key not in key_index_map
#         key_index_map[key] = i
#     return CollectSource(base_source, key_index_map, target_source)
