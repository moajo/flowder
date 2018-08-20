from flowder.source import Source


class ArraySource(Source):
    def __init__(self, contents):
        super(ArraySource, self).__init__()
        self.contents = contents

    def _calculate_size(self):
        return len(self.contents)

    def _getitem(self, item):
        return self.contents[item]

    def _iter(self):
        return iter(self.contents)


class FlatMap(Source):
    def __init__(self, parent, map):
        super(FlatMap, self).__init__(parent, has_length=False, random_access=False)
        self.map = map

    def _iter(self):
        for p in self.parent:
            yield from self._calculate_value(p)

    def _calculate_value(self, args):
        for n in self.map(args):
            yield n

    def _calculate_size(self):
        return sum(1 for _ in self)

# class CollectSource(SourceBase):
#     def __init__(self, base_source: SourceBase, key_index_map: dict, target_source: SourceBase):
#         """
#
#         :param base_source: 元となるソース
#         :param key_index_map: base_sourceの値に対応するtarget_sourceのindexを保持するdict
#         :param target_source: base_sourceと等しい値を持つindexをtarget_key_sourceから探し、そのインデックスでアクセスされるソース
#         """
#         super(CollectSource, self).__init__(base_source, target_source)
#         self.base_source = base_source
#         self.target_source = target_source
#         self.key_index_map = key_index_map
#
#     def calculate_size(self):
#         return len(self.base_source)
#
#     def __getitem__(self, item):
#         if isinstance(item, int):
#             key = self.base_source[item]
#             index = self.key_index_map[key]
#             return self.target_source[index]
#         else:
#             keys = self.base_source[item]
#             return [
#                 self.target_source[index]
#                 for index in (self.key_index_map[key] for key in keys)
#             ]
#
#     def __iter__(self):
#         for key in self.base_source:
#             index = self.key_index_map[key]
#             yield self.target_source[index]
