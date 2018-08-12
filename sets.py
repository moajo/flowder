import pathlib
import linecache

from abstracts import Dataset, Field


class Source:
    """
    データソース
    親があるならそれのみに依存。
    なければ何らかの外部データに依存する。
    （親があるけど、親以外のデータにも依存することは禁止）
    マージするときなど親は複数になる
    """

    def __init__(self, source_type, *parents):
        assert type(parents) is tuple
        self.parents: [Source] = parents or []
        self.source_type = source_type
        self.children = []
        self.size = None

    def create(self, *fields,
               return_raw_value_for_single_data=True,
               return_tuple_for_nameless_data=True
               ):
        """
        反復可能データセットを構築
        :return:
        """
        size = len(self)
        assert size is not None

        if len(fields) == 0:
            fields = [Field(self)]
        return Dataset(
            fields=fields,
            size=size,
            return_raw_value_for_single_data=return_raw_value_for_single_data,
            return_tuple_for_nameless_data=return_tuple_for_nameless_data,
        )

    def __len__(self):
        if self.size is None:
            self.size = self.calculate_size()
        return self.size

    def calculate_size(self):
        """
        データサイズを計算。createまでに呼ばれる
        :return:
        """
        raise NotImplementedError()

    def calculate_value(self, *args):
        """
        親セットの値から値（のイテレータ）を計算するステートレスな関数
        独立ソースの場合、呼ばれない。
        :param args:
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, item):
        """
        ランダムアクセスに対応する必要あり
        :param item:
        :return:
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        iterableである必要あり
        :return:
        """
        raise NotImplementedError()


class MapSource(Source):

    def __init__(self, transform, parent: Source):
        super(MapSource, self).__init__("map", parent)
        self.transform = transform

    def calculate_size(self):
        return self.parents[0].calculate_size()

    def calculate_value(self, arg):
        return self.transform(arg)

    def __getitem__(self, item):
        return self.calculate_value(
            self.parents[0][item]
        )

    def __iter__(self):
        for d in self.parents[0]:
            yield self.transform(d)


class ZipSource(Source):
    """
    ソースのサイズはすべて等しい必要がある。
    """

    def __init__(self, *parents):
        assert len(parents) != 0
        super(ZipSource, self).__init__("zip", *parents)

    def calculate_size(self):
        sizes = [p.calculate_size() for p in self.parents]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def calculate_value(self, arg):
        return arg

    def __getitem__(self, item):
        return self.calculate_value(
            *[p[item] for p in self.parents]
        )

    def __iter__(self):
        for d in zip(*self.parents):
            yield d


class StrSource(Source):
    """
    特定ファイルの各行を返すソース
    """

    def __init__(self, parent):
        assert type(parent) is TextFileSource
        super(StrSource, self).__init__("str", parent)
        self.parent = parent

    def split(self, delimiter=" "):
        return MapSource(lambda x: x.split(delimiter), self)

    def calculate_size(self):
        for f in self.parent:
            return sum(1 for _ in f)

    # def __getitem__(self, item):
    #     with self.parent_fileset.path.open() as f:
    #         for _ in range(item):
    #             f.readline()
    #         return f.readline()[:-1]  # remove tailing \n

    def calculate_value(self, file_source):
        for line in file_source:
            yield line[:-1]  # remove tailing \n

    def __getitem__(self, item):
        assert type(item) is int
        line = linecache.getline(str(self.parent.path), item + 1)
        return line
        # for f in self.parent:
        #     # line = linecache.getline('sample.txt', int(a))
        #     for line in self.calculate_value(f):
        #         yield line

    def __iter__(self):
        for f in self.parent:
            for line in self.calculate_value(f):
                yield line


class TextFileSource(Source):
    def __init__(self, path):
        super(TextFileSource, self).__init__("text_file")
        self.path = pathlib.Path(path)

    def lines(self):
        """
        str セットを返す
        :return:
        """
        return StrSource(self)

    def calculate_size(self):
        with self.path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    def __getitem__(self, item):
        if item != 0:
            raise IndexError()
        return self.path.open(encoding="utf-8")

    def __iter__(self):
        with self.path.open(encoding="utf-8") as f:
            yield f
