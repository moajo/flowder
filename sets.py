import pathlib
import linecache

from abstracts import Dataset, Field, SourceBase


class Source(SourceBase):

    def to_memory(self):
        return MemorySource(self)

    def create(self, *fields,
               return_as_tuple=False,
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
            return_as_tuple=return_as_tuple
        )

    def __len__(self):
        if self.size is None:
            self.size = self.calculate_size()
        return self.size


class MapSource(Source):

    def __init__(self, transform, parent: Source):
        super(MapSource, self).__init__(parent)
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
        super(ZipSource, self).__init__(*parents)

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


class MemorySource(Source):
    def __init__(self, parent, load_immediately=True):
        super(MemorySource, self).__init__(parent)
        self.data = None
        if load_immediately:
            self.load()

    def load(self):
        if self.data is None:
            self.data = list(self.parents[0])
            self.parents = []
        return self.data

    def calculate_size(self):
        return len(self.load())

    def __getitem__(self, item):
        return self.load()[item]

    def __iter__(self):
        return iter(self.load())


class StrSource(Source):
    """
    特定ファイルの各行を返すソース
    """

    def __init__(self, parent):
        assert type(parent) is TextFileSource
        super(StrSource, self).__init__(parent)
        self.parent = parent

    def split(self, delimiter=" "):
        return MapSource(lambda x: x.split(delimiter), self)

    def calculate_size(self):
        for f in self.parent:
            return sum(1 for _ in f)

    def calculate_value(self, file_source):
        for line in file_source:
            yield line[:-1]  # remove tailing \n

    def __getitem__(self, item):
        assert type(item) is int
        return linecache.getline(str(self.parent.path), item + 1)

    def __iter__(self):
        for f in self.parent:
            for line in self.calculate_value(f):
                yield line


class TextFileSource(Source):
    def __init__(self, path):
        super(TextFileSource, self).__init__()
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
