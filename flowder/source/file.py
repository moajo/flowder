import linecache

from flowder import Source, MapSource, SourceBase
import pathlib
import pandas as pd
from PIL import Image


class StrSource(Source):
    """
    特定ファイルの各行を返すソース
    """

    def __init__(self, path):
        super(StrSource, self).__init__()
        self.path = pathlib.Path(path)
        assert self.path.exists()

    def split(self, delimiter=" "):
        return MapSource(lambda x: x.split(delimiter), self)

    def _calculate_size(self):
        with self.path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _getitem(self, item):
        assert type(item) is int
        return linecache.getline(str(self.path), item + 1)[:-1]

    def _iter(self):
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                yield line[:-1]


class TextFileSource(Source):
    """
    iterate a single value which opened file
    """

    def __init__(self, path):
        super(TextFileSource, self).__init__()
        self.path = pathlib.Path(path)

    def lines(self):
        return StrSource(self.path)

    def csv(self, **kwargs):
        return CSVSource(self.path, **kwargs)

    def _calculate_size(self):
        return 1

    def _getitem(self, item):
        if item != 0:
            raise IndexError()
        return self.path.open(encoding="utf-8")

    def _iter(self):
        with self.path.open(encoding="utf-8") as f:
            yield f


class CSVSource(Source):
    def __init__(self, path_or_buffer, **kwargs):
        super(CSVSource, self).__init__()
        self.data_frame = pd.read_csv(path_or_buffer, **kwargs)

    def _calculate_size(self):
        return len(self.data_frame)

    def _getitem(self, item):
        if isinstance(item, int):
            v = self.data_frame[item:item + 1]
            index, data = next(v.iterrows())
            return index, {key: data[key] for key in v.keys()}
        v = self.data_frame[item]
        return [
            (
                index,
                {key: data[key] for key in data.keys()}
            )
            for index, data in v.iterrows()]

    def _iter(self):
        return (
            (
                index,
                {key: data[key] for key in data.keys()}
            )
            for index, data in self.data_frame.iterrows())


class DirectorySource(Source):
    def __init__(self, path):
        super(DirectorySource, self).__init__()
        self.path = pathlib.Path(path)

    def open(self, **kwargs):
        return MapSource(lambda p: open(p, **kwargs), self)

    def image(self):
        return ImageSource(self)

    def _calculate_size(self):
        assert self.path.exists()
        return sum(1 for _ in self.path.iterdir())

    def _getitem(self, item):
        return list(self.path.iterdir())[item]

    def _iter(self):
        return self.path.iterdir()


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


class ImageSource(Source):
    def __init__(self, path_source: SourceBase):
        super(ImageSource, self).__init__(path_source)

    def calculate_size(self):
        return len(self.parent)

    def _calculate_value(self, path):
        yield Image.open(path)

    def _getitem(self, item):
        p = self.parent[item]
        if isinstance(item, int):
            if isinstance(p, str):
                p = pathlib.Path(p)

            assert isinstance(p, pathlib.Path)
            return Image.open(p)
        else:
            ps = p
            ps = [
                pathlib.Path(p) if isinstance(p, str) else p
                for p in ps
            ]
            assert all(isinstance(p, pathlib.Path) for p in ps)

            return [Image.open(p) for p in ps]

    def _iter(self):
        for p in self.parent:
            yield Image.open(p)
