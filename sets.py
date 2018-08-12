import hashlib
import pathlib
import linecache
import pickle
import sys

from abstracts import Dataset, Field, SourceBase


class Source(SourceBase):

    def to_memory(self):
        return MemoryCacheSource(self)

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
            fields = [Field("raw", self)]
        return Dataset(
            fields=fields,
            size=size,
            return_as_tuple=return_as_tuple
        )


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


class MemoryCacheSource(Source):
    def __init__(self, parent, load_immediately=True):
        super(MemoryCacheSource, self).__init__(parent)
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


def _calc_args_hash(args):
    hs = 0
    for obj in args:
        if type(obj) == str:
            obj_hash = int(hashlib.md5(obj.encode('utf-8')).hexdigest(), 16)
            hs = (hs * 31 + obj_hash) % sys.maxsize
        elif type(obj) == int:
            hs = (hs * 31 + obj) % sys.maxsize
        else:
            raise ValueError(
                f"{obj} is not hashable.\nall arguments are needed to be hashable for caching."
            )
    return hs


class FileCacheSource(MemoryCacheSource):
    """
    dataをファイルにキャッシュする
    キャッシュファイル名はcache_group_nameとcache_argsから計算される

    """

    def __init__(self, parent, cache_group_name, *cache_args, cache_immediately=True, cache_dir=".tmp"):
        """
        キャッシュファイル名は
        `cache_group_name`_`HASH`
        となる。

        :param parent:
        :param cache_group_name: キャッシュファイルの固定プレフィックス
        :param cache_args: キャッシュファイル名にこれらのハッシュ値が使われる
        :param cache_immediately: 即座にデータをロード、キャッシュファイルを作成する
        :param cache_dir: キャッシュファイルが作られるディレクトリ
        """
        super(FileCacheSource, self).__init__(parent)
        self.data = None
        self.cache_group_name = cache_group_name
        self.cache_dir = pathlib.Path(cache_dir)
        hs = _calc_args_hash(cache_args)
        self.cache_file_path = self.cache_dir / (self.cache_group_name + "_" + str(hs))

        if cache_immediately:
            self.load()

    def clear_cache(self, remove_all=False):
        """
        キャッシュを削除する
        キャッシュファイル名が完全一致するファイルを削除する
        :param remove_all: 同一のcache_group_nameのすべてのキャッシュも削除する
        :return:
        """
        if remove_all:
            for p in self.cache_dir.glob(f"{self.cache_group_name}_*"):
                p.unlink()
        else:
            if self.cache_file_path.exists():
                self.cache_file_path.unlink()

    def _make_cache(self):
        if self.cache_file_path.exists():
            return
        if self.data is None:
            self.load(cache_if_not_yet=False)
        with self.cache_file_path.open("wb") as f:
            pickle.dump(self.data, f)
        return self.data

    def load(self, cache_if_not_yet=True):
        if self.cache_file_path.exists():
            with self.cache_file_path.open("rb") as f:
                self.data = pickle.load(f)
                self.parents = []
                return self.data
        else:
            if self.data is None:
                self.data = list(self.parents[0])
                self.parents = []
            if cache_if_not_yet:
                self._make_cache()
            return self.data


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
        return 1

    def __getitem__(self, item):
        if item != 0:
            raise IndexError()
        return self.path.open(encoding="utf-8")

    def __iter__(self):
        with self.path.open(encoding="utf-8") as f:
            yield f
