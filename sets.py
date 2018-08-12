import hashlib
import linecache
import pathlib
import pickle
import sys
from collections import OrderedDict
from queue import Queue
from typing import Iterable

from moajo_tool.utils import measure_time
from tqdm import tqdm

from abstracts import Field, SourceBase


def cache_value():
    def decorator(func):
        cache = {}

        def wrapper(arg):
            if arg not in cache:
                cache[arg] = func(arg)
            return cache[arg]

        return wrapper

    return decorator


class MapDummy:
    def __init__(self, source):
        self.source = source

    def __getitem__(self, item):
        return MapSource(lambda x: x[item], parent=self.source)

    def __getattr__(self, item):
        return MapSource(lambda x: getattr(x, item), parent=self.source)


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

    @property
    def item(self):
        return MapDummy(self)


class MapSource(Source):

    def __init__(self, transform, parent: Source):
        super(MapSource, self).__init__(parent)
        self.transform = transform

    def calculate_size(self):
        return self.parents[0].calculate_size()

    def calculate_value(self, arg):
        yield self.transform(arg)

    def __getitem__(self, item):
        return self.transform(
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
        yield arg

    def __getitem__(self, item):
        return [p[item] for p in self.parents]

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


class CachedIterator:
    """
    依存関係に応じて値の計算をキャッシュするIterator
    反復とランダムアクセスができる
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.parents = []
        self.children = []

        self.last_i = None
        self.value = None
        self.iter = None  # 独立なら単にソースのイテレータ。依存なら親データから作成した自分のデータのイテレータ
        self.reset()

    def next(self, i):
        """
        異なるiで呼ばれるごとに、値を更新して返す。
        前回と同じiで呼ばれたら前回のキャッシュ値をそのまま返す
        値の更新は独立ソースと依存ソースで異なる。

        # 独立の場合：
        datasetのnextを呼ぶ

        # 依存の場合
        親でーたセットのnext値のタプルから、calculate_valueする

        :param i:
        :return:
        """

        if self.last_i == i:
            return self.value
        self.last_i = i

        if not self.is_independent:  # 依存ソース
            if self.iter is None:
                for p in self.parents:
                    p.next(i)
                self.iter = self.dataset.calculate_value(*[p.value for p in self.parents])
            try:
                v = next(self.iter)
                self.value = v
            except StopIteration:
                for p in self.parents:
                    p.next(i)
                self.iter = self.dataset.calculate_value(*[p.value for p in self.parents])
                v = next(self.iter)
                self.value = v
        else:
            if self.iter is None:
                self.iter = iter(self.dataset)
            self.value = next(self.iter)
        return self.value

    @cache_value()
    def __getitem__(self, item):
        return self.dataset[item]  # TODO効率的な実装

    @property
    def is_independent(self):
        return len(self.parents) == 0

    def reset(self):
        if not self.is_independent:
            for p in self.parents:
                p.reset()
        self.last_i = None
        self.iter = None
        self.value = None


def create_cache_iter_tree(fields):
    cached_iters = {}
    leafs = []
    for f in fields:
        ts = f.target_source
        ts_id = id(ts)
        if ts_id not in cached_iters:
            cached_iters[ts_id] = CachedIterator(ts)
        cit = cached_iters[ts_id]
        leafs.append(cit)

        queue = Queue()
        for parent in ts.parents:
            queue.put_nowait((cit, parent))

        while queue.qsize() != 0:
            child_cached_iter, ts = queue.get_nowait()
            ts_id = id(ts)
            if ts_id not in cached_iters:  # 新規ノード
                cached_iters[ts_id] = CachedIterator(ts)
                cit = cached_iters[ts_id]
                for parent in ts.parents:
                    queue.put_nowait((cit, parent))

            cit = cached_iters[ts_id]
            cit.children.append(child_cached_iter)
            child_cached_iter.parents.append(cit)
    return leafs


class Example:
    def __init__(self, data_dict):
        self._keys = []
        for name, v in data_dict.items():
            setattr(self, name, v)
            self._keys.append(name)


def create_example(field_names, vs, return_as_tuple=True):
    """

    :param field_names: 属性名のリスト
    :param vs: 値のリスト
    :param return_as_tuple: Exampleにせずtupleのまま返すかどうか
    :return:
    """
    assert len(field_names) == len(vs)
    if return_as_tuple:
        if len(vs) == 1:
            return vs[0]
        return tuple(vs)
    return Example(OrderedDict(zip(field_names, vs)))


class Dataset(SourceBase):
    """
    fieldsとsetをつなぐ。torchのDatasetを継承。
    機能は
    - fieldsの前処理の制御
    - 全体反復
    - random access
    """

    def __init__(self, fields: Iterable, size: int, return_as_tuple=False):
        """
        note: fieldsの各要素はnameが未設定の場合、このタイミングで自動的に設定されます
        :param fields:
        :param size:
        :param return_as_tuple:
        """
        super(Dataset, self).__init__()
        self.fields = list(fields)
        self.size = size
        self._return_as_tuple = return_as_tuple

        # auto setting name to fields if not set.
        nameless_count = 1
        for f in self.fields:
            if f.name is None:
                f.name = f"attr{nameless_count}"
                nameless_count += 1
        self._memory_cache = None

    def load_to_memory(self):
        if self._memory_cache is not None:
            return
        self._memory_cache = list(self)

    @property
    def item(self):
        return MapDummy(self)

    @measure_time()
    def preprocess(self):
        """
        全fieldsのプリプロセスを実行
        :return:
        """

        fields = [
            f for f in
            tqdm(self.fields, desc="preprocess initializing")
            if f.start_preprocess_data_feed() is not False
        ]
        if len(fields) == 0:
            print("preprocess is not needed for any fields")
            return
        leaf_iterators = create_cache_iter_tree(fields)
        for i in tqdm(range(self.size), desc=f"preprocessing {len(fields)} fields"):
            for f, leaf in zip(fields, leaf_iterators):
                f.processing_data_feed(leaf.next(i))
        for f in tqdm(fields, desc="preprocess closing"):
            f.finish_preprocess_data_feed()

    def __iter__(self):
        if self._memory_cache is not None:
            return iter(self._memory_cache)
        leaf_iterators = create_cache_iter_tree(self.fields)
        for i in range(self.size):
            vs = [
                f.calculate_value(leaf.next(i))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, leaf_iterators)
            ]
            yield create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def __getitem__(self, item):  # TODO fieldまたいで値のキャッシュ
        if self._memory_cache is not None:
            return self._memory_cache[item]
        vs = [f[item] for f in self.fields]
        return create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def __len__(self):
        return self.size

    def calculate_size(self):
        return self.size
