import hashlib
import linecache
import pathlib
import pickle
import sys
from queue import Queue
from typing import Iterable

from moajo_tool.utils import measure_time
from tqdm import tqdm

from flowder.abstracts import Field, SourceBase
import pandas as pd
from PIL import Image


def cache_value(cache_arg_index=0):
    def decorator(func):
        cache = {}

        def wrapper(*args):
            arg = args[cache_arg_index]
            if arg not in cache:
                cache[arg] = func(*args)
            return cache[arg]

        return wrapper

    return decorator


class MapDummy:  # TODO equalsを実装してfilterに
    """
    to make map source easily
    """

    def __init__(self, source, root):
        self.source = source
        self.root = root
        assert isinstance(source, SourceBase)
        assert not isinstance(root, MapDummy)

    def __getitem__(self, item):
        return MapSource(lambda x: x[item], parent=self.source)

    def __getattr__(self, item):
        return MapSource(lambda x: getattr(x, item), parent=self.source)

    def __iter__(self):
        return iter(self.source)

    # def __eq__(self, other):


class Source(SourceBase):

    def on_memory(self):
        return MemoryCacheSource(self)

    def create(self, *fields, return_as_tuple=False):
        """
        create Dataset
        if fields is not given, use "raw" fields as default.
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
        """
        short-hand for Map
        ex)
        mapped = source.item[0]
        # mapped = MapSource(lambda x:x[0], source)
        mapped = source.item.hoge
        # mapped = MapSource(lambda x:x.hoge, source)
        :return:
        """
        return MapDummy(self, self)

    def map(self, transform):
        return MapSource(transform, self)

    def filter(self, pred):
        return FilterSource(pred, self)


class MapSource(Source):
    """
    this Source iterate value mapped by transform from parent source
    """

    def __init__(self, transform, parent: Source):
        super(MapSource, self).__init__(parent)
        assert isinstance(parent, SourceBase)
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
        for d in self.parent:
            yield self.transform(d)


# class EnumerableSource(Source):
#     def __init__(self, parent):
#         super(EnumerableSource, self).__init__(parent)
#
#     def calculate_size(self):
#         return len(self.parent)
#
#     def calculate_value(self, args):
#         yield args
#
#     def __getitem__(self, item):
#         assert isinstance(item, int)
#         return item, self.parent[item]
#


class FilterSource(Source):
    """
    this Source iterate value filterd by pred from parent source
    """

    def __init__(self, pred, parent: Source):
        super(FilterSource, self).__init__(parent)
        self.pred = pred

    def calculate_size(self):
        return sum(1 for _ in self)

    def calculate_value(self, arg):
        yield arg

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise NotImplementedError()

        for d in self:
            if item == 0:
                return d
            item = -1
        raise IndexError("index out of range")

    def __iter__(self):
        for d in self.parent:
            if self.pred(d):
                yield d


class ZipSource(Source):
    """
    zip to tuple
    all parents must has same size
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
        super(FileCacheSource, self).__init__(parent, load_immediately=False)
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
        """
        データが読み込み済みならそれを返す。
        読み込み済みでなければ、キャッシュファイルが存在すればそこからロードする。
        キャッシュファイルが無ければ、データを読み込む。
        その後、cache_if_not_yet=Trueなら、キャッシュファイルを作る。
        :param cache_if_not_yet:
        :return:
        """
        if self.data is not None:
            return self.data
        if self.cache_file_path.exists():
            print("[flowder.FileCacheSource]load from cache file...")
            with self.cache_file_path.open("rb") as f:
                self.data = pickle.load(f)
                self.parents = []
                return self.data
        else:
            if self.data is None:
                self.data = list(tqdm(self.parent, desc="[flowder.FileCacheSource]loading to memory..."))
                self.parents = []
            if cache_if_not_yet:
                print("[flowder.FileCacheSource]create cache file...")
                self._make_cache()
            return self.data


class StrSource(Source):
    """
    特定ファイルの各行を返すソース
    """

    def __init__(self, parent):
        assert type(parent) is TextFileSource
        super(StrSource, self).__init__(parent)

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
    """
    iterate a single value which opened file
    """

    def __init__(self, path):
        super(TextFileSource, self).__init__()
        self.path = pathlib.Path(path)

    def lines(self):
        return StrSource(self)

    def csv(self, **kwargs):
        return CSVSource(self.path, **kwargs)

    def calculate_size(self):
        return 1

    def __getitem__(self, item):
        if item != 0:
            raise IndexError()
        return self.path.open(encoding="utf-8")

    def __iter__(self):
        with self.path.open(encoding="utf-8") as f:
            yield f


class HookSource(Source):
    def __init__(self, parent, getitem_callback, iter_callback):
        super(HookSource, self).__init__(parent)
        self.getitem_callback = getitem_callback
        self.iter_callback = iter_callback

    def calculate_size(self):
        return len(self.parents[0])

    def __getitem__(self, item):
        self.getitem_callback(item)
        return self.parents[0][item]

    def __iter__(self):
        self.iter_callback()
        for v in self.parents[0]:
            yield v


class DirectorySource(Source):
    def __init__(self, path):
        super(DirectorySource, self).__init__()
        self.path = pathlib.Path(path)

    def open(self, **kwargs):
        return MapSource(lambda p: open(p, **kwargs), self)

    def image(self):
        return ImageSource(self)

    def calculate_size(self):
        assert self.path.exists()
        return sum(1 for _ in self.path.iterdir())

    def __getitem__(self, item):
        return list(self.path.iterdir())[item]

    def __iter__(self):
        return self.path.iterdir()


class ArraySource(Source):
    def __init__(self, contents):
        super(ArraySource, self).__init__()
        self.contents = contents

    def calculate_size(self):
        return len(self.contents)

    def __getitem__(self, item):
        return self.contents[item]

    def __iter__(self):
        return iter(self.contents)


class ImageSource(Source):
    def __init__(self, path_source: SourceBase):
        super(ImageSource, self).__init__(path_source)

    def calculate_size(self):
        return len(self.parent)

    def calculate_value(self, path):
        yield Image.open(path)

    def __getitem__(self, item):
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

    def __iter__(self):
        for p in self.parent:
            yield Image.open(p)


class CSVSource(Source):
    def __init__(self, path_or_buffer, **kwargs):
        super(CSVSource, self).__init__()
        self.data_frame = pd.read_csv(path_or_buffer, **kwargs)

    def calculate_size(self):
        return len(self.data_frame)

    def __getitem__(self, item):
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

    def __iter__(self):
        return (
            (
                index,
                {key: data[key] for key in data.keys()}
            )
            for index, data in self.data_frame.iterrows())


class CollectSource(SourceBase):
    def __init__(self, base_source: SourceBase, key_index_map: dict, target_source: SourceBase):
        """

        :param base_source: 元となるソース
        :param key_index_map: base_sourceの値に対応するtarget_sourceのindexを保持するdict
        :param target_source: base_sourceと等しい値を持つindexをtarget_key_sourceから探し、そのインデックスでアクセスされるソース
        """
        super(CollectSource, self).__init__(base_source, target_source)
        self.base_source = base_source
        self.target_source = target_source
        self.key_index_map = key_index_map

    def calculate_size(self):
        return len(self.base_source)

    def __getitem__(self, item):
        if isinstance(item, int):
            key = self.base_source[item]
            index = self.key_index_map[key]
            return self.target_source[index]
        else:
            keys = self.base_source[item]
            return [
                self.target_source[index]
                for index in (self.key_index_map[key] for key in keys)
            ]

    def __iter__(self):
        for key in self.base_source:
            index = self.key_index_map[key]
            yield self.target_source[index]


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

        if not self.dataset.is_independent():  # 依存ソース
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


# class Example:
#     def __init__(self, data_dict):
#         self._keys = []
#         for name, v in data_dict.items():
#             setattr(self, name, v)
#             self._keys.append(name)


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
    # return Example(OrderedDict(zip(field_names, vs)))
    return {k: v for k, v in zip(field_names, vs)}


class Dataset(Source):
    """
    fieldsとsetをつなぐSource
    複数のFieldからのデータイテレーションを最適化する
    fieldsのプリプロセスを最適化して行う

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
        self._memory_cache = None  # TODO deprecate

    def load_to_memory(self):
        if self._memory_cache is not None:
            return
        self._memory_cache = list(tqdm(self, desc="loading to memory..."))

    @property
    def item(self):
        return MapDummy(self, self)

    def is_independent(self):
        return True

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
            for f in tqdm(self.fields, desc="preprocess closing"):
                f.finish_preprocess_data_feed()
            return
        leaf_iterators = create_cache_iter_tree(fields)
        for i in tqdm(range(self.size), desc=f"preprocessing {len(fields)} fields"):
            for f, leaf in zip(fields, leaf_iterators):
                f.processing_data_feed(leaf.next(i))
        for f in tqdm(fields, desc="preprocess closing"):
            f.finish_preprocess_data_feed()

    def __iter__(self):  # TODO preprocess未処理時にエラー?
        if self._memory_cache is not None:
            return iter(self._memory_cache)
        leaf_iterators = create_cache_iter_tree(self.fields)
        for i in range(self.size):
            vs = [
                f.calculate_value(leaf.next(i))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, leaf_iterators)
            ]
            yield create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def __getitem__(self, item):  # TODO fieldまたいで値のキャッシュ,slice item
        if self._memory_cache is not None:
            return self._memory_cache[item]
        # leaf_iterators = create_cache_iter_tree(self.fields)  # todo to be instance field?
        # vs = [
        #     f.calculate_value(leaf[item])  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
        #     for f, leaf in zip(self.fields, leaf_iterators)
        # ]
        vs = [f[item] for f in self.fields]
        return create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def __len__(self):
        return self.size

    def calculate_size(self):
        return self.size
