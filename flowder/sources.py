import hashlib
import pathlib
import pickle
import sys
from queue import Queue
from typing import Iterable

from tqdm import tqdm
import inspect

from flowder.abstracts import Field, SourceBase


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


class MapDummy(SourceBase):
    """
    to make map source easily
    """

    def __init__(self, source, transform=None):
        super().__init__(source)
        self.source = source
        self.transform = transform or (lambda x: x)
        assert isinstance(source, SourceBase)

    def map(self, transform):
        def merged_transform(item):
            item = self.transform(item)
            return transform(item)

        return MapSource(merged_transform, parent=self.source)

    def filter(self, pred):
        return FilterSource(pred, MapSource(lambda x: self.transform(x), parent=self.source))

    def reduce(self):
        return MapSource(lambda x: self.transform(x), parent=self.source.reduce())

    def __getitem__(self, item):
        return MapDummy(self.source, lambda x: self.transform(x)[item])
        # return MapSource(lambda x: x[item], parent=self.source)

    def __getattr__(self, item):
        return MapDummy(self.source, lambda x: getattr(self.transform(x), item))
        # return MapSource(lambda x: getattr(x, item), parent=self.source)

    def __iter__(self):
        return iter(self.reduce())

    def __eq__(self, other):  # ==
        return FilterSource(lambda x: self.transform(x) == other, self.source)

    def __lt__(self, other):  # <
        return FilterSource(lambda x: self.transform(x) < other, self.source)

    def __le__(self, other):  # <=
        return FilterSource(lambda x: self.transform(x) <= other, self.source)

    def __gt__(self, other):  # >
        return FilterSource(lambda x: self.transform(x) > other, self.source)

    def __ge__(self, other):  # >=
        return FilterSource(lambda x: self.transform(x) >= other, self.source)

    def __ne__(self, other):  # !=
        return FilterSource(lambda x: self.transform(x) != other, self.source)


class Source(SourceBase):

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
        return MapDummy(self)

    def map(self, transform):
        return MapSource(transform, self)

    def filter(self, pred):
        return FilterSource(pred, self)

    def file_cache(self, cache_group_name, *cache_args, cache_dir=".tmp", caller_file_name=None,
                   auto_load=True,
                   show_progress_onload=True):
        """
        データのファイルキャッシュを作成します。
        キャッシュファイルは以下の要素から計算され、すべてが一致するものがキャッシュとして認識されます。
        - cache_group_name
        - cache_dir
        - caller_file_name
        - cache_argsのハッシュ

        :param cache_group_name: キャッシュの識別子
        :param cache_args: データが依存するパラメータ
        :param cache_dir: キャッシュの保存先
        :param caller_file_name: 呼び出しファイル名。(default: この関数を呼び出したファイル名)
        :param auto_load: iter時に自動的にload()する
        :param show_progress_onload: load時に進捗表示をするかどうか
        :return:
        """
        if caller_file_name is None:
            p = pathlib.Path(inspect.currentframe().f_back.f_code.co_filename)
            caller_file_name = p.name[:-len(p.suffix)]
        return FileCacheSource(
            self,
            cache_group_name,
            *cache_args,
            cache_dir=cache_dir,
            caller_file_name=caller_file_name,
            auto_load=auto_load,
            show_progress_onload=show_progress_onload,
        )


class WrapperSource(Source):
    def __init__(self, parent, has_length=True, random_access=True, auto_load=False, show_progress_onload=False):
        super(WrapperSource, self).__init__(
            parent,
            has_length=has_length,
            random_access=random_access,
            auto_load=auto_load,
            show_progress_onload=show_progress_onload
        )

    def _calculate_size(self):
        return self.parent.calculate_size()

    def _calculate_value(self, args):
        return self.parent._calculate_value(args)

    def _getitem(self, item):
        return self.parent[item]

    def _iter(self):
        return iter(self.parent)


class MapSource(Source):
    """
    this Source iterate value mapped by transform from parent source
    """

    def __init__(self, transform, parent: Source):
        assert isinstance(parent, SourceBase)
        super(MapSource, self).__init__(
            parent,
            has_length=parent.has_length,
            random_access=parent.random_access,
        )
        self.transform = transform

    def _calculate_size(self):
        return self.parent.calculate_size()

    def _calculate_value(self, arg):
        yield self.transform(arg)

    def _getitem(self, item):
        return self.transform(
            self.parent[item]
        )

    def _iter(self):
        return (self.transform(d) for d in self.parent)


class ZipSource(Source):
    """
    zip to tuple
    all parents must has same size
    """

    def __init__(self, *parents):
        assert len(parents) != 0
        super(ZipSource, self).__init__(
            *parents,
            has_length=parents[0].has_length,
            random_access=all(p.random_access for p in parents)
        )

    def _calculate_size(self):
        sizes = [p.calculate_size() for p in self.parents]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def _calculate_value(self, *arg):
        yield arg

    def _getitem(self, item):
        return tuple(p[item] for p in self.parents)

    def _iter(self):
        return zip(*self.parents)


class OnMemorySource(WrapperSource):
    """
    メモリに乗せる。
    はじめてのアクセスで自動的にロード
    """

    def __init__(self, parent, load_immediately=False, auto_load=True, show_progress_onload=True):
        super(OnMemorySource, self).__init__(
            parent,
            has_length=parent.has_length,
            random_access=parent.random_access,
            auto_load=auto_load,
            show_progress_onload=show_progress_onload

        )
        if load_immediately:
            self.load()


class FilterSource(Source):
    """
    this Source iterate value filtered by pred from parent source
    """

    def __init__(self, pred, parent: Source):
        super(FilterSource, self).__init__(
            parent,
            has_length=False,
            random_access=False,
        )
        self.pred = pred

    def _calculate_size(self):
        return sum(1 for _ in self)

    def _calculate_value(self, arg):
        yield arg

    def _getitem(self, item):
        print("TODO worning")
        if not isinstance(item, int):
            raise NotImplementedError()

        for d in self:
            if item == 0:
                return d
            item -= 1
        raise IndexError("index out of range")

    def _iter(self):
        c = 0
        for d in self.parent:
            if self.pred(d):
                c += 1
                yield d

        self._size = c


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


class FileCacheSource(WrapperSource):
    """
    dataをファイルにキャッシュする
    長さ
    """

    def __init__(self, parent, cache_group_name, *cache_args, cache_dir=".tmp", caller_file_name=None,
                 auto_load=True,
                 show_progress_onload=True):
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

        super(FileCacheSource, self).__init__(
            parent,
            has_length=parent.has_length,
            random_access=parent.has_length,
            auto_load=auto_load,
            show_progress_onload=show_progress_onload
        )

        self.cache_group_name = cache_group_name
        self.cache_dir = pathlib.Path(cache_dir)
        hs = _calc_args_hash(cache_args)
        if caller_file_name is not None:
            self.caller_file_name = caller_file_name
        else:
            p = pathlib.Path(inspect.currentframe().f_back.f_code.co_filename)
            self.caller_file_name = p.name[:-len(p.suffix)]
        cache_base_name = f"flowder.{self.caller_file_name}.{self.cache_group_name}.{str(hs)}"
        self.cache_file_path = self.cache_dir / cache_base_name
        self.cache_length_path = self.cache_dir / (cache_base_name + "_len")

        if self.cache_length_path.exists():
            with self.cache_length_path.open("rb") as f:
                self._size = pickle.load(f)
                self.has_length = True

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
        if self._data is None:
            self.load(cache_if_not_yet=False)
        with self.cache_file_path.open("wb") as f:
            pickle.dump(self._data, f)
        self.cache_length()
        return self._data

    def cache_length(self, show_progress=True):
        """
        データ読み込みはせず、長さだけキャッシュする。
        初回は全データイテレーションするため遅いが、２回目以降はロードの必要がなくなる。
        :return:
        """
        if self.cache_length_path.exists():
            return

        if self.has_length:
            l = len(self)
        else:
            if show_progress:
                desc = f"[flowder.FileCacheSource({self.cache_group_name})]loading data for cache length..."
                l = sum(1 for _ in tqdm(self._iter(), desc=desc))
            else:
                l = sum(1 for _ in self._iter())

        self._size = l
        self.has_length = True
        with self.cache_length_path.open("wb") as f:
            pickle.dump(l, f)
        return self

    def load(self, cache_if_not_yet=True):
        """
        データが読み込み済みならそれを返す。
        読み込み済みでなければ、キャッシュファイルが存在すればそこからロードする。
        キャッシュファイルが無ければ、データを読み込む。
        その後、cache_if_not_yet=Trueなら、キャッシュファイルを作る。
        :param cache_if_not_yet:
        :return:
        """
        if self.is_loaded:
            return self
        if self.cache_file_path.exists():
            print(f"[flowder.FileCacheSource({self.cache_group_name})]load from cache file...")
            with self.cache_file_path.open("rb") as f:
                data = pickle.load(f)
                self._data = data
                self.has_length = True
                self.random_access = True
                return self
        else:
            super(FileCacheSource, self).load()
            if cache_if_not_yet:
                print(f"[flowder.FileCacheSource({self.cache_group_name})]create cache file...")
                self._make_cache()
            return self


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

        if not self.dataset.is_independent:  # 依存ソース
            if self.iter is None:
                for p in self.parents:
                    p.next(i)
                self.iter = self.dataset._calculate_value(*[p.value for p in self.parents])
            try:
                v = next(self.iter)
                self.value = v
            except StopIteration:
                for p in self.parents:
                    p.next(i)
                self.iter = self.dataset._calculate_value(*[p.value for p in self.parents])
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
    # TODO important: iterを呼ばないので、データローダがイテレーションを開始しても、
    # ソースチェーン上のauto_loadなソースがロードされない。 (independentなやつしかiterされないから)
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

    @property
    def item(self):
        return MapDummy(self)

    def is_independent(self):
        return True

    def preprocess(self):
        """
        全fieldsのプリプロセスを実行
        :return:
        """

        fields = [
            f for f in
            self.fields
            if f.start_data_feed() is not False
        ]
        if len(fields) == 0:
            print("preprocess is not needed for any fields")
            for f in tqdm(self.fields, desc="preprocess closing"):
                f.finish_data_feed()
            return
        leaf_iterators = create_cache_iter_tree(fields)
        for i in range(self.size):
            for f, leaf in zip(fields, leaf_iterators):
                f.data_feed(leaf.next(i))
        for f in tqdm(fields, desc="preprocess closing"):
            f.finish_data_feed()

    def _iter(self):  # TODO preprocess未処理時にエラー?
        leaf_iterators = create_cache_iter_tree(self.fields)
        for i in range(self.size):
            vs = [
                f.calculate_value(leaf.next(i))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, leaf_iterators)
            ]
            yield create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def _getitem(self, item):  # TODO fieldまたいで値のキャッシュ,slice item
        # leaf_iterators = create_cache_iter_tree(self.fields)  # todo to be instance field?
        # vs = [
        #     f.calculate_value(leaf[item])  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
        #     for f, leaf in zip(self.fields, leaf_iterators)
        # ]
        vs = [f[item] for f in self.fields]
        return create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def _calculate_size(self):
        return self.size
