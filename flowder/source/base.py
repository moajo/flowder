import hashlib
import pathlib
import pickle
import sys
from queue import Queue
from typing import Iterable

from tqdm import tqdm
import inspect

from ..abstracts import Field, SourceBase


def _dict_to_transform(transform_dict):
    def wrapper(data):
        if isinstance(data, tuple) or isinstance(data, list):
            return tuple(
                transform_dict[i](data[i]) if i in transform_dict else data[i]
                for i in range(len(data))
            )
        elif isinstance(data, dict):
            return {
                key: transform_dict[key](data[key]) if key in transform_dict else data[key]
                for key in data
            }

    return wrapper


def _dict_to_filter(filter_dict):
    def wrapper(data):
        if isinstance(data, tuple) or isinstance(data, list):
            return all(
                filter_dict[i](data[i]) if i in filter_dict else True
                for i in range(len(data))
            )
        elif isinstance(data, dict):
            return all(
                filter_dict[key](data[key]) if key in filter_dict else True
                for key in data
            )

    return wrapper


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


class MapDummy:
    """
    to make map/filter source easily
    """

    def __init__(self, source, recursive, transform=None):
        self.source = source
        self.transform = transform or (lambda x: x)
        self.recursive = recursive
        assert isinstance(recursive, bool)
        assert isinstance(source, SourceBase)

    def map(self, transform):
        if isinstance(transform, dict):
            transform = _dict_to_transform(transform)

        def merged_transform(item):
            item = self.transform(item)
            return transform(item)

        return MapSource(merged_transform, parent=self.source)

    def filter(self, pred):
        return FilterSource(pred, MapSource(lambda x: self.transform(x), parent=self.source))

    def __getitem__(self, item):
        if self.recursive:
            return MapDummy(self.source, True, lambda x: self.transform(x)[item])
        else:
            return MapSource(lambda x: x[item], parent=self.source)

    def __getattr__(self, item):
        if self.recursive:
            return MapDummy(self.source, True, lambda x: getattr(self.transform(x), item))
        else:
            return MapSource(lambda x: getattr(x, item), parent=self.source)

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
        return MapDummy(self, False)

    @property
    def item_if(self):
        """
        short-hand for Filter
        ex)
        mapped = source.item_if[0]>0
        # mapped = FilterSource(lambda x:x[0]>0, source)
        mapped = source.item_if.user.name=="hoge"
        # mapped = FilterSource(lambda x:x.user.name=="hoge", source)
        :return:
        """
        return MapDummy(self, True)

    def map(self, transform):
        """
        if transform is dict, transform will convert only data on the key of dict

        :param transform: function or dict
        :return:
        """
        if isinstance(transform, dict):
            transform = _dict_to_transform(transform)

        return MapSource(transform, self)

    def filter(self, pred):
        if isinstance(pred, dict):
            pred = _dict_to_filter(pred)
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

    def _getitem(self, item):
        if isinstance(item, slice):
            sliced_index = list(range(len(self)))[item]
            return SlicedSource(self, sliced_index)
        else:
            return self._basic_getitem(item)

    def _basic_getitem(self, item):
        raise NotImplementedError()


class SlicedSource(Source):
    """
    this source is independent
    """

    def __init__(self, parent, sliced_index):
        super(SlicedSource, self).__init__(parent)
        self.sliced_index = sliced_index
        self.parent_source = parent

    def _calculate_size(self):
        return len(self.sliced_index)

    def _getitem(self, item):
        if isinstance(item, slice):
            sliced_index = self.sliced_index[item]
            return SlicedSource(self.parent_source, sliced_index)
        else:
            return self.parent_source[self.sliced_index[item]]

    def _iter(self):
        for i in range(len(self.sliced_index)):
            yield self[i]

    def _is_independent(self):
        return True


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

    def _basic_getitem(self, item):
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

    def _basic_getitem(self, item):
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

    def _basic_getitem(self, item):
        return tuple(p[item] for p in self.parents)

    def _iter(self):
        return zip(*self.parents)


class FilterSource(Source):  # TODO random access用のindex tableを事前計算する機能
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
        self.cache_base_name = f"flowder.{self.caller_file_name}.{self.cache_group_name}.{str(hs)}"
        self.cache_file_path = self.cache_dir / self.cache_base_name
        self.cache_length_path = self.cache_dir / (self.cache_base_name + "_len")

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
            for p in self.cache_dir.glob(f"flowder.{self.caller_file_name}.{self.cache_group_name}*"):
                p.unlink()
        else:
            if self.cache_file_path.exists():
                self.cache_file_path.unlink()

    def _make_cache(self):
        if self.cache_file_path.exists():
            return
        if self._data is None:
            self.load(cache_if_not_yet=False)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        with self.cache_file_path.open("wb") as f:
            pickle.dump(self._data, f)
        self.cache_length()
        return self._data

    def cache_length(self, show_progress=True):
        """
        データ読み込みはせず、長さだけキャッシュする。
        初回は全データイテレーションするため遅いが、２回目以降はロードの必要がなくなる。
        :return: self
        """
        if self.cache_length_path.exists():
            return self

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

    def load(self, cache_if_not_yet=True, clear_cache: str = None):
        """
        データが読み込み済みならそれを返す。
        キャッシュファイルが存在すればそこからロードする。
        キャッシュファイルが無ければ、データを読み込む。
        その後、cache_if_not_yet=Trueなら、キャッシュファイルを作る。

        clear_cacheが指定されればキャッシュを削除し、無条件でデータを読み込む。
        :param cache_if_not_yet: load後にファイルキャッシュを作成するかどうか (default: True)
        :param clear_cache: キャッシュを削除するかどうか
            "default": clear_cache(self, remove_all=False)
            "all": clear_cache(self, remove_all=True)
        :return:
        """
        if clear_cache is not None:
            assert clear_cache in ["all", "default"]
            remove_all = clear_cache == "all"
            self.clear_cache(remove_all=remove_all)
        elif self.is_loaded:
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

        self.cache_iter_inst = self._cache_iter()

    def _cache_iter(self):
        """
           異なるiで呼ばれるごとに、値を更新して返す。
           前回と同じiで呼ばれたら前回のキャッシュ値をそのまま返す
           値の更新は独立ソースと依存ソースで異なる。

           # 独立の場合：
           datasetのnextを呼ぶ

           # 依存の場合
           親データセットのnext値のタプルから、calculate_valueする

           :return:
        """

        def cache(value, i):
            while True:
                next_i = yield value
                if next_i is None or next_i == i:
                    continue
                break

        if not self.dataset.is_independent:  # 依存ソース
            def hoge():
                a = [p.cache_iter_inst for p in self.parents]
                ii = 0
                args = [next(p) for p in a]
                while True:
                    yield from cache(args, ii)
                    ii += 1
                    args = [p.send(ii) for p in a]

            ppp = hoge()
            args = next(ppp)
            parent_i = 0
            last_i = 0
            while True:
                for n in self.dataset._calculate_value(*args):
                    yield from cache(n, last_i)
                    last_i += 1

                parent_i += 1
                args = ppp.send(parent_i)
        else:
            i = 0
            for v in self.dataset:
                yield from cache(v, i)
                i += 1

    @cache_value()
    def __getitem__(self, item):
        return self.dataset[item]  # TODO効率的な実装

    @property
    def is_independent(self):
        return len(self.parents) == 0


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
        assert len(self.fields) > 0

    @property
    def item(self):
        return MapDummy(self, False)

    @property
    def item_if(self):
        return MapDummy(self, True)

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
            print("[flowder.Dataset]preprocess is not needed for any fields")
            for f in tqdm(self.fields, desc="[flowder.Dataset]preprocess closing"):
                f.finish_data_feed()
            return
        leaf_iterators = create_cache_iter_tree(fields)
        l2 = [a.cache_iter_inst for a in leaf_iterators]
        for i in tqdm(range(self.size), desc=f"[flowder.Dataset]preprocessing {len(fields)} fields"):
            for f, leaf in zip(fields, l2):
                if i == 0:
                    f.data_feed(next(leaf))
                else:
                    f.data_feed(leaf.send(i))
        for f in tqdm(fields, desc="[flowder.Dataset]preprocess closing"):
            f.finish_data_feed()

    def _iter(self):  # TODO preprocess未処理時にエラー?
        leaf_iterators = create_cache_iter_tree(self.fields)
        l2 = [a.cache_iter_inst for a in leaf_iterators]
        vs = [
            f.calculate_value(next(leaf))
            for f, leaf in zip(self.fields, l2)
        ]
        for i in range(self.size):
            yield create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)
            vs = [
                f.calculate_value(leaf.send(i + 1))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, l2)
            ]

    def _basic_getitem(self, item):  # TODO fieldまたいで値のキャッシュ,slice item
        # leaf_iterators = create_cache_iter_tree(self.fields)  # todo to be instance field?
        # vs = [
        #     f.calculate_value(leaf[item])  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
        #     for f, leaf in zip(self.fields, leaf_iterators)
        # ]
        vs = [f[item] for f in self.fields]
        return create_example([f.name for f in self.fields], vs, return_as_tuple=self._return_as_tuple)

    def _calculate_size(self):
        return self.size

    def _str(self):
        parents = [
            (str(field.target_source) + f">>({field.name})").split("\n")
            for field in self.fields
        ]
        if len(parents) == 1:
            p = parents[0]
            p[-1] += "-" + "[Dataset]"
            return "\n".join(p)
        max_width = max(len(p_lines[0]) for p_lines in parents)
        pads = [
            [
                (" " * (max_width - len(line))) + line
                for line in p_lines
            ]
            for p_lines in parents
        ]
        p_line_counts = [len(it) for it in pads]

        tails = ["┐"]
        for pl in p_line_counts:
            for _ in range(pl - 1):
                tails.append("│")
            tails.append("┤")
        tails = tails[:-2]
        tails.append("┴" + "[Dataset]")
        lines = [line for p_lines in pads for line in p_lines]
        res = [line + tail for line, tail in zip(lines, tails)]
        return "\n".join(res)
