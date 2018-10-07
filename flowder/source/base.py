import hashlib
import pathlib
import pickle
import sys

import inspect

from tqdm import tqdm

from flowder.source.depend_func import DependFunc
from flowder.source.iterable_creator import IterableCreator, ic_map, ic_filter, ic_from_array, ic_slice, ic_zip, \
    ic_from_generator, ic_concat, ic_flat_map


class PipeLine:
    """
    パイプでつなげる関数
    依存情報も保持する
    """

    def __init__(self, applications):
        # assert type(dependencies) == list
        assert type(applications) == list
        self.applications = applications
        # self.dependencies = dependencies

    def __or__(self, other):
        if isinstance(other, PipeLine):
            return self._concat(other)
        else:
            raise TypeError(f"invalid pipe type: {other}")

    def _concat(self, other):
        assert isinstance(other, PipeLine)
        return PipeLine(self.applications + other.applications)

    def _apply(self, source, key):
        assert isinstance(source, Source)

        for ap in self.applications:
            source = ap(source, key)

        # source.dependencies += self.dependencies
        return source


class FlatMapped(PipeLine):
    def __init__(self, transform, dependencies):
        """

        :param transform: func or depend_func
        :param dependencies: transformのdependenciesに追加される
        """
        assert type(dependencies) == list
        if isinstance(transform, DependFunc):
            d = transform.dependencies + dependencies
            transform = transform.func
        else:
            d = dependencies
        self.transform = transform

        def _application(source, key):
            """

            :param source:
            :param key: srcのこのkeyの部分だけをmapする
            :return:
            """
            assert isinstance(source, Source)
            if key is None:
                return source.flat_map(transform, dependencies=d)
            else:  # tuple of source
                def _m(data):
                    if type(data) in [tuple, list]:
                        assert type(key) == int
                        assert len(data) > key, f"invalid tuple mapping key(out of range): " \
                                                f"\n\tkey: {key}" \
                                                f"\n\tlen(data): {len(data)}"
                        return tuple(
                            transform(data[i]) if i == key else data[i]
                            for i in range(len(data))
                        )
                    elif type(data) == dict:
                        assert key in data, f"invalid dict mapping key(key not found): " \
                                            f"\n\tkey: {key}" \
                                            f"\n\tdata.keys(): {data.keys()}"
                        return {
                            k: transform(data[k]) if k == key else data[k]
                            for k in data
                        }

                return source.map(_m)

        super(FlatMapped, self).__init__([_application])

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class Mapped(PipeLine):
    def __init__(self, transform, dependencies):
        """

        :param transform: func or depend_func
        :param dependencies: transformのdependenciesに追加される
        """
        assert type(dependencies) == list
        if isinstance(transform, DependFunc):
            d = transform.dependencies + dependencies
            transform = transform.func
        else:
            d = dependencies
        self.transform = transform

        def _application(source, key):
            """

            :param source:
            :param key: srcのこのkeyの部分だけをmapする
            :return:
            """
            assert isinstance(source, Source)
            if key is None:
                return source.map(transform, dependencies=d)
            else:
                def _m(data):
                    if type(data) in [tuple, list]:
                        assert type(key) == int
                        assert len(data) > key, f"invalid tuple mapping key(out of range): " \
                                                f"\n\tkey: {key}" \
                                                f"\n\tlen(data): {len(data)}"
                        return tuple(
                            transform(data[i]) if i == key else data[i]
                            for i in range(len(data))
                        )
                    elif type(data) == dict:
                        assert key in data, f"invalid dict mapping key(key not found): " \
                                            f"\n\tkey: {key}" \
                                            f"\n\tdata.keys(): {data.keys()}"
                        return {
                            k: transform(data[k]) if k == key else data[k]
                            for k in data
                        }

                return source.map(_m, dependencies=d)

        super(Mapped, self).__init__([_application])

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class Filtered(PipeLine):
    def __init__(self, pred, dependencies):
        """

        :param pred: func or depend_func
        :param dependencies: predのdependenciesに追加される
        """
        assert type(dependencies) == list
        if isinstance(pred, DependFunc):
            d = pred.dependencies + dependencies
            pred = pred.func
        else:
            d = dependencies
        self.pred = pred

        def _application(source, key):
            assert isinstance(source, Source)

            if key is None:
                return source.filter(pred, dependencies=d)
            else:
                def _m(data):
                    if type(data) == dict:
                        assert key in data, f"invalid dict filtering key(key not found): " \
                                            f"\n\tkey: {key}" \
                                            f"\n\tdata.keys(): {data.keys()}"
                    else:
                        assert len(data) > key, f"invalid tuple filtering key(out of range): " \
                                                f"\n\tkey: {key}" \
                                                f"\n\tlen(data): {len(data)}"
                    return pred(data[key])

                return source.filter(_m, dependencies=d)

        super(Filtered, self).__init__([_application])

    def __call__(self, *args, **kwargs):
        return self.pred(*args, **kwargs)


def flat_mapped(convert, dependencies=None) -> FlatMapped:
    """
    FlatMap Pipeline objectを作成
    :param convert:
    :param dependencies:
    :return:
    """
    if dependencies is None:
        dependencies = []
    return FlatMapped(convert, dependencies)


def mapped(transform, dependencies=None) -> Mapped:
    """
    Map Pipeline objectを作成
    :param transform:
    :param dependencies:
    :return:
    """
    if dependencies is None:
        dependencies = []
    return Mapped(transform, dependencies)


def filtered(pred, dependencies=None) -> Filtered:
    """
    Filter Pipeline objectを作成
    :param pred:
    :param dependencies:
    :return:
    """
    if dependencies is None:
        dependencies = []
    return Filtered(pred, dependencies)


def zipped(*sources):
    for s in sources:
        assert isinstance(s, Source)
    return Source(ic_zip(*[s._raw for s in sources]), parents=list(sources))


def _pattern_to_transform(transform_dict):
    def wrapper(data):
        if isinstance(data, tuple) or isinstance(data, list):
            assert len(data) == len(transform_dict), "pattern mapping must has same length as data"
            return tuple(
                transform_dict[i](data[i]) if transform_dict[i] is not None else data[i]
                for i in range(len(data))
            )
        elif isinstance(data, dict):
            return {
                key: transform_dict[key](data[key]) if key in transform_dict else data[key]
                for key in data
            }

    return wrapper


def _pattern_to_filter(filter_dict):
    def wrapper(data):
        if isinstance(data, tuple) or isinstance(data, list):
            assert len(data) == len(filter_dict), "pattern filtering must has same length as data"
            return all(
                filter_dict[i](data[i]) if filter_dict[i] is not None else True
                for i in range(len(data))
            )
        elif isinstance(data, dict):
            return all(
                filter_dict[key](data[key]) if key in filter_dict else True
                for key in data
            )

    return wrapper


class Source:
    """
    note: operator overloadをする
    len()の実装
    hash値を持つ
    """

    def __init__(self, raw: IterableCreator, parents=None, length=None, dependencies=None):
        """

        :param raw: iteratorを返す関数
        :param parents: hashの計算やstrで使う
        :param length: 既知なら与える
        :param dependencies: 依存するパラメータ。ハッシュに使う
        """
        self.parents = parents
        self.length = length
        self._raw = raw
        self.dependencies = dependencies if dependencies is not None else []
        self.data = None  # for random access
        self._hash = None  # lazy eval

        if self.parents is not None:
            assert type(self.parents) == list
            for p in self.parents:
                assert isinstance(p, Source)
        assert self.length is None or type(self.length) == int
        if self.dependencies is not None:
            assert type(self.dependencies) == list
            for d in self.dependencies:
                assert type(d) in [str, int, bool] or isinstance(d, DependFunc)

    def __add__(self, other):  # concat Source
        assert isinstance(other, Source)
        l = None
        if self.has_length and other.has_length:
            l = len(self) + len(other)
        return Source(ic_concat(self._raw, other._raw), parents=[self, other], length=l)

    def __mul__(self, other):  # zip Srouce
        assert isinstance(other, Source)
        l = None
        if self.has_length and other.has_length:
            l = min(len(self), len(other))
        return Source(ic_zip(self._raw, other._raw), parents=[self, other], length=l)

    def __iter__(self):
        yield from self._raw(0)

    @property
    def has_length(self) -> bool:
        return self.length is not None

    @property
    def hash(self):
        """
        親のhashと自身のdependenciesからhashを計算してキャッシュする
        :return:
        """
        if self._hash is None:
            hs = 0
            if self.parents is not None:
                for p in self.parents:
                    hs = (hs * 31 + p.hash) % sys.maxsize

            if self.dependencies is not None:
                hs = (hs * 31 + _calc_args_hash(self.dependencies)) % sys.maxsize
            self._hash = hs
        return self._hash

    def __len__(self):
        if self.has_length:
            return self.length
        else:
            raise TypeError("This Source has not been defined length.")

    def __or__(self, other):
        """
        Pipeにデータを流す
        対象は Pipe or pattern(tuple/list/dict) or Callable
        Callableは暗黙にMapとみなす
        :param other:
        :return:
        """
        if isinstance(other, PipeLine):
            return other._apply(self, key=None)
        if type(other) == tuple:
            assert all(a is None or isinstance(a, PipeLine) or callable(a) for a in other)
            s = self
            for i, pipe in enumerate(other):
                if pipe is not None:
                    if not isinstance(pipe, PipeLine):
                        pipe = mapped(pipe)
                    s = pipe._apply(s, key=i)
            return s
        if type(other) == dict:
            assert all(a is None or isinstance(a, PipeLine) or callable(a) for a in other.values())
            s = self
            for key, pipe in other.items():
                if pipe is not None:
                    if not isinstance(pipe, PipeLine):
                        pipe = mapped(pipe)
                    s = pipe._apply(s, key=key)
            return s
        raise TypeError("invalid pipe operation")

    def flat_map(self, converter, dependencies=None):
        return Source(
            ic_flat_map(self._raw, converter),
            parents=[self],
            dependencies=dependencies)

    def map(self, transform, dependencies=None):
        """
        if transform is dict,list or tuple,
        transform will convert only data on the key of dict/index of list(tuple)

        :param transform: function or dict, list, tuple
        :param dependencies:
        :return:
        """
        if type(transform) in [dict, list, tuple]:
            transform = _pattern_to_transform(transform)

        return Source(ic_map(self._raw, transform),
                      parents=[self], length=self.length, dependencies=dependencies)

    def filter(self, pred, dependencies=None):
        if type(pred) in [dict, list, tuple]:
            pred = _pattern_to_filter(pred)
        return Source(ic_filter(self._raw, pred), parents=[self], dependencies=dependencies)

    def cache(self, name, cache_dir=".tmp", clear_cache="no", check_only=False, caller_file_name=None):
        """
        即座にメモリにロードし、キャッシュファイルを作成する。
        キャッシュがすでにある場合はそれをロードする。
        :param name: キャッシュ名
        :param cache_dir:
        :param check_only: Trueなら、キャッシュが存在すればTrueを返す。キャッシュの作成も削除もしない。
        :param clear_cache: ロード前にキャッシュを削除する。
        "no": なにもしない(default)
        "yes": 完全一致キャッシュを削除
        "all": キャッシュグループをすべて削除
        "clear": キャッシュグループをすべて削除し、ロードしない。
        :param caller_file_name:
        :return:
        """
        assert clear_cache in ["no", "yes", "all", "clear"]
        cache_dir = pathlib.Path(cache_dir)
        if caller_file_name is None:
            p = pathlib.Path(inspect.currentframe().f_back.f_code.co_filename)
            caller_file_name = p.name[:-len(p.suffix)]

        cache_base_name = f"flowder.{caller_file_name}.{name}.{str(self.hash)}"
        cache_file_path = cache_dir / cache_base_name

        if check_only:
            return cache_file_path.exists()

        if clear_cache == "all":  # 同一のcache_group_nameのすべてのキャッシュも削除する
            for p in cache_dir.glob(f"flowder.{caller_file_name}.{name}*"):
                p.unlink()
        elif clear_cache == "yes":  # キャッシュファイル名が完全一致するファイルを削除する
            if cache_file_path.exists():
                cache_file_path.unlink()
        elif clear_cache == "clear":
            for p in cache_dir.glob(f"flowder.{caller_file_name}.{name}*"):
                p.unlink()
            return

        if cache_file_path.exists():
            print(f"[flowder.cache({name})]load from cache file...")
            with cache_file_path.open("rb") as f:
                data = pickle.load(f)
                self.data = data
                self.length = len(data)
                self._raw = ic_from_array(data)
            return self
        else:
            if self.data is None:
                desc = f"[flowder.cache({name})]loading data..."
                if self.has_length:
                    it = tqdm(self._raw(0), total=len(self), desc=desc)
                else:
                    it = tqdm(self._raw(0), desc=desc)
                data = list(it)
                self.data = data
                self.length = len(data)
                self._raw = ic_from_array(data)

            print(f"[flowder.cache({name})]create cache file...")
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            with cache_file_path.open("wb") as f:
                pickle.dump(self.data, f)
            return self

    def mem_cache(self):
        if self.data is not None:
            return
        desc = f"[flowder.cache(?)]loading data..."
        if self.has_length:
            it = tqdm(self._raw(0), total=len(self), desc=desc)
        else:
            it = tqdm(self._raw(0), desc=desc)
        data = list(it)
        self.data = data
        self.length = len(data)
        self._raw = ic_from_array(data)

    def __getitem__(self, item):
        """
        sliceにはSourceを、intには値を返す
        :param item:
        :return:
        """
        if self.data is not None:
            return self.data[item]
        if isinstance(item, slice):
            if self.has_length:
                stop = item.stop if item.stop is not None else self.length
                start = item.start if item.start is not None else 0
                step = item.step if item.step is not None else 1
                assert step > 0
                while stop < 0:
                    stop += self.length
                while start < 0:
                    start += self.length
                stop = min(stop, self.length)
                start = min(start, stop)
                return Source(
                    ic_slice(self._raw, slice(start, stop, step)),
                    length=(stop - start) // step,
                    parents=[self])
            else:
                if item.start is not None and item.start < 0 or \
                        item.stop is not None and item.stop < 0:
                    raise IndexError(
                        "negative index does not supported on the source that has no length"
                    )
                return Source(ic_slice(self._raw, item), parents=[self])
        else:
            if not self.has_length:
                if item < 0:
                    raise IndexError(
                        "negative index does not supported on the source that has no length"
                    )
            else:
                if item < 0:
                    item += self.length
                assert 0 <= item < self.length, "index out of range"
            return next(iter(self._raw(item)))


def _calc_args_hash(args):
    hs = 0
    for obj in args:
        if type(obj) == str:
            obj_hash = int(hashlib.sha1(obj.encode('utf-8')).hexdigest(), 16)
            hs = (hs * 31 + obj_hash) % sys.maxsize
        elif type(obj) == int:
            hs = (hs * 31 + obj) % sys.maxsize
        else:
            raise ValueError(
                f"{obj} is not hashable.\nall arguments are needed to be hashable for caching."
            )
    return hs
