import pathlib
import pickle
import sys

import inspect

from tqdm import tqdm

from flowder.hash import default_hash_func, hash_max_size
from flowder.source.depend_func import DependFunc
from flowder.source.iterable_creator import IterableCreator, ic_map, ic_filter, ic_from_array, ic_slice, ic_zip, \
    ic_concat, ic_flat_map, ic_from_iterable
from flowder.source.random_access import ra_concat, RandomAccessor, ra_zip, ra_map, ra_from_array, ra_slice


def _assert_dependencies(d):
    """
    only str/int/bool+tuple/dict of they is acceptable
    :param d:
    :return:
    """
    if isinstance(d, tuple):
        [_assert_dependencies(n) for n in d]
        return
    if isinstance(d, dict):
        [type(k) == str and _assert_dependencies(v) for k, v in d.items()]
        return
    assert type(d) in [str, int, bool] or isinstance(d, DependFunc)


class PipeFunc:
    """
    |で他のPipeFuncとconcatできる関数
    """

    def __init__(self, funcs: list):
        assert type(funcs) == list
        self.funcs = funcs

    def __call__(self, arg):
        for ap in self.funcs:
            arg = ap(arg)
        return arg

    def __or__(self, other):
        if isinstance(other, PipeFunc):
            return self._concat(other)
        else:
            raise TypeError(f"invalid pipe type: {other}")

    def _concat(self, other):
        assert isinstance(other, PipeFunc)
        return PipeFunc(self.funcs + other.funcs)


class PipeLine(PipeFunc):
    """
    Sourceへの作用素applicationを持つPipeFunc
    依存情報も保持する
    関数として呼ばれるときの処理とSourceにapplyする処理を区別する。
    Sourceかlistかiterableと|でconcatできる。このときlist/iterableはSourceにラップされる
    Sourceにapplyして別のSourceを作る
    """

    def __init__(self, funcs: list, applications: list):
        assert type(applications) == list
        self.applications = applications
        super(PipeLine, self).__init__(funcs)

    def __ror__(self, other):
        if isinstance(other, list):
            return Source(ic_from_array(other), ra_from_array(other), length=len(other)) | self
        elif hasattr(other, "__iter__"):
            return Source(ic_from_iterable(other), random_accessor=None) | self
        else:
            raise TypeError(f"unsupported pipe-operation with {type(other)}")

    def _apply(self, source, key):
        assert isinstance(source, Source)

        for ap in self.applications:
            source = ap(source, key)

        return source

    def _concat(self, other):
        assert isinstance(other, PipeLine)
        return PipeLine(self.funcs + other.funcs, self.applications + other.applications)


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
        self.d = d

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
                raise TypeError("flatmap with key is not supported")

        super(FlatMapped, self).__init__([self], [_application])

    def __call__(self, source):
        if isinstance(source, list) or isinstance(source, tuple):
            source = Source(ic_from_array(source), ra_from_array(source), length=len(source))
        assert isinstance(source, Source), \
            f"Argument for FlatMapped called as function must be Source, but {type(source)} found"
        return source.flat_map(self.transform, dependencies=self.d)


class Mapped(PipeLine):
    """
    map operator
    関数として呼び出すとtransformがそのまま呼ばれる
    Sourceに適用してmapする
    """

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

        super(Mapped, self).__init__([transform], [_application])


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

        super(Filtered, self).__init__([pred], [_application])


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
    if all(s.random_accessible for s in sources):
        ra = ra_zip(*(s._random_accessor for s in sources))
        length = min(len(s) for s in sources)
    else:
        ra = None
        if all(s.has_length for s in sources):
            length = min(len(s) for s in sources)
        else:
            length = None
    return Source(ic_zip(*[s._raw for s in sources]), random_accessor=ra, parents=list(sources), length=length)


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

    def __init__(self,
                 raw: IterableCreator,
                 random_accessor: RandomAccessor = None,
                 parents=None,
                 length: int = None,
                 dependencies=None):
        """

        :param raw: iteratorを返す関数
        :param parents: hashの計算やstrで使う
        :param length: 既知なら与える
        :param dependencies: 依存するパラメータ。ハッシュに使う
        """
        self.parents = parents
        self.length = length
        self._raw = raw
        self._random_accessor = random_accessor
        self.dependencies = dependencies if dependencies is not None else []
        self.data = None  # for random access
        self._hash = None  # lazy eval

        if length is not None:
            assert length >= 0

        if random_accessor is not None:
            assert length is not None, "length must not be None if source is random accessible"

        if self.parents is not None:
            assert type(self.parents) == list
            for p in self.parents:
                assert isinstance(p, Source)
        assert self.length is None or type(self.length) == int
        if self.dependencies is not None:
            assert type(self.dependencies) == list
            for d in self.dependencies:
                _assert_dependencies(d)

    @property
    def random_accessible(self):
        return self._random_accessor is not None

    def __add__(self, other):  # concat Source
        assert isinstance(other, Source)
        if self.has_length and other.has_length:
            length = len(self) + len(other)
        else:
            length = None
        if self.random_accessible and other.random_accessible:
            ra = ra_concat(self._random_accessor, other._random_accessor, self.length)
        else:
            ra = None
        return Source(ic_concat(self._raw, other._raw), random_accessor=ra, parents=[self, other], length=length)

    def __mul__(self, other):  # zip Source
        assert isinstance(other, Source)
        if self.has_length and other.has_length:
            length = min(len(self), len(other))
        else:
            length = None
        if self.random_accessible and other.random_accessible:
            ra = ra_zip(self._random_accessor, other._random_accessor)
        else:
            ra = None
        return Source(ic_zip(self._raw, other._raw), random_accessor=ra, parents=[self, other], length=length)

    def __iter__(self):
        yield from self._raw(0)

    @property
    def has_length(self) -> bool:
        return self.length is not None

    @property
    def hash(self) -> int:
        """
        親のhashと自身のdependenciesからhashを計算してキャッシュする
        :return:
        """
        if self._hash is None:
            hs = 1
            if self.parents is not None:
                for p in self.parents:
                    hs = (hs * 31 + p.hash) % hash_max_size

            if self.dependencies is not None:
                hs = (hs * 31 + default_hash_func(self.dependencies)) % hash_max_size
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
        if callable(other):
            other = mapped(other)
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
        if dependencies is None:
            dependencies = []
        dependencies = dependencies + ["flatmap"]
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
        if dependencies is None:
            dependencies = []
        dependencies = dependencies + ["map"]
        return Source(
            ic_map(self._raw, transform),
            random_accessor=ra_map(self._random_accessor, transform) if self.random_accessible else None,
            parents=[self],
            length=self.length,
            dependencies=dependencies)

    def filter(self, pred, dependencies=None):
        if type(pred) in [dict, list, tuple]:
            pred = _pattern_to_filter(pred)
        if dependencies is None:
            dependencies = []
        dependencies = dependencies + ["filter"]
        return Source(ic_filter(self._raw, pred), parents=[self], dependencies=dependencies)

    def cache(self, name,
              cache_dir=".tmp",
              clear_cache="no",
              check_only=False,
              caller_file_name=None,
              length_only=False):
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
        :param length_only:
        Trueで長さのみをキャッシュからロードする。
        check_onlyが同時にTrueなら、length cacheの存在を確認する。
        キャッシュがない場合はこのパラメータは無視される。
        :return:
        """
        assert clear_cache in ["no", "yes", "all", "clear"]
        cache_dir = pathlib.Path(cache_dir)
        if caller_file_name is None:
            p = pathlib.Path(inspect.currentframe().f_back.f_code.co_filename)
            caller_file_name = p.name[:-len(p.suffix)]

        cache_base_name = f"flowder.{caller_file_name}.{name}.{hex(self.hash)[2:]}"
        length_cache_base_name = f"flowder.{caller_file_name}.{name}.{hex(self.hash)[2:]}.len"
        cache_file_path = cache_dir / cache_base_name
        length_cache_file_path = cache_dir / length_cache_base_name

        if check_only:
            if length_only:
                return length_cache_file_path.exists()
            else:
                return cache_file_path.exists()

        if clear_cache == "all":  # 同一のcache_group_nameのすべてのキャッシュも削除する
            for p in cache_dir.glob(f"flowder.{caller_file_name}.{name}*"):
                p.unlink()
        elif clear_cache == "yes":  # キャッシュファイル名が完全一致するファイルを削除する
            if cache_file_path.exists():
                cache_file_path.unlink()
            if length_cache_file_path.exists():
                length_cache_file_path.unlink()
        elif clear_cache == "clear":
            for p in cache_dir.glob(f"flowder.{caller_file_name}.{name}*"):
                p.unlink()
            return

        if length_only and length_cache_file_path.exists():
            # loading length from cache
            with length_cache_file_path.open("rb") as f:
                self.length = pickle.load(f)
                assert type(self.length) == int
            return self

        if cache_file_path.exists():
            print(f"[flowder.cache({name})]loading cache...\n\tcache file: {cache_file_path}")
            with cache_file_path.open("rb") as f:
                data = pickle.load(f)
                self.data = data
                self.length = len(data)
                self._raw = ic_from_array(data)
                self._random_accessor = ra_from_array(data)
            return self
        else:
            if self.data is None:
                desc = f"[flowder.cache({name})]loading data from source..."
                if self.has_length:
                    it = tqdm(self._raw(0), total=len(self), desc=desc)
                else:
                    it = tqdm(self._raw(0), desc=desc)
                data = list(it)
                self.data = data
                self.length = len(data)
                self._raw = ic_from_array(data)
                self._random_accessor = ra_from_array(data)

            print(f"[flowder.cache({name})]create cache file...\n\tcache file: {cache_file_path}")
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            with cache_file_path.open("wb") as f:
                pickle.dump(self.data, f)
            with length_cache_file_path.open("wb") as f:
                pickle.dump(len(self.data), f)
            return self

    def mem_cache(self):
        if self.data is not None:
            return
        desc = f"[flowder.cache(?)]loading data from source..."
        if self.has_length:
            it = tqdm(self._raw(0), total=len(self), desc=desc)
        else:
            it = tqdm(self._raw(0), desc=desc)
        data = list(it)
        self.data = data
        self.length = len(data)
        self._raw = ic_from_array(data)
        self._random_accessor = ra_from_array(data)

    def count(self):
        if not self.has_length:
            self.length = sum(1 for _ in self)
        return self.length

    def __getitem__(self, item):
        """
        sliceにはSourceを、intには値を返す
        :param item:
        :return:
        """

        if isinstance(item, slice):
            if self.has_length:
                stop = item.stop if item.stop is not None else self.length
                start = item.start if item.start is not None else 0
                step = item.step if item.step is not None else 1
                assert step > 0
                if stop < 0:
                    stop += self.length
                if stop < 0:
                    stop = 0
                if start < 0:
                    start += self.length
                if start < 0:
                    start = 0
                stop = min(stop, self.length)
                start = min(start, stop)
                sl = slice(start, stop, step)
                if self.data is not None:
                    ic = ic_slice(ic_from_array(self.data), sl)
                    ra = ra_slice(ra_from_array(self.data), s=sl)
                else:
                    ic = ic_slice(self._raw, sl)
                    ra = ra_slice(self._random_accessor, s=sl) if self.random_accessible else None
                return Source(
                    ic,
                    random_accessor=ra,
                    length=(stop - start) // step,
                    parents=[self],
                    dependencies=[{"slice": (start, stop, step)}],
                )
            else:
                if item.start is not None and item.start < 0 or \
                        item.stop is not None and item.stop < 0:
                    raise IndexError(
                        "negative index does not supported on the source that has no length"
                    )
                return Source(ic_slice(self._raw, item), parents=[self])
        else:
            if self.data is not None:
                return self.data[item]
            if not self.has_length:
                if item < 0:
                    raise IndexError(
                        "negative index does not supported on the source that has no length"
                    )
            else:
                if item < 0:
                    item += self.length
                if not (0 <= item < self.length):
                    raise IndexError("index out of range")
            if not self.random_accessible:
                raise IndexError("this source is not able to be random accessed")
            return self._random_accessor(item)

    def search_item(self, index):
        """
        random access
        random accessできない場合は線形探索する
        :param index:
        :return:
        """
        if not self.has_length:
            if index < 0:
                raise IndexError(
                    "negative index does not supported on the source that has no length"
                )
        else:
            if index < 0:
                index += self.length
            assert 0 <= index < self.length, "index out of range"
        if self.random_accessible:
            return self._random_accessor(index)
        return next(iter(self._raw(index)))

    def __rshift__(self, other):
        if isinstance(other, Aggregator):
            other.feed_data(self)
        else:
            raise TypeError("invalid aggregate operation")


class Aggregator:
    def __init__(self, name):
        self.name = name

    def feed_data(self, data: Source):
        raise NotImplementedError()
