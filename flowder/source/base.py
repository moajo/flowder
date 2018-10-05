import hashlib
import pathlib
import pickle
import sys

import inspect
from typing import Iterable, Callable

from tqdm import tqdm

# iterationとslice反復、indexでのrandom access
IterableCreator = Callable[[int], Iterable]


def ic_filter(ic, pred):
    def gen():
        for item in ic(0):
            if pred(item):
                yield item

    return ic_from_generator(gen)


def ic_map(ic, transform):
    def _w(start):
        for item in ic(start):
            yield transform(item)

    return _w


def ic_from_array(array) -> IterableCreator:
    def _w(start: int):
        yield from array[start:]

    return _w


def ic_slice(ic: IterableCreator, s: slice) -> IterableCreator:
    slice_step = s.step if s.step is not None else 1
    slice_start = s.start if s.start is not None else 0
    if s.stop is None:
        def _w(start):
            ds = slice_start + slice_step * start
            yield from ic(ds)

        return _w
    else:

        def _w(start):
            ds = slice_start + slice_step * start
            c = (s.stop - 1 - ds) // slice_step + 1
            for i, item in enumerate(ic(ds)):
                if i % slice_step != 0:
                    continue
                if c <= 0:
                    break
                yield item
                c -= 1

        return _w


def ic_from_generator(gen_func):
    def _w(start: int):
        for item in gen_func():
            if start > 0:
                start -= 1
                continue
            yield item

    return _w


def ic_from_iterable(iterable):
    def _w(start: int):
        for item in iterable:
            if start > 0:
                start -= 1
                continue
            yield item

    return _w


def iterator_from_iterable_creator(ic):
    def _w(start: int):
        for item in ic():
            if start > 0:
                start -= 1
                continue
            yield item

    return _w


def zip_iterator_iterable(*ic):
    def _w(start: int):
        yield from zip(*[a(start) for a in ic])

    return _w


def concat_iterator_iterable(*ic):
    def _w(start: int):
        for a in ic:
            for item in a(0):
                if start > 0:
                    start -= 1
                    continue
                yield item

    return _w


class PipeLine:
    """
    パイプでつなげる関数
    依存情報も保持する
    """

    def __init__(self, applications, dependencies):
        assert type(dependencies) == list
        assert type(applications) == list
        self.applications = applications
        self.dependencies = dependencies

    def __or__(self, other):
        if isinstance(other, PipeLine):
            return self._concat(other)

    def _concat(self, other):
        assert isinstance(other, PipeLine)
        return PipeLine(self.applications + other.applications, self.dependencies + other.dependencies)

    def _apply(self, source, key):
        assert isinstance(source, Source)

        for ap in self.applications:
            source = ap(source, key)

        source.dependencies += self.dependencies
        return source


class Mapped(PipeLine):
    def __init__(self, transform, dependencies):
        assert type(dependencies) == list
        if isinstance(transform, DependFunc):
            self.transform = transform.func
            d = transform.dependencies + dependencies
        else:
            self.transform = transform
            d = dependencies

        def _application(source, key):
            """

            :param source:
            :param key: srcのこのkeyの部分だけをmapする
            :return:
            """
            assert isinstance(source, Source)
            if key is None:
                return source.map(transform)
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

                return source.map(_m)

        super(Mapped, self).__init__([_application], d)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class Filtered(PipeLine):
    def __init__(self, pred, dependencies):
        assert type(dependencies) == list
        if isinstance(pred, DependFunc):
            self.pred = pred.func
            d = pred.dependencies + dependencies
        else:
            self.pred = pred
            d = dependencies

        def _application(source, key):
            assert isinstance(source, Source)

            if key is None:
                return source.filter(pred)
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

                return source.filter(_m)

        super(Filtered, self).__init__([_application], d)

    def __call__(self, *args, **kwargs):
        return self.pred(*args, **kwargs)


def mapped(transform, dependencies=None) -> Mapped:
    if dependencies is None:
        dependencies = []
    return Mapped(transform, dependencies)


def filtered(pred, dependencies=None) -> Filtered:
    if dependencies is None:
        dependencies = []
    return Filtered(pred, dependencies)


def zipped(*sources):
    for s in sources:
        assert isinstance(s, Source)
    return Source(zip_iterator_iterable(*[s._raw for s in sources]), parents=list(sources))


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


class DependFunc:
    def __init__(self, func, dependencies):
        self.func = func
        self.dependencies = dependencies
        assert type(dependencies) == list

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def depend(*dependencies):
    def wrapper(f):
        return DependFunc(f, list(dependencies))

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
        return Source(concat_iterator_iterable(self._raw, other._raw), parents=[self, other], length=l)

    def __mul__(self, other):  # zip Srouce
        assert isinstance(other, Source)
        l = None
        if self.has_length and other.has_length:
            l = min(len(self), len(other))
        return Source(zip_iterator_iterable(self._raw, other._raw), parents=[self, other], length=l)

    def __iter__(self):
        yield from self._raw(0)

    @property
    def has_length(self) -> bool:
        return self.length is not None

    @property
    def hash(self):
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
        # TODO: 常に定義されているとlist()などがlen()を呼んでしまい落ちる
        if self.has_length:
            return self.length
        else:
            raise TypeError("This Source has not been defined length.")

    def __or__(self, other):
        if isinstance(other, PipeLine):
            return other._apply(self, key=None)
        if type(other) == tuple:
            assert all(a is None or isinstance(a, PipeLine) for a in other)
            s = self
            for i, pipe in enumerate(other):
                if pipe is not None:
                    s = pipe._apply(s, key=i)
            return s
        if type(other) == dict:
            assert all(a is None or isinstance(a, PipeLine) for a in other.values())
            s = self
            for key, pipe in other.items():
                if pipe is not None:
                    s = pipe._apply(s, key=key)
            return s
        raise TypeError("invalid pipe operation")

    def map(self, transform, dependencies=None):
        """
        if transform is dict, transform will convert only data on the key of dict

        :param transform: function or dict
        :param dependencies:
        :return:
        """
        if isinstance(transform, dict):
            transform = _dict_to_transform(transform)

        return Source(ic_map(self._raw, transform),
                      parents=[self], length=self.length, dependencies=dependencies)

    def filter(self, pred, dependencies=None):
        if isinstance(pred, dict):
            pred = _dict_to_filter(pred)
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

            return Source(ic_slice(self._raw, item), parents=[self])
        else:
            return next(iter(self._raw(item)))


def lines(path):
    path = pathlib.Path(path)
    assert path.exists()

    hash = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(2048 * hash.block_size)
            if len(chunk) == 0:
                break
            hash.update(chunk)

    d = hash.hexdigest()

    with path.open(encoding="utf-8") as f:
        length = sum(1 for _ in f)

    def _gen():
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield line[:-1]

    obs = iterator_from_iterable_creator(_gen)

    return Source(obs, length=length, dependencies=[d])


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

#
#     def _str(self):
#         parents = [
#             (str(field.target_source) + f">>({field.name})").split("\n")
#             for field in self.fields
#         ]
#         if len(parents) == 1:
#             p = parents[0]
#             p[-1] += "-" + "[Dataset]"
#             return "\n".join(p)
#         max_width = max(len(p_lines[0]) for p_lines in parents)
#         pads = [
#             [
#                 (" " * (max_width - len(line))) + line
#                 for line in p_lines
#             ]
#             for p_lines in parents
#         ]
#         p_line_counts = [len(it) for it in pads]
#
#         tails = ["┐"]
#         for pl in p_line_counts:
#             for _ in range(pl - 1):
#                 tails.append("│")
#             tails.append("┤")
#         tails = tails[:-2]
#         tails.append("┴" + "[Dataset]")
#         lines = [line for p_lines in pads for line in p_lines]
#         res = [line + tail for line, tail in zip(lines, tails)]
#         return "\n".join(res)
