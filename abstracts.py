from collections import OrderedDict
from typing import Iterable

from moajo_tool.utils import measure_time

from processors import RawProcessor, BuildVocab
from queue import Queue
from tqdm import tqdm


class SourceBase:
    """
    データソース
    親があるならそれのみに依存。
    なければ何らかの外部データに依存する。
    （親があるけど、親以外のデータにも依存することは禁止）
    マージするときなど親は複数になる
    """

    def __init__(self, *parents):
        assert type(parents) is tuple
        self.parents: [SourceBase] = parents or []
        self.children = []
        self._size = None

    def __len__(self):
        if self._size is None:
            self._size = self.calculate_size()
        return self._size

    def calculate_size(self):
        """
        データサイズを計算。createまでに呼ばれる
        :return:
        """
        raise NotImplementedError()

    def calculate_value(self, *args):
        """
        親セットの値から値（のイテレータ）を計算するステートレスな関数
        独立ソースの場合、呼ばれない。
        :param args:
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, item):
        """
        ランダムアクセスに対応する必要あり
        :param item:
        :return:
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        iterableである必要あり
        :return:
        """
        raise NotImplementedError()


def cache_last_value():
    def decorator(func):
        cache = {}

        def wrapper(arg):
            if arg not in cache:
                for k in cache.keys():
                    del cache[k]
                cache[arg] = func(arg)
            return cache[arg]

        return wrapper

    return decorator


def cache_value():
    def decorator(func):
        cache = {}

        def wrapper(arg):
            if arg not in cache:
                cache[arg] = func(arg)
            return cache[arg]

        return wrapper

    return decorator


def create_cache_iter_tree(fields):
    cached_iters = {}
    leafs = []
    for f in fields:
        ts = f.target_set
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


class Dataset(SourceBase):
    """
    fieldsとsetをつなぐ。torchのDatasetを継承。
    機能は
    - fieldsの前処理の制御
    - 全体反復
    - random access
    """

    def __init__(self, fields: Iterable, size: int, return_as_tuple=True):
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


class Field:
    """
    データセットの各データの特定の位置に対して処理するやつTODO
    """

    def __init__(self, target_set, preprocess=None, process=None, loading_process=None, batch_process=None):
        """

        :param target_set:
        :param preprocess: 共通前処理。map。関数のリスト
        :param process: preprocessに続く前処理。Processorのリスト
        :param loading_process: 後処理。map。関数のリスト
        :param batch_process:
        """
        self.name = None
        self.target_set = target_set

        self.preprocess = preprocess or []  # list。ただの写像
        self.process = process or []  # 開始と終了通知のある関数
        self.loading_process = loading_process or []
        self.batch_process = batch_process or []

    def __getitem__(self, item):
        v = self.target_set[item]
        return self.calculate_value(v)

    def start_preprocess_data_feed(self):
        need_data_feed = False
        for p in self.process:
            if p.start_preprocess_data_feed(self) is not False:
                need_data_feed = True
        return need_data_feed

    def finish_preprocess_data_feed(self):
        for p in self.process:
            p.finish_preprocess_data_feed(self)

    def processing_data_feed(self, raw_value):
        v = raw_value
        for pre in self.preprocess:
            v = pre(v)
        for ld in self.process:
            ld(v)

    def calculate_value(self, raw_value):

        v = raw_value
        for pre in self.preprocess:
            v = pre(v)
        for ld in self.loading_process:
            v = ld(v)
        return v

    def batch_process(self, batch):
        for bp in self.batch_process:
            batch = bp(batch)
        return batch
