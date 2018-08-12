from collections import OrderedDict

from moajo_tool.utils import measure_time

from processors import RawProcessor
from queue import Queue
from tqdm import tqdm


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
    """

    :param fields:
    :return: leafs
    """
    cached_iters = {}
    leafs = []
    # queue = Queue()
    # for f in fields:
    #     queue.put_nowait(f.target_set)
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
        nameless_count = 1
        self._values = []
        self._keys = []
        for name, v in data_dict.items():
            if name is None:
                name = f"attr{nameless_count}"
                nameless_count += 1
            setattr(self, name, v)
            self._keys.append(name)
            self._values.append(v)

    def __getitem__(self, item):
        return self._values[item]


def example_from_names(names, vs):
    nameless_count = 1

    dd = []
    for name, v in zip(names, vs):
        if name is None:
            name = f"attr{nameless_count}"
            nameless_count += 1
        dd.append((name, v))
    return Example(OrderedDict(dd))


def create_example(field_names, vs, return_raw_value_for_single_data=True, return_tuple_for_nameless_data=True):
    assert len(field_names) == len(vs)
    if return_raw_value_for_single_data and len(vs) == 1:
        return vs[0]
    if return_tuple_for_nameless_data and all(name is None for name in field_names):
        return tuple(vs)
    return example_from_names(field_names, vs)


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


class Dataset:
    """
    fieldsとsetをつなぐ。torchのDatasetを継承。
    機能は
    - fieldsの前処理の制御
    - 全体反復
    - random access
    """

    def __init__(self, fields: list, size: int,
                 return_raw_value_for_single_data=True,
                 return_tuple_for_nameless_data=True):
        self.fields = fields
        self.size = size
        self._example_kwargs = {
            "return_raw_value_for_single_data": return_raw_value_for_single_data,
            "return_tuple_for_nameless_data": return_tuple_for_nameless_data,
        }

    @measure_time()
    def preprocess(self):
        """
        全fieldsのプリプロセスを実行
        :return:
        """
        leaf_iterators = create_cache_iter_tree(self.fields)
        for f in tqdm(self.fields, desc="preprocess initializing"):
            f.start_preprocess_data_feed()
        for i in tqdm(range(self.size), desc="preprocessing"):
            for f, leaf in zip(self.fields, leaf_iterators):
                f.processing_data_feed(leaf.next(i))
        for f in tqdm(self.fields, desc="preprocess closing"):
            f.finish_preprocess_data_feed()

    def __iter__(self):
        leaf_iterators = create_cache_iter_tree(self.fields)
        for i in range(self.size):
            vs = [
                f.calculate_value(leaf.next(i))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, leaf_iterators)
            ]
            yield create_example([f.name for f in self.fields], vs, **self._example_kwargs)

    def __getitem__(self, item):  # TODO fieldまたいで値のキャッシュ
        vs = [f[item] for f in self.fields]
        return create_example([f.name for f in self.fields], vs, **self._example_kwargs)

    def __len__(self):
        return self.size


class Field:
    """
    データセットの各データの特定の位置に対して処理するやつTODO
    """

    def __init__(self, target_set, preprocess=None, process=None, loading_process=None, batch_process=None):
        self.name = None
        self.target_set = target_set

        self.preprocess = preprocess or []  # list。ただの写像
        self.process = process or []  # 開始と終了通知のある関数
        self.loading_process = loading_process or []
        self.batch_process = batch_process or []

    def __getitem__(self, item):
        v = self.target_set[item]
        return self.calculate_value(v)

    def start_preprocess_data_feed(self):  # TODO withで書き直し？
        for p in self.process:
            p.start_preprocess_data_feed(self)

    def finish_preprocess_data_feed(self):
        for p in self.process:
            p.finish_preprocess_data_feed(self)

    def processing_data_feed(self, raw_value):
        v = raw_value
        for pre in self.preprocess:
            v = pre(v)
        for ld in self.process:
            v = ld(v)

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
