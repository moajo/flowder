from processors import RawProcessor
from queue import Queue


def cache_last_value():
    def decorator(func):
        last = {}

        def wrapper(arg):
            if "arg" not in last:
                last["arg"] = arg
                last["value"] = func(arg)
                return last["value"]
            if arg != last["arg"]:
                last["arg"] = arg
                last["value"] = func(arg)
            return last["value"]

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
    def __init__(self, fields, vs):
        self.nameless_values = []
        for f, v in zip(fields, vs):
            if f.name is not None:
                setattr(self, f.name, v)
            else:
                self.nameless_values.append(v)

    def __getitem__(self, item):
        return self.nameless_values[item]


def create_example(fields, vs, return_raw_value_for_single_data=True, return_tuple_for_nameless_data=True):
    assert len(fields) == len(vs)
    if return_raw_value_for_single_data and len(vs) == 1:
        return vs[0]
    if return_tuple_for_nameless_data and all(f.name is not None for f in fields):
        return tuple(vs)
    return Example(fields, vs)


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

    @cache_last_value()
    def __getitem__(self, item):
        return self.dataset[item]

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
        self.example_kwargs = {
            "return_raw_value_for_single_data": return_raw_value_for_single_data,
            "return_tuple_for_nameless_data": return_tuple_for_nameless_data,
        }

    def preprocess(self):
        """
        全fieldsのプリプロセスを実行
        :return:
        """
        pass

    def __iter__(self):
        # for i in range(self.size):
        #     yield self[i]
        self.leaf_iterators = create_cache_iter_tree(self.fields)
        for i in range(self.size):
            vs = [
                f.calculate_value(leaf.next(i))  # next(i)は終了するとStopIterationを投げるのでその場合そこで終了する
                for f, leaf in zip(self.fields, self.leaf_iterators)
            ]
            yield create_example(self.fields, vs, **self.example_kwargs)

    def __getitem__(self, item):  # TODO fieldまたいで値のキャッシュ
        vs = [f.get_value(item) for f in self.fields]
        return create_example(self.fields, vs, **self.example_kwargs)


class Field:
    """
    データセットの各データの特定の位置に対して処理するやつTODO
    """

    def __init__(self, target_set, preprocess=None, process=None, loading_process=None):
        self.name = None
        self.target_set = target_set

        root = RawProcessor(self.target_set)
        self.preprocess = preprocess
        self.process = process
        self.loading_process = loading_process or root
        # self.process = process

        # if self.process is not None:
        #     root.pipe(self.process.get_root())
        # else:
        #     self.process = root

    # def get_value(self, i):
    #     return self.process.get_value(i)

    def calculate_value(self, parent_value):
        return self.loading_process(parent_value)
