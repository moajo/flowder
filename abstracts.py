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


class Field:
    """
    データソースに変換、統計処理を行う
    複数のFieldを束ねてDatasetにする。
    """

    def __init__(self, name, target_source, preprocess=None, process=None, loading_process=None, batch_process=None):
        """

        :param name: str
        :param target_source: index accessible iterable collection
        :param preprocess: 共通前処理。map。関数のリスト
        :param process: preprocessに続く前処理。Processorのリスト
        :param loading_process: 後処理。map。関数のリスト
        :param batch_process:
        """
        assert name is not None
        assert target_source is not None

        self.name = name
        self.target_source = target_source

        self.preprocess = preprocess or []
        self.process = process or []
        self.loading_process = loading_process or []
        self.batch_process = batch_process or []

    def __getitem__(self, item):
        v = self.target_source[item]
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
