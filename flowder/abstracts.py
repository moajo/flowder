from tqdm import tqdm


class SourceBase:
    """
    データソース
    長さは計算可能かどうかを保持、calculate_sizeで計算する。未対応はhas_lengthでやる。
    ランダムアクセスも可能かどうか。
    親はマージするときなど複数になる
    親の値に対してステートレス。
    複数の親に対してはzipのみサポート
    """

    def __init__(self, *parents, has_length=True, random_access=True, auto_load=False, show_progress_onload=False):
        """

        :param parents:
        :param has_length: 長さを計算可能か
        :param random_access: 可能か
        :param auto_load: iterで自動的にロードするか
        :param show_progress_onload: load時のshow_progressのデフォルト値
        """
        assert type(parents) is tuple
        self.random_access = random_access
        self.has_length = has_length  # 長さは計算可能かわからない。
        self.parents: [SourceBase] = list(parents) or []
        self._size = None  # 長さのキャッシュ
        self._data = None
        self.auto_load = auto_load
        self.show_progress_onload = show_progress_onload

    @property
    def parent(self):
        assert len(self.parents) == 1
        return self.parents[0]

    @property
    def is_loaded(self):
        return self._data is not None

    def load(self, show_progress=None):
        if show_progress is None:
            show_progress = self.show_progress_onload
        if not self.is_loaded:
            d = self
            if show_progress:
                if d.has_length:
                    l = len(d)
                    d = tqdm(d._iter(), total=l, desc="[MemoryCacheSource]loading...")
                else:
                    d = tqdm(d._iter(), desc="[MemoryCacheSource]loading...")
            self._data = list(d)
            self.has_length = True
            self.random_access = True
        return self

    @property
    def is_independent(self):
        """override me"""
        if self.is_loaded:
            return False
        return self._is_independent()

    def __len__(self):
        if self._size is None:
            if not self.has_length:
                raise ValueError("This source has not length.(call .calculate_size() or .load() to obtain length,)")
            self._size = self.calculate_size()
        return self._size

    def to(self, source_ctor):
        """
        convert to other kind of source
        :param source_ctor:
        :return:
        """
        return source_ctor(self)

    def calculate_size(self):
        if self.is_loaded:
            return len(self._data)
        return self._calculate_size()

    def __getitem__(self, item):
        if self.is_loaded:
            return self._data[item]
        if not self.random_access:
            raise ValueError("This source is not supporting index access.(you can .load() to enable index access)")
        return self._getitem(item)

    def __iter__(self):
        if self.is_loaded:
            return iter(self._data)
        if self.auto_load:
            self.load()
        return self._iter()

    def reduce(self):
        """
        簡約。２重mapなどを一つにまとめて計算効率を高める。
        iterationの前に呼んで効率的にすることが目的
        :return:
        """
        return self

    def _iter(self):
        raise NotImplementedError()

    def _getitem(self, item):
        """
        ランダムアクセスに対応するなら実装
        :param item:
        :return:
        """
        raise NotImplementedError()

    def _calculate_value(self, args):
        """
        親セットの値から値（のイテレータ）を計算するステートレスな関数
        独立ソースの場合、呼ばれない。
        :param args:
        :return:
        """
        raise NotImplementedError()

    def _calculate_size(self):
        """
        ソースの要素数を計算する。
        has_length=Trueの場合は最初に__len__が呼ばれたときに呼ばれる
        それ以外の場合は明示的に呼ぶ
        :return:
        """
        raise NotImplementedError()

    def _is_independent(self):
        """override me"""
        return len(self.parents) == 0


class Field:
    """
    データソースに変換、統計処理を行う
    複数のFieldを束ねてDatasetにする
    """

    def __init__(self, name, target_source, preprocess=None, process=None, loading_process=None):
        """

        :param name: str
        :param target_source: index accessible iterable collection
        :param preprocess: 共通前処理。map。関数のリスト
        :param process: preprocessに続く前処理。Processorのリスト
        :param loading_process: 後処理。map。関数のリスト
        """
        assert name is not None
        assert target_source is not None
        assert isinstance(target_source, SourceBase)

        self.name = name
        self.target_source = target_source.reduce()

        self.preprocess = preprocess or []
        self.process = process or []
        self.loading_process = loading_process or []

    def __getitem__(self, item):
        v = self.target_source[item]
        return self.calculate_value(v)

    def start_data_feed(self):
        need_data_feed = False
        for p in self.process:
            if p.start_data_feed(self) is not False:
                need_data_feed = True
        return need_data_feed

    def finish_data_feed(self):
        for p in self.process:
            p.finish_data_feed(self)

    def data_feed(self, raw_value):
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

    def clone(self, dataset):
        return Field(
            self.name,
            dataset, self.preprocess,
            self.process,
            self.loading_process,
        )
