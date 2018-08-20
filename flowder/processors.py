import pickle
from collections import OrderedDict, Counter
from pathlib import Path

from torchtext.vocab import Vocab


class AggregateProcessor:
    """
    データ全体の統計をとる前処理
    複数のFieldで同時に使われることもある。
    Field.aggregate_preprocessで
    """

    def start_data_feed(self, field):
        """
        data feedの開始前に呼ばれる
        :param field:
        :return: Falseを返すとdata feedを中止する
        """
        return True

    def data_feed(self, item):
        """
        data feed。データ全件が一つづつ流れてくる
        :param item:
        :return:
        """
        raise NotImplementedError()

    def finish_data_feed(self, field):
        """
        data feedの終了後に呼ばれる
        :param field:
        :return:
        """
        pass

    def __call__(self, preprocessed_value):
        """
        実際の処理
        :param preprocessed_value:
        :return:
        """
        raise NotImplementedError()


class BuildVocab(AggregateProcessor):
    def __init__(self,
                 unk_token="<unk>",
                 pad_token="<pad>",
                 additional_special_token=None,
                 cache_file=None,
                 vocab=None,
                 auto_numericalize=False,
                 **kwargs
                 ):
        """

        :param unk_token: 未知語を表す特殊文字として使われる。必須
        :param pad_token: Noneで無視。paddingに使う。Noneでなければindexは1
        :param additional_special_token: 追加されるトークン。strまたは[str]。indexは2から割り当てられる。
        :param cache_file: 設定すればword_counterをファイルにキャッシュする。
        :param vocab: torchtext.vocab.Vocab
        :param kwargs: 作成されるVocabのコンストラクタに渡される。
        """
        assert unk_token is not None
        if isinstance(additional_special_token, list):
            assert all(isinstance(a, str) for a in additional_special_token)
        elif isinstance(additional_special_token, str):
            additional_special_token = [additional_special_token]
        elif additional_special_token is None:
            additional_special_token = []
        else:
            assert False

        assert isinstance(vocab, Vocab) or vocab is None

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.additional_special_token = additional_special_token
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.auto_numericalize = auto_numericalize

        self.vocab = vocab
        self.word_counter = Counter()
        self._kwargs = kwargs

    def clear_cache(self):
        if self.cache_file is not None and self.cache_file.exists():
            self.cache_file.unlink()

    def data_feed(self, tokenized_sentence):
        self.word_counter.update(tokenized_sentence)

    def start_data_feed(self, field):
        load_success = self.load_cache_if_exists()
        if load_success:
            return False

    def finish_data_feed(self, field):
        self.build_vocab()

    def numericalize(self, tokenized_sentence):
        return [self.vocab.stoi[word] for word in tokenized_sentence]

    def __call__(self, tokenized_sentence):
        if self.auto_numericalize:
            return self.numericalize(tokenized_sentence)
        return tokenized_sentence

    def load_cache_if_exists(self):
        if self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("rb") as f:
                self.word_counter = pickle.load(f)
                assert isinstance(self.word_counter, Counter)
            return True
        return False

    def build_vocab(self):
        specials = list(OrderedDict.fromkeys([
            tok for tok in [
                self.unk_token,
                self.pad_token,
                *self.additional_special_token
            ]
            if tok is not None
        ]))
        self.vocab = Vocab(self.word_counter, specials=specials, **self._kwargs)
        if self.cache_file is not None and not self.cache_file.exists():
            if not self.cache_file.parent.exists():
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("wb") as f:
                pickle.dump(self.word_counter, f)
            return False

    def build_from_sources(self, *sources):
        if self.load_cache_if_exists():
            return
        for s in sources:
            for d in s:
                self.data_feed(d)
        self.build_vocab()
