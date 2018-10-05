import inspect
import pathlib
import pickle
from collections import OrderedDict, Counter

from flowder.utils import map_pipe

from flowder.source import Source
from torchtext.vocab import Vocab
from tqdm import tqdm


class Aggregator:
    def __init__(self, name):
        self.name = name

    def __rrshift__(self, other):
        if isinstance(other, Source):
            self.feed_data(other)
        else:
            raise TypeError("invalid aggregate operation")

    def feed_data(self, data: Source):
        raise NotImplementedError()


class VocabBuilder(Aggregator):

    def __init__(self,
                 name: str,
                 vocab=None,
                 cache="yes",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 additional_special_token=("sos", "eos"),
                 cache_dir=".tmp",
                 caller_file_name=None,
                 **kwargs
                 ):
        """
        SourceからVocabを作成する。
        :param vocab: torchtext.vocab.Vocab。
        :param cache: WordCounterのキャッシュの設定。
        "yes": 作成後にキャッシュし、キャッシュがあればロードする。(default)
        "ignore": キャッシュを無視し、ビルド後にキャッシュを作成(上書き)する。
        "clear": キャッシュがあれば削除し、作成しない。
        "no": キャッシュを作成しない。
        :param unk_token: 未知語を表す特殊文字として使われる。indexは0。必須
        :param pad_token: Noneで無視。paddingに使う。indexは1。必須
        :param additional_special_token: 追加されるトークン。strまたは[str]。indexは2から割り当てられる。
        :param kwargs: 作成されるVocabのコンストラクタに渡される。
        """

        assert cache in ["yes", "ignore", "clear", "no"]
        assert type(unk_token) == str
        assert type(pad_token) == str
        assert isinstance(vocab, Vocab) or vocab is None

        if type(additional_special_token) in [list, tuple]:
            assert all(isinstance(a, str) for a in additional_special_token)
        elif isinstance(additional_special_token, str):
            additional_special_token = [additional_special_token]
        elif additional_special_token is None:
            additional_special_token = []
        else:
            assert False

        super(VocabBuilder, self).__init__(name)
        self.cache = cache
        cache_dir = pathlib.Path(cache_dir)

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.additional_special_token = additional_special_token

        if caller_file_name is None:
            p = pathlib.Path(inspect.currentframe().f_back.f_code.co_filename)
            caller_file_name = p.name[:-len(p.suffix)]

        cache_base_name = f"flowder.{caller_file_name}.VocabBuilder({name})"
        self.cache_file_path = cache_dir / cache_base_name

        self.vocab = vocab
        self.word_counter = Counter()
        self._kwargs = kwargs

        if cache == "yes":
            self.try_load_cache()
        elif cache == "clear":
            self.clear_cache()

    def feed_data(self, data: Source):
        if self.cache == "yes":
            load_success = self.try_load_cache()
            if load_success:
                self.build_vocab()
                return False

        desc = f"flowder.VocabBuilder({self.name}): building..."
        if data.has_length:
            it = tqdm(data, desc=desc)
        else:
            it = tqdm(iter(data), desc=desc)

        for tokenized_sentence in it:
            self.word_counter.update(tokenized_sentence)

        self.build_vocab()
        return True

    def clear_cache(self):
        if self.cache_file_path.exists():
            self.cache_file_path.unlink()

    def numericalize(self, tokenized_sentence):
        return [self.vocab.stoi[word] for word in tokenized_sentence]

    def pipe_numericalize(self):
        @map_pipe()
        def w(tokenized_sentence):
            return self.numericalize(tokenized_sentence)

        return w

    def try_load_cache(self):
        if self.cache_file_path.exists():
            print(f"flowder.VocabBuilder({self.name}): loading from cache...")
            with self.cache_file_path.open("rb") as f:
                self.word_counter = pickle.load(f)
                assert isinstance(self.word_counter, Counter)
            return True
        return False

    def build_vocab(self):
        """
        CounterからVocabを作成する
        cache_fileが指定されていれば、キャッシュを作成する
        :return: None
        """
        specials = list(OrderedDict.fromkeys([
            tok for tok in [
                self.unk_token,
                self.pad_token,
                *self.additional_special_token
            ]
            if tok is not None
        ]))
        self.vocab = Vocab(self.word_counter, specials=specials, **self._kwargs)
        if self.cache == "ignore" or (self.cache == "yes" and not self.cache_file_path.exists()):
            if not self.cache_file_path.parent.exists():
                self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file_path.open("wb") as f:
                pickle.dump(self.word_counter, f)
