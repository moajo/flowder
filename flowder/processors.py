import pickle
from collections import OrderedDict, Counter
from pathlib import Path

from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab


class Processor:
    """
    データ全件をなめる前処理
    複数のFieldで同時に使われることもある。
    """

    def __call__(self, preprocessed_value):
        raise NotImplementedError()

    def start_preprocess_data_feed(self, field):
        """
        data feedの開始前に呼ばれる
        :param field:
        :return: Falseを返すとdata feedを中止する
        """
        return True

    def finish_preprocess_data_feed(self, field):
        """
        data feedの終了後に呼ばれる
        :param field:
        :return:
        """
        pass


class BuildVocab(Processor):
    def __init__(self,
                 unk_token="<unk>",
                 pad_token="<pad>",
                 additional_special_token=None,
                 cache_file=None,
                 vocab=None,
                 **kwargs
                 ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.additional_special_token = additional_special_token or []
        self.cache_file = Path(cache_file) if cache_file is not None else None

        self.vocab = vocab
        self.word_counter = Counter()
        self._kwargs = kwargs

    def __call__(self, tokenized_sentence):
        self.word_counter.update(tokenized_sentence)

    def start_preprocess_data_feed(self, field):
        if self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("rb") as f:
                self.word_counter = pickle.load(f)
            return False

    def finish_preprocess_data_feed(self, field):
        self.build_vocab()

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


class RawProcessor(Processor):

    def __init__(self, target_set):
        super(RawProcessor, self).__init__(left=None, right=None)
        self.target_set = target_set

    def __call__(self, data):
        return data


def tensor_pad_sequence(field_name, include_length, padding_value=1):
    def wrapper(batch):
        if include_length:
            _, indices = batch[field_name][1].sort(descending=True)
            prem = [batch[field_name][0][i][:, None] for i in indices]
            padded = pad_sequence(prem, padding_value=padding_value)
            result = padded[:, indices.sort()[1], 0]
            batch[field_name][0] = result
        else:
            length = [len(a) for a in batch[field_name]]
            _, indices = length.sort(descending=True)
            prem = [batch[field_name][i][:, None] for i in indices]
            padded = pad_sequence(prem, padding_value=padding_value)
            result = padded[:, indices.sort()[1], 0]
            batch[field_name] = result
        return batch

    return wrapper
