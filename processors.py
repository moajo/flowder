from collections import OrderedDict, Counter

from torchtext.vocab import Vocab


class Processor:
    """
    前の要素から値を受け取って値を帰す関数。Fieldはこれを使って処理する
    """

    #
    # def __init__(self, left, right):
    #     assert left is Processor or left is None
    #     assert right is Processor or right is None
    #     self.left = left
    #     self.right = right

    # def get_root(self):
    #     if self.left is None:
    #         return self
    #     return self.left.get_root()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def start_preprocess_data_feed(self, field):
        pass

    def finish_preprocess_data_feed(self, field):
        pass

    # def pipe(self, parent_processor):
    #     pass  # TODO
    #
    #     # def get_value(self, i):
    #     raise NotImplementedError()

    # def calculate_value(self, parent_value):
    #     raise NotImplementedError()


class BuildVocab(Processor):
    def __init__(self,
                 unk_token="<unk>",
                 pad_token="<pad>",
                 additional_special_token=None,
                 **kwargs
                 ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.additional_special_token = additional_special_token or []

        self.vocab = None
        self.word_counter = Counter()
        self._kwargs = kwargs

    def __call__(self, tokenized_sentence):
        for word in tokenized_sentence:
            self.word_counter.update(word)

    def start_preprocess_data_feed(self, field):
        pass

    def finish_preprocess_data_feed(self, field):
        self.build_vocab()

    def build_vocab(self):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
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


class RawProcessor(Processor):

    def __init__(self, target_set):
        super(RawProcessor, self).__init__(left=None, right=None)
        self.target_set = target_set

    def __call__(self, data):
        return data
