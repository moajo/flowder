import random
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import torch
import math

from abstracts import Example


def convert_data_to_example(data):
    if type(data) is tuple:
        data = OrderedDict([(f"attr{n}", v) for n, v in enumerate(data)])

    if hasattr(data, "items"):
        data = Example(data)
    assert type(data) is Example
    return data


class Batch:
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data, dataset=None, device=None):
        """
        dataとして受け取るのは、
        - Exampleのリスト

        :param data:
        :param dataset:
        :param device:
        """
        self.batch_size = len(data)
        self.dataset = dataset

        assert len(data) != 0

        data = [convert_data_to_example(d) for d in data]

        keys = data[0]._keys
        self.keys = keys

        for k in keys:
            batch = [getattr(x, k) for x in data]
            setattr(self, k, batch)

    # self.fields = dataset.fields.keys()  # copy field names

    # for (name, field) in dataset.fields.items():
    #     if field is not None:
    #         batch = [getattr(x, name) for x in data]
    #         setattr(self, name, field.process(batch, device=device, train=train))

    # @classmethod
    # def fromvars(cls, dataset, batch_size, train=True, **kwargs):
    #     """Create a Batch directly from a number of Variables."""
    #     batch = cls()
    #     batch.batch_size = batch_size
    #     batch.dataset = dataset
    #     batch.train = train
    #     batch.fields = dataset.fields.keys()
    #     for k, v in kwargs.items():
    #         setattr(batch, k, v)
    #     return batch

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(torch.typename(self))

        var_strs = '\n'.join([f'\t[.{name}]:{_short_str(getattr(self, name))}'
                              for name in self.keys if hasattr(self, name)])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name') and
                       isinstance(self.dataset.name, str) else '')

        strt = '[{} of size {}{}]\n{}'.format(torch.typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt

    def __len__(self):
        return self.batch_size


def _short_str(tensor):
    # unwrap variable to tensor
    if not torch.is_tensor(tensor):
        # (1) unpack variable
        if hasattr(tensor, 'data'):
            tensor = getattr(tensor, 'data')
        # (2) handle include_lengths
        elif isinstance(tensor, tuple):
            return str(tuple(_short_str(t) for t in tensor))
        # (3) fallback to default str
        else:
            return str(tensor)

    # copied from torch _tensor_str
    size_str = 'x'.join(str(size) for size in tensor.size())
    device_str = '' if not tensor.is_cuda else \
        ' (GPU {})'.format(tensor.get_device())
    strt = '[{} of size {}{}]'.format(torch.typename(tensor),
                                      size_str, device_str)
    return strt


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))


class Iterator:
    """
    データセットを学習イテレーションに供給するイテレータ
    バッチ変換、パディング、シャッフル、テンソルへの変換などを行う
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None,
                 batch_constructor=Batch,
                 batch_transforms=None):
        """
        datasetはランダムアクセスできれば何でもいい。
        その値はExampleかtupleかdict
        tupleは名前を自動的につけてdictに
        dictはexampleに変換される


        :param dataset:
        :param batch_size:
        :param sort_key:
        :param device:
        :param batch_size_fn:
        :param train:
        :param repeat:
        :param shuffle:
        :param sort:
        :param sort_within_batch:
        """
        self.batch_constructor = batch_constructor
        self.batch_size, self.dataset = batch_size, dataset
        self.batch_size_fn = batch_size_fn
        self.iterations = 0
        self.repeat = repeat
        self.shuffle = shuffle
        self.sort = sort
        self.batch_transforms = batch_transforms or []
        if sort_within_batch is None:
            self.sort_within_batch = self.sort
        else:
            self.sort_within_batch = sort_within_batch
        self.sort_key = sort_key
        self.device = device
        if not torch.cuda.is_available() and self.device is None:
            self.device = -1

        self.random_shuffler = RandomShuffler()

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False
        self.batches_generator = None

    def batch_transform(self, mini_batch):
        for t in self.batch_transforms:
            mini_batch = t(mini_batch)
        return mini_batch

    def get_data_order(self):
        if self.sort:
            return [
                i for i, v in
                sorted(enumerate(self.dataset), key=lambda i, v: self.sort_key(v))
            ]
        elif self.shuffle:
            return [i for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            return list(range(len(self.dataset)))

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""

        if self._restored_from_state:
            self.random_shuffler.random_state = self._random_state_this_epoch
        else:
            self._random_state_this_epoch = self.random_shuffler.random_state

        self.batches_generator = self.create_batches()

        if self._restored_from_state:
            self._restored_from_state = False
        else:
            self._iterations_this_epoch = 0

        if not self.repeat:
            self.iterations = 0

    def create_batches(self):
        return batch(self.get_data_order(), self.dataset, self.batch_size, self.batch_size_fn)

    @property
    def epoch(self):
        return math.floor(self.iterations / len(self))

    def __len__(self):
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        B = self.batch_constructor
        while True:
            self.init_epoch()
            for idx, mini_batch in enumerate(self.batches_generator):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        mini_batch.reverse()  # because already sorted
                    else:
                        mini_batch.sort(key=self.sort_key, reverse=True)
                transformed = self.batch_transform(mini_batch)
                yield B(transformed, self.dataset, self.device)
            if not self.repeat:
                return

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "iterations_this_epoch": self._iterations_this_epoch,
            "random_state_this_epoch": self._random_state_this_epoch}

    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self._iterations_this_epoch = state_dict["iterations_this_epoch"]
        self._random_state_this_epoch = state_dict["random_state_this_epoch"]
        self._restored_from_state = True


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            return super(BucketIterator, self).create_batches()
        else:
            return pool(self.get_data_order(), self.dataset, self.batch_size,
                        self.sort_key, self.batch_size_fn,
                        random_shuffler=self.random_shuffler,
                        shuffle=self.shuffle,
                        sort_within_batch=self.sort_within_batch)


def batch(data_index, dataset, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for i in data_index:
        ex = dataset[i]
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data_index, dataset, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data_index, dataset, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b
