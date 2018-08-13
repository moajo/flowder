import random
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


# from sources import Example


def convert_data_to_example(data):
    """

    :param data: tuple,dict or Example
    :return:
    """
    if isinstance(data, tuple):
        # data = OrderedDict([(f"attr{n}", v) for n, v in enumerate(data)])
        data = {f"attr{n}": v for n, v in enumerate(data)}

    # if hasattr(data, "items"):
    #     data = Example(data)
    # assert isinstance(data, Example)
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


def default_sequence_collate(batch):
    """
    シーケンスデータに対応したdefault_collate
    listはシーケンスデータとみなす
    tupleはデータの構造とみなして再帰的に適用する
    :param batch:
    :return:
    """
    if isinstance(batch[0], tuple):
        transposed = zip(*batch)
        vs = [default_sequence_collate(samples) for samples in transposed]
        return vs
    if isinstance(batch[0], list):
        if isinstance(batch[0][0], int):
            return [torch.LongTensor(data) for data in batch]

        if isinstance(batch[0][0], float):
            return [torch.FloatTensor(data) for data in batch]
    vs = default_collate(batch)
    return vs


def create_bucket_iterator(
        dataset,
        batch_size,
        sort_key,
        sampler=None,
        num_workers=0,
        collate_fn=default_sequence_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        over_sampling_rate=100,
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size * over_sampling_rate,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
    )
    return BucketIterator(loader, batch_size, sort_key, over_sampling_rate=over_sampling_rate, collate_fn=collate_fn)


def data_to_device(data, device):
    if isinstance(data, tuple):
        return tuple(data_to_device(b, device) for b in data)
    if isinstance(data, dict):
        return {key: data_to_device(data[key], device) for key in data}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


class Iterator:
    def __init__(self, batch_iterator, sort_key_within_batch, collate_fn=default_sequence_collate, device=None):
        self.batch_iterator = batch_iterator
        self.sort_key_within_batch = sort_key_within_batch
        self.collate_fn = collate_fn
        self.device = device

    def __iter__(self):
        sort_key = self.sort_key_within_batch
        device = self.device
        collate_fn = self.collate_fn
        if sort_key is not None:
            for batch in self.batch_iterator:
                sorted_batch = sorted(batch, key=sort_key)
                if device is not None:
                    raw = data_to_device(sorted_batch, device=device)
                else:
                    raw = sorted_batch
                yield collate_fn(raw)
            else:
                for batch in self.batch_iterator:
                    if device is not None:
                        raw = data_to_device(batch, device=device)
                    else:
                        raw = batch
                    yield collate_fn(raw)


class BucketIterator:
    def __init__(self, batch_iterator, batch_size, sort_key, over_sampling_rate,
                 collate_fn=default_sequence_collate, device=None):
        self.batch_iterator = batch_iterator
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.over_sampling_rate = over_sampling_rate
        self.collate_fn = collate_fn
        self.device = device

    def __iter__(self):
        batch_size = self.batch_size
        device = self.device
        collate_fn = self.collate_fn
        for over_batch in self.batch_iterator:
            try:
                sorted_over_batch = sorted(over_batch, key=self.sort_key)
            except KeyError:
                raise KeyError("Failed to sort batch: is sort_key correct?")
            for i in np.random.permutation(range(0, len(sorted_over_batch), batch_size)):
                raw = sorted_over_batch[i:i + batch_size]
                if device is not None:
                    raw = data_to_device(raw, device)
                yield collate_fn(raw)
