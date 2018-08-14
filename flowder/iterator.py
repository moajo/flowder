import math
import random
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate

# from sources import Example
from tqdm import tqdm

from flowder.dataloader import IterDataLoader
from flowder.thread_dl import DataLoader


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
    if isinstance(batch[0], dict):
        vs = {
            key: default_sequence_collate([d[key] for d in batch])
            for key in batch[0]
        }
        return vs
    if isinstance(batch[0], list):
        if isinstance(batch[0][0], int):
            return [torch.LongTensor(data) for data in batch]

        if isinstance(batch[0][0], float):
            return [torch.FloatTensor(data) for data in batch]
    vs = default_collate(batch)
    return vs


def create_iterator(
        dataset,
        batch_size,
        shuffle,
        sort_key_within_batch,
        collate_fn=default_sequence_collate,
        device=None,
        batch_transforms=None,
        sampler=None,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
):
    batch_transforms = batch_transforms or []

    def collate(batch):
        if sort_key_within_batch is not None:
            batch = sorted(batch, key=sort_key_within_batch)
        batch = collate_fn(batch)
        for t in batch_transforms:
            batch = t(batch)
        # if device is not None:
        #     batch = data_to_device(batch, device=device)  # transfortmと先のほうがいい？
        return batch

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
    )
    return Iterator(
        loader,
        len(dataset),
        batch_size,
        device=device)


def create_bucket_iterator(
        dataset,
        batch_size,
        sort_key,
        device=None,
        batch_transforms=None,
        sampler=None,
        num_workers=1,
        collate_fn=default_sequence_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        over_sampling_rate=100,
):
    batch_transforms = batch_transforms or []

    def collate(over_batch):  # in: 100倍バッチのexample list
        try:
            sorted_over_batch = sorted(over_batch, key=sort_key)
        except KeyError:
            raise KeyError("Failed to sort batch: is sort_key correct?")

        bbb = []
        for i in np.random.permutation(range(0, len(sorted_over_batch), batch_size)):
            batch = sorted_over_batch[i:i + batch_size]
            batch = collate_fn(batch)
            for t in batch_transforms:
                batch = t(batch)
            if device is not None:
                batch = data_to_device(batch, device=device)  # transfortmと先のほうがいい？
            # return batch
            # if device is not None:
            #     batch = data_to_device(batch, device)
            # for t in self.batch_transforms:
            #     batch = t(batch)
            bbb.append(batch)
        return bbb

    loader = DataLoader(
        dataset,
        batch_size=batch_size * over_sampling_rate,
        shuffle=True,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # raw example list
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
    )
    iter_loader = IterDataLoader(
        loader,
        num_workers=num_workers,
        collate_fn=collate,  # split example list
        pin_memory=pin_memory,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
    )
    l = math.ceil(len(dataset) / batch_size)
    return BucketIterator(iter_loader, l, device)


def data_to_device(data, device):
    if isinstance(data, tuple) or isinstance(data, list):
        return tuple(data_to_device(b, device) for b in data)
    if isinstance(data, dict):
        return {key: data_to_device(data[key], device) for key in data}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


class Iterator:
    def __init__(self,
                 batch_iterator,
                 num_example,
                 batch_size,
                 device=None):
        """

        :param batch_iterator: Dataset
        :param sort_key_within_batch:
        :param device:
        """
        self.batch_iterator = batch_iterator
        self.num_example = num_example
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        device = self.device
        for b in self.batch_iterator:
            if device is not None:
                b = data_to_device(b, device=device)  # transfortmと先のほうがいい？
            yield b

    def __len__(self):
        return math.ceil(self.num_example / self.batch_size)


class BucketIterator:
    def __init__(self,
                 batch_generator_iterator,
                 length,
                 device=None):
        self.batch_generator_iterator = batch_generator_iterator
        self.length = length
        self.device = device

    def __iter__(self):
        for batch_generator in self.batch_generator_iterator:
            for batch in batch_generator:
                yield batch

    def __len__(self):
        return self.length
