import math
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


def to_device(device):
    def wrapper(batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return tuple(wrapper(b) for b in batch)
        if isinstance(batch, dict):
            return {key: wrapper(batch[key]) for key in batch}
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    return wrapper


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
        batch_transforms=(default_sequence_collate,),
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        device=None,
        prefetch_next_iterator=True,
):
    batch_transforms = batch_transforms or []

    def collate(batch):
        for t in batch_transforms:
            batch = t(batch)
        return batch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return Iterator(
        loader,
        len(dataset),
        batch_size,
        device=device,
        prefetch_next_iterator=prefetch_next_iterator and num_workers != 0,
    )


def create_bucket_iterator(
        dataset,
        batch_size,
        sort_key,
        batch_transforms=(default_sequence_collate,),
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        device=None,
        over_sampling_rate=100,
        prefetch_next_iterator=True,
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
            for t in batch_transforms:
                batch = t(batch)
            bbb.append(batch)
        return bbb

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size * over_sampling_rate,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,  # raw example list
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    l = math.ceil(len(dataset) / batch_size)
    return BucketIterator(
        loader,
        l,
        device=device,
        prefetch_next_iterator=prefetch_next_iterator and num_workers != 0,
    )


class Iterator:
    def __init__(self,
                 batch_iterator,
                 num_example,
                 batch_size,
                 device=None,
                 prefetch_next_iterator=True,
                 ):
        """

        :param batch_iterator:
        :param num_example:
        :param batch_size:
        :param device:
        """
        self.batch_iterator = batch_iterator
        self.num_example = num_example
        self.batch_size = batch_size
        self.device = device

        # prefetch iterator(start background loading process)
        self._next_iter = self._iter() if prefetch_next_iterator else None

    def _iter(self):
        if self.device is not None:
            p = to_device(self.device)
        else:
            p = lambda a: a

        batch_iterator_iter = iter(self.batch_iterator)

        def _wrapper():
            for batch in batch_iterator_iter:
                batch = p(batch)
                yield batch

        return _wrapper()

    def __iter__(self):
        if self._next_iter is not None:
            ret = self._next_iter
            self._next_iter = self._iter()
            return ret
        return self._iter()

    def __len__(self):
        return math.ceil(self.num_example / self.batch_size)


class BucketIterator:
    def __init__(self, batch_generator_iterator, length, device=None, prefetch_next_iterator=True):
        self.batch_generator_iterator = batch_generator_iterator
        self.length = length
        self.device = device

        # prefetch iterator(start background loading process)
        self._next_iter = self._iter() if prefetch_next_iterator else None

    def _iter(self):
        if self.device is not None:
            p = to_device(self.device)
        else:
            p = lambda a: a

        batch_generator_iterator_iter = iter(self.batch_generator_iterator)

        def _wrapper():
            for batch_generator in batch_generator_iterator_iter:
                for batch in batch_generator:
                    batch = p(batch)
                    yield batch

        return _wrapper()

    def __iter__(self):
        if self._next_iter is not None:
            ret = self._next_iter
            self._next_iter = self._iter()
            return ret
        return self._iter()

    def __len__(self):
        return self.length
