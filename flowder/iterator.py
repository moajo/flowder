import math
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate

import warnings
import functools


def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def to_device(device):
    def wrapper(batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return tuple(wrapper(b) for b in batch)
        if isinstance(batch, dict):
            return {key: wrapper(batch[key]) for key in batch}
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device=device, non_blocking=True)
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


@deprecated
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
    print("warning: use deprecated api")
    return Iterator(
        dataset,
        batch_size,
        shuffle,
        batch_transforms=batch_transforms,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        device=device,
        prefetch_next_iterator=prefetch_next_iterator,
    )


@deprecated
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
        batch_length_random=True,
):
    print("warning: use deprecated api")
    return BucketIterator(
        dataset,
        batch_size,
        sort_key,
        batch_transforms=batch_transforms,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        device=device,
        over_sampling_rate=over_sampling_rate,
        prefetch_next_iterator=prefetch_next_iterator,
        batch_length_random=batch_length_random,
    )


class Iterator:
    def __init__(self,
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

        batch_iterator = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.batch_iterator = batch_iterator
        self.num_example = len(dataset)
        self.batch_size = batch_size
        self.device = device

        # prefetch iterator(start background loading process)
        self._pre_fetched_next_iter = self._iter() if prefetch_next_iterator and num_workers != 0 else None

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
        if self._pre_fetched_next_iter is not None:
            ret = self._pre_fetched_next_iter
            self._pre_fetched_next_iter = self._iter()
            return ret
        return self._iter()

    def __len__(self):
        return math.ceil(self.num_example / self.batch_size)


class BucketIterator:
    """
    オーバーサンプリングしてそこから切り出すIterator
    """

    def __init__(
            self,
            dataset,
            batch_size: int,
            sort_key,
            batch_transforms=(default_sequence_collate,),
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            device=None,
            over_sampling_rate=100,
            prefetch_next_iterator=True,
            batch_length_random=True,
    ):
        assert dataset is not None
        assert isinstance(batch_size, int)
        assert sort_key is not None
        if batch_transforms is None:
            batch_transforms = []
        assert hasattr(batch_transforms, '__len__')
        batch_transforms = batch_transforms

        def collate(over_batch):  # in: 100倍バッチのexample list
            try:
                sorted_over_batch = sorted(over_batch, key=sort_key)
            except KeyError:
                raise KeyError("Failed to sort batch: is sort_key correct?")

            index_list = range(0, len(sorted_over_batch), batch_size)
            if batch_length_random:
                index_list = np.random.permutation(index_list)

            def transform(b, transforms):
                for t in transforms:
                    b = t(b)
                return b

            return [
                transform(
                    sorted_over_batch[i:i + batch_size],
                    batch_transforms,
                )
                for i in index_list
            ]

        self.batch_generator_iterator = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size * over_sampling_rate,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,  # raw example list
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.length = math.ceil(len(dataset) / batch_size)
        self.device = device

        # prefetch iterator(start background loading process)
        self._pre_fetched_next_iter = self._iter() if prefetch_next_iterator and num_workers != 0 else None

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
        if self._pre_fetched_next_iter is not None:
            ret = self._pre_fetched_next_iter
            self._pre_fetched_next_iter = self._iter()
            return ret
        return self._iter()

    def __len__(self):
        return self.length
