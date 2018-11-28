import collections
import math
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate

import warnings
import functools


def _deprecated(func):
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


def sequence_collate(batch):
    """
    シーケンスデータに対応したdefault_collate
    listはシーケンスデータとみなす
    tupleはデータの構造とみなして再帰的に適用する
    :param batch:
    :return:
    """
    if isinstance(batch[0], tuple):
        transposed = zip(*batch)
        vs = [sequence_collate(samples) for samples in transposed]
        return vs
    if isinstance(batch[0], collections.Mapping):
        vs = {
            key: sequence_collate([d[key] for d in batch])
            for key in batch[0]
        }
        return vs
    if isinstance(batch[0], list):
        if isinstance(batch[0][0], int):
            return [torch.LongTensor(data) for data in batch]

        if isinstance(batch[0][0], float):
            return [torch.FloatTensor(data) for data in batch]
    return default_collate(batch)


@_deprecated
def create_iterator(
        dataset,
        batch_size,
        shuffle,
        batch_transforms=(sequence_collate,),
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


@_deprecated
def create_bucket_iterator(
        dataset,
        batch_size,
        sort_key,
        batch_transforms=(sequence_collate,),
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


class _IteratorBase:
    def __init__(self,
                 batch_size: int,
                 num_example: int,
                 prefetch_next_iterator,
                 ):
        assert isinstance(batch_size, int)
        assert isinstance(num_example, int)
        self.batch_size = batch_size
        self.length = math.ceil(num_example / batch_size)
        self.num_example = num_example

        # prefetch iterator(start background loading process)
        self._pre_fetched_next_iter = self._iter() if prefetch_next_iterator else None

    def dispose_prefetch_iterator(self):
        self._pre_fetched_next_iter = None

    def _iter(self):
        raise NotImplementedError()

    def __iter__(self):
        if self._pre_fetched_next_iter is not None:
            ret = self._pre_fetched_next_iter
            self._pre_fetched_next_iter = self._iter()
            return ret
        return self._iter()

    def __len__(self):
        return self.length


class Iterator(_IteratorBase):
    def __init__(self,
                 dataset,
                 batch_size: int,
                 shuffle: bool,
                 batch_transforms=(sequence_collate,),
                 num_workers: int = 1,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 device=None,
                 prefetch_next_iterator: bool = True,
                 ):
        """

        :param dataset:
        :param batch_size:
        :param shuffle:
        :param batch_transforms: list or tuple or callable(batch)
        :param num_workers:
        :param pin_memory:
        :param drop_last:
        :param device:
        :param prefetch_next_iterator:
        """

        if batch_transforms is None:
            batch_transforms = []
        if not isinstance(batch_transforms, list) and not isinstance(batch_transforms, tuple):
            batch_transforms = [batch_transforms]

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
        self.device = device
        super(Iterator, self).__init__(
            batch_size,
            len(dataset),
            prefetch_next_iterator and num_workers != 0
        )

    def _iter(self):
        batch_iterator_iter = iter(self.batch_iterator)
        if self.device is not None:
            p = to_device(self.device)

            def _wrapper():
                for batch in batch_iterator_iter:
                    yield p(batch)

        else:
            def _wrapper():
                yield from batch_iterator_iter
        return _wrapper()


class BucketIterator(_IteratorBase):
    """
    オーバーサンプリングしてそこから切り出すIterator
    """

    def __init__(
            self,
            dataset,
            batch_size: int,
            sort_key,
            batch_transforms=(sequence_collate,),
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            device=None,
            over_sampling_rate=100,
            prefetch_next_iterator=True,
            batch_length_random=True,
    ):
        assert dataset is not None
        assert sort_key is not None
        if batch_transforms is None:
            batch_transforms = []
        if not isinstance(batch_transforms, list) and not isinstance(batch_transforms, tuple):
            batch_transforms = [batch_transforms]

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
        self.device = device

        super(BucketIterator, self).__init__(
            batch_size,
            len(dataset),
            prefetch_next_iterator=prefetch_next_iterator and num_workers != 0
        )

    def _iter(self):
        batch_generator_iterator_iter = iter(self.batch_generator_iterator)
        if self.device is not None:
            p = to_device(self.device)

            def _wrapper():
                for batch_generator in batch_generator_iterator_iter:
                    for batch in batch_generator:
                        yield p(batch)
        else:
            def _wrapper():
                for batch_generator in batch_generator_iterator_iter:
                    yield from batch_generator

        return _wrapper()
