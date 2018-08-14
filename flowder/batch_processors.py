import torch
from torch.utils.data.dataloader import default_collate

from flowder import default_sequence_collate
from torch.nn.utils.rnn import pad_sequence


def default_create_batch():
    """
    exampleのlistをキーごとに転置してバッチを作成する。
    数値のリストはtensorに変換する
    """

    def wrapper(batch):
        return default_sequence_collate(batch)

    return wrapper


def sort(sort_key):
    def wrapper(batch):
        try:
            return sorted(batch, key=sort_key)
        except KeyError:
            raise KeyError("Failed to sort batch: is sort_key correct?")

    return wrapper


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


def tensor_pad_sequence(field_names, include_length, padding_value=1):
    if isinstance(field_names, tuple):
        assert all(isinstance(a, str) for a in field_names)
    if isinstance(field_names, str):
        field_names = (field_names,)
    assert isinstance(field_names, tuple)

    def wrapper(batch):
        for field_name in field_names:
            if include_length:
                _, indices = batch[field_name][1].sort(descending=True)
                prem = [batch[field_name][0][i][:, None] for i in indices]
                padded = pad_sequence(prem, padding_value=padding_value)
                result = padded[:, indices.sort()[1], 0]
                batch[field_name] = result, batch[field_name][1]
            else:
                length = [len(a) for a in batch[field_name]]
                _, indices = length.sort(descending=True)
                prem = [batch[field_name][i][:, None] for i in indices]
                padded = pad_sequence(prem, padding_value=padding_value)
                result = padded[:, indices.sort()[1], 0]
                batch[field_name] = result
        return batch

    return wrapper
