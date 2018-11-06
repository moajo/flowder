import torch

from torch.nn.utils.rnn import pad_sequence
from flowder.iterator import sequence_collate
from flowder.source.base import PipeFunc


def pipe(func) -> PipeFunc:
    """
    funcをPipeFuncにラップするDecorator
    :param func:
    :return:
    """
    return PipeFunc([func])


def collate() -> PipeFunc:
    """
    数値のリストはtensorに変換する
    exampleのlistをキーごとに転置する
    """

    @pipe
    def wrapper(batch):
        return sequence_collate(batch)

    return wrapper


def sort(sort_key) -> PipeFunc:
    @pipe
    def wrapper(batch):
        try:
            return sorted(batch, key=sort_key)
        except KeyError:
            raise KeyError("Failed to sort batch: is sort_key correct?")

    return wrapper


def tensor_pad_sequence(field_names, include_length=True, batch_first=False, padding_value=1) -> PipeFunc:
    """
    可変長シーケンス列をpaddingする。
    対象はtensorのlist。各tensorはshape[0]を長さとする。
    同時にもともとの長さを1次元tensorとして作成し、元のフィールドを長さとのtupleに置き換える
    :param field_names: strまたはstrのtuple
    :param include_length: 長さ情報を残すか
    :param batch_first:
    :param padding_value:
    :return:
    """
    if isinstance(field_names, tuple):
        assert all(isinstance(a, str) for a in field_names)
    if isinstance(field_names, str):
        field_names = (field_names,)
    assert isinstance(field_names, tuple)

    @pipe
    def wrapper(batch):
        for field_name in field_names:
            result = pad_sequence(batch[field_name], padding_value=padding_value, batch_first=batch_first)
            if include_length:
                length = torch.LongTensor([len(a) for a in batch[field_name]]).contiguous()
                batch[field_name] = result, length
            else:
                batch[field_name] = result
        return batch

    return wrapper
