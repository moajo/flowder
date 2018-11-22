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
    :param field_names: (str,intまたはそれらのtuple)またはそのtuple
    strのtupleの場合、複数回のキーアクセスを示す。
    ex: ("en", 0)  ->  batch["en"][0]
    :param include_length: 長さ情報を残すか
    :param batch_first:
    :param padding_value:
    :return:
    """
    if type(field_names) in [tuple, list]:
        assert all(type(a) in [str, int, tuple] for a in field_names)
    else:
        field_names = [field_names]
    assert type(field_names) in [tuple, list]

    def get(b, f):
        if type(f) != tuple:
            return b[f]
        res = b
        for ff in f:
            res = res[ff]
        return res

    def set(b, f, value):
        if type(f) != tuple:
            b[f] = value
            return
        res = b
        for ff in f[:-1]:
            res = res[ff]
        res[f[-1]] = value

    @pipe
    def wrapper(batch):
        for field_name in field_names:
            v = get(batch, field_name)
            result = pad_sequence(v, padding_value=padding_value, batch_first=batch_first)
            if include_length:
                length = torch.LongTensor([len(a) for a in v]).contiguous()
                new_value = result, length
            else:
                new_value = result
            set(batch, field_name, new_value)
        return batch

    return wrapper
