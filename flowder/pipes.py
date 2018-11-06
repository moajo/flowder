#!/usr/bin/env python
from flowder.utils import map_pipe
from flowder.source.base import DependFunc, Mapped, mapped, PipeLine


class _AddToken(DependFunc):
    def __init__(self, token, head: bool):

        def wrapper(data: list):
            assert type(data) == list
            if head:
                return [token] + data
            else:
                return data + [token]

        super(_AddToken, self).__init__(wrapper, [token, head])


def lowercase():
    """
    小文字化
    :return:
    """

    @map_pipe(["lowercase"])
    def wrapper(tokenized):
        if type(tokenized) == str:
            tokenized = [tokenized]
        else:
            assert all(isinstance(a, str) for a in tokenized)
        return [word.lower() for word in tokenized]

    return wrapper


def split(c=None) -> Mapped:
    """
    tokenize
    空トークンはフィルタされる。
    :param c: 分割文字
    :return:
    """

    @map_pipe(["split"])
    def wrapper(s):
        assert type(s) == str
        return [word for word in s.split(c) if word != ""]

    return wrapper


def select(*keys) -> Mapped:
    """
    指定したkeyの値にmapする
    複数のkeyを指定すると、それぞれのkeyをselectしたSourceのtupleを返す。
    :param keys:
    :return:
    """
    assert len(keys) > 0
    if len(keys) == 1:
        key = keys[0]

        @map_pipe([f"select({key})"])
        def wrapper(s):
            return s[key]

        return wrapper
    else:
        def _application(source, key):
            if key is None:
                return tuple(source.map(lambda a, k=k: a[k], dependencies=[f"select({k})"]) for k in keys)
            else:
                raise ValueError("not supported")

        p = PipeLine([], [_application])
        return p


def to_dict(*keys):
    """
    tupleをkeysとzipしてdictに変換する
    :param keys:
    :return:
    """
    assert len(set(keys)) == len(keys), "keys must not contain duplicate items"

    @map_pipe(["to_dict"])
    def wrapper(s):
        assert isinstance(s, tuple), f"streaming item income to 'to_dict' must be tuple, but {type(s)} found"
        assert len(s) == len(keys), f"to_dict: keys must has same length as streaming items, but {len(keys)}!={len(s)}"
        return {
            k: v
            for k, v in zip(keys, s)
        }

    return wrapper


def add_sos(sos_token=2) -> Mapped:
    return mapped(_AddToken(sos_token, head=True), dependencies=["add_sos"])


def add_eos(eos_token=3) -> Mapped:
    return mapped(_AddToken(eos_token, head=False), dependencies=["add_sos"])
