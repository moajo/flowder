#!/usr/bin/env python
from flowder.utils import map_pipe

from flowder.source.base import DependFunc, Mapped, mapped


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

    @map_pipe()
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

    @map_pipe()
    def wrapper(s):
        assert type(s) == str
        return [word for word in s.split(c) if word != ""]

    return wrapper


def select(key) -> Mapped:
    """
    指定したkeyの値にmapする
    :param key:
    :return:
    """

    @map_pipe()
    def wrapper(s):
        return s[key]

    return wrapper


def to_dict(*keys):
    """
    tupleをkeysとzipしてdictに変換する
    :param keys:
    :return:
    """

    @map_pipe()
    def wrapper(s):
        return {
            k: v
            for k, v in zip(keys, s)
        }

    return wrapper


def add_sos(sos_token=2) -> Mapped:
    return mapped(_AddToken(sos_token, head=True))


def add_eos(eos_token=3) -> Mapped:
    return mapped(_AddToken(eos_token, head=False))
