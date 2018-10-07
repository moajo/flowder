#!/usr/bin/env python
import hashlib
import pathlib

from flowder.source.base import DependFunc, mapped, Mapped, Source, ic_from_array, filtered
from flowder.source.iterable_creator import ic_from_iterable, ic_from_generator


def map_pipe(dependencies=None):
    """
    Map Pipeに変換するdecorator
    :param dependencies:
    :return:
    """
    def wrapper(f):
        return mapped(f, dependencies)

    return wrapper


def filter_pipe(dependencies=None):
    """
    Map Pipeに変換するdecorator
    :param dependencies:
    :return:
    """
    def wrapper(f):
        return filtered(f, dependencies)

    return wrapper


def from_array(array):
    """
    create Source from list or tuple
    :param array:
    :return:
    """
    assert type(array) in [tuple, list]
    return Source(ic_from_array(array), length=len(array))


def from_items(*items):
    return Source(ic_from_array(items), length=len(items))


def from_iterable(iterable):
    """
    create Source from given iterable.
    note: Source will returned has no length.
    :param iterable:
    :return:
    """
    return Source(ic_from_iterable(iterable))


def lines(path):
    """
    return Source object contain lines in the file.
    tailing \n will be removed.
    hash of Source will calculated by sha1 using the file.
    :param path:
    :return:
    """
    path = pathlib.Path(path)
    assert path.exists()

    hash = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(2048 * hash.block_size)
            if len(chunk) == 0:
                break
            hash.update(chunk)

    d = hash.hexdigest()

    with path.open(encoding="utf-8") as f:
        length = sum(1 for _ in f)

    def _gen():
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield line[:-1]

    obs = ic_from_generator(_gen)

    return Source(obs, length=length, dependencies=[d])
