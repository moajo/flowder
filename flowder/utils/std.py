#!/usr/bin/env python
import hashlib
import linecache
import pathlib

from flowder.source.base import mapped, Source, ic_from_array, filtered
from flowder.source.iterable_creator import ic_from_iterable, ic_from_generator
from flowder.source.random_access import ra_from_array


def _cal_file_hash(path):
    hs = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(2048 * hs.block_size)
            if len(chunk) == 0:
                break
            hs.update(chunk)

    return hs.hexdigest()


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
    return Source(ic_from_array(array), ra_from_array(array), length=len(array))


def from_items(*items):
    return Source(ic_from_array(items), ra_from_array(items), length=len(items))


def from_iterable(iterable):
    """
    create Source from given iterable.
    note: Source will returned has no length.
    :param iterable:
    :return:
    """
    return Source(ic_from_iterable(iterable), None)


def lines(path):
    """
    return Source object contain lines in the file.
    tailing \n will be removed.
    hash of Source will calculated by sha1 using the file.
    :param path:
    :return:
    """
    path = pathlib.Path(path)
    assert path.exists(), "file not found"

    d = _cal_file_hash(path)

    with path.open(encoding="utf-8") as f:
        length = sum(1 for _ in f)

    def _gen():
        with path.open(encoding="utf-8") as ff:
            for line in ff:
                yield line[:-1]

    def ra(i):
        return linecache.getline(str(path), i + 1)[:-1]

    obs = ic_from_generator(_gen)

    return Source(obs, ra, length=length, dependencies=[d])


def directory(path):
    # TODO calculate hash
    path = pathlib.Path(path)

    def _gen():
        yield from path.iterdir()

    obs = ic_from_generator(_gen)
    return Source(obs)
