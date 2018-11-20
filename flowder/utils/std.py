#!/usr/bin/env python
import glob as glob_
import hashlib
import linecache
import pathlib
import gzip

from flowder.hash import default_hash_func
from flowder.source.base import mapped, Source, ic_from_array, filtered, FlatMapped
from flowder.source.iterable_creator import ic_from_iterable, ic_from_generator, ic_map
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


def from_array(array, *, hash_func=None):
    """
    create Source from list or tuple
    :param array:
    :param hash_func:
    :return:
    """
    assert type(array) in [tuple, list]
    if hash_func is not None:
        hs = hash_func(array)
        return Source(ic_from_array(array), ra_from_array(array),
                      length=len(array),
                      dependencies=[hs])
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


@map_pipe()
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


def lines_gzip(path):
    path = pathlib.Path(path)
    assert path.exists(), "file not found"

    d = _cal_file_hash(path)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        length = sum(1 for _ in f)

    def _gen():
        with gzip.open(path, "rt", encoding="utf-8") as ff:
            for line in ff:
                yield line[:-1]

    obs = ic_from_generator(_gen)

    return Source(obs, length=length, dependencies=[d])


def directory(path):
    path = pathlib.Path(path)
    files = list(path.iterdir())
    obs = ic_from_array(files)
    ra = ra_from_array(files)
    hs = default_hash_func([str(a) for a in files])
    return Source(obs, random_accessor=ra, length=len(files), dependencies=[hs])


def glob(glob_path: str):
    """
    files = glob("./*.jpg")
    :param glob_path:
    :return:
    """
    files = [pathlib.Path(a) for a in glob_.glob(glob_path)]
    obs = ic_from_array(files)
    ra = ra_from_array(files)
    hs = default_hash_func([str(a) for a in files])
    return Source(obs, random_accessor=ra, length=len(files), dependencies=[hs])


flatten: FlatMapped = FlatMapped(lambda a: a, dependencies=[])
