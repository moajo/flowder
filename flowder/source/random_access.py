#!/usr/bin/env python
from typing import TypeVar, Callable

T = TypeVar('T')
RandomAccessor = Callable[[int], T]


def ra_from_array(array):
    def _w(i):
        return array[i]

    return _w


def ra_concat(ra1, ra2, ra1_length):
    def _w(i):
        if i < ra1_length:
            return ra1(i)
        else:
            return ra2(i - ra1_length)

    return _w


def ra_zip(*ras):
    def _w(i):
        return tuple(ra(i) for ra in ras)

    return _w


def ra_map(ra1, transform):
    def _w(i):
        return transform(ra1(i))

    return _w

# def random_access_from_iterable(iterable):
#     ic = ic_from_iterable(iterable)
#     return random_access_from_ic(ic)
#
#
# def random_access_from_ic(ic):
#     def _w(i):
#         return next(iter(ic(i)))
#
#     return _w
