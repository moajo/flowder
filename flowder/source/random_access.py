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


def ra_slice(ra, s: slice):
    step = s.step
    start = s.start
    stop = s.stop

    def _w(i):
        ind = start + step * i
        if ind >= stop:
            raise IndexError("index out of range")
        return ra(ind)
    return _w
