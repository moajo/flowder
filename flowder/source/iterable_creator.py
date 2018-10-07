#!/usr/bin/env python


from typing import Callable, Iterable

"""
指定したindexからのiterationをサポート
"""
IterableCreator = Callable[[int], Iterable]


def ic_from_array(array) -> IterableCreator:
    """
    arrayからの変換
    :param array:
    :return:
    """

    def _w(start: int):
        yield from array[start:]

    return _w


def ic_from_iterable(iterable):
    """
    iterableからの変換
    :param iterable:
    :return:
    """

    def _w(start: int):
        for item in iterable:
            if start > 0:
                start -= 1
                continue
            yield item

    return _w


def ic_from_generator(gen_func):
    """
    generator functionからの変換
    :param gen_func:
    :return:
    """

    def _w(start: int):
        for item in gen_func():
            if start > 0:
                start -= 1
                continue
            yield item

    return _w


def ic_map(ic, transform):
    def _w(start):
        for item in ic(start):
            yield transform(item)

    return _w


def ic_filter(ic, pred):
    def gen():
        for item in ic(0):
            if pred(item):
                yield item

    return ic_from_generator(gen)


def ic_zip(*ic):
    def _w(start: int):
        yield from zip(*[a(start) for a in ic])

    return _w


def ic_concat(*ic):
    def _w(start: int):
        for a in ic:
            for item in a(0):
                if start > 0:
                    start -= 1
                    continue
                yield item

    return _w


def ic_flat_map(ic, converter):
    def gen():
        for seq in ic(0):
            for item in converter(seq):
                yield item

    return ic_from_generator(gen)


def ic_slice(ic: IterableCreator, s: slice) -> IterableCreator:
    """
    slice icの構成
    :param ic:
    :param s:
    :return:
    """
    slice_step = s.step if s.step is not None else 1
    slice_start = s.start if s.start is not None else 0
    if s.stop is None:
        def _w(start):
            ds = slice_start + slice_step * start
            yield from ic(ds)

        return _w
    else:
        def _w(start):
            ds = slice_start + slice_step * start
            c = (s.stop - 1 - ds) // slice_step + 1
            for i, item in enumerate(ic(ds)):
                if i % slice_step != 0:
                    continue
                if c <= 0:
                    break
                yield item
                c -= 1

        return _w
