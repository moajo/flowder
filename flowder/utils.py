#!/usr/bin/env python

from flowder.source.base import DependFunc, mapped, Mapped, Source, ic_from_array, ic_from_iterable, filtered


def map_pipe(dependencies=None):
    def wrapper(f):
        return mapped(f, dependencies)

    return wrapper


def filter_pipe(dependencies=None):
    def wrapper(f):
        return filtered(f, dependencies)

    return wrapper


def from_array(array):
    assert type(array) in [tuple, list]
    return Source(ic_from_array(array), length=len(array))


def from_items(*items):
    return Source(ic_from_array(items), length=len(items))


def from_iterable(iterable):
    return Source(ic_from_iterable(iterable))
