#!/usr/bin/env python

class DependFunc:
    """
    dependenciesを埋め込んだfunction
    """

    def __init__(self, func, dependencies):
        self.func = func
        self.dependencies = dependencies
        assert type(dependencies) == list

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def depend(*dependencies):
    """
    funcにdependenciesを埋め込むdecorator
    :param dependencies:
    :return:
    """

    def wrapper(f):
        return DependFunc(f, list(dependencies))

    return wrapper
