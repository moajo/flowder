#!/usr/bin/env python
from collections import deque
from flowder.source.base import mapped, Source, ic_from_array, filtered, _calc_args_hash, PipeLine
from flowder.source.iterable_creator import ic_from_iterable, ic_from_generator
from flowder.source.random_access import ra_from_array


class Window(PipeLine):
    def __init__(self, window_size, drop_first):
        def _application(source, key):
            assert isinstance(source, Source)
            assert key is None, "Window with key is not supported"
            return _window(source, window_size, drop_first)

        super(Window, self).__init__([_application])

    def __call__(self, *args, **kwargs):
        raise TypeError("Window never called")


def windowed(window_size, drop_first) -> Window:
    return Window(window_size, drop_first)


def _window(source: Source, window_size: int, drop_first: bool = True):
    def ic_window(ic):
        def _w(start: int):
            last_n = deque()

            if drop_first:
                for _, v in zip(range(window_size - 1), ic(start)):
                    last_n.append(v)
                start += window_size - 1
            else:
                for _ in range(window_size - 1):
                    last_n.append(None)
                st = max(0, start - window_size + 1)
                for _, v in zip(range(st, start), ic(st)):
                    last_n.append(v)

            while len(last_n) >= window_size:
                last_n.popleft()

            for a in ic(start):
                if len(last_n) != window_size - 1:
                    last_n.append(a)
                    continue
                yield tuple(last_n) + (a,)
                last_n.append(a)
                last_n.popleft()

        return _w

    def ra_window(ra, ra1_length):
        def _w(i):
            if drop_first:
                head = i
                i += window_size - 1
                ids = list(range(head, i + 1))
            else:
                head = max(0, i - window_size + 1)
                ids = list(range(head, i + 1))
                if len(ids) < window_size:
                    ids = [None for _ in range(window_size - len(ids))] + ids
            assert 0 <= i < ra1_length
            return tuple(ra(ii) if ii is not None else None for ii in ids)

        return _w

    dependencies = ["window"]
    return Source(
        ic_window(source._raw),
        random_accessor=ra_window(source._random_accessor, len(source)) if source.random_accessible else None,
        parents=[source],
        length=max(0, source.length - window_size + 1) if drop_first else source.length,
        dependencies=dependencies)
