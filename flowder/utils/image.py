#!/usr/bin/env python
from pathlib import Path

from PIL import Image
from flowder.utils import map_pipe

from flowder.source.base import Mapped


def to_image() -> Mapped:
    """
    ファイル名をImage objectにマップする
    :return:
    """

    @map_pipe()
    def wrapper(p):
        assert type(p) == str or isinstance(p, Path)
        return Image.open(p)

    return wrapper
