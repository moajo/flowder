#!/usr/bin/env python
import json as JSON
from pathlib import Path

from flowder import ra_from_array
from flowder.source.base import Source
from flowder.source.iterable_creator import ic_from_array
from flowder.utils.std import _cal_file_hash


def json(path, key=None):
    """
    return Source object contain the values in specific key of the json file
    :param path:
    :param key:
    :return:
    """
    path = Path(path)
    assert path.exists()

    d = _cal_file_hash(path)

    with path.open(encoding="utf-8") as f:
        if key is not None:
            data = JSON.load(f)[key]
        else:
            data = JSON.load(f)
    ra = ra_from_array(data)
    ic = ic_from_array(data)
    return Source(ic, random_accessor=ra, length=len(data), dependencies=[d])
