#!/usr/bin/env python

from flowder.source.base import Source
from flowder.source.iterable_creator import ic_from_generator
import pandas as pd


def csv(path_or_buffer, **kwargs):
    # TODO calculate hash
    data_frame = pd.read_csv(path_or_buffer, **kwargs)

    length = len(data_frame)

    def _gen():
        yield from ((
            index,
            {key: data[key] for key in data.keys()}
        ) for index, data in data_frame.iterrows())

    obs = ic_from_generator(_gen)
    return Source(obs, length=length)
