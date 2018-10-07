import pathlib

from flowder.pipes import split, select
from flowder.source.base import mapped
from flowder.utils import lines

ls = lines("data/kftt.ja")
assert len(ls) == 10, "there should be 10 lines"

for s in ls:
    assert isinstance(s, str), "Source iterate the raw values"
    break

for s in ls | mapped(lambda x: len(x)):
    assert isinstance(s, int), "Source iterate the raw values"
    break

for spl in ls | split():
    assert isinstance(spl, list)
    assert isinstance(spl[0], str)
    break

delimiter = "|||"
special_delimiter_text = lines("data/special_delimiter.txt") | split(delimiter)
for third_column in special_delimiter_text | select(3):
    assert isinstance(third_column, str)
    break
