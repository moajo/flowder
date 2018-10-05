import pathlib

# from flowder.utils import file
from rx import Observable, Observer

from flowder.source.base import mapped, lines, Source

# def str_source(path):
#     path = pathlib.Path(path)
#     assert path.exists()
#
#     def _gen():
#         with path.open(encoding="utf-8") as f:
#             for line in f:
#                 yield line[:-1]
#
#     return Observable.from_(_gen())


# a = str_source("01_read_lines.py")[30:]
# a.subscribe(print)

# files = lines("data/kftt.ja")
# assert len(files) == 1, "contains just a file"



ls = lines("data/kftt.ja")
assert len(ls) == 10, "there should be 10 lines"

for s in ls:
    assert isinstance(s, str), "Source iterate the raw values"
    break

for s in ls | mapped(lambda x: len(x)):
    assert isinstance(s, int), "Source iterate the raw values"
    break

# for spl in ls.split():
#     assert isinstance(spl, list)
#     assert isinstance(spl[0], str)
#     break

# dataset = ls.create()
# for example in dataset:
#     assert isinstance(example, dict), "as default, Dataset iterate dict instance"
#     assert "raw" in example, "as default, example key is 'raw'. because fields is not given on create()"

# dataset = ls.create(return_as_tuple=True)
# for example in dataset:
#     assert isinstance(example, str), "if return_as_tuple=True on create, should iterate raw value"

# delimiter = "|||"
# special_delimiter_text = file("data/special_delimiter.txt").lines().split(delimiter)
# for third_column in special_delimiter_text.item[3]:
#     assert isinstance(third_column, str)
#     break
