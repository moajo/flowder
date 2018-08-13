import sys, os

sys.path.append(os.pardir)
from flowder.utils import file

files = file("data/kftt.ja")
assert len(files) == 1, "contains just a file"

lines = files.lines()
assert len(lines) == 10, "there should be 10 lines"

for s in lines:
    assert isinstance(s, str), "Source iterate the raw values"
    break

for spl in lines.split():
    assert isinstance(spl, list)
    assert isinstance(spl[0], str)
    break

datsset = lines.create()
for example in datsset:
    assert isinstance(example, dict), "as default, Dataset iterate dict instance"
    assert "raw" in example, "as default, example key is 'raw'. because fields is not given on create()"

datsset = lines.create(return_as_tuple=True)
for example in datsset:
    assert isinstance(example, str), "if return_as_tuple=True on create, should iterate raw value"

delimiter = "|||"
special_delimiter_text = file("data/special_delimiter.txt").lines().split(delimiter)
for third_column in special_delimiter_text.item[3]:
    assert isinstance(third_column, str)
    break
