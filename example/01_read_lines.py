import sys, os

sys.path.append(os.pardir)
from abstracts import Example
from main import file

files = file("data/kftt.ja")
assert len(files) == 1, "contains just a file"

lines = files.lines()
assert len(lines) == 10, "there should be 10 lines"

for s in lines:
    assert isinstance(s, str), "Source iterate the raw values"
    break

datsset = lines.create()
for example in datsset:
    assert isinstance(example, Example), "as default, Dataset iterate Example instance"

datsset = lines.create(return_as_tuple=True)
for example in datsset:
    assert isinstance(example, str), "if return_as_tuple=True on create, should iterate raw value"
