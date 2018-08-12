import sys, os

sys.path.append(os.pardir)
from utils import zip_source, file, create_dataset
from abstracts import Example, Field

ja = file("data/kftt.ja").lines()
en = file("data/kftt.en").lines()

zipped = zip_source(ja, en)

assert len(zipped) == len(ja)

for data in zipped:
    assert isinstance(data, tuple)
    assert len(data) == 2
    j, e = data
    assert isinstance(j, str)
    assert isinstance(e, str)
    break

f1 = Field("ja", ja)
f2 = Field("en", en)
datsset = create_dataset(len(ja), f1, f2)
for example in datsset:
    assert isinstance(example, Example)
    assert hasattr(example, "ja")
    assert isinstance(example.ja, str)
    assert isinstance(example.en, str)

datsset = create_dataset(len(ja), f1, f2, return_as_tuple=True)
for example in datsset:
    assert isinstance(example, tuple) and len(example) == 2
    j, e = example
    assert isinstance(j, str)
    assert isinstance(e, str)
