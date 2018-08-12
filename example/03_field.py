import sys, os

from fields import TextField

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

f1 = TextField("ja", ja, include_length=True)
f2 = TextField("en", en, numericalize=False)
datsset = create_dataset(len(ja), f1, f2)

datsset.preprocess()
for example in datsset:
    assert isinstance(example, Example)
    assert hasattr(example, "ja")
    value, length = example.ja
    assert isinstance(length, int)
    assert isinstance(value, list)
    assert isinstance(value[0], int), "converted to word index(numericalize)"
    assert isinstance(example.en, list)
    assert isinstance(example.en[0], str), "disable numericalize"


