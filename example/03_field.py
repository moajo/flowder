import sys, os

sys.path.append(os.pardir)
from fields import TextField
from utils import zip_source, file, create_dataset

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
    assert isinstance(example, dict)
    assert "ja" in example
    value, length = example["ja"]
    assert isinstance(length, int)
    assert isinstance(value, list)
    assert isinstance(value[0], int), "converted to word index(numericalize)"
    assert isinstance(example["en"], list)
    assert isinstance(example["en"][0], str), "disable numericalize"

special_delimiter_text = file("data/special_delimiter.txt").lines().split("|||")
f1 = TextField("ja", special_delimiter_text.item[3], include_length=True)
f2 = TextField("en", special_delimiter_text.item[4], numericalize=False)
dataset = special_delimiter_text.create(f1, f2)
for data in dataset:
    assert isinstance(data["en"], list)
    assert isinstance(data["en"][0], str)
    assert isinstance(data["ja"], tuple)
    break
