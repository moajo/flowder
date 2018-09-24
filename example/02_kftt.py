from flowder.utils import zip_source, file, create_dataset
from flowder.abstracts import Field

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
dataset = create_dataset(f1, f2)
for example in dataset:
    assert isinstance(example, dict)
    assert "ja" in example
    assert isinstance(example["ja"], str)
    assert isinstance(example["en"], str)

dataset = create_dataset(f1, f2, return_as_tuple=True)
for example in dataset:
    assert isinstance(example, tuple) and len(example) == 2
    j, e = example
    assert isinstance(j, str)
    assert isinstance(e, str)

special_delimiter_text = file("data/special_delimiter.txt").lines().split("|||")
for third_column in special_delimiter_text.item[3]:
    assert isinstance(third_column, str)
    break
f = Field("ja", special_delimiter_text.item[3])
dataset = special_delimiter_text.create(f)

for japanese_column in dataset:
    pass
for japanese_column in dataset.item["ja"]:
    assert isinstance(japanese_column, str)
    assert "現在" in japanese_column
    break
