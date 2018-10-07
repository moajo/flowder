from flowder.pipes import split, select
from flowder.source.base import mapped
from flowder.utils import lines

ja = lines("data/kftt.ja")
en = lines("data/kftt.en")

zipped = ja * en
assert len(zipped) == len(ja)

for data in zipped:
    assert isinstance(data, tuple)
    assert len(data) == 2
    j, e = data
    assert isinstance(j, str)
    assert isinstance(e, str)
    break

dataset = ja * en | mapped(lambda t: {"ja": t[0], "en": t[1]})
for example in dataset:
    assert isinstance(example, dict)
    assert "ja" in example
    assert isinstance(example["ja"], str)
    assert isinstance(example["en"], str)

special_delimiter_text = lines("data/special_delimiter.txt") | split("|||")
for third_column in special_delimiter_text | select(3):
    assert isinstance(third_column, str)
    break
dataset = special_delimiter_text | select(3)

for japanese_column in dataset:
    pass
for japanese_column in dataset:
    assert isinstance(japanese_column, str)
    assert "現在" in japanese_column
    break
