from flowder.pipes import split, to_dict, select
from flowder.processors import VocabBuilder
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

v = VocabBuilder("ja")
ja >> v
ja |= split() | v.numericalizer
en |= split()
dataset = ja * en | to_dict("ja", "en")

for example in dataset:
    assert isinstance(example, dict)
    assert "ja" in example
    value = example["ja"]
    assert isinstance(value, list)
    assert isinstance(value[0], int), "converted to word index(numericalize)"
    assert isinstance(example["en"], list)
    assert isinstance(example["en"][0], str), "disable numericalize"

special_delimiter_text = lines("data/special_delimiter.txt") | split("|||")
ja = special_delimiter_text | select(3)
en = special_delimiter_text | select(4)
v = VocabBuilder("ja")
ja >> v
ja |= split() | v.numericalizer
en |= split()
dataset = ja * en | to_dict("ja", "en")
for data in dataset:
    assert isinstance(data["en"], list)
    assert isinstance(data["en"][0], str)
    assert isinstance(data["ja"], list)
    break
