from flowder.processors import AggregateProcessor

from flowder import Field
from flowder.source import ArraySource, Source
from flowder.source.util import FlatMap

from flowder.utils import file, zip_source, create_dataset

s = ArraySource(list(range(1, 11)))
ds = s.create()

for a, b in zip(ds, range(1, 11)):
    assert a["raw"] == b

fm = FlatMap(s, lambda n: list(range(n)))

reference = [
    a for b in range(1, 11) for a in range(b)
]
for a, b in zip(fm, reference):
    assert a == b

fm.load()
ds = fm.create()
for a, b in zip(fm, ds):
    assert a == b["raw"]

s = ArraySource([(n, 10 - n) for n in range(0, 11)])
a = s.item[0]
b = s.item[1]
z = zip_source(a, b)
ds = z.create()
for a, b in zip(ds, s):
    assert a["raw"] == b

data = []


class TestProcess(AggregateProcessor):

    def data_feed(self, item):
        data.append(item)

    def __call__(self, preprocessed_value):
        pass


count = TestProcess()

f = Field("test", s, process=[count])

ds = create_dataset(len(s), f)
ds.preprocess()

for a, b in zip(data, s):
    assert a == b

s = ArraySource([n for n in range(10)])
sliced = s[:5]
assert sliced[1] == s[1]
for a, b in zip(sliced, [n for n in range(10)][:5]):
    assert a == b

sliced = s[5:]
assert sliced[0] == s[5]
for a, b in zip(sliced, [n for n in range(10)][5:]):
    assert a == b

s1, s2 = s[:5], s[5:]
for a, b in zip(list(s1) + list(s2), s):
    assert a == b

for i in range(len(s1)):
    assert s1[i] == s[i]
for i in range(len(s2)):
    assert s2[i] == s[i + 5]

# MapTransform
s = ArraySource([{"a": n, "b": n * 2} for n in range(10)])
sm = s.map({
    "a": lambda a: a - 1,
    "b": lambda a: a + 1,
})
for a, b in zip(sm, [{"a": n - 1, "b": n * 2 + 1} for n in range(10)]):
    assert a == b

# zip
s1 = ArraySource([n for n in range(10)])
s2 = ArraySource([n * 2 for n in range(10)])
z = zip_source({"a": s1, "b": s2})
ref = [{"a": n, "b": n * 2} for n in range(10)]
for a, b in zip(z, ref):
    assert a == b
