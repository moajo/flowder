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
