from flowder.pipes import to_dict, select
from flowder.processors import Aggregator
from flowder.source import Source

from flowder.source.base import flat_mapped
from flowder.utils import from_array

s = from_array(list(range(1, 11)))

for a, b in zip(s, range(1, 11)):
    assert a == b

fm = s | flat_mapped(lambda n: list(range(n)))

reference = [
    a for b in range(1, 11) for a in range(b)
]
for a, b in zip(fm, reference):
    assert a == b

s = from_array([(n, 10 - n) for n in range(0, 11)])
a = s | select(0)
b = s | select(1)
z = a * b
for a, b in zip(z, s):
    assert a == b

data = []


class TestProcess(Aggregator):

    def feed_data(self, d: Source):
        for item in d:
            data.append(item)


count = TestProcess("test")

s >> count
ds = s

for a, b in zip(data, s):
    assert a == b

s = from_array([n for n in range(10)])
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
s = from_array([{"a": n, "b": n * 2} for n in range(10)])
sm = s.map({
    "a": lambda a: a - 1,
    "b": lambda a: a + 1,
})
for a, b in zip(sm, [{"a": n - 1, "b": n * 2 + 1} for n in range(10)]):
    assert a == b

# zip
s1 = from_array([n for n in range(10)])
s2 = from_array([n * 2 for n in range(10)])
z = s1 * s2 | to_dict("a", "b")
ref = [{"a": n, "b": n * 2} for n in range(10)]
for a, b in zip(z, ref):
    assert a == b
