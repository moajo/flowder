#!/usr/bin/env python
import unittest
from pathlib import Path

from flowder import Iterator, BucketIterator
from flowder.batch_processors import collate, sort, PipeFunc

from flowder.pipes import split, select, to_dict
from flowder.source import Source
from flowder.source.base import mapped, zipped, filtered, flat_mapped
from flowder.source.depend_func import depend
from flowder.utils import from_array, from_items, from_iterable, lines, lines_gzip, flatten
from flowder.utils.window import windowed


class TestIterator(unittest.TestCase):
    def test_iterator(self):
        ds = from_array(list(range(1000)))
        iter = Iterator(ds,
                        batch_size=100,
                        shuffle=False,
                        num_workers=0,
                        batch_transforms=None,
                        )
        a = list(iter)
        self.assertEqual(list, type(a[0]), )
        self.assertEqual(1000, sum(len(d) for d in a))
        self.assertEqual(list(range(100)), a[0])

        iter = Iterator(ds,
                        batch_size=300,
                        shuffle=True,
                        num_workers=0,
                        batch_transforms=None,
                        )
        a = list(iter)
        self.assertEqual(4, len(a))
        self.assertEqual(300, len(a[2]))
        self.assertEqual(100, len(a[3]))
        l = list(sorted([d for b in a for d in b]))
        self.assertEqual(list(range(1000)), l)

    def test_bucket_iterator(self):
        ds = from_array([list(range(i)) for i in range(100, 1000)])
        iter = BucketIterator(ds,
                              batch_size=100,
                              num_workers=0,
                              batch_transforms=None,
                              sort_key=lambda a: a,
                              )
        a = list(iter)
        self.assertEqual(list, type(a[0]), )
        self.assertEqual(900, sum(len(d) for d in a))

        iter = BucketIterator(ds,
                              batch_size=200,
                              num_workers=0,
                              batch_transforms=None,
                              sort_key=lambda a: a,
                              )
        a = list(iter)
        self.assertEqual(5, len(a))
        self.assertEqual([100, 200, 200, 200, 200], sorted([len(b) for b in a]))
        l = list(sorted([len(d) for b in a for d in b]))
        self.assertEqual(list(range(100, 1000)), l)


if __name__ == '__main__':
    unittest.main()
