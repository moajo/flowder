#!/usr/bin/env python
import unittest
from pathlib import Path

from flowder.batch_processors import collate, sort, PipeFunc

from flowder.pipes import split, select, to_dict
from flowder.source import Source
from flowder.source.base import mapped, zipped, filtered
from flowder.source.depend_func import depend
from flowder.utils import from_array, from_items, from_iterable, lines, lines_gzip


class TestSource(unittest.TestCase):
    def test_length(self):
        s1 = Source(lambda _: [1, 2, 3, 4, 5])
        self.assertFalse(s1.has_length)  # length is not given
        self.assertRaises(TypeError, lambda: len(s1))  # so, can not obtain length
        self.assertEqual([1, 2, 3, 4, 5], list(s1))
        self.assertRaises(IndexError, lambda: s1[0])

        s1 = Source(lambda _: [1, 2, 3, 4, 5], length=5)  # length passed
        self.assertTrue(s1.has_length)
        self.assertEqual(5, len(s1))
        self.assertEqual([1, 2, 3, 4, 5], list(s1))
        self.assertRaises(IndexError, lambda: s1[0])

    def test_simple_length(self):
        s2 = from_array([1, 2, 3, 4, 5])
        self.assertEqual(5, len(s2))
        self.assertEqual(2, s2[1])
        self.assertEqual([1, 2, 3, 4, 5], list(s2))

        s2 = from_items(1, 2, 3, 4, 5)
        self.assertEqual(5, len(s2))
        self.assertEqual(2, s2[1])
        self.assertEqual([1, 2, 3, 4, 5], list(s2))

    def test_add_mul(self):
        s1 = from_array([1, 2, 3, 4, 5])
        s2 = from_array([5, 4, 3, 2, 1, 0])
        a = s1 + s2
        self.assertEqual(11, len(a))
        self.assertEqual([1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], list(a))

        b = s1 * s2
        self.assertEqual(5, len(b))
        self.assertEqual([
            (1, 5),
            (2, 4),
            (3, 3),
            (4, 2),
            (5, 1),
        ], list(b))

    def test_random_access(self):
        s1 = from_items(1, 2, 3, 4, 5)
        self.assertEqual(1, s1[0])
        self.assertEqual(2, s1[1])
        self.assertEqual([3, 4, 5], list(s1[2:]))
        self.assertTrue(s1[2:].random_accessible)
        self.assertEqual(3, s1[2:][0])
        self.assertEqual(4, s1[2:][1])
        self.assertEqual(5, s1[2:][2])
        self.assertRaises(IndexError, lambda: s1[2:][3])
        self.assertEqual([1, 2, 3], list(s1[:3]))
        self.assertEqual([2, 3], list(s1[1:3]))

        s1 = from_array(list(range(20)))
        self.assertEqual(0, s1[0])
        self.assertEqual(1, s1[1])
        self.assertEqual([5, 7, 9], list(s1[5:10:2]))
        self.assertEqual([5, 7, 9], list(s1[5:10:2]))
        self.assertEqual(6, s1[4:10:2][1])
        self.assertEqual([4, 7], list(s1[4:10:3]))
        self.assertEqual([4, 7, 10], list(s1[4:11:3]))

        self.assertEqual(17, s1[-3])
        self.assertEqual([], list(s1[999:]))
        self.assertEqual([17, 18, 19], list(s1[-3:]))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], list(s1[:-10]))
        for i in range(-10, 30):
            self.assertEqual(list(range(20)), list(s1[:i]) + list(s1[i:]))

    def test_flat_map(self):
        s1 = from_items(1, 2, 3)
        m = s1.flat_map(lambda a: range(a))
        self.assertFalse(m.has_length)

        self.assertEqual([0, 0, 1, 0, 1, 2], list(m))

    def test_map(self):
        s1 = from_items(1, 2, 3, 4, 5)
        m = s1.map(lambda a: a * 2)
        self.assertTrue(m.has_length)
        self.assertEqual(m.parents, [s1])
        self.assertEqual(5, len(m))

        s1 = from_iterable([1, 2, 3, 4, 5])
        m = s1.map(lambda a: a * 2)
        self.assertFalse(m.has_length)

        self.assertEqual([2, 4, 6, 8, 10], list(m))

    def test_map2(self):
        """
        test for `pattern mapping`
        :return:
        """
        s1 = from_items(*range(5))

        def map_func(a):
            return a * 2

        s2 = s1.map(lambda i: (i, i * 2))
        self.assertEqual([
            (i, i * 2)
            for i in range(5)
        ], list(s2))
        map1 = s2.map((None, map_func))
        self.assertEqual([
            (i, i * 4)
            for i in range(5)
        ], list(map1))

        s2 = s1.map(lambda i: {"v": i, "double_v": i * 2})
        self.assertEqual([
            {"v": i, "double_v": i * 2}
            for i in range(5)
        ], list(s2))

        map3 = s2.map({"double_v": lambda x: x - 1})
        self.assertEqual([
            {"v": i, "double_v": i * 2 - 1}
            for i in range(5)
        ], list(map3))

    def test_filter(self):
        s1 = from_items(1, 2, 3, 4, 5)
        m = s1.filter(lambda a: a % 2 == 0)
        self.assertEqual(m.parents, [s1])
        self.assertFalse(m.has_length)

        l = [a for a in m]
        self.assertEqual([2, 4], l)

    def test_filter2(self):
        """
        test for `pattern filtering`
        :return:
        """
        s1 = from_items(*range(5))

        def filter_func(a):
            return a % 2 == 0

        s2 = s1.map(lambda i: (i, i * 2))
        map1 = s2.filter((None, filter_func))
        self.assertEqual([
            (i, i * 2)
            for i in range(5)
        ], list(map1))
        map1 = s2.filter((filter_func, None))
        self.assertEqual([
            (0, 0),
            (2, 4),
            (4, 8),
        ], list(map1))

        s2 = s1.map(lambda i: {"v": i, "double_v": i * 2})
        map3 = s2.filter({"v": filter_func})
        self.assertEqual([
            {"v": 0, "double_v": 0 * 2},
            # {"v": 1, "double_v": 1*2},
            {"v": 2, "double_v": 2 * 2},
            # {"v": 3, "double_v": 3*2},
            {"v": 4, "double_v": 4 * 2},
        ], list(map3))

    def test_hash(self):
        s1 = from_items(1, 2, 3, 4, 5)
        m = s1.filter(lambda a: a % 2 == 0)
        self.assertNotEqual(s1.hash, m.hash)
        m2 = s1.filter(lambda a: a % 2 == 0, dependencies=[2])
        self.assertNotEqual(s1.hash, m2.hash)
        self.assertNotEqual(m.hash, m2.hash)

        m3 = s1.map(lambda a: a % 2 == 0)
        self.assertNotEqual(m.hash, m3.hash)
        self.assertNotEqual(m2.hash, m3.hash)
        m4 = s1.map(lambda a: a % 2 == 0, dependencies=[2])
        self.assertNotEqual(m3.hash, m4.hash)

    def test_hash2(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = s1 | mapped(lambda a: a * 2)
        s2_2 = s1.map(lambda a: a * 2)
        self.assertEqual(s2.hash, s2_2.hash)
        self.assertNotEqual(s1.hash, s2.hash)
        s3a = s1 | mapped(lambda a: a * 2, dependencies=[2])
        s3b = s1 | mapped(lambda a: a * 3, dependencies=[3])
        self.assertNotEqual(s3a.hash, s3b.hash)

    def test_hash3(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = s1 | split()
        s3 = s1 | select("key")
        self.assertNotEqual(s1.hash, s2.hash)
        self.assertNotEqual(s1.hash, s3.hash)
        self.assertNotEqual(s2.hash, s3.hash)


class TestPipe(unittest.TestCase):

    def test_mapped(self):
        s1 = from_items(1, 2, 3, 4, 5)

        self.assertRaises(TypeError, lambda: s1 | 42)  # must has type Pipe or pattern

        m = s1 | mapped(lambda a: a + 1)
        l = [a for a in m]
        self.assertEqual([2, 3, 4, 5, 6], l)
        self.assertEqual(m.parents, [s1])
        self.assertEqual(len(m), 5)

        m = s1 | mapped(lambda a: a + 1) | mapped(lambda a: a * 2)
        l = [a for a in m]
        self.assertEqual([4, 6, 8, 10, 12], l)
        self.assertEqual(len(m), 5)
        m2 = s1 | (mapped(lambda a: a + 1) | mapped(lambda a: a * 2))
        self.assertEqual(list(m), list(m2))

        self.assertRaises(TypeError, lambda: (mapped(lambda a: a + 1) | 42))
        self.assertRaises(TypeError, lambda: 42 | (mapped(lambda a: a + 1)))

        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)

        z = zipped(s1, s2)
        self.assertEqual([
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
        ], list(z))
        # self.assertEqual((2, 1), z[1])
        # self.assertEqual((3, 1), z[2])
        # self.assertEqual((4, 1), z[3])
        # self.assertEqual((5, 1), z[4])

        r = zipped(s1, s2) | (mapped(lambda a: a + 1), mapped(lambda a: a - 1))
        self.assertEqual([
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
        ], list(r))

        r = zipped(s1, s2) | (mapped(lambda a: a + 1), None) | (mapped(lambda a: a * 2), None)
        self.assertEqual([
            (2 * 2, 1),
            (3 * 2, 1),
            (4 * 2, 1),
            (5 * 2, 1),
            (6 * 2, 1),
        ], list(r))

        r = zipped(s1, s2) | (None, mapped(lambda a: a + 1))
        self.assertEqual([
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
        ], list(r))

        r = zipped(s1, s2) | (mapped(lambda a: a + 1), None) | (None, mapped(lambda a: a * 2))
        self.assertEqual([
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
        ], list(r))

    def test_mapped_dict(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)

        r = zipped(s1, s2) | mapped(lambda a: {"a": a[0], "b": a[1]})
        self.assertEqual({"a": 3, "b": 1}, r[2])

        r2 = r | {
            "a": mapped(lambda a: a + 1),
            "b": mapped(lambda a: a * 2),
        }
        self.assertEqual([
            {"a": 2, "b": 2},
            {"a": 3, "b": 2},
            {"a": 4, "b": 2},
            {"a": 5, "b": 2},
            {"a": 6, "b": 2},
        ], list(r2))

    def test_filter(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(6, 7, 8, 9, 10)

        r = zipped(s1, s2) | (filtered(lambda a: a % 2 == 0), None)
        self.assertFalse(r.random_accessible)
        r.mem_cache()
        self.assertTrue(r.random_accessible)
        self.assertEqual((2, 7), r[0])
        self.assertEqual((4, 9), r[1])

        r2 = r | (None, filtered(lambda a: a == 7))
        self.assertEqual(1, len(list(r2)))
        self.assertEqual((2, 7), r2.search_item(0))

        r2 = zipped(s1, s2) | (filtered(lambda a: a % 2 == 0), filtered(lambda a: a % 3 == 0))
        self.assertEqual(1, len(list(r2)))
        self.assertEqual((4, 9), r2.search_item(0))

    def test_decorator(self):
        def kk(k):
            @depend(k)
            def f(a):
                return a * k

            return f

        s1 = from_items(1, 2, 3, 4, 5)
        m1 = s1 | mapped(kk(2))
        m2 = s1 | mapped(kk(3))
        self.assertNotEqual(m1.hash, m2.hash)

        m1 = s1 | mapped(lambda a: a * 2, dependencies=[2])
        m2 = s1 | mapped(lambda a: a * 3, dependencies=[3])
        self.assertNotEqual(m1.hash, m2.hash)

        m1 = s1 | mapped(lambda a: a * 2)
        m2 = s1 | mapped(lambda a: a * 3)
        self.assertEqual(m1.hash, m2.hash)

    def test_mem_cache(self):
        s1 = from_iterable([1, 2, 3, 4, 5])
        self.assertFalse(s1.has_length)
        s1.mem_cache()
        self.assertTrue(s1.has_length)
        self.assertEqual([1, 2, 3, 4, 5], list(s1))
        self.assertEqual(1, s1[0])

    def test_select(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)

        m = zipped(s1, s2) | mapped(lambda a: {"a": a[0], "b": a[1]})
        r = m | select("a")
        self.assertEqual([1, 2, 3, 4, 5], list(r))
        r = m | select("b")
        self.assertEqual([1, 1, 1, 1, 1], list(r))

    def test_to_dict(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)

        m = zipped(s1, s2) | to_dict("a", "b")
        r = m | select("a")
        self.assertEqual([1, 2, 3, 4, 5], list(r))
        r = m | select("b")
        self.assertEqual([1, 1, 1, 1, 1], list(r))


class TestCache(unittest.TestCase):
    def test_file_cache(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s1.cache("test")
        self.assertTrue(s1.cache("test", check_only=True))
        s1.cache("test", clear_cache="clear")
        self.assertFalse(s1.cache("test", check_only=True))


class TestUtil(unittest.TestCase):
    def test_lines_splits(self):
        s = lines(Path(__file__).parent / "sample_text.txt")
        self.assertEqual(4, len(s))
        self.assertEqual("hello 10", s[0])

        s = s | split()
        self.assertEqual(4, len(s))
        self.assertEqual(["hello", "10"], s[0])

        s |= (None, int)
        self.assertEqual(4, len(s))
        self.assertEqual(("hello", 10), s[0])

    def test_gzip(self):
        s = lines_gzip(Path(__file__).parent / "data" / "sample.gz")
        self.assertEqual(5, len(s))
        self.assertEqual([
            'abc',
            'def',
            'ghi',
            'jkl',
            'mno',
        ], list(s))
        self.assertFalse(s.random_accessible)


class TestBatckProcessor(unittest.TestCase):
    def test_pipe(self):
        c = collate()
        s = sort(None)

        p = c | s
        self.assertIsInstance(p, PipeFunc)


if __name__ == '__main__':
    unittest.main()
