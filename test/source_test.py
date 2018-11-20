#!/usr/bin/env python
import shutil
import unittest
from pathlib import Path

from flowder.batch_processors import collate, sort, PipeFunc

from flowder.pipes import split, select, to_dict
from flowder.source import Source
from flowder.source.base import mapped, zipped, filtered, flat_mapped
from flowder.source.depend_func import depend
from flowder.utils import from_array, from_items, from_iterable, lines, lines_gzip, flatten, choice
from flowder.utils.random import random_choice, permutation
from flowder.utils.window import windowed


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

        m2 = s1 | flat_mapped(lambda a: range(a))
        self.assertEqual([0, 0, 1, 0, 1, 2], list(m2))

        # flatmap could also flatten the Source object
        m3 = s1 | flat_mapped(lambda a: from_array(list(range(a))))
        self.assertEqual([0, 0, 1, 0, 1, 2], list(m3))

        s2 = from_items([0, 1], [2, 3, 4])
        m3 = s2 | flat_mapped(lambda a: a)
        self.assertEqual([0, 1, 2, 3, 4], list(m3))

    def test_flatten(self):
        s = [list(range(i)) for i in range(10)]

        m = flatten(s)
        self.assertEqual([n for a in s for n in a], list(m))

        m = s | flatten
        self.assertEqual([n for a in s for n in a], list(m))

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

    def test_hash_on_from_array(self):
        s1 = from_array([1, 2, 3, 4, 5], hash_func=None)
        s2 = from_array([1, 2, 3, 4, 5, 6, 7], hash_func=None)
        self.assertEqual(s1.hash, s2.hash)
        s1 = from_array([1, 2, 3, 4, 5], hash_func=lambda a: len(a))
        s2 = from_array([1, 2, 3, 4, 5, 6, 7], hash_func=lambda a: len(a))
        self.assertNotEqual(s1.hash, s2.hash)

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

    def test_hash4(self):
        s1 = from_items(1, 2, 3, 4, 5)
        self.assertNotEqual(s1.hash, s1[:].hash, "sliced source should have a different hash from the parent")
        self.assertNotEqual(s1.hash, s1[1:].hash)
        self.assertNotEqual(s1.hash, s1[:5].hash)
        self.assertNotEqual(s1[1:].hash, s1[:5].hash)

    def test_count(self):
        s1 = from_array([1, 2, 3, 4, 5])
        m = s1.flat_map(lambda a: [a])
        self.assertFalse(m.has_length)
        l = m.count()
        self.assertTrue(m.has_length)

        self.assertEqual(5, len(m))
        self.assertEqual(len(m), l)


class TestPipe(unittest.TestCase):

    def test_zipped(self):
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

        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1, 1, 1, 1, 1)
        z2 = zipped(s1, s2)
        self.assertEqual(list(z), list(z2), "zipped sequence has same length as shorter one")

    def test_mapped(self):
        s1 = from_items(1, 2, 3, 4, 5)

        self.assertRaises(TypeError, lambda: s1 | 42)  # must has type Pipe or pattern

        m = s1 | mapped(lambda a: a + 1)
        m2 = s1 | (lambda a: a + 1)
        self.assertEqual([2, 3, 4, 5, 6], list(m))
        self.assertEqual(m.parents, [s1])
        self.assertEqual(len(m), 5)
        self.assertEqual(list(m), list(m2), "callable is assumed to be mapped implicitly")

        m = s1 | mapped(lambda a: a + 1) | mapped(lambda a: a * 2)
        m2 = s1 | (lambda a: a + 1) | mapped(lambda a: a * 2)
        m3 = s1 | mapped(lambda a: a + 1) | (lambda a: a * 2)
        m4 = s1 | (mapped(lambda a: a + 1) | mapped(lambda a: a * 2))
        self.assertEqual([4, 6, 8, 10, 12], list(m))
        self.assertEqual(5, len(m))
        self.assertEqual(list(m), list(m2))
        self.assertEqual(list(m), list(m3))
        self.assertEqual(list(m), list(m4))

        self.assertRaises(TypeError, lambda: mapped(lambda a: a + 1) | 42)
        self.assertRaises(TypeError, lambda: 42 | mapped(lambda a: a + 1))

        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)
        z = zipped(s1, s2)

        r = z | (mapped(lambda a: a + 1), mapped(lambda a: a - 1))
        r2 = z | ((lambda a: a + 1), mapped(lambda a: a - 1))
        r3 = z | (mapped(lambda a: a + 1), (lambda a: a - 1))
        r4 = z | ((lambda a: a + 1), (lambda a: a - 1))
        self.assertEqual([
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
        ], list(r))
        self.assertEqual(list(r), list(r2))
        self.assertEqual(list(r), list(r3))
        self.assertEqual(list(r), list(r4))

        r = z | (mapped(lambda a: a + 1), None) | (mapped(lambda a: a * 2), None)
        self.assertEqual([
            (2 * 2, 1),
            (3 * 2, 1),
            (4 * 2, 1),
            (5 * 2, 1),
            (6 * 2, 1),
        ], list(r))

        r = z | (None, mapped(lambda a: a + 1))
        self.assertEqual([
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
        ], list(r))

        r = z | (mapped(lambda a: a + 1), None) | (None, mapped(lambda a: a * 2))
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
        r3 = r | {
            "a": lambda a: a + 1,
            "b": lambda a: a * 2,
        }
        r4 = r | mapped({
            "a": lambda a: a + 1,
            "b": lambda a: a * 2,
        })
        self.assertEqual([
            {"a": 2, "b": 2},
            {"a": 3, "b": 2},
            {"a": 4, "b": 2},
            {"a": 5, "b": 2},
            {"a": 6, "b": 2},
        ], list(r2))
        self.assertEqual(list(r2), list(r3))
        self.assertEqual(list(r2), list(r4))

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

        m = zipped(s1, s2) | to_dict("a", "b")
        a = m | select("a")
        self.assertEqual([1, 2, 3, 4, 5], list(a))
        b = m | select("b")
        self.assertEqual([1, 1, 1, 1, 1], list(b))
        a2, b2 = m | select("a", "b")
        self.assertEqual(list(a), list(a2))
        self.assertEqual(list(b), list(b2))

        a2, a3, b2 = m | select("a", "a", "b")
        self.assertEqual(list(a), list(a2))
        self.assertEqual(list(a), list(a3))
        self.assertEqual(list(b), list(b2))

    def test_to_dict(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s2 = from_items(1, 1, 1, 1, 1)

        m = zipped(s1, s2) | to_dict("a", "b")
        m2 = zipped(s1, s2) | mapped(lambda a: {"a": a[0], "b": a[1]})
        self.assertEqual(list(m2), list(m))
        self.assertRaises(AssertionError, lambda: list(zipped(s1, s2) | to_dict("a")))
        self.assertRaises(AssertionError, lambda: list(zipped(s1, s2) | to_dict("a", "b", "c")))
        self.assertRaises(AssertionError, lambda: zipped(s1, s2) | to_dict("a", "a"))

    def test_for_array_pipe(self):
        l = [1, 2, 3, 4, 5]
        m = l | mapped(lambda x: 2 * x)
        self.assertTrue(m.has_length)
        self.assertEqual([2 * n for n in l], list(m))

        l = [1, 2, 3, 4, 5]
        m = (a for a in l) | mapped(lambda x: 2 * x)
        self.assertFalse(m.has_length)
        self.assertEqual([2 * n for n in l], list(m))

    def test_pipe_combination(self):
        l = [1, 2, 3, 4, 5]
        m1 = mapped(lambda a: a * 2)
        m2 = mapped(lambda a: a + 1)
        m = m1 | m2
        self.assertEqual([a * 2 + 1 for a in l], list(l | m))
        self.assertEqual([a * 2 + 1 for a in l], list([m(a) for a in l]))

        l = [1, 2, 3, 4, 5]
        p1 = filtered(lambda a: a % 2 == 0)
        p2 = mapped(lambda a: a + 1)
        m = p1 | p2
        self.assertEqual([a + 1 for a in l if a % 2 == 0], list(l | m))
        self.assertRaises(TypeError, [m(a) for a in l])

        l = from_array([{"a": i, "b": i * 2} for i in range(10)])
        p1 = mapped(lambda a: a * 2)
        p2 = mapped(lambda a: a + 1)
        m = p1 | p2
        self.assertEqual([{"a": i * 2 + 1, "b": i * 2} for i in range(10)], list(l | {"a": m}))

        l = [1, 2, 3, 4, 5]
        p1 = mapped(lambda a: {"a": a, "b": 2 * a})
        p2 = {"a": lambda a: 3 * a}
        m = p1 | p2
        self.assertEqual([{"a": i * 3, "b": i * 2} for i in range(1, 6)], list(l | m))


class TestCache(unittest.TestCase):
    def setUp(self):
        tmp_dir = Path(".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def test_cache_and_clear(self):
        s1 = from_items(1, 2, 3, 4, 5)
        self.assertFalse(s1.cache("test", check_only=True))
        self.assertFalse(s1.cache("test", check_only=True, length_only=True))
        s1.cache("test")
        self.assertTrue(s1.cache("test", check_only=True))
        self.assertTrue(s1.cache("test", check_only=True, length_only=True))
        self.assertEqual([1, 2, 3, 4, 5], list(s1))
        self.assertEqual([1, 2, 3, 4, 5], list(s1[:]))
        self.assertEqual([1, 2], list(s1[:2]))
        s1.cache("test", clear_cache="clear")
        self.assertFalse(s1.cache("test", check_only=True))
        self.assertFalse(s1.cache("test", check_only=True, length_only=True))

    def test_length_cache(self):
        s1 = from_items(1, 2, 3, 4, 5)
        s1.cache("test", length_only=True)
        self.assertTrue(s1.cache("test", check_only=True))
        self.assertTrue(s1.cache("test", check_only=True, length_only=True))

        s1 = from_iterable([1, 2, 3, 4, 5])
        self.assertFalse(s1.has_length)
        self.assertFalse(s1.random_accessible)
        s1.cache("test", length_only=True)
        self.assertTrue(s1.has_length)
        self.assertFalse(s1.random_accessible)


class TestUtil(unittest.TestCase):
    def test_lines_splits(self):
        s = lines(Path(__file__).parent / "sample_text.txt")
        s2 = from_items(Path(__file__).parent / "sample_text.txt") | lines
        s3 = from_items(Path(__file__).parent / "sample_text.txt") | lines | flatten
        self.assertEqual(4, len(s))
        self.assertEqual("hello 10", s[0])
        self.assertEqual(list(s), list(s2[0]))
        self.assertEqual(list(s), list(s3))

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

    def test_window(self):
        def ic_ra_test(m, name):
            for i in range(-10, 20):
                self.assertEqual(list(m)[i:], list(m[i:]), f"{name}: i is {i}")

            for i in range(-len(m), len(m)):
                self.assertEqual(list(m)[i], m[i], f"{name}: i is {i}")

        s = from_array(list(range(10)))
        m = s | windowed(3, drop_first=True)
        self.assertTrue(m.has_length)
        self.assertEqual(8, len(m))
        self.assertEqual(len(m), len(list(m)))
        self.assertEqual([
            (i - 2, i - 1, i) for i in range(2, 10)
        ], list(m))
        ic_ra_test(m, "m1")

        m = s | windowed(3, drop_first=False)
        self.assertTrue(m.has_length)
        self.assertEqual(10, len(m))
        self.assertEqual([
            (i - 2 if i >= 2 else None, i - 1 if i >= 1 else None, i) for i in range(0, 10)
        ], list(m))
        ic_ra_test(m, "m2")

        m = s | windowed(1, drop_first=True)
        self.assertEqual(10, len(m))
        self.assertEqual(len(m), len(list(m)))
        self.assertEqual([
            (i,) for i in range(10)
        ], list(m))
        ic_ra_test(m, "m3")

        m4 = s | windowed(1, drop_first=False)
        self.assertEqual(10, len(m4))
        self.assertEqual(list(m), list(m4))
        ic_ra_test(m4, "m4")

        m = s | windowed(100, drop_first=True)
        self.assertEqual(0, len(m))
        self.assertEqual(len(m), len(list(m)))

        m = s | windowed(100, drop_first=False)
        self.assertEqual(10, len(m))
        self.assertEqual([
            (tuple(None for _ in range(100)) + tuple(range(i + 1)))[-100:] for i in range(10)
        ], list(m))
        self.assertEqual(list(m), list(m))
        ic_ra_test(m, "m5")


class TestStd(unittest.TestCase):
    def test_choice(self):
        s = from_array([2 * n for n in range(10)])
        ind = from_array([9, 4, 0])
        res = choice(s, ind)
        self.assertEqual([18, 8, 0], list(res))
        self.assertEqual([18, 8, 0], list(choice(from_array([2 * n for n in range(10)]), [9, 4, 0])))
        self.assertEqual([18, 8, 0], list(choice([2 * n for n in range(10)], [9, 4, 0])))

        self.assertTrue(res.has_length)
        self.assertTrue(res.random_accessible)

        s1 = from_array(list(range(10)), hash_func=lambda l: l[0])
        s2 = from_array(list(range(1, 11)), hash_func=lambda l: l[0])
        ind1 = from_array([1, 2, 3], hash_func=lambda l: l[0])
        ind2 = from_array([2, 3, 4], hash_func=lambda l: l[0])
        self.assertNotEqual(choice(s1, ind1).hash, choice(s1, ind2).hash)
        self.assertNotEqual(choice(s1, ind1).hash, choice(s2, ind1).hash)


class TestBatchProcessor(unittest.TestCase):
    def test_pipe(self):
        c = collate()
        s = sort(None)

        p = c | s
        self.assertIsInstance(p, PipeFunc)


class TestRandom(unittest.TestCase):
    def test_random_choice(self):
        s = from_array(list(range(1000)))
        res1 = random_choice(s, 3)
        res2 = random_choice(s, 3)
        self.assertNotEqual(list(res1), list(res2))
        self.assertTrue(0 < res1[0] < 1000)

        res1 = random_choice(s, 3, seed=42)
        res2 = random_choice(s, 3, seed=42)
        self.assertEqual(list(res1), list(res2))

    def test_random_permutation(self):
        s = from_array(list(range(1000)))
        res1 = permutation(s)
        res2 = permutation(s)
        self.assertNotEqual(list(res1), list(res2))
        self.assertTrue(0 < res1[0] < 1000)

        res1 = permutation(s, seed=42)
        res2 = permutation(s, seed=42)
        self.assertEqual(list(res1), list(res2))


if __name__ == '__main__':
    unittest.main()
