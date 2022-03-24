#!/usr/bin/env python3

"""This file implements unit tests for the TOSA operators."""

import binary
import unittest
import numpy as np


class TestElemBinaryOps(unittest.TestCase):
    def test_tensor_add_mismatched_ndim(self):
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

        with self.assertRaises(ValueError):
            _ = binary.add(a, b)

    def test_1d_tensor_int32_add(self):
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        b = np.array([5, 6, 7, 8], dtype=np.int32)

        c = binary.add(a, b)
        np.testing.assert_array_equal(c, [6, 8, 10, 12])

    def test_1d_tensor_fp32_add(self):
        a = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
        b = np.array([5.5, 6.6, 7.7, 8.8], dtype=np.float32)

        c = binary.add(a, b)
        np.testing.assert_allclose(c, [6.6, 8.8, 11.0, 13.2])

    def test_2d_tensor_int32_add(self):
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

        c = binary.add(a, b)
        np.testing.assert_array_equal(c, [[2, 4, 6, 8], [10, 12, 14, 16]])


if __name__ == '__main__':
    unittest.main()
