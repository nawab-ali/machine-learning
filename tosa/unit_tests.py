#!/usr/bin/env python3

"""This file implements unit tests for the TOSA operators."""

import binary
import unittest
import numpy as np


class TestElemBinaryOps(unittest.TestCase):
    def test_1d_tensor_int32_add(self):
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        b = np.array([5, 6, 7, 8], dtype=np.int32)

        c = binary.add(a, b)
        np.testing.assert_array_equal(c, [6, 8, 10, 12])


if __name__ == '__main__':
    unittest.main()
