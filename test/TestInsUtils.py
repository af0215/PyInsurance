__author__ = 'Ting'

import unittest
from lib.utils import extract_strict, extract_or_default


class TestInsUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_extract_func(self):
        colors = {"white": 0, "black": 1, "red": 2}
        rs = [extract_or_default(colors, tag, -1) for tag in ["white", "black", "red", "green", "yellow"]]
        self.assertEqual(rs, [0, 1, 2, -1, -1])
        self.assertEqual(extract_or_default(colors, "green"), None)

        rs = [extract_or_default(colors, tag) for tag in ["white", "black", "red"]]
        self.assertEqual(rs, [0, 1, 2])
        with self.assertRaises(Exception):
            extract_strict(colors, "green")

    def test_inv_return(self):
        pass

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()