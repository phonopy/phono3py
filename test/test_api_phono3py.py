import unittest
import os
import numpy as np
from phono3py.phonon3 import Phono3py

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestPhono3py(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPhono3py(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhono3py)
    unittest.TextTestRunner(verbosity=2).run(suite)
