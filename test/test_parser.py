import unittest
from f90wrap import parser
from . import test_samples_dir

class TestParser(unittest.TestCase):

    def test_parse_type_procedures(self):
        root = parser.read_files([str(test_samples_dir/'circle.f90')])
        type_procedures = root.modules[0].types[0].procedures
        self.assertEquals(len(type_procedures), 4)

