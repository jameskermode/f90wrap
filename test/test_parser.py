import unittest
from f90wrap import parser
from . import test_samples_dir


class TestParser(unittest.TestCase):

    def test_parse_type_procedures(self):
        root = parser.read_files([str(test_samples_dir/'circle.f90')])
        type_procedures = root.modules[0].types[0].procedures
        self.assertEqual(len(type_procedures), 6)
        self.assertEqual(type_procedures[0].type, 'procedure')
        self.assertEqual(type_procedures[0].name, 'get_area')
        self.assertEqual(type_procedures[0].targets, ['circle_get_area'])
        self.assertEqual(type_procedures[4].type, 'generic')
        self.assertEqual(type_procedures[4].name, 'print')
        self.assertEqual(type_procedures[4].targets, ['print_basic', 'print_tagged'])
        self.assertEqual(type_procedures[5].type, 'final')
