import unittest
from f90wrap import parser
from . import test_samples_dir


class TestParser(unittest.TestCase):

    def test_parse_type_procedures(self):
        root = parser.read_files([str(test_samples_dir/'circle.f90')])
        bindings = root.modules[0].types[0].bindings
        self.assertEqual(len(bindings), 7)
        self.assertEqual(bindings[0].type, 'procedure')
        self.assertEqual(bindings[0].name, 'get_area')
        self.assertEqual(bindings[2].attributes, ['private', 'non_overridable'])
        self.assertEqual(bindings[0].procedures[0].name, 'circle_get_area')
        self.assertEqual(bindings[4].type, 'generic')
        self.assertEqual(bindings[4].name, 'print')
        self.assertEqual(len(bindings[4].procedures), 2)
        self.assertEqual(bindings[5].type, 'final')

    def test_parse_dnad(self):
        root = parser.read_files([str(test_samples_dir/'DNAD.fpp')])
        proc_names = [ p.name for p in root.modules[0].procedures ]
        self.assertIn('abs_d', proc_names)
        self.assertIn('add_di', proc_names)
        self.assertIn('assign_di', proc_names)
