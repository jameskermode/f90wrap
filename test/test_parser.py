import unittest
import tempfile
import os
from f90wrap import parser
from . import test_samples_dir


class TestParser(unittest.TestCase):

    def test_value_attribute_parsing(self):
        '''
        Verify that the value attribute is correctly parsed and types are preserved.
        This is a regression test for issue #171.
        '''
        fortran_code = '''
module test_mod
    implicit none
contains
    subroutine foo(a, some_var, another_var)
        integer, value, intent(in) :: a
        integer, value :: some_var
        integer, intent(in) :: another_var
    end subroutine foo
end module test_mod
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(fortran_code)
            temp_path = f.name

        try:
            root = parser.read_files([temp_path])
            proc = root.modules[0].procedures[0]

            # All arguments should be integer, not real
            self.assertEqual(proc.arguments[0].type, 'integer')
            self.assertEqual(proc.arguments[1].type, 'integer')
            self.assertEqual(proc.arguments[2].type, 'integer')

            # value attribute should be preserved
            self.assertIn('value', proc.arguments[0].attributes)
            self.assertIn('value', proc.arguments[1].attributes)
            self.assertNotIn('value', proc.arguments[2].attributes)

            # intent should be preserved
            self.assertIn('intent(in)', proc.arguments[0].attributes)
            self.assertIn('intent(in)', proc.arguments[2].attributes)
        finally:
            os.unlink(temp_path)

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
