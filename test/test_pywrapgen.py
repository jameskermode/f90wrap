import unittest
import tempfile
import os
from f90wrap import fortran, parser, pywrapgen


class TestPyWrapGen(unittest.TestCase):

    def test_py_mod_names_mapping(self):
        '''
        Verify that --py-mod-names option correctly maps module instance names.
        This is a regression test for issue #269.
        '''
        # Create a simple test module
        fortran_code = '''
module simple_test
  implicit none
  integer :: count = 0
contains
  subroutine increment()
    count = count + 1
  end subroutine
end module
'''
        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(fortran_code)
            temp_path = f.name

        try:
            root = parser.read_files([temp_path])

            # Generate wrapper with py_mod_names mapping
            # Map 'simple_test' to 'my_module'
            py_mod_names = {'simple_test': 'my_module'}

            gen = pywrapgen.PythonWrapperGenerator(
                prefix='f90wrap_',
                mod_name='test_pkg',
                types={},
                py_mod_names=py_mod_names,
                class_names={},
                kind_map={},
                make_package=False,
                auto_raise='',  # empty string to avoid AttributeError
            )

            gen.visit(root)
            code = str(gen)

            # The module instance should use the mapped name 'my_module'
            # not the original Fortran name 'simple_test'
            self.assertIn('my_module = Simple_Test()', code,
                'Module instance should use mapped name from py_mod_names')
            self.assertNotIn('simple_test = Simple_Test()', code,
                'Module instance should not use original Fortran name when py_mod_names is set')
        finally:
            os.unlink(temp_path)
