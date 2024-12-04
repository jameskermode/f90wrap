import unittest

from io import StringIO
import f90wrap.parser
from f90wrap.parser import check_abstract_interface

f90wrap.parser.doc_plugin_module = None

class MockF90File:
    def __init__(self, lines):
        self.filename = 'mock_file.f90'
        self.lineno = 4
        self.lines = StringIO(lines)

    def next(self):
        self.lineno += 1
        return self.lines.readline().strip()

    def close(self):
        pass

class TestParserAbstractInterface(unittest.TestCase):
    def test_check_abstract_interface(self):
        lines = """
        abstract interface
            subroutine get_value_i(value)
                real, intent(out) :: value
            end subroutine get_value_i
            real function f(x)
                real, intent(in) :: x
            end function f
            subroutine no_args
            end subroutine no_args
        end interface
        """.strip()
        f90file = MockF90File(lines)
        check = check_abstract_interface(f90file.next(), f90file)
        interface = check[0]
        current_line = check[1]
        assert current_line == ''
        assert interface is not None
        assert 'abstract' in interface.attributes
        assert len(interface.procedures) == 3
        assert interface.procedures[0].name == 'get_value_i'
        assert interface.procedures[1].name == 'f'
        assert interface.procedures[2].name == 'no_args'


if __name__ == '__main__':
    unittest.main()
