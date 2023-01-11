import unittest
import re

from pywrapper import m_circle

circle_radius = 1.5
ball_radius = 2.5
square_size = 3.
precision = 1e-7

def clean_str(in_str):
  docstring_lines = in_str.split('\n')
  for i,line in enumerate(docstring_lines):
    docstring_lines[i] = line.strip(' \n\t')
    docstring_lines[i] = re.sub('Defined at main\\.f90 lines \\d+-\\d+', '', docstring_lines[i])
  return '\n'.join(docstring_lines)

class TestDocstring(unittest.TestCase):

  def test_module_doc(self):
    circle = m_circle.t_circle()
    docstring = m_circle.__doc__
    ref_docstring = """
      Module m_circle


      Defined at main.f90 lines 7-89

      File: main.f90
      Brief: Test program docstring
      Author: test_author
      Copyright: test_copyright
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_docstring(self):
    circle = m_circle.t_circle()
    docstring = m_circle.construct_circle.__doc__
    ref_docstring = """
    construct_circle(self, radius)


    Defined at main.f90 lines 17-20

    Parameters
    ----------
    circle : T_Circle, [in,out] t_circle to initialize
    radius : float, [in] radius of the circle

    Brief: Initialize circle
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_no_direction(self):
    circle = m_circle.t_circle()
    docstring = m_circle.no_direction.__doc__
    ref_docstring = """
    no_direction(self, radius)


    Defined at main.f90 lines 28-31

    Parameters
    ----------
    circle : T_Circle, t_circle to initialize
    radius : float, radius of the circle

    Brief: Without direction
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_docstring_incomplet(self):
    circle = m_circle.t_circle()
    docstring = m_circle.incomplete_doc_sub.__doc__
    ref_docstring = """
    incomplete_doc_sub(self, radius)


    Defined at main.f90 lines 38-41

    Parameters
    ----------
    circle : T_Circle
    radius : float, [in] radius of the circle

    Brief: Incomplete doc
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_param_return(self):
    circle = m_circle.t_circle()
    docstring = m_circle.output_1.__doc__
    ref_docstring = """
    output = output_1()


    Defined at main.f90 lines 59-61


    Returns
    -------
    output : float, [out] this is 1

    Brief: subroutine output_1 outputs 1
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_function_return(self):
    circle = m_circle.t_circle()
    docstring = m_circle.function_2.__doc__
    ref_docstring = """
    function_2 = function_2(input)


    Defined at main.f90 lines 69-71

    Parameters
    ----------
    input : str, [in] value

    Returns
    -------
    function_2 : int, return value

    Brief: this is a function
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  def test_details(self):
    circle = m_circle.t_circle()
    docstring = m_circle.details_doc.__doc__
    ref_docstring = """
    details_doc(self, radius)


    Defined at main.f90 lines 80-82

    Parameters
    ----------
    circle : T_Circle, [in,out] t_circle to initialize
    radius : float, [in] radius of the circle

    Brief: Initialize circle
    Details: Those are very informative details
    """

    assert clean_str(ref_docstring) == clean_str(docstring)

  @unittest.skip("Support for this feature is not planned for now")
  def test_doc_inside(self):
    circle = m_circle.t_circle()
    docstring = m_circle.doc_inside.__doc__
    ref_docstring = """
    doc_inside(self, radius)


    Defined at main.f90 lines 43-52

    Parameters
    ----------
    circle : T_Circle, [in,out] t_circle to initialize
    radius : float, [in] radius of the circle

    Brief: Doc inside
    """

    assert clean_str(ref_docstring) == clean_str(docstring)


if __name__ == '__main__':

    unittest.main()
