import pytest

from pywrapper import m_circle

circle_radius = 1.5
ball_radius = 2.5
square_size = 3.
precision = 1e-7

def clean_str(in_str):
  docstring_lines = in_str.split('\n')
  for i,line in enumerate(docstring_lines):
    docstring_lines[i] = line.strip(' \n\t')
  return '\n'.join(docstring_lines)

def test_docstring():
  circle = m_circle.t_circle()
  docstring = m_circle.construct_circle.__doc__
  ref_docstring = """
  construct_circle(self, radius)


  Defined at main.f90 lines 17-20

  Parameters
  ----------
  circle : T_Circle, [in,out] t_circle to initialize
  radius : float, [in] radius of the circle

  Initialize circle
  """

  assert clean_str(ref_docstring) == clean_str(docstring)

def test_no_direction():
  circle = m_circle.t_circle()
  docstring = m_circle.no_direction.__doc__
  ref_docstring = """
  no_direction(self, radius)


  Defined at main.f90 lines 28-31

  Parameters
  ----------
  circle : T_Circle, t_circle to initialize
  radius : float, radius of the circle

  Without direction
  """

  assert clean_str(ref_docstring) == clean_str(docstring)

def test_docstring_incomplet():
  circle = m_circle.t_circle()
  docstring = m_circle.incomplete_doc_sub.__doc__
  ref_docstring = """
  incomplete_doc_sub(self, radius)


  Defined at main.f90 lines 38-41

  Parameters
  ----------
  circle : T_Circle
  radius : float, [in] radius of the circle

  Incomplete doc
  """

  assert clean_str(ref_docstring) == clean_str(docstring)

def test_function_return():
  circle = m_circle.t_circle()
  docstring = m_circle.output_1.__doc__
  ref_docstring = """
  output = output_1()


  Defined at main.f90 lines 59-61


  Returns
  -------
  output : float, [out] this is 1

  subroutine output_1 outputs 1
  """

  assert clean_str(ref_docstring) == clean_str(docstring)

@pytest.mark.skip(reason="Support for this feature is not planned for now")
def test_doc_inside():
  circle = m_circle.t_circle()
  docstring = m_circle.doc_inside.__doc__
  ref_docstring = """
  doc_inside(self, radius)


  Defined at main.f90 lines 43-52

  Parameters
  ----------
  circle : T_Circle, [in,out] t_circle to initialize
  radius : float, [in] radius of the circle

  Doc inside
  """

  assert clean_str(ref_docstring) == clean_str(docstring)
