"""Test if functions with dynamically sized and fixed sized arrays are wrapped correctly.

The same situation applies to dynamically sized return values that depend on an attribute of an input type.

See issue #261 https://github.com/jameskermode/f90wrap/issues/261
"""
import numpy as np

import array_shapes

x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y = np.array([2.0, 1.0], dtype=np.float64)
result1 = 2 * x
result = np.array([3.0, 5.0, 7.0], dtype=np.float64)
result2d = np.asfortranarray([[2., 1.], [4., 2.], [6., 3.]], dtype=np.float64)

container = array_shapes.array_shapes.get_container(x)
np.testing.assert_allclose(array_shapes.array_shapes.one_array_fixed(x), result1)
np.testing.assert_allclose(array_shapes.array_shapes.one_array_fixed_range(x), result1)
np.testing.assert_allclose(array_shapes.array_shapes.one_array_dynamic(x), result1)
np.testing.assert_allclose(array_shapes.array_shapes.one_array_explicit(x, 3), result1)
np.testing.assert_allclose(array_shapes.array_shapes.one_array_explicit_range(x, 3), result1)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_fixed(y, x), result)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_dynamic(y, x), result)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_mixed(y, x), result)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_2d_fixed(y, x), result2d)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_2d_fixed_whitespace(y, x), result2d)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_2d_dynamic(y, x), result2d)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_2d_mixed(y, x), result2d)
np.testing.assert_allclose(array_shapes.array_shapes.array_container_dynamic(container, y), result)
np.testing.assert_allclose(array_shapes.array_shapes.array_container_fixed(container, y), result)
np.testing.assert_allclose(array_shapes.array_shapes.array_container_dynamic_2d(2, container, y), result2d)

print("Done")