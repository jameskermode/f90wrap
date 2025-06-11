"""Test if functions with dynamically sized and fixed sized arrays are wrapped correctly.

The same situation applies to dynamically sized return values that depend on an attribute of an input type.

See issue #261 https://github.com/jameskermode/f90wrap/issues/261
"""
import numpy as np

import array_shapes

x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y = np.array([2.0, 1.0], dtype=np.float64)

container = array_shapes.array_shapes.get_container(x)
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_dynamic(y, x), [3., 5., 7.])
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_fixed(y, x), [3., 5., 7.])
np.testing.assert_allclose(array_shapes.array_shapes.two_arrays_fixed(y, x), [3., 5., 7.])
np.testing.assert_allclose(array_shapes.array_shapes.array_container_dynamic(container, y), [3., 5., 7.])
np.testing.assert_allclose(array_shapes.array_shapes.array_container_fixed(container, y), [3., 5., 7.])

print("Done")