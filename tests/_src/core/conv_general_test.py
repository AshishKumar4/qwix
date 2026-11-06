# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for conv_general."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import conv_general
from qwix._src.core import qarray


class ConvGeneralTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='i8_i8',
          lhs_dtype=jnp.int8,
          rhs_dtype=jnp.int8,
          expected_output_dtype=jnp.int32,
      ),
      dict(
          testcase_name='i8_i8f32',
          lhs_dtype=jnp.int8,
          rhs_dtype=(jnp.int8, jnp.float32),
          expected_output_dtype=jnp.float32,
      ),
  )
  def test_output_dtype(self, lhs_dtype, rhs_dtype, expected_output_dtype):
    lhs_shape = (1, 3, 10, 10)
    rhs_shape = (5, 3, 2, 2)
    dnums = ('NCHW', 'OIHW', 'NCHW')

    if isinstance(lhs_dtype, tuple):
      lhs = qarray.QArray(
          jnp.ones(lhs_shape, lhs_dtype[0]),
          jnp.ones((1, 3, 1, 1), lhs_dtype[1]),
      )
    else:
      lhs = jnp.ones(lhs_shape, lhs_dtype)
    if isinstance(rhs_dtype, tuple):
      rhs = qarray.QArray(
          jnp.ones(rhs_shape, rhs_dtype[0]),
          jnp.ones((5, 1, 1, 1), rhs_dtype[1]),
      )
    else:
      rhs = jnp.ones(rhs_shape, rhs_dtype)

    result = conv_general.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
    )
    self.assertEqual(result.dtype, expected_output_dtype)

  def test_bfloat16_scales_multiply_the_integer_product_in_float32(self):
    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.integers(-127, 128, (2, 3, 8, 8)), jnp.int8)
    rhs = jnp.asarray(rng.integers(-127, 128, (5, 3, 3, 3)), jnp.int8)
    lhs_scale = jnp.asarray([[[[0.0123]]]], jnp.bfloat16)
    rhs_scale = jnp.asarray([[[[0.00771]]]], jnp.bfloat16)
    dnums = ('NCHW', 'OIHW', 'NCHW')
    result = conv_general.conv_general_dilated(
        qarray.QArray(lhs, lhs_scale),
        qarray.QArray(rhs, rhs_scale),
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
    )
    product = jax.lax.conv_general_dilated(
        lhs.astype(jnp.float32),
        rhs.astype(jnp.float32),
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dnums,
    )
    expected = (
        product * lhs_scale.astype(jnp.float32) * rhs_scale.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    self.assertEqual(result.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(result, expected)

  def test_an_unscaled_integer_product_stays_exact(self):
    lhs = jnp.full((1, 64, 4, 4), 127, jnp.int8)
    rhs = jnp.full((1, 64, 4, 4), 127, jnp.int8)
    result = conv_general.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )
    self.assertEqual(result.dtype, jnp.int32)
    self.assertEqual(int(result[0, 0, 0, 0]), 127 * 127 * 64 * 16)

  @parameterized.named_parameters(
      dict(testcase_name='1040_terms', channels=65, groups=1),
      dict(testcase_name='1041_terms', channels=1041, groups=1, window=1),
      dict(testcase_name='grouped_1041_terms', channels=1041, groups=2, window=1),
      dict(testcase_name='windowed_3088_terms', channels=193, groups=1),
  )
  def test_the_float32_integer_convolution_is_exact(
      self, channels, groups, window=4
  ):
    # Every code at 127 or -128, so each product is as large as int8 codes
    # make it, and a float32 sum of more than 1024 of them may pass 2**24.
    rng = np.random.default_rng(1)
    codes = np.asarray([127, -128], np.int8)
    lhs = codes[rng.integers(0, 2, (2, groups * channels, window, window))]
    rhs = codes[rng.integers(0, 2, (3 * groups, channels, window, window))]
    dnums = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, ('NCHW', 'OIHW', 'NCHW')
    )

    def convolve(lhs, rhs):
      return jax.lax.conv_general_dilated(
          lhs,
          rhs,
          window_strides=(1, 1),
          padding='VALID',
          dimension_numbers=dnums,
          feature_group_count=groups,
          preferred_element_type=jnp.float32,
      )

    result = conv_general.exact_integer_convolution(
        jnp.asarray(lhs), jnp.asarray(rhs), dnums, groups, convolve
    )
    # The host's integer convolution: each output a dot of its group's codes.
    wide_lhs = lhs.astype(np.int64).reshape(2, groups, channels, -1)
    wide_rhs = rhs.astype(np.int64).reshape(groups, 3, channels, -1)
    expected = np.einsum('bgck,gock->bgo', wide_lhs, wide_rhs).reshape(
        2, 3 * groups, 1, 1
    )
    self.assertEqual(result.dtype, jnp.int32)
    np.testing.assert_array_equal(np.asarray(result), expected)

  def test_a_window_past_one_exact_chunk_is_refused(self):
    lhs = jnp.ones((1, 1, 33, 33), jnp.int8)
    rhs = jnp.ones((1, 1, 33, 33), jnp.int8)
    dnums = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, ('NCHW', 'OIHW', 'NCHW')
    )
    with self.assertRaises(ValueError):
      conv_general.exact_integer_convolution(
          lhs, rhs, dnums, 1, lambda lhs, rhs: lhs
      )


if __name__ == '__main__':
  absltest.main()
