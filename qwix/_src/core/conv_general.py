# Copyright 2024 Google LLC
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
"""Quantized jax.lax.conv_general_dilated."""
# pylint: disable=line-too-long

from collections.abc import Sequence
from typing import Any
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import numerics
from qwix._src.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.ConvDimensionNumbers,
    for_lhs: bool,
    **kwargs: Any,
) -> qarray.HowToQuantize:
  """Gets how to quantize from conv's dimension_numbers.

  Use channelwise for batch dimension and out feature dimension.

  Args:
    dimension_numbers: The conv's dimension_numbers.
    for_lhs: Whether to quantize lhs or rhs.
    **kwargs: Additional keyword arguments to HowToQuantize.

  Returns:
    How to quantize lhs or rhs.
  """
  if for_lhs:
    channelwise_axes = [dimension_numbers.lhs_spec[0]]
  else:
    channelwise_axes = [dimension_numbers.rhs_spec[0]]
  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes={},
      **kwargs,
  )


def get_transpose(
    dimension_numbers: jax.lax.ConvDimensionNumbers, for_lhs: bool
) -> list[int | None]:
  """Returns the transpose list for the given dimension_numbers."""
  transpose = [None] * len(dimension_numbers.out_spec)
  if for_lhs:
    # Only batch dimension can be channelwise thus transposed.
    transpose[dimension_numbers.out_spec[0]] = dimension_numbers.lhs_spec[0]  # pyrefly: ignore[unsupported-operation]
  else:
    # Only out feature dimension can be channelwise thus transposed.
    transpose[dimension_numbers.out_spec[1]] = dimension_numbers.rhs_spec[0]  # pyrefly: ignore[unsupported-operation]
  return transpose  # pyrefly: ignore[bad-return]


def _slow_conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Dequantizes first then computes in floating-point types."""
  if isinstance(lhs, qarray.QArray):
    lhs = qarray.dequantize(lhs)
  if isinstance(rhs, qarray.QArray):
    rhs = qarray.dequantize(rhs)
  return jax.lax.conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation,
      rhs_dilation,
      dimension_numbers,
      feature_group_count,
      batch_group_count,
      precision,
      preferred_element_type,
      out_sharding,
  )


EXACT_FLOAT32_TERMS = 2**24 // 2**14
"""The products of two int8 codes (each at most 2**7 in magnitude, so a product
at most 2**14) a float32 sum holds exactly: every partial sum stays within
2**24."""


def exact_integer_convolution(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.ConvDimensionNumbers,
    feature_group_count: int,
    convolve,
) -> jax.Array:
  """The int32 convolution of int8 codes `lhs` and `rhs` through `convolve`, a
  float32 convolution of their values, summed exactly.

  An output's contraction, its group's input features times the kernel's window,
  is split into chunks of at most `EXACT_FLOAT32_TERMS` products: the input
  features in runs of `EXACT_FLOAT32_TERMS // window` where the window holds
  fewer, else one feature at a time over the window's positions in runs of
  `EXACT_FLOAT32_TERMS`, each run a kernel masked to its positions, which keeps
  the kernel's shape, padding and dilation. Each chunk is convolved in float32,
  exact, and the chunks are summed in int32, one convolution traced however many
  chunks there are (a `lax.scan` over them).
  """
  lhs_spec, rhs_spec, _ = dimension_numbers
  spatial = [rhs.shape[axis] for axis in rhs_spec[2:]]
  window = int(np.prod(spatial))
  features = rhs.shape[rhs_spec[1]]
  width = min(features, max(1, EXACT_FLOAT32_TERMS // window))
  runs = -(-features // width)
  # The window's position runs, each `positions` of the kernel's flattened ones.
  positions = min(window, EXACT_FLOAT32_TERMS)
  position_runs = -(-window // positions)
  # Each group's features padded with zero codes to whole runs, which add nothing.
  padded = runs * width
  grouped = jnp.moveaxis(lhs, lhs_spec[1], -1)
  leading = grouped.shape[:-1]
  grouped = grouped.reshape(*leading, feature_group_count, features)
  grouped = jnp.pad(
      grouped, [(0, 0)] * (grouped.ndim - 1) + [(0, padded - features)]
  ).astype(jnp.float32)
  kernel = jnp.moveaxis(rhs, (rhs_spec[1], *rhs_spec[2:]), (0, *range(1, 1 + len(spatial))))
  kernel = jnp.pad(
      kernel, [(0, padded - features)] + [(0, 0)] * (kernel.ndim - 1)
  ).astype(jnp.float32)

  def chunk(at):
    run, position_run = at // position_runs, at % position_runs
    lhs_part = jax.lax.dynamic_slice_in_dim(grouped, run * width, width, axis=-1)
    lhs_part = jnp.moveaxis(
        lhs_part.reshape(*leading, feature_group_count * width), -1, lhs_spec[1]
    )
    rhs_part = jax.lax.dynamic_slice_in_dim(kernel, run * width, width, axis=0)
    # The run's positions of the window, its kernel masked to them.
    flat = jnp.arange(window)
    mask = (flat >= position_run * positions) & (
        flat < (position_run + 1) * positions
    )
    rhs_part = rhs_part * mask.astype(jnp.float32).reshape(
        (1, *spatial) + (1,) * (kernel.ndim - 1 - len(spatial))
    )
    rhs_part = jnp.moveaxis(
        rhs_part, (0, *range(1, 1 + len(spatial))), (rhs_spec[1], *rhs_spec[2:])
    )
    return convolve(lhs_part, rhs_part).astype(jnp.int32)

  chunks = runs * position_runs
  if chunks == 1:
    return chunk(0)

  def add(total, at):
    return total + chunk(at), None

  first = chunk(0)
  total, _ = jax.lax.scan(add, first, jnp.arange(1, chunks))
  return total


def _fast_conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Quantized jax.lax.conv_general_dilated. Both sides must be QArrays."""
  dimension_numbers = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )
  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  if isinstance(lhs, qarray.QArray):
    lhs_value = lhs.qvalue
    lhs_scale = lhs.scale
    lhs_zero_point = lhs.zero_point
    if qarray.get_tiled_axes(lhs):
      raise ValueError('subchannel is not supported for conv_general_dilated.')
  else:
    lhs_value = lhs
    lhs_scale = None
    lhs_zero_point = None

  if isinstance(rhs, qarray.QArray):
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
    rhs_zero_point = rhs.zero_point
    if qarray.get_tiled_axes(rhs):
      raise ValueError('subchannel is not supported for conv_general_dilated.')
  else:
    rhs_value = rhs
    rhs_scale = None
    rhs_zero_point = None

  if rhs_zero_point is not None:
    raise ValueError('Asymmetric quantization for rhs is not supported.')

  # The scales multiply the product in at least float32, whatever the operands'
  # dtype: a bfloat16 product of int8 codes would round before its scales, and
  # XLA:GPU lowers an integer convolution with a float32 result alone. An
  # unscaled integer product stays integer.
  scaled_type = (
      jnp.promote_types(result_type, jnp.float32)
      if jnp.issubdtype(result_type, jnp.floating)
      else result_type
  )

  def convolved(lhs_value, rhs_value, element_type):
    return jax.lax.conv_general_dilated(
        lhs_value,
        rhs_value,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        preferred_element_type=element_type,
        out_sharding=out_sharding,
    )

  def product(lhs_value, rhs_value, convolve):
    res = convolve(lhs_value, rhs_value)
    if lhs_zero_point is not None:
      # TODO(zhuyunx): This value can be constant folded in SRQ scenarios.
      res -= convolve(
          jnp.broadcast_to(lhs_zero_point, lhs_value.shape).astype(
              lhs_value.dtype
          ),
          rhs_value,
      )
    return res.astype(scaled_type)

  if jnp.issubdtype(preferred_element_type, jnp.integer):
    # XLA:GPU runs no s8 x s8 -> s32 convolution (UNIMPLEMENTED on an A100 and
    # an H100) and runs its float32 one, which sums the integer products
    # exactly in chunks (`exact_integer_convolution`). Every other platform
    # convolves the integers.
    def in_float32(lhs, rhs):
      return exact_integer_convolution(
          lhs,
          rhs,
          dimension_numbers,
          feature_group_count,
          lambda lhs, rhs: convolved(lhs, rhs, jnp.float32),
      ).astype(preferred_element_type)

    res = jax.lax.platform_dependent(
        lhs_value,
        rhs_value,
        cuda=lambda lhs, rhs: product(lhs, rhs, in_float32),
        default=lambda lhs, rhs: product(
            lhs, rhs, lambda lhs, rhs: convolved(lhs, rhs, preferred_element_type)
        ),
    )
  else:
    res = product(
        lhs_value,
        rhs_value,
        lambda lhs, rhs: convolved(lhs, rhs, preferred_element_type),
    )

  if lhs_scale is not None:
    transpose = get_transpose(dimension_numbers, for_lhs=True)
    lhs_scale = qarray.transpose_array(lhs_scale, transpose)
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, lhs_scale)

  if rhs_scale is not None:
    transpose = get_transpose(dimension_numbers, for_lhs=False)
    rhs_scale = qarray.transpose_array(rhs_scale, transpose)
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, rhs_scale)

  return res.astype(result_type)


def conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Computes a general dilated convolution with support for ``QArray`` inputs.

  This function serves as a drop-in replacement for
  `jax.lax.conv_general_dilated
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html>`_.

  It automatically dispatches to a quantized implementation if inputs are
  compatible ``QArray``s. Otherwise, it dequantizes inputs and falls back to the
  standard floating-point JAX implementation.

  Args:
    lhs: The left-hand side, either a jax.Array or QArray.
    rhs: The right-hand side, either a jax.Array or QArray.
    window_strides: A sequence of integers specifying the stride of the
      convolution window.
    padding: The padding algorithm (e.g., 'SAME', 'VALID') or explicit padding
      amounts.
    lhs_dilation: Dilation factors for the input (lhs) spatial dimensions.
    rhs_dilation: Dilation factors for the kernel (rhs) spatial dimensions.
    dimension_numbers: A structure specifying the dimension layout for lhs, rhs,
      and the output.
    feature_group_count: The number of feature groups for grouped convolution.
    batch_group_count: The number of batch groups.
    precision: The numerical precision configuration for the computation.
    preferred_element_type: The target data type for accumulation.
    out_sharding: Optional sharding spec for the output array.

  Returns:
    An Array containing the convolution result.
  """
  use_fast_path = True

  for x in (lhs, rhs):
    if isinstance(x, jax.Array) and numerics.should_quantize(x.dtype):
      use_fast_path = False
      break

  if use_fast_path:
    return _fast_conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        preferred_element_type,
        out_sharding,
    )
  else:
    return _slow_conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type,
        out_sharding,
    )
