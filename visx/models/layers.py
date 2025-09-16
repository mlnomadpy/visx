"""Core model layers and utilities for VISX."""

from __future__ import annotations

import typing as tp
import jax
import jax.numpy as jnp
from jax import lax
import opt_einsum

from flax.core.frozen_dict import FrozenDict
from flax import nnx
from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
  PromoteDtypeFn,
  EinsumT,
)

Array = jax.Array
Axis = int
Size = int

# Default initializers
default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return ((padding, padding),) * rank
  if isinstance(padding, (list, tuple)) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, (list, tuple)) and len(p) == 2:
        new_pad.append(tuple(p))
      else:
        break
    else:
      return tuple(new_pad)
  raise ValueError(
    f"Invalid padding format: {padding}. Padding must be an int, a string, "
    f"a sequence of {rank} ints, or a sequence of {rank} pairs of ints."
  )


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class YatConv(Module):
  """YAT Convolutional layer with alpha scaling."""
  
  in_features: int
  out_features: int
  kernel_size: tp.Sequence[int] | int
  strides: tp.Sequence[int] | int
  padding: PaddingLike
  input_dilation: tp.Sequence[int] | int
  kernel_dilation: tp.Sequence[int] | int
  feature_group_count: int
  use_bias: bool
  mask: Array | None
  dtype: Dtype | None
  param_dtype: Dtype
  precision: PrecisionLike
  kernel_init: Initializer
  bias_init: Initializer
  conv_general_dilated: ConvGeneralDilatedT
  promote_dtype: PromoteDtypeFn
  epsilon: float
  use_alpha: bool
  alpha_init: Initializer

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: tp.Sequence[int] | int,
    rngs: rnglib.Rngs,
    strides: tp.Sequence[int] | int = 1,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Sequence[int] | int = 1,
    kernel_dilation: tp.Sequence[int] | int = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    mask: Array | None = None,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1/137,
    use_alpha: bool = True,
    alpha_init: Initializer = default_alpha_init,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    self.kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel = nnx.Param(kernel_init(kernel_key, self.kernel_shape, param_dtype))

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None

    self.alpha: nnx.Param[jax.Array] | None
    if use_alpha:
      alpha_key = rngs.params()
      self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
    else:
      self.alpha = None

    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.mask = mask
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated
    self.promote_dtype = promote_dtype
    self.epsilon = epsilon
    self.use_alpha = use_alpha
    self.alpha_init = alpha_init

  def __call__(self, inputs: Array) -> Array:
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None
    alpha = self.alpha.value if self.alpha is not None else None

    def maybe_broadcast(
      x: Array | None, target_shape: tp.Sequence[int]
    ) -> Array | None:
      if x is None:
        return None
      return jnp.broadcast_to(x, target_shape)

    kernel = maybe_broadcast(kernel, self.kernel_shape)
    bias = maybe_broadcast(bias, (self.out_features,))

    if self.mask is not None and self.mask.shape != kernel.shape:
      raise ValueError('Mask must have the same shape as weights.')

    padding = canonicalize_padding(self.padding, len(self.kernel_size))
    if self.mask is not None:
      kernel *= self.mask

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    # YAT-specific modification: Apply YAT transformation
    y = self.conv_general_dilated(
      inputs,
      kernel,
      self.strides,
      padding,
      lhs_dilation=self.input_dilation,
      rhs_dilation=self.kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    # Distance-based YAT computation
    input_expanded = jnp.expand_dims(inputs, axis=-1)  # Add channel dimension for broadcasting
    kernel_expanded = jnp.expand_dims(kernel, axis=0)  # Add batch dimension for broadcasting
    
    # Compute squared distances (simplified for conv)
    inputs_squared_sum = jnp.sum(inputs**2, axis=(1, 2), keepdims=True)
    kernel_squared_sum = jnp.sum(kernel**2, axis=(0, 1), keepdims=True)
    
    # Apply YAT transformation: y^2 / (distance + epsilon)
    distances = inputs_squared_sum + kernel_squared_sum - 2 * y
    y = y**2 / (distances + self.epsilon)

    inputs, kernel, bias, alpha = self.promote_dtype(
      (inputs, kernel, bias, alpha), dtype=self.dtype
    )

    assert self.use_bias == (bias is not None)
    assert self.use_alpha == (alpha is not None)

    if bias is not None:
      if bias.ndim != y.ndim:
        bias_shape = tuple([1] * (y.ndim - bias.ndim) + list(bias.shape))
        bias = jnp.reshape(bias, bias_shape)
      y += bias

    if alpha is not None:
      scale = (jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)) ** alpha
      y = y * scale

    return y


class YatNMN(Module):
  """
  YAT Neural Memory Network layer - a specialized dense layer with YAT transformations.
  
  Attributes:
    in_features: number of input features.
    out_features: number of output features.
    use_bias: whether to add a bias to the output (default: True).
    use_alpha: whether to use alpha scaling (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    dot_general: dot product function.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias')

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rngs: rnglib.Rngs,
    *,
    use_bias: bool = True,
    use_alpha: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1/137,
  ):

    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )
    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
    else:
      self.bias = None

    self.alpha: nnx.Param[jax.Array] | None
    if use_alpha:
      alpha_key = rngs.params()
      self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
    else:
      self.alpha = None

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.use_alpha = use_alpha
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.alpha_init = alpha_init
    self.dot_general = dot_general
    self.promote_dtype = promote_dtype
    self.epsilon = epsilon

  def __call__(self, inputs: Array) -> Array:
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None
    alpha = self.alpha.value if self.alpha is not None else None

    inputs, kernel, bias, alpha = self.promote_dtype(
      (inputs, kernel, bias, alpha), dtype=self.dtype
    )
    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )

    assert self.use_bias == (bias is not None)
    assert self.use_alpha == (alpha is not None)

    inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)
    kernel_squared_sum = jnp.sum(kernel**2, axis=0, keepdims=True)
    distances = inputs_squared_sum + kernel_squared_sum - 2 * y

    # YAT transformation
    y = y ** 2 /  (distances + self.epsilon)

    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if alpha is not None:
      scale = (jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)) ** alpha
      y = y * scale

    return y