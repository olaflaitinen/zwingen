# stubs/optax.pyi
# PEP 561 type stubs for the subset of optax consumed by zwingen.
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Scope:
#   - GradientTransformation as a tight Protocol (init / update pair).
#   - All optimisers / schedulers wired into `zwingen.optim`.
#   - `chain` returns a single GradientTransformation so that nested chains
#     remain assignment-compatible.
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, NamedTuple, Protocol, TypeAlias, TypeVar

import jax
from jax import Array

__version__: Final[str]

_Params: TypeAlias = Any  # arbitrary pytree of jax.Array
_Updates: TypeAlias = Any
_OptState: TypeAlias = Any
_T = TypeVar("_T")

Scalar: TypeAlias = float | Array
Schedule: TypeAlias = Callable[[int | Array], Array]

# ---------------------------------------------------------------------------
# Core protocol
# ---------------------------------------------------------------------------

class TransformInitFn(Protocol):
    def __call__(self, params: _Params) -> _OptState: ...

class TransformUpdateFn(Protocol):
    def __call__(
        self,
        updates: _Updates,
        state: _OptState,
        params: _Params | None = ...,
    ) -> tuple[_Updates, _OptState]: ...

class GradientTransformation(NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn

class GradientTransformationExtraArgs(NamedTuple):
    init: TransformInitFn
    update: Callable[..., tuple[_Updates, _OptState]]

EmptyState: TypeAlias = tuple[()]

# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def chain(*args: GradientTransformation) -> GradientTransformation: ...
def apply_if_finite(
    inner: GradientTransformation, max_consecutive_errors: int
) -> GradientTransformation: ...
def multi_transform(
    transforms: dict[str, GradientTransformation],
    param_labels: Any,
) -> GradientTransformation: ...
def masked(
    inner: GradientTransformation,
    mask: Any,
    *,
    mask_compatible_extra_args: bool = ...,
) -> GradientTransformation: ...

# ---------------------------------------------------------------------------
# Optimisers consumed by zwingen.optim
# ---------------------------------------------------------------------------

def sgd(
    learning_rate: Scalar | Schedule,
    momentum: Scalar | None = ...,
    nesterov: bool = ...,
    accumulator_dtype: Any = ...,
) -> GradientTransformation: ...

def adam(
    learning_rate: Scalar | Schedule,
    b1: Scalar = ...,
    b2: Scalar = ...,
    eps: Scalar = ...,
    eps_root: Scalar = ...,
    mu_dtype: Any = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...

def adamw(
    learning_rate: Scalar | Schedule,
    b1: Scalar = ...,
    b2: Scalar = ...,
    eps: Scalar = ...,
    eps_root: Scalar = ...,
    mu_dtype: Any = ...,
    weight_decay: Scalar = ...,
    mask: Any = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...

def adafactor(
    learning_rate: Scalar | Schedule | None = ...,
    min_dim_size_to_factor: int = ...,
    decay_rate: Scalar = ...,
    decay_offset: int = ...,
    multiply_by_parameter_scale: bool = ...,
    clipping_threshold: Scalar | None = ...,
    momentum: Scalar | None = ...,
    dtype_momentum: Any = ...,
    weight_decay_rate: Scalar | None = ...,
    eps: Scalar = ...,
    factored: bool = ...,
    weight_decay_mask: Any = ...,
) -> GradientTransformation: ...

def rmsprop(
    learning_rate: Scalar | Schedule,
    decay: Scalar = ...,
    eps: Scalar = ...,
    initial_scale: Scalar = ...,
    centered: bool = ...,
    momentum: Scalar | None = ...,
    nesterov: bool = ...,
) -> GradientTransformation: ...

def lion(
    learning_rate: Scalar | Schedule,
    b1: Scalar = ...,
    b2: Scalar = ...,
    weight_decay: Scalar = ...,
    mask: Any = ...,
) -> GradientTransformation: ...

def yogi(
    learning_rate: Scalar | Schedule,
    b1: Scalar = ...,
    b2: Scalar = ...,
    eps: Scalar = ...,
    initial_accumulator_value: Scalar = ...,
) -> GradientTransformation: ...

# ---------------------------------------------------------------------------
# Schedulers (used in zwingen.optim.schedulers)
# ---------------------------------------------------------------------------

def constant_schedule(value: Scalar) -> Schedule: ...
def linear_schedule(
    init_value: Scalar,
    end_value: Scalar,
    transition_steps: int,
    transition_begin: int = ...,
) -> Schedule: ...
def exponential_decay(
    init_value: Scalar,
    transition_steps: int,
    decay_rate: Scalar,
    transition_begin: int = ...,
    staircase: bool = ...,
    end_value: Scalar | None = ...,
) -> Schedule: ...
def cosine_decay_schedule(
    init_value: Scalar,
    decay_steps: int,
    alpha: Scalar = ...,
    exponent: Scalar = ...,
) -> Schedule: ...
def warmup_cosine_decay_schedule(
    init_value: Scalar,
    peak_value: Scalar,
    warmup_steps: int,
    decay_steps: int,
    end_value: Scalar = ...,
    exponent: Scalar = ...,
) -> Schedule: ...
def join_schedules(schedules: Sequence[Schedule], boundaries: Sequence[int]) -> Schedule: ...

# ---------------------------------------------------------------------------
# Building-block transformations
# ---------------------------------------------------------------------------

def scale(step_size: Scalar) -> GradientTransformation: ...
def scale_by_adam(
    b1: Scalar = ...,
    b2: Scalar = ...,
    eps: Scalar = ...,
    eps_root: Scalar = ...,
    mu_dtype: Any = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...
def scale_by_schedule(step_size_fn: Schedule) -> GradientTransformation: ...
def scale_by_trust_ratio(
    min_norm: Scalar = ...,
    trust_coefficient: Scalar = ...,
    eps: Scalar = ...,
) -> GradientTransformation: ...
def add_decayed_weights(
    weight_decay: Scalar = ...,
    mask: Any = ...,
) -> GradientTransformation: ...
def trace(
    decay: Scalar,
    nesterov: bool = ...,
    accumulator_dtype: Any = ...,
) -> GradientTransformation: ...
def ema(
    decay: Scalar,
    debias: bool = ...,
    accumulator_dtype: Any = ...,
) -> GradientTransformation: ...
def clip(max_delta: Scalar) -> GradientTransformation: ...
def clip_by_global_norm(max_norm: Scalar) -> GradientTransformation: ...
def adaptive_grad_clip(clipping: Scalar, eps: Scalar = ...) -> GradientTransformation: ...
def zero_nans() -> GradientTransformation: ...
def identity() -> GradientTransformation: ...

# ---------------------------------------------------------------------------
# Loss helpers (consumed by zwingen.bayes.vi and zwingen.uq.calibration)
# ---------------------------------------------------------------------------

def softmax_cross_entropy(logits: Array, labels: Array) -> Array: ...
def softmax_cross_entropy_with_integer_labels(logits: Array, labels: Array) -> Array: ...
def sigmoid_binary_cross_entropy(logits: Array, labels: Array) -> Array: ...
def l2_loss(predictions: Array, targets: Array | None = ...) -> Array: ...
def huber_loss(
    predictions: Array, targets: Array | None = ..., delta: Scalar = ...
) -> Array: ...
def cosine_similarity(predictions: Array, targets: Array, epsilon: Scalar = ...) -> Array: ...
def cosine_distance(predictions: Array, targets: Array, epsilon: Scalar = ...) -> Array: ...
def kl_divergence(log_predictions: Array, targets: Array) -> Array: ...

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_updates(params: _Params, updates: _Updates) -> _Params: ...
def incremental_update(
    new_tensors: _Params, old_tensors: _Params, step_size: Scalar
) -> _Params: ...
def periodic_update(
    new_tensors: _Params,
    old_tensors: _Params,
    steps: Array,
    update_period: int,
) -> _Params: ...

__all__ = [
    "GradientTransformation",
    "GradientTransformationExtraArgs",
    "Schedule",
    "adafactor",
    "adam",
    "adaptive_grad_clip",
    "adamw",
    "add_decayed_weights",
    "apply_if_finite",
    "apply_updates",
    "chain",
    "clip",
    "clip_by_global_norm",
    "constant_schedule",
    "cosine_decay_schedule",
    "cosine_distance",
    "cosine_similarity",
    "ema",
    "exponential_decay",
    "huber_loss",
    "identity",
    "incremental_update",
    "join_schedules",
    "kl_divergence",
    "l2_loss",
    "linear_schedule",
    "lion",
    "masked",
    "multi_transform",
    "periodic_update",
    "rmsprop",
    "scale",
    "scale_by_adam",
    "scale_by_schedule",
    "scale_by_trust_ratio",
    "sgd",
    "sigmoid_binary_cross_entropy",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "trace",
    "warmup_cosine_decay_schedule",
    "yogi",
    "zero_nans",
]
