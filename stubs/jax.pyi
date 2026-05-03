# stubs/jax.pyi
# PEP 561 type stubs for the subset of JAX consumed by zwingen.
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Scope:
#   - Public symbols imported anywhere under src/zwingen/.
#   - Tighter Array typing than upstream (which currently re-exports `Any`).
#   - PRNGKey is modeled as an opaque, hashable handle so that misuse (e.g.
#     reusing a consumed key without splitting) can be caught by static analysis.
#
# Out of scope:
#   - jax.experimental.* (consumed only behind `zwingen.experimental`; covered
#     in a separate stub file if and when needed).
#   - jax2tf, jax.checkpoint policies (not used by zwingen).
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np

from . import lax as lax
from . import nn as nn
from . import numpy as numpy
from . import random as random
from . import scipy as scipy
from . import sharding as sharding
from . import tree_util as tree_util

__version__: Final[str]

# ---------------------------------------------------------------------------
# Core array protocol
# ---------------------------------------------------------------------------

_T = TypeVar("_T")
_DType = TypeVar("_DType", bound=np.dtype[Any])
_Shape: TypeAlias = tuple[int, ...]
_Device: TypeAlias = "Device"

class Device(Protocol):
    id: int
    platform: str
    device_kind: str
    process_index: int
    def __repr__(self) -> str: ...

@runtime_checkable
class Array(Protocol):
    @property
    def shape(self) -> _Shape: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def device(self) -> Device: ...
    @property
    def sharding(self) -> sharding.Sharding | None: ...
    @property
    def weak_type(self) -> bool: ...
    def __array__(self, dtype: np.dtype[Any] | None = ...) -> np.ndarray[Any, Any]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __getitem__(self, key: Any) -> Array: ...
    def __add__(self, other: Array | float | int) -> Array: ...
    def __radd__(self, other: Array | float | int) -> Array: ...
    def __sub__(self, other: Array | float | int) -> Array: ...
    def __rsub__(self, other: Array | float | int) -> Array: ...
    def __mul__(self, other: Array | float | int) -> Array: ...
    def __rmul__(self, other: Array | float | int) -> Array: ...
    def __matmul__(self, other: Array) -> Array: ...
    def __rmatmul__(self, other: Array) -> Array: ...
    def __truediv__(self, other: Array | float | int) -> Array: ...
    def __rtruediv__(self, other: Array | float | int) -> Array: ...
    def __pow__(self, other: Array | float | int) -> Array: ...
    def __neg__(self) -> Array: ...
    def __pos__(self) -> Array: ...
    def __abs__(self) -> Array: ...
    def __eq__(self, other: object) -> Array: ...  # type: ignore[override]
    def __ne__(self, other: object) -> Array: ...  # type: ignore[override]
    def __lt__(self, other: Array | float | int) -> Array: ...
    def __le__(self, other: Array | float | int) -> Array: ...
    def __gt__(self, other: Array | float | int) -> Array: ...
    def __ge__(self, other: Array | float | int) -> Array: ...
    def reshape(self, *shape: int | _Shape) -> Array: ...
    def transpose(self, *axes: int) -> Array: ...
    def astype(self, dtype: np.dtype[Any] | type[Any] | str) -> Array: ...
    def block_until_ready(self) -> Array: ...
    def copy(self) -> Array: ...
    def sum(self, axis: int | _Shape | None = ...) -> Array: ...
    def mean(self, axis: int | _Shape | None = ...) -> Array: ...
    def max(self, axis: int | _Shape | None = ...) -> Array: ...
    def min(self, axis: int | _Shape | None = ...) -> Array: ...

ArrayLike: TypeAlias = Array | np.ndarray[Any, Any] | int | float | bool | complex

# Opaque PRNG key. Modeled as a NewType over Array so that downstream code
# cannot accidentally feed a plain Array into a function expecting a key.
PRNGKey = NewType("PRNGKey", Array)
KeyArray: TypeAlias = PRNGKey

# ---------------------------------------------------------------------------
# Transformations: jit, vmap, pmap, grad, value_and_grad
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])

class _Jitted(Protocol[_F]):
    __wrapped__: _F
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def lower(self, *args: Any, **kwargs: Any) -> Any: ...
    def trace(self, *args: Any, **kwargs: Any) -> Any: ...

def jit(
    fun: _F,
    *,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    inline: bool = ...,
    backend: str | None = ...,
    device: Device | None = ...,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
) -> _Jitted[_F]: ...

def vmap(
    fun: _F,
    in_axes: int | None | Sequence[int | None] = ...,
    out_axes: int | None | Sequence[int | None] = ...,
    *,
    axis_name: str | None = ...,
    axis_size: int | None = ...,
    spmd_axis_name: str | tuple[str, ...] | None = ...,
) -> _F: ...

def pmap(
    fun: _F,
    axis_name: str | None = ...,
    *,
    in_axes: int | None | Sequence[int | None] = ...,
    out_axes: int | None | Sequence[int | None] = ...,
    static_broadcasted_argnums: int | Sequence[int] | None = ...,
    devices: Sequence[Device] | None = ...,
    backend: str | None = ...,
    axis_size: int | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
) -> _F: ...

def grad(
    fun: Callable[..., Array],
    argnums: int | Sequence[int] = ...,
    has_aux: bool = ...,
    holomorphic: bool = ...,
    allow_int: bool = ...,
    reduce_axes: Sequence[str] = ...,
) -> Callable[..., Array]: ...

def value_and_grad(
    fun: Callable[..., Array],
    argnums: int | Sequence[int] = ...,
    has_aux: bool = ...,
    holomorphic: bool = ...,
    allow_int: bool = ...,
) -> Callable[..., tuple[Array, Array]]: ...

def jacfwd(
    fun: Callable[..., Array],
    argnums: int | Sequence[int] = ...,
    holomorphic: bool = ...,
    allow_int: bool = ...,
) -> Callable[..., Array]: ...

def jacrev(
    fun: Callable[..., Array],
    argnums: int | Sequence[int] = ...,
    holomorphic: bool = ...,
    allow_int: bool = ...,
) -> Callable[..., Array]: ...

def hessian(
    fun: Callable[..., Array],
    argnums: int | Sequence[int] = ...,
) -> Callable[..., Array]: ...

def jvp(
    fun: Callable[..., Array],
    primals: Sequence[Array],
    tangents: Sequence[Array],
    has_aux: bool = ...,
) -> tuple[Array, Array]: ...

def vjp(
    fun: Callable[..., Array],
    *primals: Array,
    has_aux: bool = ...,
) -> tuple[Array, Callable[..., tuple[Array, ...]]]: ...

def linearize(
    fun: Callable[..., Array],
    *primals: Array,
) -> tuple[Array, Callable[..., Array]]: ...

# ---------------------------------------------------------------------------
# Device and host helpers
# ---------------------------------------------------------------------------

def devices(backend: str | None = ...) -> list[Device]: ...
def local_devices(backend: str | None = ...) -> list[Device]: ...
def device_count(backend: str | None = ...) -> int: ...
def local_device_count(backend: str | None = ...) -> int: ...
def process_count() -> int: ...
def process_index() -> int: ...
def default_backend() -> str: ...

def block_until_ready(x: Any) -> Any: ...
def device_put(
    x: ArrayLike,
    device: Device | sharding.Sharding | None = ...,
    *,
    src: Device | None = ...,
    donate: bool = ...,
) -> Array: ...
def device_get(x: Any) -> Any: ...

# ---------------------------------------------------------------------------
# Configuration and debugging
# ---------------------------------------------------------------------------

class _Config(Protocol):
    def update(self, name: str, value: Any) -> None: ...
    def read(self, name: str) -> Any: ...
    def define_bool_state(self, name: str, default: bool, help: str) -> None: ...
    def define_string_state(self, name: str, default: str, help: str) -> None: ...
    def define_int_state(self, name: str, default: int, help: str) -> None: ...
    jax_enable_x64: bool
    jax_platform_name: str
    jax_default_matmul_precision: Literal["default", "high", "highest"] | None

config: _Config

def debug_nans(enabled: bool = ...) -> None: ...
def debug_infs(enabled: bool = ...) -> None: ...
def disable_jit(disable: bool = ...) -> Any: ...  # context manager
def enable_checks(enable: bool = ...) -> Any: ...

# ---------------------------------------------------------------------------
# Pytree alias re-exports (full surface lives in tree_util.pyi)
# ---------------------------------------------------------------------------

def tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = ...,
) -> Any: ...
def tree_leaves(tree: Any, is_leaf: Callable[[Any], bool] | None = ...) -> list[Any]: ...
def tree_structure(tree: Any, is_leaf: Callable[[Any], bool] | None = ...) -> Any: ...
def tree_flatten(
    tree: Any, is_leaf: Callable[[Any], bool] | None = ...
) -> tuple[list[Any], Any]: ...
def tree_unflatten(treedef: Any, leaves: Iterable[Any]) -> Any: ...

__all__ = [
    "Array",
    "ArrayLike",
    "Device",
    "KeyArray",
    "PRNGKey",
    "block_until_ready",
    "config",
    "debug_infs",
    "debug_nans",
    "default_backend",
    "device_count",
    "device_get",
    "device_put",
    "devices",
    "disable_jit",
    "enable_checks",
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "jit",
    "jvp",
    "lax",
    "linearize",
    "local_device_count",
    "local_devices",
    "nn",
    "numpy",
    "pmap",
    "process_count",
    "process_index",
    "random",
    "scipy",
    "sharding",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_structure",
    "tree_unflatten",
    "tree_util",
    "value_and_grad",
    "vjp",
    "vmap",
]
