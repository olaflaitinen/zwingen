# stubs/numpyro.pyi
# PEP 561 type stubs for the subset of numpyro consumed by zwingen.
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Scope:
#   - Public sampling primitives (sample, plate, deterministic, factor, param,
#     prng_key, subsample, scope, condition, do).
#   - The Distribution protocol surface used by zwingen.bayes.
#   - Inference engines: NUTS, HMC, SA, MCMC, SVI, autoguides.
#   - Diagnostics consumed by zwingen.bayes.diagnostics.
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Final, Generic, Literal, Protocol, TypeAlias, TypeVar

import jax
from jax import Array

from . import distributions as distributions
from . import handlers as handlers
from . import infer as infer
from . import optim as optim

__version__: Final[str]

_T = TypeVar("_T")
_Sample: TypeAlias = Array
_Params: TypeAlias = Mapping[str, Array]

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def sample(
    name: str,
    fn: distributions.Distribution,
    obs: Array | None = ...,
    rng_key: Array | None = ...,
    sample_shape: tuple[int, ...] = ...,
    infer: Mapping[str, Any] | None = ...,
    obs_mask: Array | None = ...,
) -> _Sample: ...

def param(
    name: str,
    init_value: Array | Callable[[Array], Array] | None = ...,
    *,
    constraint: distributions.constraints.Constraint = ...,
    event_dim: int | None = ...,
) -> Array: ...

def deterministic(name: str, value: Array) -> Array: ...

def factor(name: str, log_factor: Array, *, has_rsample: bool = ...) -> None: ...

def plate(
    name: str,
    size: int,
    subsample_size: int | None = ...,
    dim: int | None = ...,
) -> AbstractContextManager[Array]: ...

def plate_stack(
    prefix: str,
    sizes: Sequence[int],
    rightmost_dim: int = ...,
) -> AbstractContextManager[None]: ...

def subsample(data: Array, event_dim: int) -> Array: ...

def prng_key() -> Array: ...

def module(
    name: str,
    nn_module: Any,
    input_shape: tuple[int, ...] | None = ...,
) -> Callable[..., Array]: ...

class _Scope(AbstractContextManager[None], Protocol):
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

def scope(
    fn: Callable[..., _T] | None = ...,
    prefix: str = ...,
    divider: str = ...,
    *,
    hide_types: list[str] | None = ...,
) -> Callable[..., _T]: ...

# ---------------------------------------------------------------------------
# Effect handlers (the ones zwingen actually pulls in)
# ---------------------------------------------------------------------------

class _Trace(dict[str, dict[str, Any]]):
    def format_shapes(self, *, last_site: str | None = ...) -> str: ...

class trace(AbstractContextManager["trace"]):
    def __init__(self, fn: Callable[..., Any] | None = ...) -> None: ...
    def __enter__(self) -> _Trace: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...
    def get_trace(self, *args: Any, **kwargs: Any) -> _Trace: ...

class seed(AbstractContextManager["seed"]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = ...,
        rng_seed: int | Array | None = ...,
    ) -> None: ...
    def __enter__(self) -> seed: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class condition(AbstractContextManager["condition"]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = ...,
        data: Mapping[str, Array] | None = ...,
        condition_fn: Callable[[dict[str, Any]], dict[str, Array]] | None = ...,
    ) -> None: ...
    def __enter__(self) -> condition: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class do(AbstractContextManager["do"]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = ...,
        data: Mapping[str, Array] | None = ...,
    ) -> None: ...
    def __enter__(self) -> do: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class substitute(AbstractContextManager["substitute"]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = ...,
        data: Mapping[str, Array] | None = ...,
        substitute_fn: Callable[[dict[str, Any]], Array | None] | None = ...,
    ) -> None: ...
    def __enter__(self) -> substitute: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class block(AbstractContextManager["block"]):
    def __init__(
        self,
        fn: Callable[..., Any] | None = ...,
        hide_fn: Callable[[dict[str, Any]], bool] | None = ...,
        hide: list[str] | None = ...,
        expose_types: list[str] | None = ...,
        expose: list[str] | None = ...,
    ) -> None: ...
    def __enter__(self) -> block: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

# ---------------------------------------------------------------------------
# Diagnostics consumed by zwingen.bayes.diagnostics
# ---------------------------------------------------------------------------

class _Diagnostics(Protocol):
    def gelman_rubin(self, x: Array) -> Array: ...
    def split_gelman_rubin(self, x: Array) -> Array: ...
    def effective_sample_size(self, x: Array) -> Array: ...
    def autocorrelation(self, x: Array, axis: int = ...) -> Array: ...
    def autocovariance(self, x: Array, axis: int = ...) -> Array: ...
    def hpdi(self, x: Array, prob: float = ..., axis: int = ...) -> Array: ...
    def summary(
        self,
        samples: Mapping[str, Array],
        prob: float = ...,
        group_by_chain: bool = ...,
    ) -> dict[str, dict[str, Array]]: ...

diagnostics: _Diagnostics

# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def enable_x64(use_x64: bool = ...) -> None: ...
def set_platform(platform: Literal["cpu", "gpu", "tpu"]) -> None: ...
def set_host_device_count(n: int) -> None: ...
def get_host_device_count() -> int: ...

__all__ = [
    "block",
    "condition",
    "deterministic",
    "diagnostics",
    "distributions",
    "do",
    "enable_x64",
    "factor",
    "get_host_device_count",
    "handlers",
    "infer",
    "module",
    "optim",
    "param",
    "plate",
    "plate_stack",
    "prng_key",
    "sample",
    "scope",
    "seed",
    "set_host_device_count",
    "set_platform",
    "substitute",
    "subsample",
    "trace",
]
