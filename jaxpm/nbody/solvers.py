from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF
from diffrax import LocalLinearInterpolation, RESULTS, AbstractTerm, AbstractSolver
from .ode import LeapFrogODETerm
import jax_cosmo as jc
from jax import lax

_ErrorEstimate: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "y"]]
Yb: TypeAlias = PyTree[Float[ArrayLike, "y"]]

_SolverState: TypeAlias = tuple[Ya, Yb]

class EfficientLeapFrog(AbstractSolver):
    """Semi-implicit Euler's method.

    Symplectic method. Does not support adaptive step sizing. Uses 1st order local
    linear interpolation for dense/ts output.
    """
    initial_t0 : RealScalarLike
    final_t1: RealScalarLike
    cosmo: jc.Cosmology  # Declares cosmology object as a data member
    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 2

    def init(
        self,
        terms: tuple[LeapFrogODETerm, LeapFrogODETerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        term_1, _ = terms
        y0_1, y0_2 = y0

        # Compute forces (kick update)
        control = term_1.contr(t0, t1, action="FK", cosmo=self.cosmo)
        y1_2 = (y0_2**ω + term_1.vf_prod(t0, y0_1, args, control) ** ω).ω

        return (y0_1, y1_2)


    def step(
        self,
        terms: tuple[LeapFrogODETerm, LeapFrogODETerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump

        term_1, term_2 = terms
        y0_1, y0_2 = lax.cond(t0 == self.initial_t0, lambda _ : solver_state , lambda _ : y0, None)
        t0t1 = (t0 * t1) ** 0.5

        # Drift
        control1 = term_2.contr(t0, t1, action="D", cosmo=self.cosmo)
        y1_1 = (y0_1**ω + term_2.vf_prod(t0t1, y0_2, args, control1) ** ω).ω

        # Double kick or last kick
        control2 = term_1.contr(
            t0, t1, action="K", cosmo=self.cosmo, cond=(t1 == self.final_t1)
        )
        y1_2 = (y0_2**ω + term_1.vf_prod(t1, y1_1, args, control2) ** ω).ω

        y1 = (y1_1, y1_2)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, solver_state, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.vf(t0, y0_2, args)
        f2 = term_2.vf(t0, y0_1, args)
        return f1, f2
