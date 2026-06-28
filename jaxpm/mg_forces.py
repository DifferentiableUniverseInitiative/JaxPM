"""Multigrid (real-space) gravitational force solver for JaxPM, as a drop-in alternative
to the FFT Poisson solve in `pm_forces`.

The FFT path computes the potential as ifft(delta_k * invlaplace_kernel) and the force as
ifft(-gradient_kernel * pot_k). Here we instead:
  1. solve the *discrete* periodic Poisson  lap(phi) = delta  with the halo multigrid solver
     (matches `invlaplace_kernel(fd=True)`), optionally **warm-started** from a previous phi;
  2. take the force as the 4th-order finite-difference gradient -grad(phi) -- the same stencil
     `gradient_kernel(..., order=1)` represents -- via jnp.roll (works on any sharding; XLA
     inserts the needed comm on sharded axes).

MG is iterative, so in a simulation where phi varies slowly between steps a
recycled/extrapolated previous potential cuts the solve to ~1 cycle. The FFT is a direct solver
and cannot exploit that. Pass `u0` (e.g. growth-scaled previous phi) to warm-start.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from jaxpm import multigrid as _mg

# 3D-tuned multigrid defaults (see mg_comparison/FINDINGS.md): omega~6/7, wide-halo CA smoother.
_mg.CA_OMEGA = 0.857
_mg.CA_WMAX = 4


def infer_levels(shape, pdims=None):
    """Number of factor-2 MG levels; if sharded, keep every coarse grid divisible by pdims."""
    levels, dims = 0, list(shape)
    while all(d % 2 == 0 and d >= 4 for d in dims):
        nxt = [d // 2 for d in dims]
        if pdims is not None and (nxt[0] % pdims[0] or nxt[1] % pdims[1]):
            break
        levels += 1
        dims = nxt
    return max(levels, 1)


def mg_potential(delta,
                 *,
                 levels=None,
                 v1=4,
                 v2=4,
                 cycles=2,
                 mu=1,
                 omega=0.857,
                 agg_n=0,
                 mesh=None,
                 u0=None):
    """Solve the discrete periodic Poisson  lap(phi) = (delta - <delta>)  via halo multigrid.

    `u0` is an optional warm-start initial guess (previous-step phi). Returns phi (real, f32).
    """
    _mg.CA_OMEGA = float(omega)
    _mg.set_field_axes(mesh)  # match MG halo specs to the field's mesh axes
    delta = delta - jnp.mean(
        delta)  # solvability: zero-mean source (FFT zeros k=0)
    F = delta.astype(jnp.float32)
    if levels is None:
        levels = infer_levels(F.shape)
    U0 = jnp.zeros_like(F) if u0 is None else u0.astype(jnp.float32)
    return _mg.poisson_multigrid_halo(F,
                                      U0,
                                      l=int(levels),
                                      v1=int(v1),
                                      v2=int(v2),
                                      mu=int(mu),
                                      iter_cycle=int(cycles),
                                      h=1.0,
                                      mesh=mesh,
                                      agg_n=int(agg_n))


def fd_gradient(phi, axis, order=4):
    """Finite-difference d phi / dx_axis (grid units). order=4 matches gradient_kernel(order=1).

    Uses jnp.roll, so it is correct on a single device and on sharded arrays alike (XLA inserts
    the halo comm on sharded axes). roll(phi, -1) brings phi_{n+1} to index n.
    """
    if order == 2:
        return 0.5 * (jnp.roll(phi, -1, axis) - jnp.roll(phi, 1, axis))
    # 4th-order central:  (8(f_{n+1}-f_{n-1}) - (f_{n+2}-f_{n-2})) / 12
    return (8.0 * (jnp.roll(phi, -1, axis) - jnp.roll(phi, 1, axis)) -
            (jnp.roll(phi, -2, axis) - jnp.roll(phi, 2, axis))) / 12.0


def mg_force_field(delta, *, grad_order=4, **mg_kwargs):
    """Return (force_mesh[...,3], phi) from a real-space density `delta` via multigrid.

    force_i = -d phi / dx_i. Pass `u0=` in mg_kwargs to warm-start; reuse the returned phi
    as next step's u0 (optionally growth-scaled).
    """
    phi = mg_potential(delta, **mg_kwargs)
    forces = jnp.stack(
        [-fd_gradient(phi, i, order=grad_order) for i in range(3)], axis=-1)
    return forces, phi
