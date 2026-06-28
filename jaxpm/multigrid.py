"""Experimental halo-exchange multigrid kernels for sharded JAX arrays.

This is intentionally local to the benchmark directory. It replaces the global
`jnp.roll` stencil operations in the DiffHydro prototype with jaxdecomp halo
exchange on the sharded x/y axes. The z axis is assumed unsharded and still uses
local periodic rolls.

Current scope:
  - periodic 3D only
  - sharding compatible with the benchmark's NamedSharding(mesh, P("z","y",None))
  - halo-backed Laplacian, residual, Jacobi smoother, full-weighting restriction,
    and fused trilinear prolong-add
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

Array = jnp.ndarray
FIELD_SPEC = P("z", "y", None)


def set_field_axes(mesh):
    """Align the multigrid field PartitionSpec to the host code's mesh axis names.

    JaxPM shards the 3D field as P(<axis0>, <axis1>, None) over a 2D device mesh whose axes may
    be named differently (e.g. ('x','y')) than this module's default ('z','y'). All shard_map
    in/out specs read the module-global FIELD_SPEC at trace time, so setting it once here (before
    the solve is traced) makes the halo kernels match the field's actual sharding."""
    global FIELD_SPEC
    if mesh is not None and not mesh.empty:
        names = tuple(mesh.axis_names)
        FIELD_SPEC = P(names[0], names[1], None)
    return FIELD_SPEC


def _has_mesh(mesh: Optional[Mesh]) -> bool:
    return mesh is not None and not mesh.empty


def _pad_local_xy(u: Array, mesh: Mesh, width: int = 1) -> Array:
    """Pad each local shard along x/y before halo exchange."""

    @partial(shard_map, mesh=mesh, in_specs=FIELD_SPEC, out_specs=FIELD_SPEC)
    def _pad(a):
        return jnp.pad(a, ((width, width), (width, width), (0, 0)),
                       mode="wrap")

    return _pad(u)


def _exchange_xy(
    u: Array,
    mesh: Optional[Mesh],
    halo_backend: str = "jax",
    width: int = 1,
) -> Array:
    """Return local x/y padded array with exchanged periodic ghost zones."""
    if not _has_mesh(mesh):
        return jnp.pad(u, ((width, width), (width, width), (0, 0)),
                       mode="wrap")

    from jaxdecomp import halo_exchange

    padded = _pad_local_xy(
        u, mesh, width=width)  # global shape includes per-shard halos.
    return halo_exchange(
        padded,
        halo_extents=(width, width),
        halo_periods=(True, True),
        backend=halo_backend,
    )


def _halo_slices_xy(h: Array, mesh: Optional[Mesh]):
    """Extract local core and x/y neighbors from a per-shard halo-padded array."""
    if not _has_mesh(mesh):
        c = h[1:-1, 1:-1, :]
        return c, h[:-2, 1:-1, :], h[2:, 1:-1, :], h[1:-1, :-2, :], h[1:-1,
                                                                      2:, :]

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=FIELD_SPEC,
        out_specs=(FIELD_SPEC, FIELD_SPEC, FIELD_SPEC, FIELD_SPEC, FIELD_SPEC),
    )
    def _slices(a):
        return (
            a[1:-1, 1:-1, :],
            a[:-2, 1:-1, :],
            a[2:, 1:-1, :],
            a[1:-1, :-2, :],
            a[1:-1, 2:, :],
        )

    return _slices(h)


def _laplace_from_halo(hx: Array, mesh: Optional[Mesh]) -> Array:
    """Compute neighbor sum minus 6*center from an exchanged halo array."""
    if not _has_mesh(mesh):
        center = hx[1:-1, 1:-1, :]
        sum_n = (hx[:-2, 1:-1, :] + hx[2:, 1:-1, :] + hx[1:-1, :-2, :] +
                 hx[1:-1, 2:, :] + jnp.roll(center, 1, 2) +
                 jnp.roll(center, -1, 2))
        return sum_n - 6.0 * center

    @partial(shard_map, mesh=mesh, in_specs=FIELD_SPEC, out_specs=FIELD_SPEC)
    def _local(a):
        center = a[1:-1, 1:-1, :]
        sum_n = (a[:-2, 1:-1, :] + a[2:, 1:-1, :] + a[1:-1, :-2, :] +
                 a[1:-1, 2:, :] + jnp.roll(center, 1, 2) +
                 jnp.roll(center, -1, 2))
        return sum_n - 6.0 * center

    return _local(hx)


def _jacobi_from_halo(hx: Array, F: Array, h: float, mesh: Optional[Mesh],
                      omega: float) -> Array:
    if not _has_mesh(mesh):
        center = hx[1:-1, 1:-1, :]
        sum_n = (hx[:-2, 1:-1, :] + hx[2:, 1:-1, :] + hx[1:-1, :-2, :] +
                 hx[1:-1, 2:, :] + jnp.roll(center, 1, 2) +
                 jnp.roll(center, -1, 2))
        u_star = (sum_n - (h * h) * F) / 6.0
        return (1.0 - omega) * center + omega * u_star

    @partial(shard_map,
             mesh=mesh,
             in_specs=(FIELD_SPEC, FIELD_SPEC),
             out_specs=FIELD_SPEC)
    def _local(a, f):
        center = a[1:-1, 1:-1, :]
        sum_n = (a[:-2, 1:-1, :] + a[2:, 1:-1, :] + a[1:-1, :-2, :] +
                 a[1:-1, 2:, :] + jnp.roll(center, 1, 2) +
                 jnp.roll(center, -1, 2))
        u_star = (sum_n - (h * h) * f) / 6.0
        return (1.0 - omega) * center + omega * u_star

    return _local(hx, F)


def _smooth_axis_from_halo(hx: Array, axis: int,
                           mesh: Optional[Mesh]) -> Array:
    """Apply 1D [1/4, 1/2, 1/4] smoothing along x or y from halo."""
    if axis not in (0, 1):
        raise ValueError(axis)
    if not _has_mesh(mesh):
        c = hx[1:-1, 1:-1, :]
        if axis == 0:
            return 0.25 * hx[:-2, 1:-1, :] + 0.5 * c + 0.25 * hx[2:, 1:-1, :]
        return 0.25 * hx[1:-1, :-2, :] + 0.5 * c + 0.25 * hx[1:-1, 2:, :]

    @partial(shard_map, mesh=mesh, in_specs=FIELD_SPEC, out_specs=FIELD_SPEC)
    def _local_x(a):
        c = a[1:-1, 1:-1, :]
        return 0.25 * a[:-2, 1:-1, :] + 0.5 * c + 0.25 * a[2:, 1:-1, :]

    @partial(shard_map, mesh=mesh, in_specs=FIELD_SPEC, out_specs=FIELD_SPEC)
    def _local_y(a):
        c = a[1:-1, 1:-1, :]
        return 0.25 * a[1:-1, :-2, :] + 0.5 * c + 0.25 * a[1:-1, 2:, :]

    return _local_x(hx) if axis == 0 else _local_y(hx)


def apply_poisson_halo(u: Array,
                       h: float,
                       mesh: Optional[Mesh] = None,
                       halo_backend: str = "jax") -> Array:
    """A u = periodic 6-point Laplacian, using exchanged halos for sharded x/y."""
    if u.ndim != 3:
        raise ValueError(
            f"apply_poisson_halo expects 3D arrays, got shape {u.shape}")

    hx = _exchange_xy(u, mesh, halo_backend)
    return _laplace_from_halo(hx, mesh) / (h * h)


def residual_halo(F: Array,
                  u: Array,
                  h: float,
                  mesh: Optional[Mesh] = None,
                  halo_backend: str = "jax") -> Array:
    return F - apply_poisson_halo(u, h, mesh=mesh, halo_backend=halo_backend)


def jacobi_sweep_halo(
    u: Array,
    F: Array,
    h: float,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
    omega: float = 2.0 / 3.0,
) -> Array:
    hx = _exchange_xy(u, mesh, halo_backend)
    return _jacobi_from_halo(hx, F, h, mesh, omega)


# Communication-avoiding halo width: exchange a width-W halo once, then do up to W
# local Jacobi sweeps with no further communication. Caps the per-exchange halo so
# wrap-pad stays valid on small/coarse grids; sweeps beyond W are done in extra chunks.
CA_WMAX = 4
# Damped-Jacobi relaxation factor. 2/3 is 1D-optimal; the 3D 7-point model problem smooths
# best near ~6/7. Overridable from the sweep driver to tune convergence-per-cycle.
CA_OMEGA = 2.0 / 3.0


def _ca_local_sweeps(u_pad: Array, F_pad: Array, w: int, h: float,
                     omega: float) -> Array:
    """Do `w` weighted-Jacobi sweeps on an x/y width-w halo-padded, z-periodic block.
    Returns the valid core [w:-w, w:-w, :]. Numerically identical to w global sweeps."""
    hh = h * h
    inv6 = 1.0 / 6.0
    u = u_pad
    fc = F_pad[1:-1, 1:-1, :]
    for _ in range(w):
        c = u[1:-1, 1:-1, :]
        sum_n = (u[:-2, 1:-1, :] + u[2:, 1:-1, :] + u[1:-1, :-2, :] +
                 u[1:-1, 2:, :] + jnp.roll(c, 1, 2) + jnp.roll(c, -1, 2))
        u_star = (sum_n - hh * fc) * inv6
        u = u.at[1:-1, 1:-1, :].set((1.0 - omega) * c + omega * u_star)
    return u[w:-w, w:-w, :]


def smooth_weighted_jacobi_halo(
    u0: Array,
    F: Array,
    h: float,
    iters: int,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
    omega: float = None,
) -> Array:
    """Communication-avoiding weighted-Jacobi: one wide halo exchange per W sweeps."""
    if omega is None:
        omega = CA_OMEGA
    iters = int(iters)
    if iters <= 0:
        return u0
    u = u0
    rem = iters
    while rem > 0:
        w = min(rem, CA_WMAX)
        u_pad = _exchange_xy(u, mesh, halo_backend, width=w)
        F_pad = _exchange_xy(F, mesh, halo_backend, width=w)
        if not _has_mesh(mesh):
            u = _ca_local_sweeps(u_pad, F_pad, w, h, omega)
        else:

            @partial(shard_map,
                     mesh=mesh,
                     in_specs=(FIELD_SPEC, FIELD_SPEC),
                     out_specs=FIELD_SPEC)
            def _local(a, f, _w=w):
                return _ca_local_sweeps(a, f, _w, h, omega)

            u = _local(u_pad, F_pad)
        rem -= w
    return u


# When True, restrict by computing the fine residual once and applying a SEPARABLE
# [1/4,1/2,1/4]^3 full-weighting, instead of the fused 27-sample restrict_residual_halo.
# Mathematically identical full-weighting of the residual, but it streams the ~1GB/GPU
# fine arrays a handful of times rather than ~27x -- a large HBM-bandwidth win at the
# finest level, where the bandwidth-bound 2048^3 V-cycle spends most of its time.
CA_CHEAP_RESTRICT = False


def restrict_full_weighting_halo(
    fine: Array,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
) -> Array:
    """Separable [1/4,1/2,1/4] full-weighting using halo neighbors on x/y."""
    x = fine
    x = _smooth_axis_from_halo(_exchange_xy(x, mesh, halo_backend), 0, mesh)
    x = _smooth_axis_from_halo(_exchange_xy(x, mesh, halo_backend), 1, mesh)
    x = 0.25 * jnp.roll(x, 1, 2) + 0.5 * x + 0.25 * jnp.roll(x, -1, 2)
    return x[::2, ::2, ::2]


def _restrict(F, U, h, mesh=None, halo_backend="jax"):
    """Restricted full-weighted residual; cheap (residual+separable) or fused 27-term."""
    if CA_CHEAP_RESTRICT:
        res = residual_halo(F, U, h, mesh=mesh, halo_backend=halo_backend)
        return restrict_full_weighting_halo(res,
                                            mesh=mesh,
                                            halo_backend=halo_backend)
    return restrict_residual_halo(F,
                                  U,
                                  h,
                                  mesh=mesh,
                                  halo_backend=halo_backend)


def restrict_residual_halo(
    F: Array,
    U: Array,
    h: float,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
) -> Array:
    """Return full-weighted residual without materializing the fine residual."""
    F_halo = _exchange_xy(F, mesh, halo_backend, width=1)
    U_halo = _exchange_xy(U, mesh, halo_backend, width=2)
    inv_h2 = 1.0 / (h * h)

    def _sample_z(a: Array, zoff: int, nz_coarse: int) -> Array:
        return jnp.roll(a, -zoff, axis=2)[..., :2 * nz_coarse:2]

    def _residual_sample(fh: Array, uh: Array, xoff: int, yoff: int,
                         zoff: int) -> Array:
        nx_coarse = (fh.shape[0] - 2) // 2
        ny_coarse = (fh.shape[1] - 2) // 2
        nz_coarse = fh.shape[2] // 2

        def _xy(a: Array, xstart: int, ystart: int) -> Array:
            return a[
                xstart:xstart + 2 * nx_coarse:2,
                ystart:ystart + 2 * ny_coarse:2,
                :,
            ]

        f = _sample_z(_xy(fh, 1 + xoff, 1 + yoff), zoff, nz_coarse)
        uc_xy = _xy(uh, 2 + xoff, 2 + yoff)
        uxm_xy = _xy(uh, 1 + xoff, 2 + yoff)
        uxp_xy = _xy(uh, 3 + xoff, 2 + yoff)
        uym_xy = _xy(uh, 2 + xoff, 1 + yoff)
        uyp_xy = _xy(uh, 2 + xoff, 3 + yoff)

        uc = _sample_z(uc_xy, zoff, nz_coarse)
        lap = (_sample_z(uxm_xy, zoff, nz_coarse) +
               _sample_z(uxp_xy, zoff, nz_coarse) +
               _sample_z(uym_xy, zoff, nz_coarse) +
               _sample_z(uyp_xy, zoff, nz_coarse) +
               _sample_z(uc_xy, zoff - 1, nz_coarse) +
               _sample_z(uc_xy, zoff + 1, nz_coarse) - 6.0 * uc)
        return f - lap * inv_h2

    def _restrict_local(fh: Array, uh: Array) -> Array:
        out = None
        for xoff, wx in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            for yoff, wy in ((-1, 0.25), (0, 0.5), (1, 0.25)):
                for zoff, wz in ((-1, 0.25), (0, 0.5), (1, 0.25)):
                    term = (wx * wy * wz) * _residual_sample(
                        fh, uh, xoff, yoff, zoff)
                    out = term if out is None else out + term
        return out

    if not _has_mesh(mesh):
        return _restrict_local(F_halo, U_halo)

    @partial(shard_map,
             mesh=mesh,
             in_specs=(FIELD_SPEC, FIELD_SPEC),
             out_specs=FIELD_SPEC)
    def _local(fh, uh):
        return _restrict_local(fh, uh)

    return _local(F_halo, U_halo)


def _prolong_add_from_halo(U: Array, Ec_halo: Array,
                           mesh: Optional[Mesh]) -> Array:
    """Add trilinear prolongation of halo-padded coarse correction into fine U."""
    if not _has_mesh(mesh):
        c = Ec_halo[1:-1, 1:-1, :]
        xp = Ec_halo[2:, 1:-1, :]
        yp = Ec_halo[1:-1, 2:, :]
        xyp = Ec_halo[2:, 2:, :]

        cz = jnp.roll(c, -1, 2)
        xpz = jnp.roll(xp, -1, 2)
        ypz = jnp.roll(yp, -1, 2)
        xypz = jnp.roll(xyp, -1, 2)

        out = U
        out = out.at[0::2, 0::2, 0::2].add(c)
        out = out.at[1::2, 0::2, 0::2].add(0.5 * (c + xp))
        out = out.at[0::2, 1::2, 0::2].add(0.5 * (c + yp))
        out = out.at[0::2, 0::2, 1::2].add(0.5 * (c + cz))
        out = out.at[1::2, 1::2, 0::2].add(0.25 * (c + xp + yp + xyp))
        out = out.at[1::2, 0::2, 1::2].add(0.25 * (c + xp + cz + xpz))
        out = out.at[0::2, 1::2, 1::2].add(0.25 * (c + yp + cz + ypz))
        out = out.at[1::2, 1::2, 1::2].add(
            0.125 * (c + xp + yp + xyp + cz + xpz + ypz + xypz))
        return out

    @partial(shard_map,
             mesh=mesh,
             in_specs=(FIELD_SPEC, FIELD_SPEC),
             out_specs=FIELD_SPEC)
    def _local(u, ec):
        c = ec[1:-1, 1:-1, :]
        xp = ec[2:, 1:-1, :]
        yp = ec[1:-1, 2:, :]
        xyp = ec[2:, 2:, :]

        cz = jnp.roll(c, -1, 2)
        xpz = jnp.roll(xp, -1, 2)
        ypz = jnp.roll(yp, -1, 2)
        xypz = jnp.roll(xyp, -1, 2)

        out = u
        out = out.at[0::2, 0::2, 0::2].add(c)
        out = out.at[1::2, 0::2, 0::2].add(0.5 * (c + xp))
        out = out.at[0::2, 1::2, 0::2].add(0.5 * (c + yp))
        out = out.at[0::2, 0::2, 1::2].add(0.5 * (c + cz))
        out = out.at[1::2, 1::2, 0::2].add(0.25 * (c + xp + yp + xyp))
        out = out.at[1::2, 0::2, 1::2].add(0.25 * (c + xp + cz + xpz))
        out = out.at[0::2, 1::2, 1::2].add(0.25 * (c + yp + cz + ypz))
        out = out.at[1::2, 1::2, 1::2].add(
            0.125 * (c + xp + yp + xyp + cz + xpz + ypz + xypz))
        return out

    return _local(U, Ec_halo)


def prolong_add_halo(
    U: Array,
    Ec: Array,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
) -> Array:
    """Add trilinear prolongation of Ec into U without materializing correction."""
    Ec_halo = _exchange_xy(Ec, mesh, halo_backend)
    return _prolong_add_from_halo(U, Ec_halo, mesh)


def _agg_coarse_solve(
    F: Array,
    U: Array,
    level: int,
    h: float,
    *,
    mesh: Mesh,
    halo_backend: str,
    v1: int,
    v2: int,
    mu: int,
) -> Array:
    """Coarse-grid agglomeration: gather the (now small) coarse grid to every device,
    solve the remaining V-cycle levels single-device via the local jnp.roll path (NO
    inter-node halo exchange), then reshard. Numerically identical to the sharded recursion
    (same operators / sweeps / bottom solve) -- only the communication pattern changes. The
    sharded path's deepest levels are a few cells/device, so their halo exchanges are pure
    inter-node latency; doing them on one replicated copy removes that. Must run inside a jit."""
    rep = NamedSharding(mesh, P(None, None, None))
    shd = NamedSharding(mesh, FIELD_SPEC)
    F_r = jax.lax.with_sharding_constraint(F, rep)
    U_r = jax.lax.with_sharding_constraint(U, rep)
    # On the replicated copy we are no longer bound by the sharded-axis divisibility that caps
    # `level` (a >=4-way sharded axis stops the V-cycle early, e.g. at 12^3, leaving the long-
    # wavelength mode unresolved -> low residual but large solution error). Coarsen as deeply as
    # clean factor-2 allows so the smoothest mode is actually solved.
    g = min(int(s) for s in F.shape)
    local_l, t = 0, g
    while t % 2 == 0 and t > 3:
        t //= 2
        local_l += 1
    local_l = max(local_l, int(level))
    E_r = poisson_multigrid_halo(
        F_r,
        U_r,
        l=local_l,
        v1=v1,
        v2=v2,
        mu=mu,
        iter_cycle=1,
        h=h,
        mesh=None,
        halo_backend=halo_backend,
    )
    return jax.lax.with_sharding_constraint(E_r, shd)


def _cycle_halo(
    F: Array,
    U: Array,
    level: int,
    h: float,
    *,
    mesh: Optional[Mesh],
    halo_backend: str,
    v1: int,
    v2: int,
    mu: int,
    agg_n: int = 0,
) -> Array:
    # Agglomerate the coarse levels onto one replicated copy once the grid is small enough.
    if _has_mesh(mesh) and agg_n > 0 and 2 < min(U.shape) <= agg_n:
        return _agg_coarse_solve(F,
                                 U,
                                 level,
                                 h,
                                 mesh=mesh,
                                 halo_backend=halo_backend,
                                 v1=v1,
                                 v2=v2,
                                 mu=mu)
    if level <= 0 or min(U.shape) <= 2:
        return smooth_weighted_jacobi_halo(U,
                                           F,
                                           h,
                                           iters=16,
                                           mesh=mesh,
                                           halo_backend=halo_backend)

    U = smooth_weighted_jacobi_halo(U,
                                    F,
                                    h,
                                    iters=v1,
                                    mesh=mesh,
                                    halo_backend=halo_backend)
    Rc = _restrict(F, U, h, mesh=mesh, halo_backend=halo_backend)

    Ec = jnp.zeros_like(Rc, dtype=U.dtype)
    for _ in range(int(mu)):
        Ec = _cycle_halo(
            Rc,
            Ec,
            level - 1,
            2.0 * h,
            mesh=mesh,
            halo_backend=halo_backend,
            v1=v1,
            v2=v2,
            mu=mu,
            agg_n=agg_n,
        )

    U = prolong_add_halo(U, Ec, mesh=mesh, halo_backend=halo_backend)
    return smooth_weighted_jacobi_halo(U,
                                       F,
                                       h,
                                       iters=v2,
                                       mesh=mesh,
                                       halo_backend=halo_backend)


# Full-multigrid (nested iteration) toggle. FMG restricts the RHS to the bottom, solves, and
# prolongs upward doing `n_fmg` V-cycles per level. The finest grid is visited only once (n_fmg
# V-cycles) instead of `iter_cycle` times from a zero guess, so the dominant finest-level
# restriction/prolong/transfer cost is ~halved while reaching truncation-level accuracy.
CA_FMG = False
CA_FMG_NCYCLE = 1
# FMG RHS restriction: full-weighting (separable, accurate but expensive at the fine level) vs
# injection (plain decimation, ~free). The nested V-cycles fix up the coarser RHS error, so
# injection usually keeps FMG accuracy while removing the costly full-fine separable passes.
CA_FMG_INJECT = False


def _restrict_rhs(F, mesh, halo_backend):
    if CA_FMG_INJECT:
        return F[::2, ::2, ::2]
    return restrict_full_weighting_halo(F,
                                        mesh=mesh,
                                        halo_backend=halo_backend)


def _fmg_halo(
    F: Array,
    level: int,
    h: float,
    *,
    mesh: Optional[Mesh],
    halo_backend: str,
    v1: int,
    v2: int,
    mu: int,
    agg_n: int,
    n_fmg: int,
) -> Array:
    """Full-multigrid V-cycle. Returns an approximate solution U (not a correction)."""
    # Agglomerate the whole FMG tail once the grid is small (mirrors _agg_coarse_solve): do the
    # nested iteration single-device on a replicated copy, no inter-node halo exchange.
    if _has_mesh(mesh) and agg_n > 0 and 2 < min(F.shape) <= agg_n:
        rep = NamedSharding(mesh, P(None, None, None))
        shd = NamedSharding(mesh, FIELD_SPEC)
        F_r = jax.lax.with_sharding_constraint(F, rep)
        g = min(int(s) for s in F.shape)
        local_l, t = 0, g
        while t % 2 == 0 and t > 3:
            t //= 2
            local_l += 1
        local_l = max(local_l, int(level))
        U_r = _fmg_halo(
            F_r,
            local_l,
            h,
            mesh=None,
            halo_backend=halo_backend,
            v1=v1,
            v2=v2,
            mu=mu,
            agg_n=0,
            n_fmg=n_fmg,
        )
        return jax.lax.with_sharding_constraint(U_r, shd)

    if level <= 0 or min(F.shape) <= 2:
        return smooth_weighted_jacobi_halo(jnp.zeros_like(F),
                                           F,
                                           h,
                                           iters=16,
                                           mesh=mesh,
                                           halo_backend=halo_backend)

    # Restrict the RHS, FMG-solve the coarse problem, prolong as the fine initial guess.
    Fc = _restrict_rhs(F, mesh, halo_backend)
    Uc = _fmg_halo(
        Fc,
        level - 1,
        2.0 * h,
        mesh=mesh,
        halo_backend=halo_backend,
        v1=v1,
        v2=v2,
        mu=mu,
        agg_n=agg_n,
        n_fmg=n_fmg,
    )
    U = prolong_add_halo(jnp.zeros_like(F),
                         Uc,
                         mesh=mesh,
                         halo_backend=halo_backend)
    for _ in range(int(n_fmg)):
        U = _cycle_halo(
            F,
            U,
            level,
            h,
            mesh=mesh,
            halo_backend=halo_backend,
            v1=v1,
            v2=v2,
            mu=mu,
            agg_n=agg_n,
        )
    return U


def poisson_multigrid_halo(
    F: Array,
    U: Array,
    *,
    l: int,
    v1: int,
    v2: int,
    mu: int,
    iter_cycle: int,
    h: float = 1.0,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
    agg_n: int = 0,
) -> Array:
    """Solve A phi = F with fixed halo-backed V/W-cycles (or FMG when CA_FMG)."""
    if CA_FMG:
        U_out = _fmg_halo(
            F,
            int(l),
            h,
            mesh=mesh,
            halo_backend=halo_backend,
            v1=int(v1),
            v2=int(v2),
            mu=int(mu),
            agg_n=int(agg_n),
            n_fmg=int(CA_FMG_NCYCLE),
        )
        # Optional extra full V-cycles after the FMG pass (iter_cycle-1 of them).
        for _ in range(int(iter_cycle) - 1):
            U_out = _cycle_halo(
                F,
                U_out,
                int(l),
                h,
                mesh=mesh,
                halo_backend=halo_backend,
                v1=int(v1),
                v2=int(v2),
                mu=int(mu),
                agg_n=int(agg_n),
            )
        return U_out
    U_out = U
    for _ in range(int(iter_cycle)):
        U_out = _cycle_halo(
            F,
            U_out,
            int(l),
            h,
            mesh=mesh,
            halo_backend=halo_backend,
            v1=int(v1),
            v2=int(v2),
            mu=int(mu),
            agg_n=int(agg_n),
        )
    return U_out


def make_poisson_mg_halo_solver(
    *,
    l: int,
    v1: int,
    v2: int,
    mu: int,
    iter_cycle: int,
    h: float,
    mesh: Optional[Mesh] = None,
    halo_backend: str = "jax",
    jit_solver: bool = True,
    agg_n: int = 0,
):
    """Return a halo-backed MG solve(F) function.

    `jit_solver=False` keeps the recursive V-cycle out of one monolithic XLA
    executable. It still stages the expensive MG kernels with per-kernel JIT so
    large intermediate buffers can die between dispatches without falling back to
    very slow op-by-op eager execution.

    `agg_n>0` enables coarse-grid agglomeration: once a coarse grid is <= agg_n cells per
    side, gather it to a replicated copy and finish the V-cycle single-device, removing the
    latency-bound inter-node halo exchanges on the deepest (few-cells/device) levels.
    """

    def _solve(F: Array, U0: Optional[Array] = None) -> Array:
        # Warm start: pass U0 = last step's potential (optionally growth-scaled,
        # U0 = (D_n/D_{n-1})*phi_prev) to cut the cycles needed when phi varies slowly.
        # The FFT cannot exploit this. Omit U0 for a cold (zero) start.
        U_init = jnp.zeros_like(F, dtype=jnp.float32) if U0 is None else U0
        return poisson_multigrid_halo(
            F,
            U_init,
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            h=h,
            mesh=mesh,
            halo_backend=halo_backend,
            agg_n=agg_n,
        )

    if jit_solver:
        return jax.jit(_solve)

    @partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0, ))
    def _smooth_staged(U: Array, F: Array, h_level: float,
                       iters: int) -> Array:
        return smooth_weighted_jacobi_halo(
            U,
            F,
            h_level,
            iters=iters,
            mesh=mesh,
            halo_backend=halo_backend,
        )

    @partial(jax.jit, static_argnums=(2, ))
    def _restrict_staged(F: Array, U: Array, h_level: float) -> Array:
        return _restrict(F, U, h_level, mesh=mesh, halo_backend=halo_backend)

    @partial(jax.jit, donate_argnums=(0, 1))
    def _prolong_staged(U: Array, Ec: Array) -> Array:
        return prolong_add_halo(U, Ec, mesh=mesh, halo_backend=halo_backend)

    @partial(jax.jit, static_argnums=(2, 3))
    def _agg_staged(F: Array, U: Array, level: int, h_level: float) -> Array:
        return _agg_coarse_solve(F,
                                 U,
                                 level,
                                 h_level,
                                 mesh=mesh,
                                 halo_backend=halo_backend,
                                 v1=v1,
                                 v2=v2,
                                 mu=mu)

    def _cycle_staged(F: Array, U: Array, level: int, h_level: float) -> Array:
        # Agglomerate onto one replicated copy once the grid is small (kills the deepest,
        # latency-bound sharded halo exchanges); finish those levels single-device.
        if _has_mesh(mesh) and agg_n > 0 and 2 < min(U.shape) <= agg_n:
            return _agg_staged(F, U, level, h_level)
        if level <= 0 or min(U.shape) <= 2:
            return _smooth_staged(U, F, h_level, 16)

        U = _smooth_staged(U, F, h_level, int(v1))
        Rc = _restrict_staged(F, U, h_level)

        Ec = jnp.zeros_like(Rc, dtype=U.dtype)
        for _ in range(int(mu)):
            Ec = _cycle_staged(Rc, Ec, level - 1, 2.0 * h_level)

        U = _prolong_staged(U, Ec)
        return _smooth_staged(U, F, h_level, int(v2))

    def _solve_staged(F: Array) -> Array:
        U_out = jnp.zeros_like(F, dtype=jnp.float32)
        for _ in range(int(iter_cycle)):
            U_out = _cycle_staged(F, U_out, int(l), h)
        return U_out

    return _solve_staged
