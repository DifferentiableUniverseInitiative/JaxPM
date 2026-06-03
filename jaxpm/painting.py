import warnings
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, fft3d, get_halo_size, halo_exchange,
                               ifft3d, slice_pad, slice_unpad)
from jaxpm.kernels import compensation_kernel, fftk, resolve_order
from jaxpm.painting_utils import gather, scatter


def _cic_paint_impl(grid_mesh, positions, weight=1.):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    displacement field: [nx, ny, nz, 3]
    """

    positions = positions.reshape([-1, 3])
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    if jnp.isscalar(weight):
        kernel = jnp.multiply(jnp.expand_dims(weight, axis=-1), kernel)
    else:
        kernel = jnp.multiply(weight.reshape(*positions.shape[:-1]), kernel)

    neighboor_coords = jnp.mod(
        neighboor_coords.reshape([-1, 8, 3]).astype('int32'),
        jnp.array(grid_mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1, 2),
                                            scatter_dims_to_operand_dims=(0, 1,
                                                                          2))
    mesh = lax.scatter_add(grid_mesh, neighboor_coords,
                           kernel.reshape([-1, 8]), dnums)
    return mesh


@partial(jax.jit, static_argnums=(3, 4))
def cic_paint(grid_mesh, positions, weight=1., halo_size=0, sharding=None):
    warnings.warn(
        "cic_paint is deprecated; use paint(positions, grid_mesh, "
        "order='cic') instead.",
        DeprecationWarning,
        stacklevel=2)

    if sharding is not None:
        print("""
            WARNING : absolute painting is not recommended in multi-device mode.
            Please use relative painting instead.
            """)

    positions = positions.reshape((*grid_mesh.shape, 3))

    halo_size, halo_extents = get_halo_size(halo_size, sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding)

    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    weight_spec = P() if jnp.isscalar(weight) else spec

    grid_mesh = autoshmap(_cic_paint_impl,
                          gpu_mesh=gpu_mesh,
                          in_specs=(spec, spec, weight_spec),
                          out_specs=spec)(grid_mesh, positions, weight)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)

    return grid_mesh


def _cic_read_impl(grid_mesh, positions):
    """ Paints positions onto mesh
    mesh: [nx, ny, nz]
    positions: [nx,ny,nz, 3]
    """
    # Save original shape for reshaping output later
    original_shape = positions.shape
    # Reshape positions to a flat list of 3D coordinates
    positions = positions.reshape([-1, 3])
    # Expand dimensions to calculate neighbor coordinates
    positions = jnp.expand_dims(positions, 1)
    # Floor the positions to get the base grid cell for each particle
    floor = jnp.floor(positions)
    # Define connections to calculate all neighbor coordinates
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])
    # Calculate the 8 neighboring coordinates
    neighboor_coords = floor + connection
    # Calculate kernel weights based on distance from each neighboring coordinate
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    # Modulo operation to wrap around edges if necessary
    neighboor_coords = jnp.mod(neighboor_coords.astype('int32'),
                               jnp.array(grid_mesh.shape))
    # Ensure grid_mesh shape is as expected
    # Retrieve values from grid_mesh at each neighboring coordinate and multiply by kernel
    return (grid_mesh[neighboor_coords[..., 0],
                      neighboor_coords[..., 1],
                      neighboor_coords[..., 2]] * kernel).sum(axis=-1).reshape(original_shape[:-1]) # yapf: disable


@partial(jax.jit, static_argnums=(2, 3))
def cic_read(grid_mesh, positions, halo_size=0, sharding=None):
    warnings.warn(
        "cic_read is deprecated; use readout(grid_mesh, positions, "
        "order='cic') instead.",
        DeprecationWarning,
        stacklevel=2)

    original_shape = positions.shape
    positions = positions.reshape((*grid_mesh.shape, 3))

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()

    displacement = autoshmap(_cic_read_impl,
                             gpu_mesh=gpu_mesh,
                             in_specs=(spec, spec),
                             out_specs=spec)(grid_mesh, positions)

    return displacement.reshape(original_shape[:-1])


def cic_paint_2d(mesh, positions, weight):
    """ Paints positions onto a 2d mesh
    mesh: [nx, ny]
    positions: [..., 2]  (arbitrary batch dims preserved until scatter)
    weight: [...] or None
    """
    positions = jnp.expand_dims(positions,
                                -2)  # (*batch, 1, 2) — preserves sharding
    floor = jnp.floor(positions)
    connection = jnp.array([[0, 0], [1., 0], [0., 1], [1., 1]])

    neighboor_coords = floor + connection  # (*batch, 4, 2)
    kernel = 1. - jnp.abs(positions - neighboor_coords)  # (*batch, 4, 2)
    kernel = kernel[..., 0] * kernel[..., 1]  # (*batch, 4)
    if weight is not None:
        kernel = kernel * weight[
            ..., None]  # (*batch, 1) broadcasts with (*batch, 4)

    # Flatten batch dims for scatter — communication is unavoidable here
    neighboor_coords = jnp.mod(neighboor_coords.astype('int32'),
                               jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(),
                                            inserted_window_dims=(0, 1),
                                            scatter_dims_to_operand_dims=(0,
                                                                          1))
    mesh = lax.scatter_add(mesh, neighboor_coords, kernel, dnums)
    return mesh


def _cic_paint_dx_impl(displacements,
                       weight=1.,
                       halo_size=0,
                       chunk_size=2**24):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = displacements.shape
    particle_mesh = jnp.zeros(original_shape[:-1], dtype=displacements.dtype)
    if not jnp.isscalar(weight):
        if weight.shape != original_shape[:-1]:
            raise ValueError("Weight shape must match particle shape")
        else:
            weight = weight.flatten()
    # Padding is forced to be zero in a single gpu run

    a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                           jnp.arange(particle_mesh.shape[1]),
                           jnp.arange(particle_mesh.shape[2]),
                           indexing='ij')

    particle_mesh = jnp.pad(particle_mesh, halo_size)
    pmid = jnp.stack([a + halo_x, b + halo_y, c], axis=-1)
    return scatter(pmid.reshape([-1, 3]),
                   displacements.reshape([-1, 3]),
                   particle_mesh,
                   chunk_size=chunk_size,
                   val=weight)


@partial(jax.jit, static_argnums=(1, 2, 4))
def cic_paint_dx(displacements,
                 halo_size=0,
                 sharding=None,
                 weight=1.0,
                 chunk_size=2**24):
    warnings.warn(
        "cic_paint_dx is deprecated; use paint(displacements, "
        "initial_particles='uniform', order='cic') instead.",
        DeprecationWarning,
        stacklevel=2)

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)

    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    weight_spec = P() if jnp.isscalar(weight) else spec
    grid_mesh = autoshmap(partial(_cic_paint_dx_impl,
                                  halo_size=halo_size,
                                  chunk_size=chunk_size),
                          gpu_mesh=gpu_mesh,
                          in_specs=(spec, weight_spec),
                          out_specs=spec)(displacements, weight)

    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_size, sharding)
    return grid_mesh


def _cic_read_dx_impl(grid_mesh, disp, halo_size):

    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]

    original_shape = [
        dim - 2 * halo[0] for dim, halo in zip(grid_mesh.shape, halo_size)
    ]
    a, b, c = jnp.meshgrid(jnp.arange(original_shape[0]),
                           jnp.arange(original_shape[1]),
                           jnp.arange(original_shape[2]),
                           indexing='ij')

    pmid = jnp.stack([a + halo_x, b + halo_y, c], axis=-1)

    pmid = pmid.reshape([-1, 3])
    disp = disp.reshape([-1, 3])

    return gather(pmid, disp, grid_mesh).reshape(original_shape)


@partial(jax.jit, static_argnums=(2, 3))
def cic_read_dx(grid_mesh, disp, halo_size=0, sharding=None):
    warnings.warn(
        "cic_read_dx is deprecated; use readout(grid_mesh, disp, "
        "initial_particles='uniform', order='cic') instead.",
        DeprecationWarning,
        stacklevel=2)

    halo_size, halo_extents = get_halo_size(halo_size, sharding=sharding)
    # Halo size is halved for the read operation
    # We only need to read the density field
    # while in the painting operation we need to exchange and reduce the halo
    # We chose to do that since it is much easier to write a custom jvp rule for exchange
    # while it is a bit harder if there is a reduction involved
    halo_size = jax.tree.map(lambda x: x // 2, halo_size)
    grid_mesh = slice_pad(grid_mesh, halo_size, sharding=sharding)
    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    displacements = autoshmap(partial(_cic_read_dx_impl, halo_size=halo_size),
                              gpu_mesh=gpu_mesh,
                              in_specs=(spec),
                              out_specs=spec)(grid_mesh, disp)

    return displacements


def compensate_cic(field):
    """
    Compensate for CiC painting
    Args:
      field: input 3D cic-painted field
    Returns:
      compensated_field
    """
    delta_k = fft3d(field)

    kvec = fftk(delta_k)
    delta_k = compensation_kernel(kvec, 2) * delta_k
    return ifft3d(delta_k)


# ===========================================================================
# Higher-order mass assignment: NGP / CIC / TSC / PCS
# ---------------------------------------------------------------------------
# A single order-parameterized ``paint`` / ``readout`` pair that supersedes the
# CIC-specific functions above. The per-particle neighbour tensor is never
# materialized: we ``lax.scan`` over the ``order**3`` stencil offsets and
# accumulate into the mesh (peak memory O(N + mesh), no per-particle chunking).
# The scan body is rematerialized (``jax.checkpoint``) so reverse-mode AD keeps
# the same memory bound -- important because this runs inside ``pm_forces``.
# ===========================================================================


def _bspline_weight(s, order):
    """B-spline assignment weight at distance ``s`` for a static ``order``.

    ``s`` is the non-negative distance between a particle and a candidate grid
    cell. Matches jax-power's ``_resampler_kernels``.
    """
    if order == 1:  # NGP
        return jnp.ones_like(s)
    elif order == 2:  # CIC
        return 1.0 - s
    elif order == 3:  # TSC
        return jnp.where(s <= 0.5, 0.75 - s**2, 0.5 * (1.5 - s)**2)
    else:  # order == 4, PCS
        return jnp.where(s <= 1.0, (4.0 - 6.0 * s**2 + 3.0 * s**3) / 6.0,
                         (2.0 - s)**3 / 6.0)


def _stencil_offsets(order):
    """Static ``[order**3, 3]`` integer offsets ``{0, ..., order-1}**3``."""
    rng = np.arange(order)
    a, b, c = np.meshgrid(rng, rng, rng, indexing='ij')
    return jnp.asarray(np.stack(
        [a.ravel(), b.ravel(), c.ravel()], axis=-1),
                       dtype='int32')


def _assign_setup(positions, order):
    """Lowest contributing cell ``i0`` and fractional offset ``frac``.

    ``i0 = floor(pos - order/2 + 1)`` yields the correct odd/even stencil
    centering for every order (NGP -> round(x), CIC -> floor(x), ...) and
    ``frac = pos - i0`` keeps every per-cell distance inside the B-spline
    support. Works for negative ``pos`` too (``floor`` + ``mod`` downstream).
    """
    i0 = jnp.floor(positions - order / 2.0 + 1.0)
    frac = positions - i0
    return i0.astype('int32'), frac


def _paint_order_impl(grid_mesh, positions, weight, order):
    """Scatter absolute ``positions`` onto ``grid_mesh`` (order-``order``)."""
    positions = positions.reshape([-1, 3])
    mesh_shape = jnp.array(grid_mesh.shape)
    i0, frac = _assign_setup(positions, order)

    weight = jnp.asarray(weight)
    wpart = weight if weight.ndim == 0 else weight.reshape(-1)

    offsets = _stencil_offsets(order)

    @jax.checkpoint
    def body(mesh, off):
        idx = jnp.mod(i0 + off, mesh_shape)
        s = jnp.abs(frac - off)
        w = (_bspline_weight(s[:, 0], order) * _bspline_weight(s[:, 1], order)
             * _bspline_weight(s[:, 2], order)) * wpart
        mesh = mesh.at[idx[:, 0], idx[:, 1], idx[:, 2]].add(w)
        return mesh, None

    grid_mesh, _ = lax.scan(body, grid_mesh, offsets)
    return grid_mesh


def _readout_order_impl(grid_mesh, positions, order):
    """Gather values from ``grid_mesh`` at absolute ``positions`` (order-``order``)."""
    original_shape = positions.shape
    positions = positions.reshape([-1, 3])
    mesh_shape = jnp.array(grid_mesh.shape)
    i0, frac = _assign_setup(positions, order)

    offsets = _stencil_offsets(order)

    @jax.checkpoint
    def body(val, off):
        idx = jnp.mod(i0 + off, mesh_shape)
        s = jnp.abs(frac - off)
        w = (_bspline_weight(s[:, 0], order) *
             _bspline_weight(s[:, 1], order) * _bspline_weight(s[:, 2], order))
        val = val + grid_mesh[idx[:, 0], idx[:, 1], idx[:, 2]] * w
        return val, None

    val0 = jnp.zeros(positions.shape[0], dtype=grid_mesh.dtype)
    val, _ = lax.scan(body, val0, offsets)
    return val.reshape(original_shape[:-1])


def _uniform_pmid(shape, halo_size):
    """Uniform integer particle grid for displacement painting, with halo offset.

    Confined to the local device domain (built inside ``shard_map``), so a
    particle's index only needs to be known within its local slice.
    """
    halo_x, _ = halo_size[0]
    halo_y, _ = halo_size[1]
    a, b, c = jnp.meshgrid(jnp.arange(shape[0]),
                           jnp.arange(shape[1]),
                           jnp.arange(shape[2]),
                           indexing='ij')
    return jnp.stack([a + halo_x, b + halo_y, c], axis=-1)


def _paint_dx_order_impl(displacements, weight, halo_size, order):
    """Displacement-mode scatter: add a uniform grid to ``displacements``."""
    local_shape = displacements.shape[:-1]
    particle_mesh = jnp.zeros(local_shape, dtype=displacements.dtype)
    particle_mesh = jnp.pad(particle_mesh, halo_size)
    pmid = _uniform_pmid(local_shape, halo_size)
    positions = pmid + displacements
    return _paint_order_impl(particle_mesh, positions, weight, order)


def _readout_dx_order_impl(grid_mesh, disp, halo_size, order):
    """Displacement-mode gather: read at uniform-grid + ``disp`` positions."""
    local_shape = [
        dim - 2 * halo[0] for dim, halo in zip(grid_mesh.shape, halo_size)
    ]
    pmid = _uniform_pmid(local_shape, halo_size)
    positions = pmid + disp
    return _readout_order_impl(grid_mesh, positions, order)


def _deconvolve(field, order):
    """Divide a painted real field by the assignment window (one factor)."""
    delta_k = fft3d(field)
    kvec = fftk(delta_k)
    return ifft3d(compensation_kernel(kvec, order) * delta_k)


def _check_initial_particles(initial_particles):
    if initial_particles not in (None, 'uniform'):
        raise ValueError(
            "initial_particles must be None (absolute positions) or 'uniform' "
            f"(displacements), got {initial_particles!r}")


@partial(jax.jit,
         static_argnames=('initial_particles', 'order', 'deconvolution',
                          'halo_size', 'sharding'))
def paint(positions,
          grid_mesh=None,
          initial_particles=None,
          weight=1.0,
          order='CIC',
          deconvolution=False,
          halo_size=0,
          sharding=None):
    """Paint particles onto a 3D mesh with an NGP/CIC/TSC/PCS scheme.

    Parameters
    ----------
    positions : array
        Absolute positions ``(..., 3)`` if ``initial_particles is None``, or
        displacements ``(nx, ny, nz, 3)`` from a uniform grid if
        ``initial_particles == 'uniform'``.
    grid_mesh : array, optional
        Pre-allocated output mesh (absolute mode only). If ``None`` a zero mesh
        is allocated (and sharding-constrained when ``sharding`` is given).
    initial_particles : {None, 'uniform'}
        ``None`` -> ``positions`` are absolute. ``'uniform'`` -> ``positions``
        are displacements; a uniform integer grid local to each device is added.
    weight : float or array
        Per-particle weight (scalar) or array matching the particle grid.
    order : int or str
        Assignment order: NGP=1, CIC=2, TSC=3, PCS=4 (name or integer).
    deconvolution : bool
        If True, divide the painted field by the assignment window
        ``compensation_kernel(k, order)`` in Fourier space before returning.
    halo_size, sharding :
        Distributed halo width and JAX sharding (machinery unchanged from CIC).

    Returns
    -------
    grid_mesh : array
        The painted (and optionally deconvolved) density field.
    """
    order = resolve_order(order)
    _check_initial_particles(initial_particles)

    halo_tuple, halo_extents = get_halo_size(halo_size, sharding)
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    weight_spec = P() if jnp.ndim(weight) == 0 else spec

    if initial_particles is None:
        if sharding is not None:
            warnings.warn(
                "Absolute painting is not recommended in multi-device mode; "
                "use initial_particles='uniform' (displacements) instead.",
                stacklevel=2)
        if grid_mesh is None:
            grid_mesh = jnp.zeros(positions.shape[:-1], dtype=positions.dtype)
        positions = positions.reshape((*grid_mesh.shape, 3))
        if sharding is not None:
            positions = jax.lax.with_sharding_constraint(positions, sharding)
            grid_mesh = jax.lax.with_sharding_constraint(grid_mesh, sharding)
        grid_mesh = slice_pad(grid_mesh, halo_tuple, sharding)
        grid_mesh = autoshmap(partial(_paint_order_impl, order=order),
                              gpu_mesh=gpu_mesh,
                              in_specs=(spec, spec, weight_spec),
                              out_specs=spec)(grid_mesh, positions, weight)
    else:
        if sharding is not None:
            positions = jax.lax.with_sharding_constraint(positions, sharding)
        grid_mesh = autoshmap(partial(_paint_dx_order_impl,
                                      halo_size=halo_tuple,
                                      order=order),
                              gpu_mesh=gpu_mesh,
                              in_specs=(spec, weight_spec),
                              out_specs=spec)(positions, weight)

    grid_mesh = halo_exchange(grid_mesh,
                              halo_extents=halo_extents,
                              halo_periods=(True, True))
    grid_mesh = slice_unpad(grid_mesh, halo_tuple, sharding)

    if deconvolution:
        # Real-space round-trip (FFT -> window^-1 -> iFFT) so paint() returns a
        # deconvolved *field*. Callers already working in Fourier space (e.g.
        # pm_forces) skip this and multiply delta_k by compensation_kernel(k,
        # order) directly -- one fewer FFT/iFFT pair.
        grid_mesh = _deconvolve(grid_mesh, order)

    return grid_mesh


@partial(jax.jit,
         static_argnames=('initial_particles', 'order', 'halo_size',
                          'sharding'))
def readout(grid_mesh,
            positions,
            initial_particles=None,
            order='CIC',
            halo_size=0,
            sharding=None):
    """Read mesh values at particle positions with an NGP/CIC/TSC/PCS scheme.

    Mirror of :func:`paint` (gather instead of scatter). ``positions`` are
    absolute if ``initial_particles is None`` or displacements from a uniform
    grid if ``initial_particles == 'uniform'``. No deconvolution (readout
    returns sampled values, e.g. forces, not a field to compensate).
    """
    order = resolve_order(order)
    _check_initial_particles(initial_particles)

    halo_tuple, halo_extents = get_halo_size(halo_size, sharding=sharding)
    gpu_mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()

    if initial_particles is None:
        original_shape = positions.shape
        positions = positions.reshape((*grid_mesh.shape, 3))
        grid_mesh = slice_pad(grid_mesh, halo_tuple, sharding=sharding)
        grid_mesh = halo_exchange(grid_mesh,
                                  halo_extents=halo_extents,
                                  halo_periods=(True, True))
        out = autoshmap(partial(_readout_order_impl, order=order),
                        gpu_mesh=gpu_mesh,
                        in_specs=(spec, spec),
                        out_specs=spec)(grid_mesh, positions)
        return out.reshape(original_shape[:-1])
    else:
        # Read only gathers (no halo reduction), so it needs only half the halo
        # *padding* width of a paint -- same convention as the deprecated
        # cic_read_dx. This is distinct from the exchange *extent*, which
        # get_halo_size already halves (get_halo_size returns the full padding).
        halo_tuple = jax.tree.map(lambda x: x // 2, halo_tuple)
        grid_mesh = slice_pad(grid_mesh, halo_tuple, sharding=sharding)
        grid_mesh = halo_exchange(grid_mesh,
                                  halo_extents=halo_extents,
                                  halo_periods=(True, True))
        return autoshmap(partial(_readout_dx_order_impl,
                                 halo_size=halo_tuple,
                                 order=order),
                         gpu_mesh=gpu_mesh,
                         in_specs=(spec, spec),
                         out_specs=spec)(grid_mesh, positions)
