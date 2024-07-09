from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import scan
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomp import halo_exchange

from jaxpm._src.spmd_config import (CallBackOperator, CustomPartionedOperator,
                                    ShardedOperator, register_operator)
from jaxpm.ops import slice_pad, slice_unpad


class CICPaintOperator(ShardedOperator):

    name = 'cic_paint'

    def single_gpu_impl(particle_mesh: jnp.ndarray,
                        positions: jnp.ndarray,
                        halo_size=0):

        del halo_size

        positions = positions.reshape([-1, 3])

        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)

        connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                                 [1., 1, 0], [1., 0, 1], [0., 1, 1],
                                 [1., 1, 1]]])

        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords_mod = jnp.mod(
            neighboor_coords.reshape([-1, 8, 3]).astype('int32'),
            jnp.array(particle_mesh.shape))

        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2))
        particle_mesh = lax.scatter_add(particle_mesh, neighboor_coords_mod,
                                        kernel.reshape([-1, 8]), dnums)

        return particle_mesh

    def multi_gpu_impl(particle_mesh: jnp.ndarray,
                       positions: jnp.ndarray,
                       halo_size=8,
                       __aux_input=None):

        rank = jax.process_index()
        correct_y = -particle_mesh.shape[1] * (rank // __aux_input[0])
        correct_z = -particle_mesh.shape[0] * (rank % __aux_input[1])
        # Get positions relative to the start of each slice
        positions = positions.at[:, :, :, 1].add(correct_y)
        positions = positions.at[:, :, :, 0].add(correct_z)
        positions = positions.reshape([-1, 3])

        halo_tuple = (halo_size, halo_size)
        if __aux_input[0] == 1:
            halo_width = ((0, 0), halo_tuple, (0, 0))
            halo_start = [0, halo_size, 0]
        elif __aux_input[1] == 1:
            halo_width = (halo_tuple, (0, 0), (0, 0))
            halo_start = [halo_size, 0, 0]
        else:
            halo_width = (halo_tuple, halo_tuple, (0, 0))
            halo_start = [halo_size, halo_size, 0]

        particle_mesh = jnp.pad(particle_mesh, halo_width)
        positions += jnp.array(halo_start).reshape([-1, 3])

        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)

        connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                                 [1., 1, 0], [1., 0, 1], [0., 1, 1],
                                 [1., 1, 1]]])

        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords_mod = jnp.mod(
            neighboor_coords.reshape([-1, 8, 3]).astype('int32'),
            jnp.array(particle_mesh.shape))

        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2))
        particle_mesh = lax.scatter_add(particle_mesh, neighboor_coords_mod,
                                        kernel.reshape([-1, 8]), dnums)

        return particle_mesh, halo_size

    def multi_gpu_epilog(particle_mesh, halo_size, __aux_input=None):

        if __aux_input[0] == 1:
            halo_width = (0, halo_size, 0)
            halo_extents = (0, halo_size // 2, 0)
        elif __aux_input[1] == 1:
            halo_width = (halo_size, 0, 0)
            halo_extents = (halo_size // 2, 0, 0)
        else:
            halo_width = (halo_size, halo_size, 0)
            halo_extents = (halo_size // 2, halo_size // 2, 0)

        particle_mesh = halo_exchange(particle_mesh,
                                      halo_extents=halo_extents,
                                      halo_periods=(True, True, True))
        particle_mesh = slice_unpad(particle_mesh, pad_width=halo_width)

        return particle_mesh

    def get_aux_input_from_base_sharding(base_sharding):

        def get_axis_size(sharding, index):
            axis_name = sharding.spec[index]
            if axis_name == None:
                return 1
            else:
                return sharding.mesh.shape[sharding.spec[index]]

        return [get_axis_size(base_sharding, i) for i in range(2)]

    def infer_sharding_from_base_sharding(base_sharding):

        in_specs = base_sharding.spec, base_sharding.spec, P()
        out_specs = base_sharding.spec

        return in_specs, out_specs


class CICReadOperator(ShardedOperator):

    name = 'cic_read'

    def single_gpu_impl(particle_mesh: jnp.ndarray,
                        positions: jnp.ndarray,
                        halo_size=0):
        del halo_size

        original_shape = positions.shape
        positions = positions.reshape([-1, 3])
        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)
        connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                                 [1., 1, 0], [1., 0, 1], [0., 1, 1],
                                 [1., 1, 1]]])

        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords = jnp.mod(neighboor_coords.astype('int32'),
                                   jnp.array(particle_mesh.shape))

        particles = (
            particle_mesh[neighboor_coords[..., 0], neighboor_coords[..., 1],
                          neighboor_coords[..., 3]] * kernel).sum(axis=-1)
        return particles.reshape(original_shape)

    def multi_gpu_prolog(particle_mesh: jnp.ndarray,
                         positions: jnp.ndarray,
                         halo_size=0,
                         __aux_input=None):

        halo_tuple = (halo_size, halo_size)
        if __aux_input[0] == 1:
            halo_width = ((0, 0), halo_tuple, (0, 0))
            halo_extents = (0, halo_size // 2, 0)
        elif __aux_input[1] == 1:
            halo_width = (halo_tuple, (0, 0), (0, 0))
            halo_extents = (halo_size // 2, 0, 0)
        else:
            halo_width = (halo_tuple, halo_tuple, (0, 0))
            halo_extents = (halo_size // 2, halo_size // 2, 0)

        particle_mesh = slice_pad(particle_mesh, pad_width=halo_width)
        particle_mesh = halo_exchange(particle_mesh,
                                      halo_extents=halo_extents,
                                      halo_periods=(True, True, True))

        return particle_mesh, positions, halo_size

    def multi_gpu_impl(particle_mesh: jnp.ndarray,
                       positions: jnp.ndarray,
                       halo_size=0,
                       __aux_input=None):

        original_shape = positions.shape
        positions = positions.reshape([-1, 3])
        if __aux_input[0] == 1:
            halo_start = [0, halo_size, 0]
        elif __aux_input[1] == 1:
            halo_start = [halo_size, 0, 0]
        else:
            halo_start = [halo_size, halo_size, 0]

        positions += jnp.array(halo_start).reshape([-1, 3])

        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)
        connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                                 [1., 1, 0], [1., 0, 1], [0., 1, 1],
                                 [1., 1, 1]]])

        neighboor_coords = floor + connection
        kernel = 1. - jnp.abs(positions - neighboor_coords)
        kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

        neighboor_coords = jnp.mod(neighboor_coords.astype('int32'),
                                   jnp.array(particle_mesh.shape))

        particles = (
            particle_mesh[neighboor_coords[..., 0], neighboor_coords[..., 1],
                          neighboor_coords[..., 3]] * kernel).sum(axis=-1)
        return particles.reshape(original_shape)

    def get_aux_input_from_base_sharding(base_sharding):

        def get_axis_size(sharding, index):
            axis_name = sharding.spec[index]
            if axis_name == None:
                return 1
            else:
                return sharding.mesh.shape[sharding.spec[index]]

        return [get_axis_size(base_sharding, i) for i in range(2)]

    def infer_sharding_from_base_sharding(base_sharding):

        in_specs = base_sharding.spec, base_sharding.spec, P()
        out_specs = base_sharding.spec

        return in_specs, out_specs


def _chunk_split(ptcl_num, chunk_size, *arrays):
    """Split and reshape particle arrays into chunks and remainders, with the remainders
    preceding the chunks. 0D ones are duplicated as full arrays in the chunks."""
    chunk_size = ptcl_num if chunk_size is None else min(chunk_size, ptcl_num)
    remainder_size = ptcl_num % chunk_size
    chunk_num = ptcl_num // chunk_size

    remainder = None
    chunks = arrays
    if remainder_size:
        remainder = [x[:remainder_size] if x.ndim != 0 else x for x in arrays]
        chunks = [x[remainder_size:] if x.ndim != 0 else x for x in arrays]

    # `scan` triggers errors in scatter and gather without the `full`
    chunks = [
        x.reshape(chunk_num, chunk_size, *x.shape[1:])
        if x.ndim != 0 else jnp.full(chunk_num, x) for x in chunks
    ]

    return remainder, chunks


def enmesh(i1, d1, a1, s1, b12, a2, s2):
    """Multilinear enmeshing."""
    i1 = jnp.asarray(i1)
    d1 = jnp.asarray(d1)
    a1 = jnp.float64(a1) if a2 is not None else jnp.array(a1, dtype=d1.dtype)
    if s1 is not None:
        s1 = jnp.array(s1, dtype=i1.dtype)
    b12 = jnp.float64(b12)
    if a2 is not None:
        a2 = jnp.float64(a2)
    if s2 is not None:
        s2 = jnp.array(s2, dtype=i1.dtype)

    dim = i1.shape[1]
    neighbors = (jnp.arange(2**dim, dtype=i1.dtype)[:, jnp.newaxis] >>
                 jnp.arange(dim, dtype=i1.dtype)) & 1

    if a2 is not None:
        P = i1 * a1 + d1 - b12
        P = P[:, jnp.newaxis]  # insert neighbor axis
        i2 = P + neighbors * a2  # multilinear

        if s1 is not None:
            L = s1 * a1
            i2 %= L

        i2 //= a2
        d2 = P - i2 * a2

        if s1 is not None:
            d2 -= jnp.rint(d2 / L) * L  # also abs(d2) < a2 is expected

        i2 = i2.astype(i1.dtype)
        d2 = d2.astype(d1.dtype)
        a2 = a2.astype(d1.dtype)

        d2 /= a2
    else:
        i12, d12 = jnp.divmod(b12, a1)
        i1 -= i12.astype(i1.dtype)
        d1 -= d12.astype(d1.dtype)

        # insert neighbor axis
        i1 = i1[:, jnp.newaxis]
        d1 = d1[:, jnp.newaxis]

        # multilinear
        d1 /= a1
        i2 = jnp.floor(d1).astype(i1.dtype)
        i2 += neighbors
        d2 = d1 - i2
        i2 += i1

        if s1 is not None:
            i2 %= s1

    f2 = 1 - jnp.abs(d2)

    if s1 is None and s2 is not None:  # all i2 >= 0 if s1 is not None
        i2 = jnp.where(i2 < 0, s2, i2)

    f2 = f2.prod(axis=-1)

    return i2, f2


def _scatter_chunk(carry, chunk):
    mesh, offset, cell_size, mesh_shape = carry
    pmid, disp, val = chunk
    spatial_ndim = pmid.shape[1]
    spatial_shape = mesh.shape

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, cell_size, mesh_shape, offset, cell_size,
                       spatial_shape)
    # scatter
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    mesh = mesh.at[ind].add(val * frac)

    carry = mesh, offset, cell_size, mesh_shape
    return carry, None


def scatter(pmid,
            disp,
            mesh,
            chunk_size=2**24,
            val=1.,
            offset=0,
            cell_size=1.):

    ptcl_num, spatial_ndim = pmid.shape
    val = jnp.asarray(val)
    mesh = jnp.asarray(mesh)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)
    carry = mesh, offset, cell_size, mesh.shape
    if remainder is not None:
        carry = _scatter_chunk(carry, remainder)[0]
    carry = scan(_scatter_chunk, carry, chunks)[0]
    mesh = carry[0]
    return mesh


def _chunk_cat(remainder_array, chunked_array):
    """Reshape and concatenate one remainder and one chunked particle arrays."""
    array = chunked_array.reshape(-1, *chunked_array.shape[2:])

    if remainder_array is not None:
        array = jnp.concatenate((remainder_array, array), axis=0)

    return array


def _gather(pmid, disp, mesh, chunk_size=2**24, val=1, offset=0, cell_size=1.):
    ptcl_num, spatial_ndim = pmid.shape

    mesh = jnp.asarray(mesh)

    val = jnp.asarray(val)

    if mesh.shape[spatial_ndim:] != val.shape[1:]:
        raise ValueError('channel shape mismatch: '
                         f'{mesh.shape[spatial_ndim:]} != {val.shape[1:]}')

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    carry = mesh, offset, cell_size, mesh.shape
    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(carry, remainder)[1]
    val = scan(_gather_chunk, carry, chunks)[1]

    val = _chunk_cat(val_0, val)

    return val


def _gather_chunk(carry, chunk):
    mesh, offset, cell_size, mesh_shape = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, cell_size, mesh_shape, offset, cell_size,
                       spatial_shape)

    # gather
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    frac = jnp.expand_dims(frac, chan_axis)
    val += (mesh.at[ind].get(mode='drop', fill_value=0) * frac).sum(axis=1)

    return carry, val


class CICPaintDXOperator(ShardedOperator):

    name = 'cic_paint_dx'

    def single_gpu_impl(displacement, halo_size=0):

        del halo_size

        original_shape = displacement.shape

        particle_mesh = jnp.zeros(original_shape[:-1], dtype='float32')

        a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                               jnp.arange(particle_mesh.shape[1]),
                               jnp.arange(particle_mesh.shape[2]),
                               indexing='ij')

        pmid = jnp.stack([a, b, c], axis=-1)
        pmid = pmid.reshape([-1, 3])
        return scatter(pmid, displacement.reshape([-1, 3]), particle_mesh)

    def multi_gpu_impl(displacement, halo_size=0, __aux_input=None):

        original_shape = displacement.shape
        particle_mesh = jnp.zeros(original_shape[:-1], dtype='float32')

        halo_tuple = (halo_size, halo_size)
        if __aux_input[0] == 1:
            halo_width = ((0, 0), halo_tuple, (0, 0))
        elif __aux_input[1] == 1:
            halo_width = (halo_tuple, (0, 0), (0, 0))
        else:
            halo_width = (halo_tuple, halo_tuple, (0, 0))

        particle_mesh = jnp.pad(particle_mesh, halo_width)

        a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                               jnp.arange(particle_mesh.shape[1]),
                               jnp.arange(particle_mesh.shape[2]),
                               indexing='ij')

        pmid = jnp.stack([b + halo_size, a + halo_size, c], axis=-1)
        pmid = pmid.reshape([-1, 3])
        return scatter(pmid, displacement.reshape([-1, 3]),
                       particle_mesh), halo_size

    def multi_gpu_epilog(particle_mesh, halo_size, __aux_input=None):

        if __aux_input[0] == 1:
            halo_width = (0, halo_size, 0)
            halo_extents = (0, halo_size // 2, 0)
        elif __aux_input[1] == 1:
            halo_width = (halo_size, 0, 0)
            halo_extents = (halo_size // 2, 0, 0)
        else:
            halo_width = (halo_size, halo_size, 0)
            halo_extents = (halo_size // 2, halo_size // 2, 0)

        particle_mesh = halo_exchange(particle_mesh,
                                      halo_extents=halo_extents,
                                      halo_periods=(True, True, True))
        particle_mesh = slice_unpad(particle_mesh, pad_width=halo_width)

        return particle_mesh

    def get_aux_input_from_base_sharding(base_sharding):

        def get_axis_size(sharding, index):
            axis_name = sharding.spec[index]
            if axis_name == None:
                return 1
            else:
                return sharding.mesh.shape[sharding.spec[index]]

        return [get_axis_size(base_sharding, i) for i in range(2)]

    def infer_sharding_from_base_sharding(base_sharding):

        in_specs = base_sharding.spec, P()
        out_specs = base_sharding.spec

        return in_specs, out_specs


class CICReadDXOperator(ShardedOperator):

    name = 'cic_read_dx'

    def single_gpu_impl(particle_mesh, halo_size=0):

        del halo_size

        original_shape = particle_mesh.shape

        a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                               jnp.arange(particle_mesh.shape[1]),
                               jnp.arange(particle_mesh.shape[2]),
                               indexing='ij')

        pmid = jnp.stack([b, a, c], axis=-1)
        pmid = pmid.reshape([-1, 3])
        positions = _gather(pmid, jnp.zeros_like(pmid), particle_mesh)

        return positions.reshape(original_shape)

    def multi_gpu_prolog(particle_mesh, halo_size=0, __aux_input=None):

        halo_tuple = (halo_size, halo_size)
        if __aux_input[0] == 1:
            halo_width = ((0, 0), halo_tuple, (0, 0))
            halo_extents = (0, halo_size // 2, 0)
        elif __aux_input[1] == 1:
            halo_width = (halo_tuple, (0, 0), (0, 0))
            halo_extents = (halo_size // 2, 0, 0)
        else:
            halo_width = (halo_tuple, halo_tuple, (0, 0))
            halo_extents = (halo_size // 2, halo_size // 2, 0)

        particle_mesh = slice_pad(particle_mesh, pad_width=halo_width)
        particle_mesh = halo_exchange(particle_mesh,
                                      halo_extents=halo_extents,
                                      halo_periods=(True, True, True))

        return particle_mesh, halo_size

    def multi_gpu_impl(particle_mesh, halo_size, __aux_input=None):

        original_shape = particle_mesh.shape

        a, b, c = jnp.meshgrid(jnp.arange(particle_mesh.shape[0]),
                               jnp.arange(particle_mesh.shape[1]),
                               jnp.arange(particle_mesh.shape[2]),
                               indexing='ij')

        pmid = jnp.stack([b + halo_size, a + halo_size, c], axis=-1)
        pmid = pmid.reshape([-1, 3])
        positions = _gather(pmid, jnp.zeros_like(pmid), particle_mesh)

        return positions.reshape(original_shape)

    def get_aux_input_from_base_sharding(base_sharding):

        def get_axis_size(sharding, index):
            axis_name = sharding.spec[index]
            if axis_name == None:
                return 1
            else:
                return sharding.mesh.shape[sharding.spec[index]]

        return [get_axis_size(base_sharding, i) for i in range(2)]

    def infer_sharding_from_base_sharding(base_sharding):

        in_specs = base_sharding.spec, P()
        out_specs = base_sharding.spec

        return in_specs, out_specs


register_operator(CICPaintOperator)
register_operator(CICReadOperator)
register_operator(CICPaintDXOperator)
register_operator(CICReadDXOperator)
