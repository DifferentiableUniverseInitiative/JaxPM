import jax
import jax.numpy as jnp
from jax.lax import scan


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


def enmesh(base_indices, displacements, cell_size, base_shape, offset,
           new_cell_size, new_shape):
    """Multilinear enmeshing."""
    base_indices = jax.tree.map(jnp.asarray , base_indices)
    displacements = jax.tree.map(jnp.asarray , displacements)
    with jax.experimental.enable_x64():
        cell_size = jnp.float64(
            cell_size) if new_cell_size is not None else jnp.array(
                cell_size, dtype=displacements.dtype)
        if base_shape is not None:
            base_shape = jnp.array(base_shape, dtype=base_indices.dtype)
        offset = jnp.float64(offset)
        if new_cell_size is not None:
            new_cell_size = jnp.float64(new_cell_size)
        if new_shape is not None:
            new_shape = jnp.array(new_shape, dtype=base_indices.dtype)

    spatial_dim = base_indices.shape[1]
    neighbor_offsets = (
        jnp.arange(2**spatial_dim, dtype=base_indices.dtype)[:, jnp.newaxis] >>
        jnp.arange(spatial_dim, dtype=base_indices.dtype)) & 1

    if new_cell_size is not None:
        particle_positions = base_indices * cell_size + displacements - offset
        particle_positions = particle_positions[:, jnp.
                                                newaxis]  # insert neighbor axis
        new_indices = particle_positions + neighbor_offsets * new_cell_size  # multilinear

        if base_shape is not None:
            grid_length = base_shape * cell_size
            new_indices %= grid_length

        new_indices //= new_cell_size
        new_displacements = particle_positions - new_indices * new_cell_size

        if base_shape is not None:
            new_displacements -= jax.tree.map(jnp.rint ,
                new_displacements / grid_length
            ) * grid_length  # also abs(new_displacements) < new_cell_size is expected

        new_indices = new_indices.astype(base_indices.dtype)
        new_displacements = new_displacements.astype(displacements.dtype)
        new_cell_size = new_cell_size.astype(displacements.dtype)

        new_displacements /= new_cell_size
    else:
        offset_indices, offset_displacements = jnp.divmod(offset, cell_size)
        base_indices -= offset_indices.astype(base_indices.dtype)
        displacements -= offset_displacements.astype(displacements.dtype)

        # insert neighbor axis
        base_indices = base_indices[:, jnp.newaxis]
        displacements = displacements[:, jnp.newaxis]

        # multilinear
        displacements /= cell_size
        new_indices = jnp.floor(displacements).astype(base_indices.dtype)
        new_indices += neighbor_offsets
        new_displacements = displacements - new_indices
        new_indices += base_indices

        if base_shape is not None:
            new_indices %= base_shape

    weights = 1 - jax.tree.map(jnp.abs , new_displacements)

    if base_shape is None and new_shape is not None:  # all new_indices >= 0 if base_shape is not None
        new_indices = jnp.where(new_indices < 0, new_shape, new_indices)

    weights = weights.prod(axis=-1)

    return new_indices, weights


def _scatter_chunk(carry, chunk):
    mesh, offset, cell_size, mesh_shape = carry
    pmid, disp, val = chunk
    spatial_ndim = pmid.shape[1]
    spatial_shape = mesh.shape

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, cell_size, mesh_shape, offset, cell_size,
                       spatial_shape)
    # scatter
    ind = jax.tree.map(lambda x : tuple(x[..., i] for i in range(spatial_ndim)) , ind)
    mesh_structure = jax.tree.structure(mesh)
    val_flat = jax.tree.leaves(val)
    val_tree = jax.tree.unflatten(mesh_structure, val_flat)
    mesh = jax.tree.map(lambda m , v , i, f : m.at[i].add(jnp.multiply(jnp.expand_dims(v, axis=-1), f)) , mesh , val_tree ,ind ,  frac)
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
    val = jax.tree.map(jnp.asarray , val)
    mesh = jax.tree.map(jnp.asarray , mesh)
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


def gather(pmid, disp, mesh, chunk_size=2**24, val=0, offset=0, cell_size=1.):
    ptcl_num, spatial_ndim = pmid.shape

    mesh = jax.tree.map(jnp.asarray , mesh)

    val = jax.tree.map(jnp.asarray , val)

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
    ind = jax.tree.map(lambda x : tuple(x[..., i] for i in range(spatial_ndim)) , ind)
    frac = jax.tree.map(lambda x: jnp.expand_dims(x, chan_axis), frac)
    ind_structure = jax.tree.structure(ind)
    frac_structure = jax.tree.structure(frac)
    mesh_structure = jax.tree.structure(mesh)
    val += jax.tree.map(lambda m , i , f : (m.at[i].get(mode='drop', fill_value=0) * f).sum(axis=1) , mesh , ind , frac)

    return carry, val
