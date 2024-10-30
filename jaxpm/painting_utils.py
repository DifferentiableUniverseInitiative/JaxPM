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


def enmesh(i1, d1, a1, s1, b12, a2, s2):
    """Multilinear enmeshing."""
    i1 = jnp.asarray(i1)
    d1 = jnp.asarray(d1)
    with jax.experimental.enable_x64():
        a1 = jnp.float64(a1) if a2 is not None else jnp.array(a1,
                                                              dtype=d1.dtype)
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


def gather(pmid, disp, mesh, chunk_size=2**24, val=1, offset=0, cell_size=1.):
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
