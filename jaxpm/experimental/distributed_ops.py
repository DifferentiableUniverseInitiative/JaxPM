from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.maps import xmap
from jax.experimental.pjit import PartitionSpec, pjit

import jaxpm.painting as paint

# TODO: add a way to configure axis resources from command line
axis_resources = {'x': 'nx', 'y': 'ny'}
mesh_size = {'nx': 2, 'ny': 2}


@partial(xmap,
         in_axes=({
             0: 'x',
             2: 'y'
         }, {
             0: 'x',
             2: 'y'
         }, {
             0: 'x',
             2: 'y'
         }),
         out_axes=({
             0: 'x',
             2: 'y'
         }),
         axis_resources=axis_resources)
def stack3d(a, b, c):
    return jnp.stack([a, b, c], axis=-1)


@partial(xmap,
         in_axes=({
             0: 'x',
             2: 'y'
         }, [...]),
         out_axes=({
             0: 'x',
             2: 'y'
         }),
         axis_resources=axis_resources)
def scalar_multiply(a, factor):
    return a * factor


@partial(xmap,
         in_axes=({
             0: 'x',
             2: 'y'
         }, {
             0: 'x',
             2: 'y'
         }),
         out_axes=({
             0: 'x',
             2: 'y'
         }),
         axis_resources=axis_resources)
def add(a, b):
    return a + b


@partial(xmap,
         in_axes=['x', 'y', ...],
         out_axes=['x', 'y', ...],
         axis_resources=axis_resources)
def fft3d(mesh):
    """ Performs a 3D complex Fourier transform

    Args:
        mesh: a real 3D tensor of shape [Nx, Ny, Nz]

    Returns:
        3D FFT of the input, note that the dimensions of the output
        are tranposed.
    """
    mesh = jnp.fft.fft(mesh)
    mesh = lax.all_to_all(mesh, 'x', 0, 0)
    mesh = jnp.fft.fft(mesh)
    mesh = lax.all_to_all(mesh, 'y', 0, 0)
    return jnp.fft.fft(mesh)  # Note the output is transposed # [z, x, y]


@partial(xmap,
         in_axes=['x', 'y', ...],
         out_axes=['x', 'y', ...],
         axis_resources=axis_resources)
def ifft3d(mesh):
    mesh = jnp.fft.ifft(mesh)
    mesh = lax.all_to_all(mesh, 'y', 0, 0)
    mesh = jnp.fft.ifft(mesh)
    mesh = lax.all_to_all(mesh, 'x', 0, 0)
    return jnp.fft.ifft(mesh).real


def normal(key, shape=[]):

    @partial(xmap,
             in_axes=['x', 'y', ...],
             out_axes={
                 0: 'x',
                 2: 'y'
             },
             axis_resources=axis_resources)
    def fn(key):
        """ Generate a distributed random normal distributions
        Args:
            key: array of random keys with same layout as computational mesh
            shape: logical shape of array to sample
        """
        return jax.random.normal(
            key,
            shape=[shape[0] // mesh_size['nx'], shape[1] // mesh_size['ny']] +
            shape[2:])

    return fn(key)


@partial(xmap,
         in_axes=(['x', 'y', ...], [['x'], ['y'], [...]], [...], [...]),
         out_axes=['x', 'y', ...],
         axis_resources=axis_resources)
@jax.jit
def scale_by_power_spectrum(kfield, kvec, k, pk):
    kx, ky, kz = kvec
    kk = jnp.sqrt(kx**2 + ky**2 + kz**2)
    return kfield * jc.scipy.interpolate.interp(kk, k, pk)


@partial(xmap,
         in_axes=(['x', 'y', 'z'], [['x'], ['y'], ['z']]),
         out_axes=(['x', 'y', 'z']),
         axis_resources=axis_resources)
def gradient_laplace_kernel(kfield, kvec):
    kx, ky, kz = kvec
    kk = (kx**2 + ky**2 + kz**2)
    kernel = jnp.where(kk == 0, 1., 1. / kk)
    return (kfield * kernel * 1j * 1 / 6.0 *
            (8 * jnp.sin(ky) - jnp.sin(2 * ky)), kfield * kernel * 1j * 1 /
            6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz)), kfield * kernel * 1j *
            1 / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)))


@partial(xmap,
         in_axes=([...]),
         out_axes={
             0: 'x',
             2: 'y'
         },
         axis_sizes={
             'x': mesh_size['nx'],
             'y': mesh_size['ny']
         },
         axis_resources=axis_resources)
def meshgrid(x, y, z):
    """ Generates a mesh grid of appropriate size for the
    computational mesh we have.
    """
    return jnp.stack(jnp.meshgrid(x, y, z), axis=-1)


def cic_paint(pos, mesh_shape, halo_size=0):

    @partial(xmap,
             in_axes=({
                 0: 'x',
                 2: 'y'
             }),
             out_axes=({
                 0: 'x',
                 2: 'y'
             }),
             axis_resources=axis_resources)
    def fn(pos):

        mesh = jnp.zeros([
            mesh_shape[0] // mesh_size['nx'] +
            2 * halo_size, mesh_shape[1] // mesh_size['ny'] + 2 * halo_size
        ] + mesh_shape[2:])

        # Paint particles
        mesh = paint.cic_paint(
            mesh,
            pos.reshape(-1, 3) +
            jnp.array([halo_size, halo_size, 0]).reshape([-1, 3]))

        # Perform halo exchange
        # Halo exchange along x
        left = lax.pshuffle(mesh[-2 * halo_size:],
                            perm=range(mesh_size['nx'])[::-1],
                            axis_name='x')
        right = lax.pshuffle(mesh[:2 * halo_size],
                             perm=range(mesh_size['nx'])[::-1],
                             axis_name='x')
        mesh = mesh.at[:2 * halo_size].add(left)
        mesh = mesh.at[-2 * halo_size:].add(right)

        # Halo exchange along y
        left = lax.pshuffle(mesh[:, -2 * halo_size:],
                            perm=range(mesh_size['ny'])[::-1],
                            axis_name='y')
        right = lax.pshuffle(mesh[:, :2 * halo_size],
                             perm=range(mesh_size['ny'])[::-1],
                             axis_name='y')
        mesh = mesh.at[:, :2 * halo_size].add(left)
        mesh = mesh.at[:, -2 * halo_size:].add(right)

        # removing halo and returning mesh
        return mesh[halo_size:-halo_size, halo_size:-halo_size]

    return fn(pos)


def cic_read(mesh, pos, halo_size=0):

    @partial(xmap,
             in_axes=(
                 {
                     0: 'x',
                     2: 'y'
                 },
                 {
                     0: 'x',
                     2: 'y'
                 },
             ),
             out_axes=({
                 0: 'x',
                 2: 'y'
             }),
             axis_resources=axis_resources)
    def fn(mesh, pos):

        # Halo exchange to grab neighboring borders
        # Exchange along x
        left = lax.pshuffle(mesh[-halo_size:],
                            perm=range(mesh_size['nx'])[::-1],
                            axis_name='x')
        right = lax.pshuffle(mesh[:halo_size],
                             perm=range(mesh_size['nx'])[::-1],
                             axis_name='x')
        mesh = jnp.concatenate([left, mesh, right], axis=0)
        # Exchange along y
        left = lax.pshuffle(mesh[:, -halo_size:],
                            perm=range(mesh_size['ny'])[::-1],
                            axis_name='y')
        right = lax.pshuffle(mesh[:, :halo_size],
                             perm=range(mesh_size['ny'])[::-1],
                             axis_name='y')
        mesh = jnp.concatenate([left, mesh, right], axis=1)

        # Reading field at particles positions
        res = paint.cic_read(
            mesh,
            pos.reshape(-1, 3) +
            jnp.array([halo_size, halo_size, 0]).reshape([-1, 3]))

        return res.reshape(pos.shape[:-1])

    return fn(mesh, pos)


@partial(pjit,
         in_axis_resources=PartitionSpec('nx', 'ny'),
         out_axis_resources=PartitionSpec('nx', None, 'ny', None))
def reshape_dense_to_split(x):
    """ Redistribute data from [x,y,z] convention to [Nx,x,Ny,y,z]
    Changes the logical shape of the array, but no shuffling of the
    data should be necessary
    """
    shape = list(x.shape)
    return x.reshape([
        mesh_size['nx'], shape[0] //
        mesh_size['nx'], mesh_size['ny'], shape[2] // mesh_size['ny']
    ] + shape[2:])


@partial(pjit,
         in_axis_resources=PartitionSpec('nx', None, 'ny', None),
         out_axis_resources=PartitionSpec('nx', 'ny'))
def reshape_split_to_dense(x):
    """ Redistribute data from [Nx,x,Ny,y,z] convention to [x,y,z]
    Changes the logical shape of the array, but no shuffling of the
    data should be necessary
    """
    shape = list(x.shape)
    return x.reshape([shape[0] * shape[1], shape[2] * shape[3]] + shape[4:])
