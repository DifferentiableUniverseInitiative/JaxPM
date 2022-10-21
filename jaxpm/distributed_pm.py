import jax
from jax.lax import linear_solve_p
import jax.numpy as jnp
from jax.experimental.maps import xmap
from functools import partial
import jax_cosmo as jc

from jaxpm.kernels import fftk
import jaxpm.distributed_ops as dops
from jaxpm.growth import growth_factor, growth_rate, dGfa


def pm_forces(positions, mesh_shape=None, delta_k=None, halo_size=16):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        mesh_shape = delta_k.shape
    kvec = [k.squeeze() for k in fftk(mesh_shape)]

    if delta_k is None:
        delta = dops.cic_paint(positions, mesh_shape, halo_size)
        delta_k = dops.fft3d(dops.reshape_split_to_dense(delta))

    forces_k = dops.gradient_laplace_kernel(delta_k, kvec)

    # Recovers forces at particle positions
    forces = [dops.cic_read(dops.reshape_dense_to_split(dops.ifft3d(f)),
                            positions, halo_size) for f in forces_k]

    return dops.stack3d(*forces)


def linear_field(cosmo, mesh_shape, box_size, seed, return_Fourier=True):
    """
    Generate initial conditions.
    Seed should have the dimension of the computational mesh
    """

    # Sample normal field
    field = dops.normal(seed, mesh_shape)

    # Go to Fourier space
    field = dops.fft3d(dops.reshape_split_to_dense(field))

    # Rescaling k to physical units
    kvec = [k.squeeze() / box_size[i] * mesh_shape[i]
            for i, k in enumerate(fftk(mesh_shape, symmetric=False))]
    k = jnp.logspace(-4, 2, 256)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk = pk * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
               ) / (box_size[0] * box_size[1] * box_size[2])

    field = dops.scale_by_power_spectrum(field, kvec, k, jnp.sqrt(pk))

    if return_Fourier:
        return field
    else:
        return dops.reshape_dense_to_split(dops.ifft3d(field))


def lpt(cosmo, initial_conditions, positions, a):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(positions, delta_k=initial_conditions)
    a = jnp.atleast_1d(a)
    dx = dops.scalar_multiply(initial_force * growth_factor(cosmo, a))
    p = dops.scalar_multiply(dx, a**2 * growth_rate(cosmo, a) *
                             jnp.sqrt(jc.background.Esqr(cosmo, a)))
    return dx, p


def make_ode_fn(mesh_shape):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape=mesh_shape) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = dops.scalar_multiply(
            vel, 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))))

        # Computes the update of velocity (kick)
        dvel = dops.scalar_multiply(
            forces, 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))))

        return dpos, dvel

    return nbody_ode
