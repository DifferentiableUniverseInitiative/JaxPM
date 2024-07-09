import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.sharding import PartitionSpec as P

from jaxpm.growth import dGfa, growth_factor, growth_rate
from jaxpm.kernels import (PGD_kernel, fftk, gradient_kernel, laplace_kernel,
                           longrange_kernel)
from jaxpm.painting import cic_paint, cic_read
from jaxpm.distributed import fft3d, ifft3d, autoshmap, get_local_shape

from functools import partial

def pm_forces(positions, mesh_shape=None, delta=None, r_split=0):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        mesh_shape = delta.shape
    kvec = fftk(mesh_shape)

    if delta is None:
        delta_k = fft3d(cic_paint(jnp.zeros(mesh_shape), positions))
    else:
        delta_k = fft3d(delta)

    # Computes gravitational potential
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec,
                                                              r_split=r_split)
    # Computes gravitational forces
    return jnp.stack([
        cic_read(ifft3d(gradient_kernel(kvec, i) * pot_k), positions)
        for i in range(3)
    ],
                     axis=-1)


def lpt(cosmo, initial_conditions, a, particles_shape=None):
    """
    Computes first order LPT displacement
    """
    if particles_shape is None:
        particles_shape = initial_conditions.shape
    local_mesh_shape = get_local_shape(particles_shape)
    displacement = autoshmap(
      partial(jnp.zeros, shape=local_mesh_shape+[3], dtype='float32'),
      in_specs=(),
      out_specs=P('x', 'y'))()  # yapf: disable

    initial_force = pm_forces(displacement, delta=initial_conditions)
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo,
                                                                   a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo,
                                                             a) * initial_force
    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    # Initialize a random field with one slice on each gpu
    local_mesh_shape = get_local_shape(mesh_shape)
    field = autoshmap(
      partial(jax.random.normal, shape=local_mesh_shape, dtype='float32'),
      in_specs=P(None),
      out_specs=P('x', 'y'))(seed)  # yapf: disable

    field = fft3d(field) * pkmesh**0.5
    field = ifft3d(field)
    return field


def make_ode_fn(mesh_shape):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape=mesh_shape) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode


def pgd_correction(pos, params):		
    """		
    improve the short-range interactions of PM-Nbody simulations with potential gradient descent method, based on https://arxiv.org/abs/1804.00671		
    args:		
      pos: particle positions [npart, 3]		
      params: [alpha, kl, ks] pgd parameters		
    """		
    kvec = fftk(mesh_shape)		

    delta = cic_paint(jnp.zeros(mesh_shape), pos)		
    alpha, kl, ks = params		
    delta_k = jnp.fft.rfftn(delta)		
    PGD_range = PGD_kernel(kvec, kl, ks)		

    pot_k_pgd = (delta_k * laplace_kernel(kvec)) * PGD_range		

    forces_pgd = jnp.stack([		
        cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k_pgd), pos)		
        for i in range(3)		
    ],		
                           axis=-1)		

    dpos_pgd = forces_pgd * alpha		

    return dpos_pgd