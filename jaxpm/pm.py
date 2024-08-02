from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import (autoshmap, fft3d, get_local_shape, ifft3d,
                               normal_field)
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, fftk, gradient_kernel, laplace_kernel,
                           longrange_kernel)
from jaxpm.painting import cic_paint, cic_paint_dx, cic_read, cic_read_dx


def pm_forces(positions, mesh_shape=None, delta=None, r_split=0, halo_size=0):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        assert (delta is not None
                ), "If mesh_shape is not provided, delta should be provided"
        mesh_shape = delta.shape
    kvec = fftk(mesh_shape)

    if delta is None:
        delta_k = fft3d(cic_paint_dx(positions, halo_size=halo_size))
    else:
        delta_k = fft3d(delta)

    # Computes gravitational potential
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec,
                                                              r_split=r_split)
    # Computes gravitational forces
    forces = jnp.stack([
        cic_read_dx(ifft3d(gradient_kernel(kvec, i) * pot_k),
                    halo_size=halo_size) for i in range(3)
    ],
                       axis=-1)

    return forces


def lpt2_source(mesh_size, initial_conditions):

    kvec = fftk(mesh_size)
    # TODO : this has already been done for LPT1, we should reuse it
    delta_k = fft3d(initial_conditions)

    source = jnp.zeros_like(delta_k)

    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    # laplace_kernel should be actually inv laplace_kernel
    # adding a minus sign here that will be negated when computing forces
    # because F = -grad(phi)
    # and phi = -laplace_kernel(delta_k)
    pot_k = delta_k * laplace_kernel(delta_k)

    nabla_i_nabla_i = [
        ifft3d(gradient_kernel(kvec, i)**2 * pot_k) for i in range(3)
    ]
    # for diagonal terms
    source += nabla_i_nabla_i[D1[0]] * nabla_i_nabla_i[D2[0]]
    source += nabla_i_nabla_i[D1[1]] * nabla_i_nabla_i[D2[1]]
    source += nabla_i_nabla_i[D1[2]] * nabla_i_nabla_i[D2[2]]

    # off diag terms
    for i in range(3):
        nabla_i_nabla_j = gradient_kernel(kvec, D1[i]) * gradient_kernel(
            kvec, D2[i])
        phi = ifft3d(nabla_i_nabla_j * pot_k)
        source -= phi**2

    return source


def lpt(cosmo, initial_conditions, a, halo_size=0):
    """
    Computes first order LPT displacement
    """
    local_mesh_shape = (*get_local_shape(initial_conditions.shape), 3)
    displacement = autoshmap(
      partial(jnp.zeros, shape=(local_mesh_shape), dtype='float32'),
      in_specs=(),
      out_specs=P('x', 'y'))()  # yapf: disable


    initial_force = pm_forces(displacement,
                              delta=initial_conditions,
                              halo_size=halo_size)
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo,
                                                                   a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo,
                                                             a) * initial_force
    return dx, p, f


# @Credit Hugo Simon https://github.com/hsimonfroy/montecosmo
def lpt2(cosmo, initial_conditions, dx, p, f, a, halo_size=0):

    mesh_size = initial_conditions.shape
    local_mesh_shape = (*get_local_shape(initial_conditions.shape), 3)
    # TODO
    # Displacements have been created in the previous step
    # find a way to reuse them
    displacement = autoshmap(
      partial(jnp.zeros, shape=(local_mesh_shape), dtype='float32'),
      in_specs=(),
      out_specs=P('x', 'y'))()  # yapf: disable

    lpt2_delta = lpt2_source(mesh_size, initial_conditions)
    delta2_k = fft3d(lpt2_delta)

    lpt2_forces = pm_forces(displacement,
                            mesh_size,
                            delta_k=delta2_k,
                            halo_size=halo_size)
    dx2 = 3 / 7 * growth_factor_second(cosmo, a) * lpt2_forces
    p2 = a**2 * growth_rate_second(cosmo, a) * jnp.sqrt(
        jc.background.Esqr(cosmo, a)) * dx2
    f2 = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGf2a(cosmo,
                                                               a) * lpt2_forces

    dx += dx2
    p += p2
    f += f2

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
    field = normal_field(mesh_shape, seed=seed)
    field = fft3d(field) * pkmesh**0.5
    field = ifft3d(field)
    return field


def make_ode_fn(mesh_shape, halo_size=0):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape=mesh_shape,
                           halo_size=halo_size) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode


def pgd_correction(pos, mesh_shape, params):
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
    PGD_range=PGD_kernel(kvec, kl, ks)
    
    pot_k_pgd=(delta_k * laplace_kernel(kvec))*PGD_range

    forces_pgd= jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k_pgd), pos) 
                      for i in range(3)],axis=-1)
    
    dpos_pgd = forces_pgd*alpha
   
    return dpos_pgd


def make_neural_ode_fn(model, mesh_shape):
    def neural_nbody_ode(state, a, cosmo, params):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state
        kvec = fftk(mesh_shape)

        delta = cic_paint(jnp.zeros(mesh_shape), pos)

        delta_k = jnp.fft.rfftn(delta)

        # Computes gravitational potential
        pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

        # Apply a correction filter
        kk = jnp.sqrt(sum((ki/jnp.pi)**2 for ki in kvec))
        pot_k = pot_k *(1. + model.apply(params, kk, jnp.atleast_1d(a)))

        # Computes gravitational forces
        forces = jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k), pos) 
                          for i in range(3)],axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel
    return neural_nbody_ode
