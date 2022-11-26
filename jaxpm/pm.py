import jax
from jax.experimental.maps import xmap
import jax.numpy as jnp

import jax_cosmo as jc

from jaxpm.ops import fft3d, ifft3d, zeros, normal
from jaxpm.kernels import fftk, apply_gradient_laplace
from jaxpm.painting import cic_paint, cic_read
from jaxpm.growth import growth_factor, growth_rate, dGfa


def pm_forces(positions, mesh_shape=None, delta_k=None, halo_size=0, sharding_info=None):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if delta_k is None:
        delta = cic_paint(zeros(mesh_shape, sharding_info=sharding_info),
                          positions,
                          halo_size=halo_size, sharding_info=sharding_info)
        delta_k = fft3d(delta, sharding_info=sharding_info)

    # Computes gravitational forces
    kvec = fftk(delta_k.shape, symmetric=False, sharding_info=sharding_info)
    forces_k = apply_gradient_laplace(delta_k, kvec)

    # Interpolate forces at the position of particles
    return jnp.stack([cic_read(ifft3d(forces_k[..., i], sharding_info=sharding_info).real,
                               positions, halo_size=halo_size, sharding_info=sharding_info)
                      for i in range(3)], axis=-1)


def lpt(cosmo, positions, initial_conditions, a, halo_size=0, sharding_info=None):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(
        positions, delta_k=initial_conditions, halo_size=halo_size, sharding_info=sharding_info)
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * \
        jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * \
        dGfa(cosmo, a) * initial_force
    return dx, p, f


def linear_field(cosmo, mesh_shape, box_size, key, sharding_info=None):
    """
    Generate initial conditions in Fourier space.
    """
    # Sample normal field
    field = normal(key, mesh_shape, sharding_info=sharding_info)

    # Transform to Fourier space
    kfield = fft3d(field, sharding_info=sharding_info)

    # Rescaling k to physical units
    kvec = [k / box_size[i] * mesh_shape[i]
            for i, k in enumerate(fftk(kfield.shape,
                                       symmetric=False,
                                       sharding_info=sharding_info))]

    # Evaluating linear matter powerspectrum
    k = jnp.logspace(-4, 2, 256)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk = pk * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
               ) / (box_size[0] * box_size[1] * box_size[2])

    # Multipliyng the field by the proper power spectrum
    kfield = xmap(lambda kfield, kx, ky, kz:
                  kfield * jc.scipy.interpolate.interp(jnp.sqrt(kx**2+ky**2+kz**2),
                                                       k, jnp.sqrt(pk)),
                  in_axes=(('x', 'y', ...), ['x'], ['y'], [...]),
                  out_axes=('x', 'y', ...))(kfield, kvec[0], kvec[1], kvec[2])

    return kfield


def make_ode_fn(mesh_shape, halo_size=0, sharding_info=None):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape=mesh_shape,
                           halo_size=halo_size, sharding_info=sharding_info) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode
