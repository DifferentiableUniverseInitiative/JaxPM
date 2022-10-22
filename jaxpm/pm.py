import jax
import jax.numpy as jnp

import jax_cosmo as jc

from jaxpm.ops import fft3d, ifft3d, zeros
from jaxpm.kernels import fftk, apply_gradient_laplace
from jaxpm.painting import cic_paint, cic_read
from jaxpm.growth import growth_factor, growth_rate, dGfa

def pm_forces(positions, mesh_shape=None, delta_k=None, halo_size=0, token=None, comms=None):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        mesh_shape = delta_k.shape

    kvec = fftk(mesh_shape, comms=comms)

    if delta_k is None:
        delta, token = cic_paint(zeros(mesh_shape,comms=comms), 
                                positions, 
                                halo_size=halo_size, token=token, comms=comms)
        delta_k, token = fft3d(delta, token=token, comms=comms)

    # Computes gravitational potential
    forces_k = apply_gradient_laplace(kfield, kvec)

    # Computes gravitational forces
    fx, token = ifft3d(forces_k[...,0], token=token, comms=comms)
    fx, token = cic_read(fx, positions, halo_size=halo_size, comms=comms)

    fy, token = ifft3d(forces_k[...,1], token=token, comms=comms)
    fy, token = cic_read(fy, positions, halo_size=halo_size, comms=comms)

    fz, token = ifft3d(forces_k[...,2], token=token, comms=comms)
    fz, token = cic_read(fz, positions, halo_size=halo_size, comms=comms)

    return jnp.stack([fx,fy,fz],axis=-1), token

def lpt(cosmo, initial_conditions, positions, a, token=token, comms=comms):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(positions, delta=initial_conditions, token=token, comms=comms)
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo, a) * initial_force
    return dx, p, f, comms

def linear_field(mesh_shape, box_size, pk, seed):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])

    field = jax.random.normal(seed, mesh_shape)
    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)
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
    PGD_range=PGD_kernel(kvec, kl, ks)
    
    pot_k_pgd=(delta_k * laplace_kernel(kvec))*PGD_range

    forces_pgd= jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k_pgd), pos) 
                      for i in range(3)],axis=-1)
    
    dpos_pgd = forces_pgd*alpha
   
    return dpos_pgd