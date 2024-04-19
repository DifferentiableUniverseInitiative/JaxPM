import jax
from jax.experimental.maps import xmap
import jax.numpy as jnp

import jax_cosmo as jc

from jaxpm.ops import fft3d, ifft3d, zeros, normal
from jaxpm.kernels import fftk, apply_gradient_laplace
from jaxpm.painting import cic_paint, cic_read
from jaxpm.growth import growth_factor, growth_rate, dGfa
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental.shard_map import shard_map
from functools import partial



def pm_forces(mesh , positions, mesh_shape=None, delta_k=None, halo_size=0, sharding_info=None):
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

    local_kx = kvec[0]
    local_ky = kvec[1]
    replicated_kz = kvec[2]

    gspmd_kx = multihost_utils.host_local_array_to_global_array(local_kx ,mesh, P('z'))
    gspmd_ky = multihost_utils.host_local_array_to_global_array(local_ky ,mesh, P('y'))

    @partial(jax.jit,static_argnums=(1))
    def ifft3d_c2r(forces_k , i):
        return ifft3d(forces_k[..., i], sharding_info=sharding_info).real
    
    forces = []
    with mesh:
        forces_k = apply_gradient_laplace(delta_k, gspmd_kx , gspmd_ky , replicated_kz)
        # Interpolate forces at the position of particles

    for i in range(3):
        with mesh:
            ifft_forces = ifft3d_c2r(forces_k , i)

        force = cic_read(mesh , ifft_forces, positions, halo_size=halo_size, sharding_info=sharding_info)
        forces.append(force)

    return jnp.stack(forces , axis=-1)



def lpt(mesh ,cosmo, positions, initial_conditions, a, halo_size=0, sharding_info=None):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(mesh,
        positions, delta_k=initial_conditions, halo_size=halo_size, sharding_info=sharding_info)
    a = jnp.atleast_1d(a)


    @jax.jit
    def compute_dx(cosmo , i_force):
        return growth_factor(cosmo, a) * i_force

    @jax.jit
    def compute_p(cosmo , dx):
        return a**2 * growth_rate(cosmo, a) * \
            jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    
    @jax.jit
    def compute_f(cosmo , initial_force):
        return a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * \
            dGfa(cosmo, a) * initial_force

    with mesh:
        dx = compute_dx(cosmo , initial_force)
        p  = compute_p(cosmo , dx)
        f = compute_f(cosmo , initial_force)



    return dx, p, f


@jax.jit
def interpolate(kfield, kx, ky, kz , k , pk):
    return kfield * jc.scipy.interpolate.interp(jnp.sqrt(kx**2+ky**2+kz**2), k, jnp.sqrt(pk))


def linear_field(cosmo, mesh, mesh_shape, box_size, key, sharding_info=None):
    """
    Generate initial conditions in Fourier space.
    """
    # Sample normal field
    pdims = sharding_info.pdims
    slice_shape = (mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0],mesh_shape[2])

    slice_field = normal(key, slice_shape, sharding_info=sharding_info)

    field = multihost_utils.host_local_array_to_global_array(
      slice_field, mesh, P('z', 'y'))

    # Transform to Fourier space
    with mesh :
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

    local_kx = kvec[0]
    local_ky = kvec[1]
    replicated_kz = kvec[2]

    gspmd_kx = multihost_utils.host_local_array_to_global_array(local_kx ,mesh, P('z'))
    gspmd_ky = multihost_utils.host_local_array_to_global_array(local_ky ,mesh, P('y'))


    with mesh:
        kfield = interpolate(kfield,gspmd_kx, gspmd_ky, replicated_kz ,k, pk)

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
