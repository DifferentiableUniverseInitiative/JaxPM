import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


import os
os.environ["EQX_ON_ERROR"] = "nan"
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.debug import visualize_array_sharding

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.painting import cic_paint_dx , cic_read_dx , cic_paint , cic_read
from jaxpm.pm import linear_field, lpt, make_diffrax_ode
from functools import partial
from diffrax import ConstantStepSize, LeapfrogMidpoint, ODETerm, SaveAt, diffeqsolve
from jaxpm.distributed import uniform_particles

#assert jax.device_count() >= 8, "This notebook requires a TPU or GPU runtime with 8 devices"



from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

all_gather = partial(process_allgather, tiled=False)

pdims = (2, 4)
#devices = create_device_mesh(pdims)
#mesh = Mesh(devices, axis_names=('x', 'y'))
#sharding = NamedSharding(mesh, P('x', 'y'))
sharding = None


from typing import NamedTuple
from jaxdecomp import ShardedArray

mesh_shape = 64
box_size = 64.
halo_size = 2
snapshots = (0.5, 1.0)

class Params(NamedTuple):
    omega_c: float
    sigma8: float
    initial_conditions : jnp.ndarray

mesh_shape = (mesh_shape,) * 3
box_size = (box_size,) * 3
omega_c = 0.25
sigma8 = 0.8
# Create a small function to generate the matter power spectrum
k = jnp.logspace(-4, 1, 128)
pk = jc.power.linear_matter_power(
    jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

initial_conditions = linear_field(mesh_shape,
                                    box_size,
                                    pk_fn,
                                    seed=jax.random.PRNGKey(0),
                                    sharding=sharding)


#initial_conditions = ShardedArray(initial_conditions, sharding)

params = Params(omega_c, sigma8, initial_conditions)



@partial(jax.jit , static_argnums=(1 , 2,3,4 ))
def forward_model(params , mesh_shape,box_size,halo_size , snapshots):

    # Create initial conditions
    cosmo = jc.Planck15(Omega_c=params.omega_c, sigma8=params.sigma8)
    particles = uniform_particles(mesh_shape , sharding) 
    ic_structure = jax.tree.structure(params.initial_conditions)
    particles = jax.tree.unflatten(ic_structure , jax.tree.leaves(particles))
    # Initial displacement
    dx, p, f = lpt(cosmo,
                   params.initial_conditions,
                   particles,
                   a=0.1,
                   order=2,
                   halo_size=halo_size,
                   sharding=sharding)

    # Evolve the simulation forward
    ode_fn = ODETerm(
        make_diffrax_ode(mesh_shape, paint_absolute_pos=True,halo_size=halo_size,sharding=sharding))
    solver = LeapfrogMidpoint()

    y0 = jax.tree.map(lambda particles , dx , p : jnp.stack([particles  + dx ,p],axis=0) , particles , dx , p)
    print(f"y0 structure: {jax.tree.structure(y0)}")

    stepsize_controller = ConstantStepSize()
    res = diffeqsolve(ode_fn,
                      solver,
                      t0=0.1,
                      t1=1.,
                      dt0=0.01,
                      y0=y0,
                      args=cosmo,
                      saveat=SaveAt(ts=snapshots),
                      stepsize_controller=stepsize_controller)
    ode_solutions = [sol[0] for sol in res.ys]
    
    ode_field = cic_paint(jnp.zeros(mesh_shape, jnp.float32), ode_solutions[-1])
    return particles + dx , ode_field


    ode_field = cic_paint_dx(ode_solutions[-1])
    return dx , ode_field



lpt_particles , ode_field = forward_model(params , mesh_shape,box_size,halo_size , snapshots)


import matplotlib.pyplot as plt

lpt_field = cic_paint(jnp.zeros(mesh_shape, jnp.float32), lpt_particles)
#lpt_field = cic_paint_dx(lpt_particles)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(lpt_field.sum(axis=0) , cmap='magma')
plt.colorbar()
plt.title('LPT field')
plt.subplot(122)
plt.imshow(ode_field.sum(axis=0) , cmap='magma')
plt.colorbar()
plt.title('ODE field')
plt.show()
plt.close()

#particles = jax.random.uniform(jax.random.PRNGKey(0), (4 , 4 ,4 , 3), minval=0.1, maxval=0.9)
#field = jax.random.uniform(jax.random.PRNGKey(0), (4, 4, 4))
#
#partiles = ShardedArray(particles, sharding)
#field = ShardedArray(field, sharding)
#
#
#cic_read_dx(field , particles )