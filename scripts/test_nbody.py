from mpi4py import MPI
import os
import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp
import jaxdecomp
from jaxpm.ops import fft3d, ifft3d, normal, meshgrid3d, zeros, ShardingInfo
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.painting import cic_paint
from jax.experimental.ode import odeint
import jax_cosmo as jc
import time
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from functools import partial
### Setting up a whole bunch of things #######
# Create communicators
world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

jax.config.update("jax_enable_x64", True)

# Here we assume clients are on the same node, so we restrict which device
# they can use based on their rank
jax.distributed.initialize()

# Setup random keys
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]
################################################

# Size and parameters of the simulation volume
N = 256
mesh_shape = (N, N, N)
box_size = [500, 500, 500]  # Mpc/h
halo_size = 32
sharding_info = ShardingInfo(global_shape=mesh_shape,
                             pdims=(2,2),
                             halo_extents=(halo_size, halo_size, 0),
                             rank=rank)
cosmo = jc.Planck15()
a = 0.1


devices = mesh_utils.create_device_mesh(sharding_info.pdims[::-1])
mesh = Mesh(devices, axis_names=('z', 'y'))

initial_conditions = linear_field(cosmo, mesh, mesh_shape, box_size, key,
                                sharding_info=sharding_info)

@jax.jit
def ifft3d_c2r(initial_conditions):
    return ifft3d(initial_conditions, sharding_info=sharding_info).real

def run_sim(mesh , initial_conditions, cosmo, key):
    
    with mesh:
        init_field = ifft3d_c2r(initial_conditions)

        # Initialize particles
        pos = meshgrid3d(mesh_shape, sharding_info=sharding_info)

    # Initial displacement by LPT
    cosmo = jc.Planck15()

    dx, p, f = lpt(mesh , cosmo, pos, initial_conditions, a, halo_size=halo_size, sharding_info=sharding_info)

    # And now, we run an actual nbody
    #res = odeint(make_ode_fn(mesh_shape, halo_size, sharding_info),
    #             [pos+dx, p], jnp.linspace(0.1, 1.0, 2), cosmo,
    #             rtol=1e-3, atol=1e-3)
    ## Painting on a new mesh
    with mesh:
        displacement = jit(jnp.add)(p , dx)

    empty_field = zeros(mesh , mesh_shape, sharding_info=sharding_info)

    field = cic_paint(mesh , empty_field,
                displacement, halo_size, sharding_info=sharding_info)

    return init_field, field

# initial_conditions = linear_field(cosmo, mesh_shape, box_size, key,
#                                   sharding_info=sharding_info)

# init_field = ifft3d(initial_conditions, sharding_info=sharding_info).real

# print("hello", init_field.shape)

# cosmo = jc.Planck15()
# pos = meshgrid3d(mesh_shape, sharding_info=sharding_info)
# dx, p, f = lpt(cosmo, pos, initial_conditions, a, sharding_info=sharding_info)

# #dx = 3*jax.random.normal(key=key, shape=[1048576, 3])
# # Initialize particles
# # pos = meshgrid3d(mesh_shape, sharding_info=sharding_info)

# field = cic_paint(zeros(mesh_shape, sharding_info=sharding_info),
#                         pos+dx, halo_size, sharding_info=sharding_info)

# # Recover the real space initial conditions
init_field, field = run_sim(mesh , initial_conditions,cosmo, key)
#init_field, field = run_sim(mesh , initial_conditions,cosmo, key)

# import jaxdecomp
# field = jaxdecomp.halo_exchange(field,
#                    halo_extents=sharding_info.halo_extents,
#                     halo_periods=(True,True,True),
#                     pdims=sharding_info.pdims,
#                     global_shape=sharding_info.global_shape)

# time1 = time.time()
# init_field, field = run_sim(cosmo, key)
# init_field.block_until_ready()
# time2 = time.time()



# if rank == 0:
onp.save('simulation_init_field_float16_%d.npy'%rank, init_field.addressable_data(0).astype(onp.float16))
onp.save('simulation_field_float16_%d.npy'%rank, field.addressable_data(0).astype(onp.float16))

# print('Done in', time2-time1)

print("Done")

jaxdecomp.finalize()