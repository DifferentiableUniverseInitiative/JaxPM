from dataclasses import fields
from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as onp
import mpi4jax
from jaxpm.ops import fft3d, ifft3d, normal, meshgrid3d, zeros
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.painting import cic_paint
from jax.experimental.ode import odeint
import jax_cosmo as jc


### Setting up a whole bunch of things #######
# Create communicators
world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

cart_comm = MPI.COMM_WORLD.Create_cart(dims=[2, 2],
                                       periods=[True, True])
comms = [cart_comm.Sub([True, False]),
         cart_comm.Sub([False, True])]

# Setup random keys
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]
################################################

# Size and parameters of the simulation volume
N = 256
mesh_shape = [N, N, N]
box_size = [205, 205, 205]  # Mpc/h
cosmo = jc.Planck15()
halo_size = 16
a = 0.1


@jax.jit
def run_sim(cosmo, key):
    initial_conditions = linear_field(cosmo, mesh_shape, box_size, key,
                                      comms=comms)
    init_field = ifft3d(initial_conditions, comms=comms).real

    # Initialize particles
    pos = meshgrid3d(mesh_shape, comms=comms)

    # Initial displacement by LPT
    cosmo = jc.Planck15()
    dx, p, f = lpt(cosmo, pos, initial_conditions, a, comms=comms)

    # And now, we run an actual nbody
    res = odeint(make_ode_fn(mesh_shape, halo_size, comms),
                 [pos+dx, p], jnp.linspace(0.1, 1.0, 2), cosmo,
                 rtol=1e-5, atol=1e-5)

    # Painting on a new mesh
    field = cic_paint(zeros(mesh_shape, comms=comms),
                      res[0][-1], halo_size, comms=comms)

    return init_field, field


# Recover the real space initial conditions
init_field, field = run_sim(cosmo, key)

# Testing that the result is actually  looking like what we expect
total_array, token = mpi4jax.allgather(field, comm=comms[0])
total_array = total_array.reshape([N, N//2, N])
total_array, token = mpi4jax.allgather(
    total_array.transpose([1, 0, 2]), comm=comms[1], token=token)
total_array = total_array.reshape([N, N, N])
total_array = total_array.transpose([1, 0, 2])

if rank == 0:
    onp.save('simulation.npy', total_array)

print('Done !')
