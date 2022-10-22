from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
from jaxpm.ops import fft3d, ifft3d, normal

# Create communicators
world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

cart_comm = MPI.COMM_WORLD.Create_cart(dims=[2, 2],
                                       periods=[True, True])
comms = [cart_comm.Sub([True, False]),
         cart_comm.Sub([False, True])]

if rank == 0:
    print("Communication setup done!")


# Setup random keys
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]

# Size of the FFT
N = 256
mesh_shape = [N, N, N]

# Generate a random gaussian variable for the global
# mesh shape
original_array = normal(key, mesh_shape, comms=comms)

# Run a forward FFT
karray = jax.jit(lambda x: fft3d(x, comms=comms))(original_array)
rarray = jax.jit(lambda x: ifft3d(x, comms=comms))(karray)

# Testing that the fft is indeed invertible
print("I'm ", rank, abs(rarray.real - original_array).mean())


# Testing that the FFT is actually what we expect
total_array, token = mpi4jax.allgather(original_array, comm=comms[0])
total_array = total_array.reshape([N, N//2, N])
total_array, token = mpi4jax.allgather(
    total_array.transpose([1, 0, 2]), comm=comms[1], token=token)
total_array = total_array.reshape([N, N, N])
total_array = total_array.transpose([1, 0, 2])

total_karray, token = mpi4jax.allgather(karray, comm=comms[0], token=token)
total_karray = total_karray.reshape([N, N//2, N])
total_karray, token = mpi4jax.allgather(
    total_karray.transpose([1, 0, 2]), comm=comms[1], token=token)
total_karray = total_karray.reshape([N, N, N])
total_karray = total_karray.transpose([1, 0, 2])

print('FFT test:', rank, abs(jnp.fft.fftn(
    total_array).transpose([2, 0, 1]) - total_karray).mean())

if rank == 0:
    print("For reference, the mean value of the fft is", jnp.abs(jnp.fft.fftn(
        total_array)).mean())
