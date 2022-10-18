# Can be executed with:
# srun  -n 4 -c 32 --gpus-per-task 1 --gpu-bind=none python test_pfft.py
import jax 
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
from jax.experimental.maps import xmap
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec, pjit
from functools import partial

jax.distributed.initialize()

cube_size = 2048

@partial(xmap,
         in_axes=[...],
         out_axes=['x','y', ...],
         axis_sizes={'x':cube_size, 'y':cube_size},
         axis_resources={'x': 'nx', 'y':'ny',
                         'key_x':'nx', 'key_y':'ny'})
def pnormal(key):
    return jax.random.normal(key, shape=[cube_size])

@partial(xmap,
         in_axes={0:'x', 1:'y'},
         out_axes=['x','y', ...],
         axis_resources={'x': 'nx', 'y': 'ny'})
@jax.jit
def pfft3d(mesh):
    # [x, y, z]
    mesh = jnp.fft.fft(mesh) # Transform on z
    mesh = lax.all_to_all(mesh, 'x', 0, 0) # Now x is exposed, [z,y,x]
    mesh = jnp.fft.fft(mesh) # Transform on x
    mesh = lax.all_to_all(mesh, 'y', 0, 0) # Now y is exposed, [z,x,y]
    mesh = jnp.fft.fft(mesh) # Transform on y
    # [z, x, y]
    return mesh

@partial(xmap,
         in_axes={0:'x', 1:'y'},
         out_axes=['x','y', ...],
         axis_resources={'x': 'nx', 'y': 'ny'})
@jax.jit
def pifft3d(mesh):
    # [z, x, y]
    mesh = jnp.fft.ifft(mesh) # Transform on y
    mesh = lax.all_to_all(mesh, 'y', 0, 0) # Now x is exposed, [z,y,x]
    mesh = jnp.fft.ifft(mesh) # Transform on x
    mesh = lax.all_to_all(mesh, 'x', 0, 0) # Now z is exposed, [x,y,z]
    mesh = jnp.fft.ifft(mesh) # Transform on z
    # [x, y, z]
    return mesh

key = jax.random.PRNGKey(42)
# keys = jax.random.split(key, 4).reshape((2,2,2))

# We reshape all our devices to the mesh shape we want
devices = np.array(jax.devices()).reshape((2, 4))

with Mesh(devices, ('nx', 'ny')):
    mesh = pnormal(key)
    kmesh = pfft3d(mesh)
    kmesh.block_until_ready()

# jax.profiler.start_trace("tensorboard")
# with Mesh(devices, ('nx', 'ny')):
#     mesh = pnormal(key)
#     kmesh = pfft3d(mesh)
#     kmesh.block_until_ready()
# jax.profiler.stop_trace()    

print('Done')