# Start this script with:
# mpirun -np 4 python test_script.py
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
import matplotlib.pylab as plt
import jax 
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental.maps import mesh, xmap
from jax.experimental.pjit import PartitionSpec, pjit
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions

def cic_paint(mesh, positions):
  """ Paints positions onto mesh
  mesh: [nx, ny, nz]
  positions: [npart, 3]
  """
  positions = jnp.expand_dims(positions, 1)
  floor = jnp.floor(positions)
  connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], 
                           [0., 0, 1], [1., 1, 0], [1., 0, 1], 
                           [0., 1, 1], [1., 1, 1]]])

  neighboor_coords = floor + connection
  kernel = 1. - jnp.abs(positions - neighboor_coords)
  kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  

  dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0, 1, 2),
    scatter_dims_to_operand_dims=(0, 1, 2))
  mesh = lax.scatter_add(mesh, 
                         neighboor_coords.reshape([-1,8,3]).astype('int32'), 
                         kernel.reshape([-1,8]),
                         dnums)
  return mesh

# And let's draw some points from some 3D distribution
dist = tfd.MultivariateNormalDiag(loc=[16.,16.,16.], scale_identity_multiplier=3.)
pos = dist.sample(1e4, seed=jax.random.PRNGKey(0))

f = pjit(lambda x: cic_paint(x, pos),
         in_axis_resources=PartitionSpec('x', 'y', 'z'), 
         out_axis_resources=None)

devices = np.array(jax.devices()).reshape((2, 2, 1))

# Let's import the mesh
m = jnp.zeros([32, 32, 32])

with mesh(devices, ('x', 'y', 'z')):
  # Shard the mesh, I'm not sure this is absolutely necessary
  m = pjit(lambda x: x,
           in_axis_resources=None,
           out_axis_resources=PartitionSpec('x', 'y', 'z'))(m)

  # Apply the sharded CiC function
  res = f(m)

plt.imshow(res.sum(axis=2))
plt.show()