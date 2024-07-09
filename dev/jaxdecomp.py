import argparse
import jax
import numpy as np

# Setting up distributed jax
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.pm import linear_field, lpt
from jaxpm.painting import cic_paint
from jax.experimental import mesh_utils
from jax.sharding import Mesh

mesh_shape= [256, 256, 256]
box_size  = [256.,256.,256.]
snapshots = jnp.linspace(0.1, 1., 2)

@jax.jit
def run_simulation(omega_c, sigma8, seed):
    # Create a cosmology
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(mesh_shape, box_size, pk_fn, seed=seed)
    
    # Initialize particle displacements 
    dx, p, f = lpt(cosmo, initial_conditions, 1.0)

    field = cic_paint(jnp.zeros_like(initial_conditions), dx)
    return field

def main(args):
  # Setting up distributed random numbers
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]

  # Create computing mesh and sharding information
  devices = mesh_utils.create_device_mesh((2,2))
  mesh = Mesh(devices.T, axis_names=('x', 'y'))

  # Run the simulation on the compute mesh
  with mesh:
    field = run_simulation(0.32, 0.8, key)

  print('done')
  np.save(f'field_{rank}.npy', field.addressable_data(0))
  
  # Closing distributed jax
  jax.distributed.shutdown()

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Distributed LPT N-body simulation.")
  args = parser.parse_args()
  main(args)