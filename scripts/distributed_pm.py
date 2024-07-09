import jax
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

print(f"Started process {rank} of {size}")

import diffrax
import jaxpm as jpm
from jaxpm import solvers
import jax_cosmo as jc
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import jaxpm as jaxpm


cosmo = jc.Planck15(Omega_c=0.25, sigma8=0.8)
# Create initial field
size = 256
mesh_shape = (size, size, size)
box_size = [float(size), float(size), float(size)]


def gen_input():

    key = jax.random.PRNGKey(0)
    initial_field = jpm.ops.normal(mesh_shape,key)
    kvec = jpm.ops.fftk(mesh_shape , symmetric=False)

    return initial_field , kvec

@jax.jit
def fn(cosmo , initial_field , kvec):
    solver = solvers.FastPM()
    particles = jpm.ops.generate_initial_positions(mesh_shape)


    state = solver.init_state(cosmo , particles , kvec , initial_field , box_size)

    state = solver.lpt(state , a=0.1)

    diffsolver = diffrax.Dopri5()
    step_size = diffrax.PIDController(rtol=1e-3,atol=1e-3)

    state = solver.nbody(state , solver=diffsolver , stepsize_controller=step_size,
                t0=0.1,
                t1=1,
                dt0=0.01)

    final_field = jpm.painting.cic_paint_dx(state.displacements)
    return final_field



# One GPU
initial_field , kvec = gen_input()
final_field = fn(cosmo , initial_field , kvec)
np.save('file.npy',final_field)

# Multiple GPUs
# pdims = (2 , 2)
# devices = mesh_utils.create_device_mesh(pdims)
# mesh = Mesh(devices, axis_names=('y', 'z'))
# sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))


# with jaxpm.SPMDConfig(sharding):
#     initial_field , kvec = gen_input()
#     final_field = fn(cosmo , initial_field , kvec)

# final_field = multihost_utils.process_allgather(final_field , tiled=True)
# np.save('file_spmd.npy',final_field)
