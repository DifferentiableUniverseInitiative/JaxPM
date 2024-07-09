from typing import Any, Callable, Hashable

Specs = Any
AxisName = Hashable

try:
    import jaxdecomp
    distributed = True
except ImportError:
    print("jaxdecomp not installed. Distributed functions will not work.")
    distributed = False

import jax.numpy as jnp
from jax._src import mesh as mesh_lib
from jax.experimental.shard_map import shard_map


def autoshmap(f: Callable,
          in_specs: Specs,
          out_specs: Specs,
          check_rep: bool = True,
          auto: frozenset[AxisName] = frozenset()):
    """Helper function to wrap the provided function in a shard map if 
    the code is being executed in a mesh context."""
    mesh = mesh_lib.thread_resources.env.physical_mesh
    if mesh.empty:
        return f
    else:
        return shard_map(f, mesh, in_specs, out_specs, check_rep, auto)

def fft3d(x):
    if distributed and not(mesh_lib.thread_resources.env.physical_mesh.empty):
        return jaxdecomp.pfft3d(x.astype(jnp.complex64))
    else:
        return jnp.fft.rfftn(x)

def ifft3d(x):
    if distributed and not(mesh_lib.thread_resources.env.physical_mesh.empty):
        return jaxdecomp.pifft3d(x).real
    else:
        return jnp.fft.irfftn(x)

def get_local_shape(mesh_shape):
  """ Helper function to get the local size of a mesh given the global size.
  """
  if mesh_lib.thread_resources.env.physical_mesh.empty:
    return mesh_shape
  else:
    pdims = mesh_lib.thread_resources.env.physical_mesh.devices.shape
    return [mesh_shape[0] // pdims[0], mesh_shape[1] // pdims[1], mesh_shape[2]]