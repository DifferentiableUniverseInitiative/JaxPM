import os

os.environ["EQX_ON_ERROR"] = "nan"  # avoid an allgather caused by diffrax
import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

from functools import partial

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import (ConstantStepSize, LeapfrogMidpoint, ODETerm, SaveAt,
                     diffeqsolve)
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt, make_ode_fn

all_gather = partial(process_allgather, tiled=True)

pdims = (2, 4)
devices = create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('x', 'y'))
sharding = NamedSharding(mesh, P('x', 'y'))

mesh_shape = [2024, 1024, 1024]
box_size = [1024., 1024., 1024.]
halo_size = 512
snapshots = jnp.linspace(0.1, 1., 2)


@jax.jit
def run_simulation(omega_c, sigma8):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(
        jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

    # Create initial conditions
    initial_conditions = linear_field(mesh_shape,
                                      box_size,
                                      pk_fn,
                                      seed=jax.random.PRNGKey(0),
                                      sharding=sharding)

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),
                          axis=-1).reshape([-1, 3])
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    # Initial displacement
    dx, p, _ = lpt(cosmo,
                   initial_conditions,
                   particles,
                   0.1,
                   halo_size=halo_size,
                   sharding=sharding)

    # Evolve the simulation forward
    ode_fn = make_ode_fn(mesh_shape, halo_size=halo_size, sharding=sharding)
    term = ODETerm(
        lambda t, state, args: jnp.stack(ode_fn(state, t, args), axis=0))
    solver = LeapfrogMidpoint()

    stepsize_controller = ConstantStepSize()
    res = diffeqsolve(term,
                      solver,
                      t0=0.1,
                      t1=1.,
                      dt0=0.01,
                      y0=jnp.stack([dx, p], axis=0),
                      args=cosmo,
                      saveat=SaveAt(ts=snapshots),
                      stepsize_controller=stepsize_controller)

    return initial_conditions, dx, res.ys, res.stats


initial_conditions, lpt_displacements, ode_solutions, solver_stats = run_simulation(
    0.25, 0.8)
print(f"[{rank}] Simulation completed")
print(f"[{rank}] Solver stats: {solver_stats}")

# Gather the results
initial_conditions = all_gather(initial_conditions)
lpt_displacements = all_gather(lpt_displacements)
ode_solutions = [all_gather(sol) for sol in ode_solutions]

if rank == 0:
    np.savez("multihost_pm.npz",
             initial_conditions=initial_conditions,
             lpt_displacements=lpt_displacements,
             ode_solutions=ode_solutions,
             solver_stats=solver_stats)

print(f"[{rank}] Simulation results saved")
