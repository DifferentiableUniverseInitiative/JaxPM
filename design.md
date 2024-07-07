# Design Document for JaxPM

This document aims to detail some of the API, implementation choices, and internal mechanism.

## Objective

Provide a user-friendly framework for distributed Particle-Mesh N-body simulations.

## Related Work

This project would be the latest iteration of a number of past libraries that have provided differentiable N-body models.

- [FlowPM](https://github.com/DifferentiableUniverseInitiative/flowpm): TensorFlow
- [vmad FastPM](https://github.com/rainwoodman/vmad/blob/master/vmad/lib/fastpm.py): VMAD
- Borg


In addition, a number of fast N-body simulation projets exist out there:
- [FastPM](https://github.com/fastpm/fastpm)
- ...

## Design Overview

### Coding principles

Following recent trends and JAX philosophy, the library should have a functional programming type of interface.


### Illustration of API

Here is a potential illustration of what the user interface could be for the simulation code:
```python
import jaxpm as jpm
import jax_cosmo as jc

# Instantiate differentiable cosmology object
cosmo = jc.Planck()

# Creates initial conditions
inital_conditions = jpm.generate_ic(cosmo, boxsize, nmesh, dtype='float32')

# Create a particular solver
solver = jpm.solvers.fastpm(cosmo, B=1)

# Initialize and run the simulation
state = solver.init(initial_conditions)
state = solver.nbody(state)

# Painting the results
density = jpm.zeros(boxsize, nmesh)
density = jpm.paint(density, state.positions)
```



### Distributed implementation


```python
import jaxpm as jpm
from jpm import SPMDConfig
import jax_cosmo as jc
import jax.numpy as jnp
import diffrax

jax.distributed.initialize()

pdims = (4, 4) # Got this from autotuning
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('y', 'z'))
sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

# Creates initial conditions
boxsize = [1024., 1024., 1024.]
nmesh = [1024, 1024, 1024]

# Create distributed frequencies
kvec = jpm.fftk(nmesh, sharding)
# Generate initial positions
particles = jpm.generate_initial_positions(nmesh, sharding)

# This contains the mesh, pdims that are necessary to create the shard_mapped functions
spmd_config = SPMDConfig(sharding)

# fftk and generate_initial_positions cannot be jitted because otherwise
# We will be closing on non addressable array which is not allowed
# https://github.com/google/jax/issues/22218

# Instantiate differentiable cosmology object
cosmo = jc.Planck(Omega_c=0.3,, sigma8= 0.8)
snapshots = jnp.linespace(0.1, 1, 10)


@jax.jit
def high_level_api_simulation(cosmo, kvec, particles)
    # Initial conditions is a distributed 3D array
    inital_conditions = jpm.generate_ic(cosmo, boxsize, nmesh,sharding, kvec , dtype='float32')

    # Create a particular solver
    solver = jpm.solvers.fastpm(cosmo, B=1)

    # Initialize and run the simulation
    state = solver.init(initial_conditions)
    state = solver.nbody(state) # Will use base leapfrog integrator
    # OR
    diffrax_solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5,atol=1e-5)
    state = solver.nbody(state , diffrax_solver, stepsize_controller, t0=0.1, t1=1 , dt=0.01 , s=snapshots)

    # Painting the results
    # User defined function to compute weights
    weights = get_weights(particles)
    density = jpm.cic_paint(state.positions , weights)

    return density

with spmd_config:
    density = high_level_api_simulation(cosmo)
    density_dt = jax.grad(high_level_api_simulation)(cosmo)

@jax.jit
def mid_level_api_simulation(cosmo, kvec, particles)
    # Initial conditions is a distributed 3D array
    inital_conditions = jpm.generate_ic(cosmo, boxsize, nmesh,sharding, kvec , dtype='float32')

    # Create a particular solver
    solver = jpm.solvers.fastpm(cosmo, B=1)

    # Initialize and run the simulation
    state = solver.init(initial_conditions , kvec)

    state = solver.lpt(state , a=0.1)
    # OR
    state = solver.lpt2(state , a=0.1) # Does both LPT1 and LPT2

    # Run the nbody simulation
    state = solver.Euler(state , t0=0.1, t1=1 , dt=0.01)
    # OR
    state = solver.Euler(state , t0=0.1, t1=1 , ts=snaphots)

    # Also provide a leapfrog integrator

    # Or use external integrator
    ode_fn = jpm.make_ode_fn(nmesh)
    term = ODETerm(lambda t, state, args: ode_fn(state, t, args))
    diffrax_solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5,atol=1e-5)

    ode_solution = diffrax.diffeqsolve(term,
                        diffrax_solver,
                        t0=0.1,
                        t1=1.,
                        dt0=0.01,
                        y0=state,
                        saveat=SaveAt(t0=False,t1=True,ts=snapshots),
                        args=cosmo,
                        stepsize_controller=stepsize_controller)

    final_state = ode_solution.ys[-1]
    # User defined function to compute weights
    weights = get_weights(particles)
    density = jpm.cic_paint(final_state.positions , weights)

    return state

with spmd_config:
    density = mid_level_api_simulation(cosmo)
    density_dt = jax.grad(mid_level_api_simulation)(cosmo)

@jax.jit
def low_level_api_simulation(cosmo, kvec, particles, a=0.1)
    # Initial conditions is a distributed 3D array
    inital_conditions = jpm.generate_ic(cosmo, boxsize, nmesh,sharding, kvec , dtype='float32')

    # Create a particular solver
    initial_force = pm_forces(inital_conditions,nmesh)
    # First order LPT
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    # LPT2
    delta2 = jpm.generate_2lpt(inital_conditions, nmesh,kvec)
    init_force2 = pm_forces(delta2,nmesh)
    # Taken from Hugo Simon
    dx2 = 3/7 * growth_factor_second(cosmo, a) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
    p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2

    dx += dx2
    p  += p2

    state = jpm.empty_state(dx , p , kvec)

    def nbody_ode(state, a, cosmo):

        pos, vel , kvec= state.positions, state.velocities , state.kvec

        forces = pm_forces(pos, mesh_shape=nmesh) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    term = ODETerm(lambda t, state, args: nbody_ode(state, t, args))
    diffrax_solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5,atol=1e-5)


    ode_solution = diffeqsolve(term,
                        diffrax_solver,
                        t0=0.1,
                        t1=1.,
                        dt0=0.01,
                        y0=state,
                        saveat=SaveAt(t0=False,t1=True,ts=snapshots),
                        args=cosmo,
                        stepsize_controller=stepsize_controller)

    final_state = ode_solution.ys[-1]
    # User defined function to compute weights
    weights = get_weights(particles)
    density = jpm.cic_paint(final_state.positions , weights)

    return state

with spmd_config:
    density = low_level_api_simulation(cosmo)
    density_dt = jax.grad(low_level_api_simulation)(cosmo)

```

Users can also mix and match the different levels of API to create their own custom simulations.

### TODOs

- [ ] Implement tsc_paint and tsc_compenstate
- [ ] Implement a distributed power spectrum calculation
