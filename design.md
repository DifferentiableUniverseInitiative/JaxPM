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
