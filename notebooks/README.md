# Particle Mesh Simulation with JAXPM on Multi-GPU and Multi-Host Systems

This collection of notebooks demonstrates how to perform Particle Mesh (PM) simulations using **JAXPM**, leveraging JAX for efficient computation on multi-GPU and multi-host systems. Each notebook progressively covers different setups, from single-GPU simulations to advanced, distributed, multi-host simulations across multiple nodes.

## Table of Contents

1. **[Single-GPU Particle Mesh Simulation](01-Introduction.ipynb)**
   - Introduction to basic PM simulations on a single GPU.
   - Uses JAXPM to run simulations with absolute particle positions and Cloud-in-Cell (CIC) painting.

2. **[Advanced Particle Mesh Simulation on a Single GPU](02-Advanced_usage.ipynb)**
   - Explore using diffrax solvers in the ODE step.
   - Explores second order Lagrangian Perturbation Theory (LPT) simulations.
   - Introduces weighted density field projections

3. **[Multi-GPU Particle Mesh Simulation with Halo Exchange](03-MultiGPU_PM_Halo.ipynb)**
   - Extends PM simulation to multi-GPU setups with halo exchange.
   - Uses sharding and device mesh configurations to manage distributed data across GPUs.

4. **[Multi-GPU Particle Mesh Simulation with Advanced Solvers](04-MultiGPU_PM_Solvers.ipynb)**
   - Compares different ODE solvers (Leapfrog and Dopri5) in multi-GPU simulations.
   - Highlights performance, memory considerations, and solver impact on simulation quality.

5. **[Multi-Host Particle Mesh Simulation](05-MultiHost_PM.ipynb)**
   - Extends PM simulations to multi-host, multi-GPU setups for large-scale simulations.
   - Guides through job submission, device initialization, and retrieving results across nodes.

## Getting Started

Each notebook includes installation instructions and guidelines for configuring JAXPM and required dependencies. Follow the setup instructions in each notebook to ensure an optimal environment.

## Requirements

- **JAXPM** (included in the installation commands within notebooks)
- **Diffrax** for ODE solvers
- **JAX** with CUDA support for multi-GPU or TPU setups
- **SLURM** for job scheduling on clusters (if running multi-host setups)

> **Note**: These notebooks are tested on the **Jean Zay** supercomputer and may require configuration changes for different HPC clusters.

## Caveats

### Cloud-in-Cell (CIC) Painting (Single Device)

There is two ways to perform the CIC painting in JAXPM. The first one is to use the `cic_paint` which paints absolute particle positions to the mesh. The second one is to use the `cic_paint_dx` which paints relative particle positions to the mesh (using uniform particles). The absolute version is faster at the cost of more memory usage.

inorder to use relative painting you need to :

 - Set the `particles` argument in `lpt` function from `jaxpm.pm` to `None`
 - Set `paint_absolute_pos` to `False` in `make_ode_fn` or `make_diffrax_ode` function from `jaxpm.pm` (it is True by default)

Otherwise you set `particles` to the starting particles of your choice and leave `paint_absolute_pos` to `True` (default value).

### Cloud-in-Cell (CIC) Painting (Multi Device)

Both `cic_paint` and `cic_paint_dx` functions are available in multi-device mode.

You need to set the arguments `sharding` and `halo_size` which is explained in the notebook [03-MultiGPU_PM_Halo.ipynb](03-MultiGPU_PM_Halo.ipynb).

One thing to note that `cic_paint` is not as accurate as `cic_paint_dx` in multi-device mode and therefor is not recommended.

Using relative painting in multi-device mode is just like in single device mode.\
You need to set the `particles` argument in `lpt` function from `jaxpm.pm` to `None` and set `paint_absolute_pos` to `False`

### Distributed PM

To run a distributed PM follow the examples in notebooks [03](03-MultiGPU_PM_Halo.ipynb) and [05](05-MultiHost_PM.ipynb) for multi-host.

In short you need to set the arguments `sharding` and `halo_size` in `lpt` , `linear_field` the `make_ode` functions and `pm_forces` if you use it.

Missmatching the shardings will give you errors and unexpected results.

You can also use `normal_field` and `uniform_particles` from `jaxpm.pm.distributed` to create the fields and particles with a sharding.

### Choosing the right pdims

pdims are processor dimensions.\
Explained more in the jaxdecomp paper [here](https://github.com/DifferentiableUniverseInitiative/jaxDecomp).

For 8 devices there are three decompositions that are possible:
- (1 , 8)
- (2 , 4) , (4 , 2)
- (8 , 1)

(1 , X) should be the fastest (2 , X) or (X , 2) is more accurate but slightly slower.\
and (X , 1) is giving the least accurate results for some reason so it is not recommended.
