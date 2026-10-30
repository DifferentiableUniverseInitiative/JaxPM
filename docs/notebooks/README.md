# JaxPM tutorial notebooks

This collection demonstrates Particle-Mesh (PM) cosmological simulations with
**JaxPM**, leveraging JAX for efficient computation from a single GPU up to
distributed multi-GPU / multi-host systems. These notebooks are the source of the
[ReadTheDocs documentation](../index.rst); each can also be opened directly in Colab
via the badge at the top of the notebook.

## Table of contents

### Getting started (single GPU)
1. **[Introduction](01-Introduction.ipynb)** — end-to-end intro: linear field → LPT →
   a few symplectic PM steps → `paint` density. Uses the current
   `paint` / `symplectic_ode` + diffrax `SemiImplicitEuler` API.
2. **[Advanced usage](02-Advanced_usage.ipynb)** — 1st/2nd-order LPT, displacement vs
   absolute positions (`initial_particles`), and weighted projections.
3. **[Painting & deconvolution](02b-Painting_and_Deconvolution.ipynb)** — mass-assignment
   windows (NGP/CIC/TSC/PCS), 3D power spectra via `jaxpm.utils.power_spectrum`, and
   window deconvolution / compensation.

### Multi-GPU / multi-host
4. **[Multi-GPU PM with halo exchange](03-MultiGPU_PM_Halo.ipynb)** — sharding, device
   mesh, and halo size.
5. **[Multi-GPU PM solvers](04-MultiGPU_PM_Solvers.ipynb)** — comparing ODE solvers in a
   distributed setup.
6. **[Multi-host PM](05-MultiHost_PM.ipynb)** — multi-node runs (job submission, device
   init, gathering results).
7. **[Animating PM fields](06-Animating_PM_Fields.ipynb)** — rendering an animation from
   saved fields.

> **Note:** notebooks 03–06 still demonstrate the older `cic_paint` / `make_diffrax_ode`
> API and are scheduled for migration to `paint` / `symplectic_ode`.

### Spherical painting & lensing
8. **[Spherical painting methods](07-Spherical_Painting_Methods.ipynb)** — HEALPix
   painting (NGP / RBF), and window deconvolution with `jaxpm.spherical.deconvolve_map`.
9. **[Convergence vs GLASS](08-convergence-vs-glass.ipynb)** — Born weak-lensing
   convergence from a lightcone, validated against GLASS.

## Painting API (current)

Mass assignment is done through a single `paint` entry point (`jaxpm.painting`):

- **Absolute positions:** `paint(positions, grid_mesh=..., order='cic')`.
- **Displacements (memory-efficient):** run the simulation with
  `initial_particles='uniform'` / `lpt(..., particles=None)` so the state is stored as
  displacements `dx`, then `paint(dx, initial_particles='uniform', order='cic')`. The
  uniform integer grid is added internally at paint time.
- `order` selects the window: `'ngp'`, `'cic'`, `'tsc'`, `'pcs'`. `readout` is the
  adjoint (mesh → particles).

Absolute painting is faster but uses more memory; the displacement path is preferred in
distributed settings, where absolute `cic_paint` is also less accurate than the
displacement path.

## Distributed PM

For multi-GPU / multi-host runs, pass matching `sharding` and `halo_size` to
`linear_field`, `lpt`, the ODE construction, and `pm_forces`. Mismatched shardings lead
to errors or wrong results. Use `normal_field` / `uniform_particles` from
`jaxpm.distributed` to create fields and particles with a given sharding. See notebooks
[03](03-MultiGPU_PM_Halo.ipynb) and [05](05-MultiHost_PM.ipynb).

### Choosing `pdims`

`pdims` are the processor dimensions of the device mesh (see the
[jaxDecomp paper](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)). For 8
devices: `(1, 8)` is typically fastest, `(2, 4)` / `(4, 2)` are more accurate but
slightly slower, and `(8, 1)` is least accurate and not recommended.

## Requirements

- **JaxPM** (install commands are inside each notebook)
- **diffrax** for the ODE solvers
- **JAX** with CUDA for multi-GPU / TPU setups
- **`jaxpm[spherical]`** (jax-healpy + healpy) for notebooks 07–08
- **SLURM** for multi-host job scheduling on clusters

> These notebooks are tested on the **Jean Zay** supercomputer and may need
> configuration changes on other HPC clusters.
