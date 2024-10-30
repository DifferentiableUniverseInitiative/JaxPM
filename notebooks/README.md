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
