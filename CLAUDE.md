# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxPM is a JAX-powered Cosmological Particle-Mesh N-body Solver that provides differentiable particle-mesh simulations for cosmological applications. The project supports both single-GPU and distributed multi-GPU/multi-host execution using jaxDecomp.

## Development Commands

### Installation
```bash
# Basic installation
pip install jaxpm

# Development installation
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test markers
pytest -m single_device
pytest -m distributed

# Run specific test files
pytest tests/test_gradients.py
pytest tests/test_distributed_pm.py
```

### Build
```bash
# Build package
pip install build
python -m build
```

## Core Architecture

### Main Components

- **`jaxpm/pm.py`**: Core particle-mesh functionality including `pm_forces()` and ODE integration
- **`jaxpm/distributed.py`**: Multi-GPU distribution utilities (`fft3d`, `ifft3d`, `normal_field`)
- **`jaxpm/lensing.py`**: Gravitational lensing implementations including ray tracing
- **`jaxpm/spherical.py`**: Spherical coordinate transformations and HEALPix mapping
- **`jaxpm/painting.py`**: Cloud-in-Cell (CIC) painting functions for particle-to-mesh operations
- **`jaxpm/kernels.py`**: FFT kernels for gravitational and cosmological calculations
- **`jaxpm/growth.py`**: Cosmological growth functions and perturbation theory
- **`jaxpm/ode.py`**: ODE solvers and integration utilities

### Key Design Patterns

1. **Sharding Strategy**: Multi-GPU operations require consistent `sharding` and `halo_size` parameters across functions
2. **Painting Modes**: Two CIC painting approaches:
   - `cic_paint`: Absolute particle positions (faster, more memory)
   - `cic_paint_dx`: Relative positions from uniform grid (more accurate in multi-device)
3. **Distributed Operations**: Functions like `pm_forces`, `lpt`, `linear_field` accept sharding parameters for multi-GPU execution

### Dependencies

- **JAX** (>=0.4.35): Core computational framework
- **jax_cosmo**: Cosmological calculations
- **jaxdecomp** (>=0.2.3): Multi-GPU domain decomposition
- **jax_healpy**: For spherical coordinate operations

## Multi-GPU Development

### Key Considerations
- Always pass consistent `sharding` and `halo_size` parameters
- Use processor dimensions (pdims) like (1,8), (2,4), (4,2) for 8 devices
- Avoid (X,1) decompositions as they give less accurate results
- For relative painting, set `particles=None` in `lpt()` and `paint_absolute_pos=False`

### Distribution Utilities
```python
from jaxpm.distributed import uniform_particles, normal_field, fft3d, ifft3d
```

## Notebook Examples

The `notebooks/` directory contains comprehensive examples:
- `01-Introduction.ipynb`: Basic single-GPU PM simulation
- `02-Advanced_usage.ipynb`: Second-order LPT and diffrax solvers
- `03-MultiGPU_PM_Halo.ipynb`: Multi-GPU with halo exchange
- `04-MultiGPU_PM_Solvers.ipynb`: Solver comparisons
- `05-MultiHost_PM.ipynb`: Multi-host distributed simulations
- `06-RayTracing.ipynb`: Gravitational lensing ray tracing
- `07-SphericalRayTracing.ipynb`: Spherical coordinate ray tracing

## Current Development Branch

Working on branch `41-spherical-lensing` with focus on spherical ray tracing capabilities and power spectrum analysis.