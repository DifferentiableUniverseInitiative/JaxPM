#!/usr/bin/env python
"""Timing-comparison harness for JaxPM gravitational force solvers: FFT vs multigrid vs
warm-started multigrid. Built for lots of quick comparison runs.

Single GPU (default):     python bench/bench_forces.py --mesh 256
Pick solvers/knobs:       python bench/bench_forces.py --mesh 256 --solvers fft mg mgwarm \
                                   --cycles 2 --v 4 --nsteps 8
Multi-GPU (sharded):      srun python bench/bench_forces.py --mesh 512 --pdims 1 4

For `mgwarm` each step warm-starts the MG solve from the previous step's potential (growth-scaled);
the FFT solver is direct and re-does the full transform every step regardless. We advance the
particles by a small leapfrog drift each step so the field genuinely evolves between solves.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=256, help="cubic mesh size N (N^3)")
    ap.add_argument("--pdims", type=int, nargs=2, default=None, help="2D device grid, e.g. 1 4")
    ap.add_argument("--solvers", nargs="+", default=["fft", "mg", "mgwarm"],
                    choices=["fft", "mg", "mgwarm"])
    ap.add_argument("--nsteps", type=int, default=6, help="evolution steps to time per solver")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats per step")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--v", type=int, default=4, help="pre/post smoothing sweeps")
    ap.add_argument("--omega", type=float, default=0.857)
    ap.add_argument("--agg-n", type=int, default=16,
                    help="MG coarse-grid agglomeration threshold (needed at >=2-way sharding)")
    ap.add_argument("--grad-order", type=int, default=4, choices=[2, 4])
    ap.add_argument("--warm-cycles", type=int, default=1, help="MG cycles when warm-started")
    ap.add_argument("--dt", type=float, default=0.02, help="leapfrog drift to evolve the field")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fd", action="store_true", default=True,
                    help="FFT uses the discrete-Laplacian inverse (matches MG operator; default on)")
    ap.add_argument("--no-fd", dest="fd", action="store_false",
                    help="FFT uses the continuum -1/k^2 (JaxPM default); shows the operator gap vs MG")
    return ap.parse_args()


def maybe_init_distributed():
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        jax.distributed.initialize()
        return jax.process_index() == 0
    return True


def make_sharding(pdims):
    if pdims is None:
        return None
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental import mesh_utils
    devs = mesh_utils.create_device_mesh(tuple(pdims))
    mesh = jax.sharding.Mesh(devs, axis_names=("x", "y"))
    return NamedSharding(mesh, P("x", "y"))


def main():
    args = parse_args()
    is_coord = maybe_init_distributed()
    def cp(*a, **k):
        if is_coord: print(*a, **k, flush=True)

    from jaxpm.pm import pm_forces
    from jaxpm.distributed import uniform_particles

    N = args.mesh
    mesh_shape = (N, N, N)
    sharding = make_sharding(args.pdims)
    cp(f"mesh={mesh_shape} devices={jax.device_count()} pdims={args.pdims} solvers={args.solvers}")

    # Initial particle state: uniform grid + small random displacement (a PM-like configuration).
    pos = uniform_particles(mesh_shape, sharding=sharding).reshape(-1, 3)
    rng = np.random.default_rng(args.seed)
    disp = jnp.asarray(0.3 * rng.standard_normal((pos.shape[0], 3)).astype(np.float32))
    if sharding is not None:
        disp = jax.device_put(disp, sharding.mesh and jax.sharding.NamedSharding(
            sharding.mesh, jax.sharding.PartitionSpec(None, None)))
    pos = (pos + disp) % N
    vel = jnp.zeros_like(pos)

    mg_params = dict(cycles=args.cycles, v1=args.v, v2=args.v, omega=args.omega,
                     agg_n=args.agg_n, grad_order=args.grad_order)
    mg_params_warm = dict(mg_params); mg_params_warm["cycles"] = args.warm_cycles

    @jax.jit
    def forces_fft(pos):
        return pm_forces(pos, mesh_shape=mesh_shape, halo_size=0, sharding=sharding,
                         solver="fft", fd=args.fd)

    @partial(jax.jit, static_argnames=("warm",))
    def forces_mg(pos, u0, warm):
        p = mg_params_warm if warm else mg_params
        return pm_forces(pos, mesh_shape=mesh_shape, halo_size=0, sharding=sharding,
                         solver="mg", mg_params=p, u0=u0, return_potential=True)

    def drift(pos, f, dt):                       # tiny leapfrog to evolve the field between steps
        return (pos + dt * f) % N

    def block(x):
        return jax.block_until_ready(x)

    # Reference (FFT) forces along an evolving trajectory, for accuracy comparison.
    ref_traj = []
    p = pos
    for s in range(args.nsteps):
        f = block(forces_fft(p))
        ref_traj.append((p, f))
        p = drift(p, f, args.dt)

    def relerr(a, b):
        return float(jnp.linalg.norm((a - b).ravel()) / (jnp.linalg.norm(b.ravel()) + 1e-30))

    results = {}
    for solver in args.solvers:
        # warmup/compile
        if solver == "fft":
            block(forces_fft(pos))
        else:
            block(forces_mg(pos, jnp.zeros(mesh_shape, jnp.float32), solver == "mgwarm"))
        # Walk the SAME (FFT) trajectory positions so accuracy is apples-to-apples; warm-start
        # recycles phi from the previous reference position (a real between-step change).
        times, accs = [], []
        phi_prev = None
        for s in range(args.nsteps):
            p = ref_traj[s][0]
            if solver == "fft":
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    f = block(forces_fft(p))
                dt_ms = (time.perf_counter() - t0) / args.repeats * 1e3
            else:
                warm = solver == "mgwarm"
                u0 = phi_prev if (warm and phi_prev is not None) else jnp.zeros(mesh_shape, jnp.float32)
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    f, phi = forces_mg(p, u0, warm)
                    block((f, phi))
                dt_ms = (time.perf_counter() - t0) / args.repeats * 1e3
                phi_prev = phi
            accs.append(relerr(f, ref_traj[s][1]))
            times.append(dt_ms)
        results[solver] = (float(np.median(times[1:] or times)), np.array(accs))
        cp(f"[{solver:7s}] median {np.median(times[1:] or times):7.2f} ms/solve   "
           f"force relerr vs FFT: step0={accs[0]:.2e} last={accs[-1]:.2e}")

    if "fft" in results and ("mg" in results or "mgwarm" in results):
        tf = results["fft"][0]
        cp("\nspeedups (FFT / solver):")
        for s in args.solvers:
            if s != "fft":
                cp(f"  {s:7s}: {tf / results[s][0]:.2f}x   ({results[s][0]:.2f} ms vs FFT {tf:.2f} ms)")


if __name__ == "__main__":
    main()
