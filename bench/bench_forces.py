#!/usr/bin/env python
"""Scaling benchmark for JaxPM gravitational force solvers: multigrid (MG) vs FFT.

One solver per job, timed with ``jax_hpc_profiler.JaxTimer`` and written as a single
CSV row -- structured like the jaxDecomp benchmark (``../jaxDecomp/benchmarks/bench.py``)
so the strong/weak-scaling SLURM submitters drop straight in.

Single GPU:        python bench/bench_forces.py --global_shape 256 256 256 --pdims 1 -b mg
Multi-GPU (srun):  srun python bench/bench_forces.py --global_shape 1024 1024 1024 \
                       --pdims 64 -b fft -n 16 --gpus-per-node 4 -o results/strong

Solvers (``-b/--solver``):
  fft      direct FFT Poisson solve (jaxdecomp pfft), re-done every call.
  mg       real-space halo multigrid, cold start (u0 = 0) every call.
  mgwarm   multigrid warm-started from a pre-converged potential (steady-state cost).

The decomposition is always a 1D slab along x (``pdims = n x 1``): the first spatial axis is
split across all ``n`` devices, the others stay local. Particles are always relative
displacements (painted with the fast stencil ``order`` scheme, default CIC) and forces are
read back with the matching ``readout``; both solvers share the same paint/readout so the
MG-vs-FFT comparison is apples-to-apples.

The CIC halo is a fraction of the local slab (fli-style: halo = local_mesh x multiplier),
auto-picked as the smallest of HALO_FRACTIONS that covers the rms 1D displacement sigma for the
``--box-size`` box and mesh resolution. As GPUs increase the local slab (N/n) shrinks, so the
fraction needed to cover sigma grows -- the "starved ghost zone" of jax-fli's resolution-
convergence experiment.
"""
import argparse
import math
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["--xla_gpu_nccl_termination_timeout_seconds"] = "100"
os.environ["--xla_gpu_executable_warn_stuck_timeout"] = "60"


def _maybe_init_distributed():
    if (int(os.environ.get("SLURM_NTASKS", 0)) > 1
            or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
            or int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)) > 1
            or int(os.environ.get("PMI_SIZE", 0)) > 1):
        for key in ("VSCODE_PROXY_URI", "no_proxy", "NO_PROXY"):
            os.environ.pop(key, None)
        # print time stamp
        from datetime import datetime

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detected multi-host environment, initializing JAX distributed ..."
        )
        import jax

        jax.distributed.initialize()
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] JAX distributed initialized with {jax.process_count()} processes, rank {jax.process_index()}"
        )


_maybe_init_distributed()

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental import multihost_utils
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import JaxTimer

from jaxpm.distributed import normal_field
from jaxpm.pm import pm_forces

# Candidate halo sizes, as a fraction of the local slab. The smallest one that covers the
# displacement sigma is used (fli-style halo = local_mesh x multiplier).
HALO_FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


def parse_args():
    ap = argparse.ArgumentParser(
        description="JaxPM MG-vs-FFT force solver benchmark")
    ap.add_argument(
        "--pdims",
        type=int,
        required=True,
        help=
        "number of devices along x; the run is always a 1D slab (pdims = n x 1)"
    )
    ap.add_argument(
        "--global_shape",
        nargs=3,
        type=int,
        help="Global mesh shape, e.g. 1024 1024 1024 (strong scaling)")
    ap.add_argument(
        "--local_shape",
        nargs=3,
        type=int,
        help=
        "Per-device slab (lx ly lz) for weak scaling; global = (lx*n, ly, lz)")
    ap.add_argument("-n", "--nb_nodes", type=int, default=1)
    ap.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node; drives the hybrid (NVLink/IB) device mesh")
    ap.add_argument("-o", "--output_path", type=str, default=".")
    ap.add_argument("-pr",
                    "--precision",
                    type=str,
                    default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("-i", "--iterations", type=int, default=5)
    ap.add_argument("-b",
                    "--solver",
                    type=str,
                    default="fft",
                    choices=["fft", "mg", "mgwarm"],
                    help="force solver to time")
    # painting
    ap.add_argument("--order",
                    type=str,
                    default="CIC",
                    choices=["NGP", "CIC", "TSC", "PCS"],
                    help="mass-assignment scheme")
    ap.add_argument(
        "--box-size",
        type=float,
        default=2000.0,
        help=
        "physical box size (Mpc/h); with the mesh resolution this sets the rms "
        "displacement sigma the halo must cover")
    # multigrid knobs
    ap.add_argument("--cycles", type=int, default=2, help="MG V-cycles (cold)")
    ap.add_argument("--warm-cycles",
                    type=int,
                    default=1,
                    help="MG V-cycles when warm-started")
    ap.add_argument("--v",
                    type=int,
                    default=4,
                    help="pre/post smoothing sweeps")
    ap.add_argument("--omega", type=float, default=0.857)
    ap.add_argument(
        "--agg-n",
        type=int,
        default=16,
        help=
        "MG coarse-grid agglomeration threshold (needed at >=2-way sharding)")
    ap.add_argument("--grad-order", type=int, default=4, choices=[2, 4])
    # FFT operator
    ap.add_argument(
        "--fd",
        action="store_true",
        default=True,
        help="FFT uses the discrete-Laplacian inverse (matches MG; default on)"
    )
    ap.add_argument(
        "--no-fd",
        dest="fd",
        action="store_false",
        help="FFT uses the continuum -1/k^2 (shows the operator gap vs MG)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def make_sharding(n, gpus_per_node):
    """1D slab device mesh over ('x','y') with pdims = (n, 1): the first spatial axis is
    split across all n devices, the rest stays local. Topology-aware across nodes.

    Single node (uniform NVLink fabric): a plain mesh. Several nodes (non-uniform
    bandwidth -- fast NVLink inside a node, slower InfiniBand between): tile with
    ``create_hybrid_device_mesh`` so each node's GPUs stay contiguous on the x-axis
    (over NVLink) and only the slab boundaries between nodes cross the network.
    See jax-fli docs/2-advanced-usage/11-multi-host-pm.md.
    """
    if n == 1:
        return None
    devices = jax.devices()
    multi_node = hasattr(devices[0], "slice_index")
    if multi_node and n % gpus_per_node == 0:
        intra = (gpus_per_node, 1
                 )  # GPUs within one node (NVLink), contiguous on x
        inter = (n // gpus_per_node, 1)  # across nodes (InfiniBand)
        mesh = Mesh(create_hybrid_device_mesh(intra, inter),
                    axis_names=("x", "y"))
    else:
        # Single node, or a slab whose x-extent is not a whole number of nodes:
        # fall back to a flat auto mesh (still correct, just not NVLink-tiled).
        mesh = jax.make_mesh((n, 1), ("x", "y"),
                             axis_types=(AxisType.Auto, AxisType.Auto))
    return NamedSharding(mesh, P("x", "y"))


def displacement_sigma_cells(box, n_cells):
    """rms 1D Zel'dovich displacement at z=0 (Planck15), expressed in mesh cells.

    sigma_1D = sqrt( int P(k) dk / (6 pi^2) ) [Mpc/h] (linear theory, a=1), divided by the
    cell size box/n_cells. Mirrors jax-fli docs/5-experiments/01-resolution-convergence
    ``_sigma_displacement``; the halo must cover this many cells.
    """
    cosmo = jc.Planck15()
    k = jnp.logspace(-4, 1.3, 4000)
    pk = jc.power.linear_matter_power(cosmo, k, a=1.0)
    sigma_phys = float(jnp.sqrt(jnp.trapezoid(pk, k) /
                                (6 * jnp.pi**2)))  # Mpc/h
    return sigma_phys / (box / n_cells)


def main():
    args = parse_args()
    is_coord = jax.process_index() == 0

    def cp(*a, **k):
        if is_coord:
            print(*a, **k, flush=True)

    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if args.precision == "float64" else jnp.float32

    # Always a 1D slab: pdims = (n, 1), only the x-axis is split across devices.
    n = args.pdims
    # Resolve global mesh shape (strong: --global_shape; weak: --local_shape -> *n on x).
    if (args.global_shape is None) == (args.local_shape is None):
        raise SystemExit(
            "Provide exactly one of --global_shape / --local_shape")
    if args.local_shape is not None:
        mesh_shape = (args.local_shape[0] * n, args.local_shape[1],
                      args.local_shape[2])
    else:
        mesh_shape = tuple(args.global_shape)
    if mesh_shape[0] % n != 0:
        raise SystemExit(
            f"mesh x-dim {mesh_shape[0]} not divisible by pdims n={n}")

    os.makedirs(args.output_path, exist_ok=True)
    sharding = make_sharding(n, args.gpus_per_node)
    N = mesh_shape[0]
    cp(f"mesh={mesh_shape} devices={jax.device_count()} pdims=({n}, 1) "
       f"solver={args.solver} order={args.order} precision={args.precision}")

    # Halo = a fraction of the local slab, auto-picked to cover the rms displacement sigma.
    # sigma (cells) is set by the box + mesh resolution; the local slab N/n shrinks with more
    # GPUs, so the fraction of it needed to cover sigma grows with the GPU count -- and can
    # eventually exceed even the largest fraction (the "starved ghost zone").
    sigma_cells = displacement_sigma_cells(args.box_size, N)
    local_mesh = mesh_shape[0] // n
    if n == 1:
        halo_size = 0  # single device: no seams
    else:
        for mult in HALO_FRACTIONS:
            halo_size = math.ceil(mult * local_mesh)
            if halo_size >= sigma_cells:
                break
        else:
            cp(f"WARNING: even {HALO_FRACTIONS[-1]:.0%} of the local slab ({halo_size} cells) "
               f"< sigma {sigma_cells:.1f} cells -- starved ghost zone.")
    pct = 100.0 * halo_size / local_mesh if local_mesh else 0.0
    cp(f"box={args.box_size} Mpc/h  sigma={sigma_cells:.2f} cells  local_mesh={local_mesh}  "
       f"halo={halo_size} cells ({pct:.0f}% of local slab)")

    disp = normal_field(jax.random.PRNGKey(args.seed), (*mesh_shape, 3),
                        sharding=sharding,
                        dtype=dtype)

    mg_params = dict(cycles=args.cycles,
                     v1=args.v,
                     v2=args.v,
                     omega=args.omega,
                     agg_n=max(args.agg_n, n),  # agglomeration must fire before the
                                                # sharded axis coarsens below 1 cell/shard
                                                # (needs agg_n >= pdims); see multigrid._cycle_halo
                     grad_order=args.grad_order)
    mg_params_warm = {**mg_params, "cycles": args.warm_cycles}

    common = dict(mesh_shape=mesh_shape,
                  paint_absolute_pos=False,
                  order=args.order,
                  halo_size=halo_size,
                  sharding=sharding)

    @jax.jit
    def f_fft(field):
        return pm_forces(field, solver="fft", fd=args.fd, **common)

    @jax.jit
    def f_mg(field, u0):
        return pm_forces(field,
                         solver="mg",
                         mg_params=mg_params,
                         u0=u0,
                         return_potential=True,
                         **common)

    @jax.jit
    def f_mgwarm(field, u0):
        return pm_forces(field,
                         solver="mg",
                         mg_params=mg_params_warm,
                         u0=u0,
                         return_potential=True,
                         **common)

    # Pick the timed function + its (fixed) arguments for this solver.
    if args.solver == "fft":
        fn, fn_args = f_fft, (disp, )
    elif args.solver == "mg":
        u0 = jnp.zeros(mesh_shape, dtype)
        if sharding is not None:
            u0 = jax.lax.with_sharding_constraint(u0, sharding)
        fn, fn_args = f_mg, (disp, u0)
    else:  # mgwarm: pre-solve a cold MG once for a near-converged u0 (steady-state warm cost;
        # best-case -- the field does not drift between the seed solve and the timed solves).
        u0 = jnp.zeros(mesh_shape, dtype)
        if sharding is not None:
            u0 = jax.lax.with_sharding_constraint(u0, sharding)
        _, u0 = jax.block_until_ready(f_mg(disp, u0))
        fn, fn_args = f_mgwarm, (disp, u0)

    timer = JaxTimer(save_jaxpr=False)
    out = timer.chrono_jit(fn, *fn_args)  # warmup + compile
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        multihost_utils.sync_global_devices("warmup")
    for _ in range(args.iterations):
        out = timer.chrono_fun(fn, *fn_args)
    del out

    jit_ms = timer.jit_time  # report() resets timer state, so capture it first
    timer.report(f"{args.output_path}/forces.csv",
                 function=f"{args.solver}-forces",
                 x=N,
                 y=mesh_shape[1],
                 z=mesh_shape[2],
                 precision=args.precision,
                 px=n,
                 py=1,
                 backend=args.solver,
                 nodes=args.nb_nodes)
    cp(f"[{args.solver:7s}] done -> {args.output_path}/forces.csv (jit {jit_ms:.1f} ms)"
       )


if __name__ == "__main__":
    main()
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        multihost_utils.sync_global_devices("end")
