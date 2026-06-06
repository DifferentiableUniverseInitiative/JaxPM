import os

os.environ["EQX_ON_ERROR"] = "nan"  # avoid an allgather caused by diffrax
import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
if rank == 0:
    print(f"SIZE is {jax.device_count()}")

import argparse
from functools import partial

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import (
    ConstantStepSize,
    ODETerm,
    SaveAt,
    SemiImplicitEuler,
    diffeqsolve,
)
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.ode import symplectic_fpm_ode
from jaxpm.painting import paint
from jaxpm.pm import linear_field, lpt

all_gather = partial(process_allgather, tiled=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a multi-host cosmological PM simulation with JAX (FastPM solver)."
    )
    parser.add_argument(
        "-p",
        "--pdims",
        type=int,
        nargs=2,
        default=[1, jax.device_count()],
        help="Processor grid dimensions as two integers (e.g., 2 4).",
    )
    parser.add_argument(
        "-m",
        "--mesh_shape",
        type=int,
        nargs=3,
        default=[512, 512, 512],
        help="Shape of the simulation mesh as three values (e.g., 512 512 512).",
    )
    parser.add_argument(
        "-b",
        "--box_size",
        type=float,
        nargs=3,
        default=[500.0, 500.0, 500.0],
        help="Box size of the simulation as three values (e.g., 500.0 500.0 1000.0).",
    )
    parser.add_argument(
        "-st",
        "--snapshots",
        type=int,
        default=2,
        help="Number of snapshots to save during the simulation.",
    )
    parser.add_argument(
        "-H", "--halo_size", type=int, default=64, help="Halo size for the simulation."
    )
    return parser.parse_args()


def create_sharding(pdims):
    # Build the mesh with *Auto* axis types: jaxDecomp's distributed FFT and halo-exchange
    # primitives rely on JAX's automatic (compiler-managed) sharding. An Explicit-axis
    # mesh (the default of jax.make_mesh without axis_types) would break them. See:
    # https://jaxdecomp.readthedocs.io/en/latest/07-xla_sharding_configuration.html#auto-vs-explicit-axis-types
    mesh = jax.make_mesh(
        pdims, axis_names=("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto)
    )
    return NamedSharding(mesh, P("x", "y"))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def run_simulation(
    omega_c,
    sigma8,
    mesh_shape,
    box_size,
    halo_size,
    nb_snapshots,
    sharding,
):
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

    initial_conditions = linear_field(
        mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(0), sharding=sharding
    )

    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    dx, p, _ = lpt(
        cosmo, initial_conditions, a=0.1, halo_size=halo_size, sharding=sharding
    )

    # FastPM solver: symplectic kick/drift operators integrated by a fixed-step
    # symplectic Euler. The FastPM factors are per unit dt0, so a constant step that
    # equals dt0 is required to reproduce the FastPM scheme.
    t0, t1, dt0 = 0.1, 1.0, 0.05
    drift, kick, first_kick = symplectic_fpm_ode(
        mesh_shape,
        dt0,
        cosmo,
        initial_particles="uniform",
        halo_size=halo_size,
        sharding=sharding,
    )
    # FastPM initial half-kick to stagger the velocities.
    p = p + dt0 * first_kick(t0, dx, cosmo)

    res = diffeqsolve(
        (ODETerm(drift), ODETerm(kick)),
        SemiImplicitEuler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=(dx, p),
        args=cosmo,
        saveat=SaveAt(ts=jnp.linspace(0.2, 1.0, nb_snapshots)),
        stepsize_controller=ConstantStepSize(),
    )

    # res.ys = (positions, velocities); positions are stored as displacements.
    ode_fields = [
        paint(
            pos,
            initial_particles="uniform",
            order="cic",
            halo_size=halo_size,
            sharding=sharding,
        )
        for pos in res.ys[0]
    ]
    lpt_field = paint(
        dx,
        initial_particles="uniform",
        order="cic",
        halo_size=halo_size,
        sharding=sharding,
    )
    return initial_conditions, lpt_field, ode_fields, res.stats


def main():
    args = parse_arguments()
    mesh_shape = args.mesh_shape
    box_size = args.box_size
    halo_size = args.halo_size
    nb_snapshots = args.snapshots

    sharding = create_sharding(args.pdims)

    initial_conditions, lpt_displacements, ode_solutions, solver_stats = run_simulation(
        0.25,
        0.8,
        tuple(mesh_shape),
        tuple(box_size),
        halo_size,
        nb_snapshots,
        sharding,
    )

    if rank == 0:
        os.makedirs("fields", exist_ok=True)
        print(f"[{rank}] Simulation done")
        print(f"Solver stats: {solver_stats}")

    # Save initial conditions
    initial_conditions_g = all_gather(initial_conditions)
    if rank == 0:
        print(f"[{rank}] Saving initial_conditions")
        np.save("fields/initial_conditions.npy", initial_conditions_g)
        print(f"[{rank}] initial_conditions saved")
    del initial_conditions_g, initial_conditions

    # Save LPT displacements
    lpt_displacements_g = all_gather(lpt_displacements)
    if rank == 0:
        print(f"[{rank}] Saving lpt_displacements")
        np.save("fields/lpt_displacements.npy", lpt_displacements_g)
        print(f"[{rank}] lpt_displacements saved")
    del lpt_displacements_g, lpt_displacements

    # Save each ODE solution separately
    for i, sol in enumerate(ode_solutions):
        sol_g = all_gather(sol)
        if rank == 0:
            print(f"[{rank}] Saving ode_solution_{i}")
            np.save(f"fields/ode_solution_{i}.npy", sol_g)
            print(f"[{rank}] ode_solution_{i} saved")
        del sol_g


if __name__ == "__main__":
    main()
