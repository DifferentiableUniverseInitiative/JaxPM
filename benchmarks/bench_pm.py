import os

os.environ["EQX_ON_ERROR"] = "nan"  # avoid an allgather caused by diffrax
import jax

jax.distributed.initialize()

rank = jax.process_index()
size = jax.process_count()

import argparse
import time

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from cupy.cuda.nvtx import RangePop, RangePush
from diffrax import (ConstantStepSize, Dopri5, LeapfrogMidpoint, ODETerm,
                     PIDController, SaveAt, Tsit5, diffeqsolve)
from hpc_plotter.timer import Timer
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt, make_ode_fn


def run_simulation(mesh_shape,
                   box_size,
                   halo_size,
                   solver_choice,
                   iterations,
                   pdims=None):

    @jax.jit
    def simulate(omega_c, sigma8):
        # Create a small function to generate the matter power spectrum
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(
            jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
        pk_fn = lambda x: interpolate_power_spectrum(x, k, pk)

        # Create initial conditions
        initial_conditions = linear_field(mesh_shape,
                                          box_size,
                                          pk_fn,
                                          seed=jax.random.PRNGKey(0))

        # Create particles
        cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
        dx, p, _ = lpt(cosmo, initial_conditions, 0.1, halo_size=halo_size)
        if solver_choice == "Dopri5":
            solver = Dopri5()
        elif solver_choice == "LeapfrogMidpoint":
            solver = LeapfrogMidpoint()
        elif solver_choice == "Tsit5":
            solver = Tsit5()
        elif solver_choice == "lpt":
            lpt_field = cic_paint_dx(dx, halo_size=halo_size)
            print(f"TYPE of lpt_field: {type(lpt_field)}")
            return lpt_field, {"num_steps": 0}
        else:
            raise ValueError(
                "Invalid solver choice. Use 'Dopri5' or 'LeapfrogMidpoint'.")
        # Evolve the simulation forward
        ode_fn = make_ode_fn(mesh_shape, halo_size=halo_size)
        term = ODETerm(
            lambda t, state, args: jnp.stack(ode_fn(state, t, args), axis=0))

        if solver_choice == "Dopri5" or solver_choice == "Tsit5":
            stepsize_controller = PIDController(rtol=1e-4, atol=1e-4)
        elif solver_choice == "LeapfrogMidpoint" or solver_choice == "Euler":
            stepsize_controller = ConstantStepSize()
        res = diffeqsolve(term,
                          solver,
                          t0=0.1,
                          t1=1.,
                          dt0=0.01,
                          y0=jnp.stack([dx, p], axis=0),
                          args=cosmo,
                          saveat=SaveAt(t1=True),
                          stepsize_controller=stepsize_controller)

        # Return the simulation volume at requested
        state = res.ys[-1]
        final_field = cic_paint_dx(state[0], halo_size=halo_size)

        return final_field, res.stats

    def run():
        # Warm start
        chrono_fun = Timer()
        RangePush("warmup")
        final_field, stats = chrono_fun.chrono_jit(simulate,
                                                   0.32,
                                                   0.8,
                                                   ndarray_arg=0)
        RangePop()
        sync_global_devices("warmup")
        for i in range(iterations):
            RangePush(f"sim iter {i}")
            final_field, stats = chrono_fun.chrono_fun(simulate,
                                                       0.32,
                                                       0.8,
                                                       ndarray_arg=0)
            RangePop()
        return final_field, stats, chrono_fun

    if jax.device_count() > 1:
        devices = mesh_utils.create_device_mesh(pdims)
        mesh = Mesh(devices.T, axis_names=('x', 'y'))
        with mesh:
            # Warm start
            final_field, stats, chrono_fun = run()
    else:
        final_field, stats, chrono_fun = run()

    return final_field, stats, chrono_fun


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='JAX Cosmo Simulation Benchmark')
    parser.add_argument('-m',
                        '--mesh_size',
                        type=int,
                        help='Mesh size',
                        required=True)
    parser.add_argument('-b',
                        '--box_size',
                        type=float,
                        help='Box size',
                        required=True)
    parser.add_argument('-p',
                        '--pdims',
                        type=str,
                        help='Processor dimensions',
                        default=None)
    parser.add_argument(
        '-pr',
        '--precision',
        type=str,
        help='Precision',
        choices=["float32", "float64"],
    )
    parser.add_argument('-hs',
                        '--halo_size',
                        type=int,
                        help='Halo size',
                        required=True)
    parser.add_argument('-s',
                        '--solver',
                        type=str,
                        help='Solver',
                        choices=[
                            "Dopri5", "dopri5", "d5", "Tsit5", "tsit5", "t5",
                            "LeapfrogMidpoint", "leapfrogmidpoint", "lfm",
                            "lpt"
                        ],
                        required=True)
    parser.add_argument('-i',
                        '--iterations',
                        type=int,
                        help='Number of iterations',
                        default=10)
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        help='Output path',
                        default=".")
    parser.add_argument('-f',
                        '--save_fields',
                        action='store_true',
                        help='Save fields')
    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        help='Number of nodes',
                        default=1)

    args = parser.parse_args()
    mesh_size = args.mesh_size
    box_size = [args.box_size] * 3
    halo_size = args.halo_size
    solver_choice = args.solver
    iterations = args.iterations
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    print(f"solver choice: {solver_choice}")
    match solver_choice:
        case "Dopri5" | "dopri5" | "d5":
            solver_choice = "Dopri5"
        case "Tsit5" | "tsit5" | "t5":
            solver_choice = "Tsit5"
        case "LeapfrogMidpoint" | "leapfrogmidpoint" | "lfm":
            solver_choice = "LeapfrogMidpoint"
        case "lpt":
            solver_choice = "lpt"
        case _:
            raise ValueError(
                "Invalid solver choice. Use 'Dopri5', 'Tsit5', 'LeapfrogMidpoint' or 'lpt"
            )
    if args.precision == "float32":
        jax.config.update("jax_enable_x64", False)
    elif args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    if args.pdims:
        pdims = tuple(map(int, args.pdims.split("x")))
    else:
        pdims = (1, 1)

    mesh_shape = [mesh_size] * 3

    final_field, stats, chrono_fun = run_simulation(mesh_shape, box_size,
                                                    halo_size, solver_choice,
                                                    iterations, pdims)

    print(
        f"shape of final_field {final_field.shape} and sharding spec {final_field.sharding} and local shape {final_field.addressable_data(0).shape}"
    )

    metadata = {
        'rank': rank,
        'function_name': f'JAXPM-{solver_choice}',
        'precision': args.precision,
        'x': str(mesh_size),
        'y': str(mesh_size),
        'z': str(stats["num_steps"]),
        'px': str(pdims[0]),
        'py': str(pdims[1]),
        'backend': 'NCCL',
        'nodes': str(args.nodes)
    }
    # Print the results to a CSV file
    chrono_fun.print_to_csv(f'{output_path}/jaxpm_benchmark.csv', **metadata)

    # Save the final field
    nb_gpus = jax.device_count()
    pdm_str = f"{pdims[0]}x{pdims[1]}"
    field_folder = f"{output_path}/final_field/jaxpm/{nb_gpus}/{mesh_size}_{int(box_size[0])}/{pdm_str}/{solver_choice}/halo_{halo_size}"
    os.makedirs(field_folder, exist_ok=True)
    with open(f'{field_folder}/jaxpm.log', 'w') as f:
        f.write(f"Args: {args}\n")
        f.write(f"JIT time: {chrono_fun.jit_time:.4f} ms\n")
        for i, time in enumerate(chrono_fun.times):
            f.write(f"Time {i}: {time:.4f} ms\n")
        f.write(f"Stats: {stats}\n")
    if args.save_fields:
        np.save(f'{field_folder}/final_field_0_{rank}.npy',
                final_field.addressable_data(0))

    print(f"Finished! ")
    print(f"Stats {stats}")
    print(f"Saving to {output_path}/jax_pm_benchmark.csv")
    print(f"Saving field and logs in {field_folder}")
