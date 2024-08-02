import os

# Change JAX GPU memory preallocation fraction
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM,
    boltzmann, linear_power, growth,
    white_noise, linear_modes,
    lpt, nbody, scatter
)
from pmwd.pm_util import fftinv
from pmwd.spec_util import powspec
from pmwd.vis_util import simshow
from hpc_plotter.timer import Timer

# Simulation configuration
def run_pmwd_simulation(ptcl_grid_shape, ptcl_spacing, solver ,  iterations):

    @jax.jit
    def simulate(omega_m, sigma8):
        
        
        conf = Configuration(ptcl_spacing, ptcl_grid_shape=ptcl_grid_shape, mesh_shape=1,lpt_order=1,a_nbody_maxstep=1/91)
        print(conf)
        print(f'Simulating {conf.ptcl_num} particles with a {conf.mesh_shape} mesh for {conf.a_nbody_num} time steps.')

        cosmo = Cosmology(conf, A_s_1e9=2.0, n_s=0.96, Omega_m=omega_m, Omega_b=sigma8, h=0.7)
        print(cosmo)

        # Boltzmann calculation
        cosmo = boltzmann(cosmo, conf)
        print("Boltzmann calculation completed.")

        # Generate white noise field and scale with the linear power spectrum
        seed = 0
        modes = white_noise(seed, conf)
        modes = linear_modes(modes, cosmo, conf)
        print("Linear modes generated.")

        # Solve LPT at some early time
        ptcl, obsvbl = lpt(modes, cosmo, conf)
        print("LPT solved.")
        
        if solver == "lfm":
          # N-body time integration from LPT initial conditions
          ptcl, obsvbl = jax.block_until_ready(nbody(ptcl, obsvbl, cosmo, conf))
          print("N-body time integration completed.")

        # Scatter particles to mesh to get the density field
        dens = scatter(ptcl, conf)
        return dens
    
    chrono_timer = Timer()
    final_field = chrono_timer.chrono_jit(simulate, 0.3, 0.05)
    
    for _ in range(iterations):
        final_field = chrono_timer.chrono_fun(simulate, 0.3, 0.05)

    return final_field , chrono_timer
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PMWD Simulation')
    parser.add_argument('-m', '--mesh_size', type=int, help='Mesh size', required=True)
    parser.add_argument('-b', '--box_size', type=float, help='Box size', required=True)
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations', default=10)
    parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
    parser.add_argument('-f', '--save_fields', action='store_true', help='Save fields')
    parser.add_argument('-s', '--solver', type=str, help='Solver', choices=["lfm" , "lpt"])
    parser.add_argument('-pr',
                          '--precision',
                          type=str,
                          help='Precision',
                          choices=["float32", "float64"],)


    args = parser.parse_args()
    
    mesh_shape = [args.mesh_size] * 3
    ptcl_spacing = args.box_size /args.mesh_size 
    iterations = args.iterations
    solver = args.solver
    output_path = args.output_path
    if args.precision == "float32":
        jax.config.update("jax_enable_x64", False)
    elif args.precision == "float64":  
        jax.config.update("jax_enable_x64", True)


    os.makedirs(output_path, exist_ok=True)
    
    final_field , chrono_fun = run_pmwd_simulation(mesh_shape, ptcl_spacing, solver, iterations)
    print("PMWD simulation completed.")


    metadata = {
            'rank': 0,
            'function_name': f'PMWD-{solver}',
            'precision': args.precision,
            'x': str(mesh_shape[0]),
            'y': str(mesh_shape[0]),
            'z': str(mesh_shape[0]),
            'px': "1",
            'py': "1",
            'backend': 'NCCL',
            'nodes': "1"
        }
    chrono_fun.print_to_csv(f"{output_path}/pmwd.csv", **metadata)
    field_folder = f"{output_path}/final_field/pmwd/1/{args.mesh_size}_{int(args.box_size)}/1x1/{args.solver}/halo_0"
    os.makedirs(field_folder, exist_ok=True)
    with open(f"{field_folder}/pmwd.log", "w") as f:
        f.write(f"PMWD simulation completed.\n")
        f.write(f"Args : {args}\n")
        f.write(f"JIT time: {chrono_fun.jit_time:.4f} ms\n")
        for i , time in enumerate(chrono_fun.times):
          f.write(f"Time {i}: {time:.4f} ms\n")
        if args.save_fields:
            np.save(f"{field_folder}/final_field_0_0.npy", final_field)
            print("Fields saved.")
    
    
    print(f"saving to {output_path}/pmwd.csv")
    print(f"saving field and logs to {field_folder}/pmwd.log")


