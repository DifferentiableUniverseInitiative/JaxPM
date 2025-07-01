#!/usr/bin/env python3
"""
Unified Power Spectrum Analysis and Theory Comparison

This script combines:
1. Angular power spectrum computation from convergence maps (from analyze_power_spectrum.py)
2. Theoretical comparison using jax-cosmo (from compare_with_theory.py)

Provides a complete pipeline for analyzing convergence maps and comparing with theory.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
from functools import partial

class PowerSpectrumAnalyzer:
    """
    Unified power spectrum analysis class.
    """
    
    def __init__(self, results_dir="experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up cosmology for theoretical calculations
        self.cosmo = self._setup_cosmology()
        
    def _setup_cosmology(self):
        """Set up jax-cosmo cosmology matching simulation parameters."""
        Planck18 = partial(
            jc.Cosmology,
            # Omega_m = 0.3111
            Omega_c=0.2607,
            Omega_b=0.0490,
            Omega_k=0.0,
            h=0.6766,
            n_s=0.9665,
            sigma8=0.8102,
            w0=-1.0,
            wa=0.0,
        )

        return Planck18()

    def compute_angular_power_spectrum(self, kappa_map, lmax=None):
        """
        Compute angular power spectrum C(ℓ) from convergence map using anafast directly.
        
        Parameters:
        -----------
        kappa_map : array
            HEALPix convergence map
        lmax : int
            Maximum multipole (default: 3*nside-1)
            
        Returns:
        --------
        ell : array
            Multipole values
        cl : array
            C(ℓ) values (raw from anafast)
        """
        print(f"Computing power spectrum for map with {len(kappa_map)} pixels")
        nside = hp.npix2nside(len(kappa_map))
        print(f"NSIDE = {nside}")
        
        # Compute power spectrum using HEALPix anafast
        if lmax is None:
            lmax = 3 * nside - 1
        
        print(f"Computing C(ℓ) up to ℓ_max = {lmax}")
        cl = hp.anafast(kappa_map, use_weights=True, lmax=lmax)
        ell = np.arange(len(cl))
        
        print(f"Power spectrum computed: {len(ell)} multipoles")
        print(f"ℓ range: 0 to {ell[-1]}")
        print(f"C(ℓ) range: {cl.min():.2e} to {cl.max():.2e}")
        
        return ell, cl
    
    def compute_theoretical_lensing_cl(self, z_source, ell_grid):
        """
        Compute theoretical lensing convergence C(ℓ) using jax-cosmo.
        
        Parameters:
        -----------
        z_source : float
            Source redshift
        ell_grid : array
            Multipole values
            
        Returns:
        --------
        cl_theory : array
            Theoretical C(ℓ) values
        """
        print(f"Computing theoretical lensing C(ℓ) for z_source = {z_source}")
        
        # Create lensing tracer
        tracer = jc.probes.WeakLensing(
            [jc.redshift.delta_nz(z_source)],
            sigma_e=0.0  # No shape noise for theoretical calculation
        )
        
        cl_theory = jc.angular_cl.angular_cl(
            cosmo=self.cosmo,
            ell=ell_grid,
            probes=[tracer] 
        )
        
        cl_theory = cl_theory[0]
        
        print(f"Theoretical C(ℓ) computed for {len(ell_grid)} multipoles")
        print(f"Range: {float(jnp.min(cl_theory)):.2e} to {float(jnp.max(cl_theory)):.2e}")
        
        return np.array(cl_theory)
    
    def plot_power_spectrum_comparison(self, ell_sim, cl_sim, ell_theory, cl_theory, 
                                     z_source=1.0, map_name="", save_path=None):
        """
        Create comprehensive comparison plot between theory and simulation.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Skip ell=0,1 for plotting (monopole/dipole)
        valid_sim = ell_sim >= 2
        valid_theory = ell_theory >= 2
        
        ell_sim_plot = ell_sim[valid_sim]
        cl_sim_plot = cl_sim[valid_sim]
        ell_theory_plot = ell_theory[valid_theory]  
        cl_theory_plot = cl_theory[valid_theory]
        
        # Main comparison plot - raw C(ℓ) values
        ax1.loglog(ell_sim_plot, cl_sim_plot, 'o-', label='Simulation (anafast)', color='blue', 
                   markersize=2, alpha=0.8, linewidth=1)
        ax1.loglog(ell_theory_plot, cl_theory_plot, '--', label='Theory (jax-cosmo)', 
                   color='red', linewidth=2)
        
        ax1.set_xlabel(r'Multipole $\ell$')
        ax1.set_ylabel(r'$C(\ell)$')
        ax1.set_title(f'Lensing Power Spectrum Comparison\n{map_name} (z_source = {z_source})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([2, min(ell_sim_plot[-1], 500)])
        
        # Ratio plot
        cl_theory_interp = np.interp(ell_sim_plot, ell_theory_plot, cl_theory_plot)
        ratio = cl_sim_plot / cl_theory_interp
        
        ax2.semilogx(ell_sim_plot, ratio, 'o-', color='green', markersize=2, linewidth=1)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(ell_sim_plot, 0.5, 2.0, alpha=0.2, color='gray', 
                        label='Factor of 2 range')
        
        ax2.set_xlabel(r'Multipole $\ell$')
        ax2.set_ylabel('Simulation / Theory')
        ax2.set_title('Ratio: Simulation / Theory')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([2, min(ell_sim_plot[-1], 500)])
        ax2.set_ylim([0.1, 10])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        valid = np.isfinite(ratio)
        mean_ratio = np.mean(ratio[valid])
        median_ratio = np.median(ratio[valid])
        std_ratio = np.std(ratio[valid])
        
        print(f"\nComparison statistics:")
        print(f"  Mean ratio (sim/theory): {mean_ratio:.3f}")
        print(f"  Median ratio: {median_ratio:.3f}")
        print(f"  Standard deviation: {std_ratio:.3f}")
        print(f"  Factor of 2 agreement: {np.sum((ratio > 0.5) & (ratio < 2.0))/len(ratio)*100:.1f}%")
        
        return fig, mean_ratio, median_ratio, std_ratio
    
    def analyze_convergence_map(self, map_file, z_source=1.0):
        """
        Complete analysis pipeline for a convergence map.
        
        Parameters:
        -----------
        map_file : str or Path
            Path to convergence map file
        z_source : float
            Source redshift
            
        Returns:
        --------
        results : dict
            Dictionary containing all analysis results
        """
        map_file = Path(map_file)
        print(f"\n{'='*60}")
        print(f"=== Analyzing: {map_file.name} ===")
        print(f"{'='*60}")
        
        # Load convergence map
        print(f"Loading convergence map: {map_file}")
        kappa_map = np.load(map_file)
        
        print(f"Map statistics:")
        print(f"  Shape: {kappa_map.shape}")
        print(f"  Min/Max: {kappa_map.min():.6f} / {kappa_map.max():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(kappa_map**2)):.6f}")
        
        # Compute simulation power spectrum
        print(f"\n--- Computing Simulation Power Spectrum ---")
        ell_sim, cl_sim = self.compute_angular_power_spectrum(kappa_map)
        
        # Compute theoretical power spectrum
        print(f"\n--- Computing Theoretical Power Spectrum ---")
        ell_theory = np.logspace(np.log10(2), np.log10(min(ell_sim[-1], 500)), 100)
        cl_theory = self.compute_theoretical_lensing_cl(z_source, ell_theory)
        
        # Create output filenames
        base_name = map_file.stem.replace('_physical', '')
        ps_data_file = self.results_dir / f"{base_name}_power_spectrum.txt"
        comparison_plot = self.results_dir / f"{base_name}_theory_comparison.png"
        
        # Save power spectrum data
        ps_data = np.column_stack([ell_sim, cl_sim])
        np.savetxt(ps_data_file, ps_data, 
                   header="ell  C(ell)\nSimulation power spectrum from anafast")
        print(f"Power spectrum data saved to {ps_data_file}")
        
        # Create comparison plot
        print(f"\n--- Creating Theory Comparison ---")
        fig, mean_ratio, median_ratio, std_ratio = self.plot_power_spectrum_comparison(
            ell_sim, cl_sim, ell_theory, cl_theory, 
            z_source=z_source, map_name=map_file.stem, save_path=comparison_plot)
        
        # Print key values for comparison
        print(f"\nKey power spectrum values:")
        idx_10 = np.argmin(np.abs(ell_sim - 10))
        idx_100 = np.argmin(np.abs(ell_sim - 100))
        
        if idx_10 < len(cl_sim):
            print(f"  C(ℓ=10) ≈ {cl_sim[idx_10]:.2e}")
        if idx_100 < len(cl_sim):
            print(f"  C(ℓ=100) ≈ {cl_sim[idx_100]:.2e}")
        
        # Compile results
        results = {
            'map_file': map_file,
            'z_source': z_source,
            'map_stats': {
                'shape': kappa_map.shape,
                'min': kappa_map.min(),
                'max': kappa_map.max(),
                'rms': np.sqrt(np.mean(kappa_map**2))
            },
            'power_spectrum': {
                'ell_sim': ell_sim,
                'cl_sim': cl_sim,
                'ell_theory': ell_theory,
                'cl_theory': cl_theory
            },
            'comparison_stats': {
                'mean_ratio': mean_ratio,
                'median_ratio': median_ratio,
                'std_ratio': std_ratio
            },
            'output_files': {
                'power_spectrum': ps_data_file,
                'comparison_plot': comparison_plot
            }
        }
        
        return results
    
    def create_combined_comparison_plot(self, all_results, save_path=None):
        """
        Create a single figure showing all convergence map comparisons.
        
        Parameters:
        -----------
        all_results : list
            List of results dictionaries from analyze_convergence_map
        save_path : str, optional
            Path to save the combined plot
        """
        n_maps = len(all_results)
        if n_maps == 0:
            print("No results to plot!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, n_maps, figsize=(4*n_maps, 8))
        if n_maps == 1:
            axes = axes.reshape(2, 1)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, results in enumerate(all_results):
            ps = results['power_spectrum']
            ell_sim = ps['ell_sim']
            cl_sim = ps['cl_sim']
            ell_theory = ps['ell_theory']
            cl_theory = ps['cl_theory']
            
            # Skip ell=0,1 for plotting
            valid_sim = ell_sim >= 2
            valid_theory = ell_theory >= 2
            
            ell_sim_plot = ell_sim[valid_sim]
            cl_sim_plot = cl_sim[valid_sim]
            ell_theory_plot = ell_theory[valid_theory]
            cl_theory_plot = cl_theory[valid_theory]
            
            color = colors[i % len(colors)]
            map_name = results['map_file'].stem.replace('_physical', '')
            
            # Power spectrum comparison
            ax1 = axes[0, i]
            ax1.loglog(ell_sim_plot, cl_sim_plot, 'o-', label='Simulation', 
                      color=color, markersize=2, alpha=0.8, linewidth=1)
            ax1.loglog(ell_theory_plot, cl_theory_plot, '--', label='Theory', 
                      color='black', linewidth=2, alpha=0.7)
            
            ax1.set_xlabel(r'Multipole $\ell$')
            ax1.set_ylabel(r'$C(\ell)$')
            ax1.set_title(f'{map_name}')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            ax1.set_xlim([2, min(ell_sim_plot[-1], 50)])
            
            # Ratio plot
            ax2 = axes[1, i]
            cl_theory_interp = np.interp(ell_sim_plot, ell_theory_plot, cl_theory_plot)
            ratio = cl_sim_plot / cl_theory_interp
            
            ax2.semilogx(ell_sim_plot, ratio, 'o-', color=color, markersize=2, linewidth=1)
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax2.fill_between(ell_sim_plot, 0.5, 2.0, alpha=0.2, color='gray')
            
            ax2.set_xlabel(r'Multipole $\ell$')
            ax2.set_ylabel('Sim/Theory')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([2, min(ell_sim_plot[-1], 50)])
            ax2.set_ylim([0.01, 100])
            
            # Add ratio statistics as text
            stats = results['comparison_stats']
            ax2.text(0.05, 0.95, f"Mean: {stats['mean_ratio']:.1f}\nRMS: {results['map_stats']['rms']:.1e}", 
                    transform=ax2.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Convergence Map Power Spectrum Comparisons', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Combined comparison plot saved to {save_path}")
        
        plt.show()
        return fig

    def analyze_all_maps(self):
        """
        Analyze all convergence maps found in the results directory.
        
        Returns:
        --------
        all_results : list
            List of results dictionaries for each map
        """
        print("=== Power Spectrum Analysis Pipeline ===")
        
        # Look for convergence map files - get unique maps (avoid duplicates)
        map_files = list(self.results_dir.glob("convergence_map_*_physical.npy"))
        
        # Remove duplicates by keeping only the first 4 maps (0,1,2,3)
        unique_maps = []
        seen_indices = set()
        for map_file in map_files:
            # Extract map index from filename
            parts = map_file.stem.split('_')
            if len(parts) >= 3:
                try:
                    idx = int(parts[2])
                    if idx not in seen_indices:
                        unique_maps.append(map_file)
                        seen_indices.add(idx)
                except ValueError:
                    continue
        
        # Sort by index
        unique_maps.sort(key=lambda x: int(x.stem.split('_')[2]))
        
        if not unique_maps:
            print("No convergence map files found!")
            print("Run spherical_ray_trace.py first to generate convergence maps.")
            return []
        
        all_results = []
        
        # Analyze each map
        for map_file in unique_maps:
            # Extract z_source from filename
            if "_z1.0_" in map_file.name:
                z_source = 1.0
            elif "_z2.0_" in map_file.name:
                z_source = 2.0
            else:
                z_source = 1.0  # Default
            
            try:
                results = self.analyze_convergence_map(map_file, z_source=z_source)
                all_results.append(results)
                print(f"✓ Analysis completed for {map_file.name}")
                
            except Exception as e:
                print(f"✗ Error analyzing {map_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create combined comparison plot
        if all_results:
            print(f"\n--- Creating Combined Comparison Plot ---")
            combined_plot_path = self.results_dir / "all_maps_comparison.png"
            self.create_combined_comparison_plot(all_results, save_path=combined_plot_path)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"=== Analysis Summary ===")
        print(f"{'='*60}")
        print(f"Successfully analyzed {len(all_results)} convergence maps")
        
        for results in all_results:
            stats = results['comparison_stats']
            print(f"\n{results['map_file'].name}:")
            print(f"  z_source = {results['z_source']}")
            print(f"  Map RMS = {results['map_stats']['rms']:.6f}")
            print(f"  Theory ratio = {stats['mean_ratio']:.3f} ± {stats['std_ratio']:.3f}")
        
        return all_results

def main():
    """
    Main function to run complete power spectrum analysis.
    """
    analyzer = PowerSpectrumAnalyzer()
    results = analyzer.analyze_all_maps()
    
    if results:
        print(f"\nAnalysis complete! Generated {len(results)} comparisons.")
        print("Check experiments/results/ for output files.")
    else:
        print("No convergence maps found to analyze.")

if __name__ == "__main__":
    main()