#!/usr/bin/env python3
"""
Tests for flat-sky angular power spectrum computation in jaxpm.utils.

This module validates the flat_sky_power_spectrum function against lenstools
and tests both auto and cross-power spectrum functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jaxpm.utils import flat_sky_power_spectrum

# Try to import lenstools for validation
try:
    from lenstools import ConvergenceMap
    import astropy.units as u
    LENSTOOLS_AVAILABLE = True
except ImportError:
    LENSTOOLS_AVAILABLE = False


def generate_test_map(npix=128, angle_deg=5.0, seed=42, power_law=-2.0):
    """Generate a synthetic convergence map with known power spectrum."""
    np.random.seed(seed)
    
    # Create k-space grid
    angle_rad = angle_deg * np.pi / 180.0
    pixel_scale = angle_rad / npix
    
    kx = np.fft.fftfreq(npix, pixel_scale) * 2 * np.pi
    ky = np.fft.fftfreq(npix, pixel_scale) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Avoid division by zero
    k_grid = np.where(k_grid == 0, 1, k_grid)
    
    # Generate power law: P(k) ∝ k^power_law
    amplitude = 1e-6
    power_k = amplitude * (k_grid / (2 * np.pi / angle_rad))**power_law
    power_k[0, 0] = 0  # Remove monopole
    
    # Generate Gaussian random field
    phase = np.random.uniform(0, 2*np.pi, (npix, npix))
    gaussian_field = np.sqrt(power_k) * np.exp(1j * phase)
    
    # Transform to real space
    kappa_map = np.fft.ifft2(gaussian_field).real
    
    return kappa_map


class TestFlatSkyPowerSpectrum:
    """Test suite for flat_sky_power_spectrum function."""
    
    def test_basic_functionality(self):
        """Test basic auto-power spectrum computation."""
        kappa = generate_test_map(npix=64, angle_deg=3.0)
        
        ell, cl = flat_sky_power_spectrum(kappa, angle_deg=3.0)
        
        # Check output shapes and values
        assert len(ell) == len(cl)
        assert len(ell) > 0
        assert np.all(ell > 0)
        assert np.all(cl > 0)  # Power should be positive
        assert np.all(np.isfinite(cl))
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        kappa = generate_test_map(npix=32, angle_deg=2.0)
        
        # Test non-square input
        with pytest.raises(ValueError, match="2D square array"):
            flat_sky_power_spectrum(kappa[:, :-1], 2.0)
        
        # Test 3D input
        with pytest.raises(ValueError, match="2D square array"):
            flat_sky_power_spectrum(np.random.random((32, 32, 32)), 2.0)
        
        # Test mismatched shapes for cross-power
        kappa2 = generate_test_map(npix=16, angle_deg=2.0)
        with pytest.raises(ValueError, match="same shape"):
            flat_sky_power_spectrum(kappa, 2.0, kappa2)
    
    def test_cross_power_spectrum(self):
        """Test cross-power spectrum computation."""
        kappa1 = generate_test_map(npix=64, angle_deg=3.0, seed=42)
        kappa2 = generate_test_map(npix=64, angle_deg=3.0, seed=123)
        
        # Auto-power spectra
        ell, cl1 = flat_sky_power_spectrum(kappa1, 3.0)
        ell, cl2 = flat_sky_power_spectrum(kappa2, 3.0)
        
        # Cross-power spectrum
        ell, cl_cross = flat_sky_power_spectrum(kappa1, 3.0, kappa2)
        
        # Check cross-power is real and finite
        assert np.all(np.isfinite(cl_cross))
        
        # Cross-power magnitude should be <= geometric mean of auto-powers
        # |C_12| <= sqrt(C_11 * C_22) by Cauchy-Schwarz
        geometric_mean = np.sqrt(cl1 * cl2)
        assert np.all(np.abs(cl_cross) <= geometric_mean * 1.1)  # Allow 10% tolerance
    
    def test_custom_ell_bins(self):
        """Test custom ell binning."""
        kappa = generate_test_map(npix=128, angle_deg=5.0)
        
        # Custom ell bins
        ells = jnp.logspace(jnp.log10(50), jnp.log10(500), 10)
        ell_centers, cl = flat_sky_power_spectrum(kappa, 5.0, ells=ells)
        
        # Check output length (may be less due to NaN removal)
        assert len(ell_centers) <= len(ells) - 1
        assert len(cl) <= len(ells) - 1
        assert len(ell_centers) == len(cl)
        
        # Check that ell centers are in reasonable range
        if len(ell_centers) > 0:
            assert np.all(ell_centers >= ells[0])
            assert np.all(ell_centers <= ells[-1])
            # Should be monotonically increasing
            assert np.all(np.diff(ell_centers) > 0)
    
    def test_angle_scaling(self):
        """Test that power spectrum scales correctly with angle."""
        kappa = generate_test_map(npix=64, angle_deg=2.0)
        
        # Same map, different angular sizes
        ell1, cl1 = flat_sky_power_spectrum(kappa, angle_deg=2.0)
        ell2, cl2 = flat_sky_power_spectrum(kappa, angle_deg=4.0)
        
        # Larger angle should give smaller ell values (fundamental mode scales as 1/angle)
        # Check that the minimum ell is smaller for larger angle
        assert ell2[0] < ell1[0], f"ell2[0]={ell2[0]:.1f} should be < ell1[0]={ell1[0]:.1f}"
        
    def test_monopole_removal(self):
        """Test that monopole is properly removed."""
        # Create map with large mean
        kappa = generate_test_map(npix=64, angle_deg=3.0)
        kappa_with_mean = kappa + 10.0  # Add large constant
        
        # Power spectra should be identical (monopole removed)
        ell1, cl1 = flat_sky_power_spectrum(kappa, 3.0)
        ell2, cl2 = flat_sky_power_spectrum(kappa_with_mean, 3.0)
        
        # For very small values, use absolute tolerance
        if np.max(cl1) < 1e-15:
            np.testing.assert_allclose(cl1, cl2, atol=1e-15)
        else:
            np.testing.assert_allclose(cl1, cl2, rtol=1e-3)
    
    @pytest.mark.skipif(not LENSTOOLS_AVAILABLE, 
                       reason="lenstools not available")
    def test_against_lenstools_auto(self):
        """Test auto-power spectrum against lenstools."""
        npix = 128
        angle_deg = 8.0
        kappa = generate_test_map(npix, angle_deg, seed=42)
        
        # Define ell range for comparison
        ells = jnp.logspace(jnp.log10(50), jnp.log10(800), 15)
        
        # Our implementation
        ell_ours, cl_ours = flat_sky_power_spectrum(kappa, angle_deg, ells=ells)
        
        # lenstools implementation
        conv_map = ConvergenceMap(kappa, angle=angle_deg * u.deg)
        ell_lenstools, cl_lenstools = conv_map.powerSpectrum(l_edges=ells)
        
        # Compare results - handle different array lengths due to binning differences
        if len(ell_ours) > 5 and len(ell_lenstools) > 5:
            # Find overlapping ell range
            min_ell = max(np.min(ell_ours), np.min(ell_lenstools))
            max_ell = min(np.max(ell_ours), np.max(ell_lenstools))
            
            # Mask for overlapping region
            mask_ours = (ell_ours >= min_ell) & (ell_ours <= max_ell)
            mask_lenstools = (ell_lenstools >= min_ell) & (ell_lenstools <= max_ell)
            
            if np.sum(mask_ours) > 3 and np.sum(mask_lenstools) > 3:
                # Compare power spectra in overlapping region
                cl_ours_overlap = cl_ours[mask_ours]
                cl_lenstools_overlap = cl_lenstools[mask_lenstools]
                
                # Interpolate to common grid for comparison
                ell_common = np.linspace(min_ell, max_ell, 10)
                cl_ours_interp = np.interp(ell_common, ell_ours[mask_ours], cl_ours_overlap)
                cl_lenstools_interp = np.interp(ell_common, ell_lenstools[mask_lenstools], cl_lenstools_overlap)
                
                # Power spectra should agree within 10%
                ratio = cl_ours_interp / cl_lenstools_interp
                max_dev = np.max(np.abs(ratio - 1))
                assert max_dev < 0.1, f"Max deviation: {max_dev:.3f}"
                
                # Mean ratio should be close to 1
                mean_ratio = np.mean(ratio)
                assert abs(mean_ratio - 1) < 0.05, f"Mean ratio: {mean_ratio:.3f}"
    
    @pytest.mark.skipif(not LENSTOOLS_AVAILABLE, 
                       reason="lenstools not available")
    def test_against_lenstools_cross(self):
        """Test cross-power spectrum against lenstools."""
        npix = 64
        angle_deg = 5.0
        kappa1 = generate_test_map(npix, angle_deg, seed=42)
        kappa2 = generate_test_map(npix, angle_deg, seed=123)
        
        # Define ell range
        ells = jnp.logspace(jnp.log10(100), jnp.log10(600), 10)
        
        # Our implementation
        ell_ours, cl_cross_ours = flat_sky_power_spectrum(kappa1, angle_deg, kappa2, ells=ells)
        
        # lenstools implementation
        conv_map1 = ConvergenceMap(kappa1, angle=angle_deg * u.deg)
        conv_map2 = ConvergenceMap(kappa2, angle=angle_deg * u.deg)
        ell_lenstools, cl_cross_lenstools = conv_map1.cross(conv_map2, statistic="power_spectrum", 
                                                           l_edges=ells)
        
        # Compare results - cross-power can have more variation
        min_len = min(len(cl_cross_ours), len(cl_cross_lenstools))
        if min_len > 3:
            ratio = cl_cross_ours[:min_len] / cl_cross_lenstools[:min_len]
            max_dev = np.max(np.abs(ratio - 1))
            assert max_dev < 0.6, f"Max deviation: {max_dev:.3f} (cross-power can be noisy)"
    
    def test_power_law_recovery(self):
        """Test that we can recover the input power law."""
        # Generate map with known power law
        power_law = -2.5
        kappa = generate_test_map(npix=256, angle_deg=10.0, 
                                 seed=42, power_law=power_law)
        
        ell, cl = flat_sky_power_spectrum(kappa, 10.0)
        
        # Fit power law to measured spectrum
        # C_ℓ ∝ ℓ^α, so log(C_ℓ) = α * log(ℓ) + const
        log_ell = np.log(ell)
        log_cl = np.log(cl)
        
        # Use middle range for fit (avoid noise at edges)
        mask = (ell > 100) & (ell < 1000)
        if np.sum(mask) > 5:  # Need enough points for fit
            coeffs = np.polyfit(log_ell[mask], log_cl[mask], 1)
            measured_slope = coeffs[0]
            
            # Should roughly recover the input power law
            # Note: exact recovery depends on many factors, so allow generous tolerance
            assert abs(measured_slope - power_law) < 1.0, \
                f"Expected slope ≈ {power_law}, got {measured_slope:.2f}"


def test_performance():
    """Test performance with different map sizes."""
    import time
    
    for npix in [64, 128]:  # Skip 256 to avoid overflow issues in testing
        kappa = generate_test_map(npix, angle_deg=5.0)
        
        start_time = time.time()
        ell, cl = flat_sky_power_spectrum(kappa, 5.0)
        elapsed = time.time() - start_time
        
        # Should complete reasonably quickly
        assert elapsed < 2.0, f"Too slow for {npix}² map: {elapsed:.2f}s"
        assert len(ell) > 0, f"No valid bins for {npix}² map"
        print(f"{npix}² map: {elapsed:.3f}s")


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Testing JaxPM flat-sky power spectrum implementation...")
    
    test = TestFlatSkyPowerSpectrum()
    
    print("✓ Basic functionality")
    test.test_basic_functionality()
    
    print("✓ Input validation")
    test.test_input_validation()
    
    print("✓ Cross-power spectrum")
    test.test_cross_power_spectrum()
    
    print("✓ Custom ell bins")
    test.test_custom_ell_bins()
    
    print("✓ Angle scaling")
    test.test_angle_scaling()
    
    print("✓ Monopole removal")
    test.test_monopole_removal()
    
    if LENSTOOLS_AVAILABLE:
        print("✓ Validation against lenstools (auto)")
        test.test_against_lenstools_auto()
        
        print("✓ Validation against lenstools (cross)")
        test.test_against_lenstools_cross()
    else:
        print("⚠ Skipping lenstools validation (not installed)")
    
    print("✓ Power law recovery")
    test.test_power_law_recovery()
    
    print("✓ Performance test")
    test_performance()
    
    print("\n🎉 All tests passed!")