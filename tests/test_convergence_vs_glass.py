import os
from functools import partial

import camb
# Glass and CAMB imports
import glass
import healpy as hp
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest
from cosmology.compat.camb import Cosmology
from diffrax import (ConstantStepSize, ODETerm, SaveAt, SemiImplicitEuler,
                     diffeqsolve)
from numpy.testing import assert_allclose

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.lensing import convergence_Born, spherical_density_fn
from jaxpm.ode import symplectic_ode
# JaxPM imports
from jaxpm.pm import linear_field, lpt


@pytest.fixture(scope="module")
def cosmo_fixture():
    """Cosmology fixture for convergence tests"""
    Planck18 = partial(
        jc.Cosmology,
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


@pytest.fixture(scope="module")
def convergence_test_config():
    """Configuration for convergence vs Glass tests"""
    return {
        'mesh_size': 128,
        'z_sources': [0.3, 0.5, 0.8],
        'nside': 256,
        'n_shells': 40,
        'seed': 42,
        't0': 0.1,
        't1': 1.0,
        'dt0': 0.05,
        'low_ell_cutoff': 2,
        'observer_position_in_box': [0.5, 0.5, 0.5]
    }


@pytest.fixture(scope="module")
def nbody_density_planes(cosmo_fixture, convergence_test_config):
    """Run JaxPM N-body simulation and extract spherical density planes for multi-redshift lensing"""
    cosmo = cosmo_fixture
    config = convergence_test_config

    # Compute simulation geometry
    z_max = max(config['z_sources'])
    r_comoving = jc.background.radial_comoving_distance(
        cosmo, jc.utils.z2a(z_max)).squeeze()
    cosmo._workspace = {}

    factors = jnp.clip(jnp.array(config['observer_position_in_box']), 0.0, 0.5)
    factors = 1.0 + 2.0 * jnp.minimum(factors, 1.0 - factors)

    box_size = tuple(map(float, factors * r_comoving))
    observer_position = jnp.array(
        config['observer_position_in_box']) * jnp.array(box_size)
    d_R = (r_comoving / config['n_shells']).squeeze()
    mesh_shape = (config['mesh_size'], config['mesh_size'],
                  config['mesh_size'])

    # Create initial conditions
    k = jnp.logspace(-3, 1, 256)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding=None)
    cosmo._workspace = {}

    initial_conditions = linear_field(mesh_shape,
                                      box_size,
                                      pk_fn,
                                      seed=jax.random.PRNGKey(config['seed']),
                                      sharding=None)

    # LPT displacements
    dx, p, f = lpt(cosmo,
                   initial_conditions,
                   particles=None,
                   a=config['t0'],
                   order=1,
                   sharding=None,
                   halo_size=0)
    cosmo._workspace = {}
    # Setup time evolution
    drift, kick = symplectic_ode(mesh_shape,
                                 paint_absolute_pos=False,
                                 halo_size=0,
                                 sharding=None)
    ode_fn = ODETerm(kick), ODETerm(drift)
    solver = SemiImplicitEuler()

    # Define spherical shells
    n_lens = int((box_size[-1] - observer_position[-1]) / d_R)
    r = jnp.linspace(0.0, box_size[-1] - observer_position[-1], n_lens)
    r_center = 0.5 * (r[1:] + r[:-1])
    a_center = jc.background.a_of_chi(cosmo, r_center)
    cosmo._workspace = {}
    time_steps = a_center[::-1]  # Reverse order for time evolution

    saveat = SaveAt(
        ts=time_steps,
        fn=lambda t, y, args: spherical_density_fn(mesh_shape,
                                                   box_size,
                                                   config['nside'],
                                                   observer_position,
                                                   d_R,
                                                   sharding=None)
        (t, y[1], args),
    )

    # Run simulation
    y0 = (p, dx)
    res = diffeqsolve(
        ode_fn,
        solver,
        t0=config['t0'],
        t1=config['t1'],
        dt0=config['dt0'],
        y0=y0,
        args=cosmo,
        saveat=saveat,
        stepsize_controller=ConstantStepSize(),
    )

    density_planes = res.ys

    return {
        'density_planes': density_planes,
        'r_center': r_center,
        'a_center': a_center,
        'd_R': d_R,
        'box_size': box_size,
        'observer_position': observer_position,
        'mesh_shape': mesh_shape
    }


@pytest.fixture(scope="module")
def jaxpm_convergence_maps(nbody_density_planes, cosmo_fixture,
                           convergence_test_config):
    """Compute JaxPM Born convergence for multiple redshifts"""
    cosmo = cosmo_fixture
    config = convergence_test_config
    planes_data = nbody_density_planes

    # Reverse time ordering for convergence calculation
    lightcone = planes_data['density_planes'][::-1]

    # Compute convergence using JaxPM's Born approximation for multiple source redshifts
    convergence_jaxpm_multi = convergence_Born(cosmo, lightcone,
                                               planes_data['r_center'],
                                               planes_data['a_center'],
                                               jnp.array(config['z_sources']),
                                               planes_data['d_R'])

    return convergence_jaxpm_multi, config['z_sources']


@pytest.fixture(scope="module")
def glass_tophat_maps(nbody_density_planes, cosmo_fixture,
                      convergence_test_config):
    """Compute Glass convergence with top-hat windows"""
    cosmo = cosmo_fixture
    config = convergence_test_config
    planes_data = nbody_density_planes

    # Convert JaxPM density planes to Glass format
    lightcone = planes_data['density_planes'][::-1]
    density_planes_glass = []
    for plane in lightcone:
        plane_np = np.array(plane)
        mean_density = np.mean(plane_np)
        if mean_density > 0:
            delta = plane_np / mean_density - 1.0
        else:
            delta = np.zeros_like(plane_np)
        density_planes_glass.append(delta)

    # Setup Glass cosmology to match jax_cosmo parameters
    h = cosmo.h
    omega_m = cosmo.Omega_c + cosmo.Omega_b
    Oc = cosmo.Omega_c
    Ob = cosmo.Omega_b

    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)
    glass_cosmo = Cosmology(results)

    # Glass top-hat convergence calculation with multi-redshift support
    z_targets = np.array(config['z_sources'], dtype=float)
    sort_idx = np.argsort(z_targets)
    sorted_targets = z_targets[sort_idx]
    stored_maps = {}
    target_ptr = 0

    convergence_glass_tophat_calc = glass.MultiPlaneConvergence(glass_cosmo)
    r_edges = jnp.linspace(
        0.0,
        planes_data['box_size'][-1] - planes_data['observer_position'][-1],
        len(planes_data['r_center']) + 2)
    z_edges = np.array(jc.utils.a2z(jc.background.a_of_chi(cosmo, r_edges)))
    shells_tophat = glass.tophat_windows(z_edges)

    prev_zeff = None
    prev_kappa = None
    tol = 5e-3

    for i, win in enumerate(shells_tophat):
        if i >= len(density_planes_glass):
            break

        convergence_glass_tophat_calc.add_window(density_planes_glass[i], win)
        current_kappa = np.array(convergence_glass_tophat_calc.kappa,
                                 copy=True)
        current_zeff = win.zeff

        if prev_kappa is None:
            prev_kappa = np.zeros_like(current_kappa)
        if prev_zeff is None:
            prev_zeff = 0.0

        while target_ptr < len(
                sorted_targets
        ) and current_zeff + tol >= sorted_targets[target_ptr]:
            target_z = sorted_targets[target_ptr]
            choose_current = abs(current_zeff - target_z) <= abs(
                prev_zeff - target_z) if prev_zeff is not None else True
            stored_maps[
                target_z] = current_kappa if choose_current else prev_kappa
            target_ptr += 1

        prev_zeff = current_zeff
        prev_kappa = current_kappa

    if target_ptr < len(sorted_targets):
        if prev_kappa is None:
            raise ValueError("No GLASS top-hat convergence maps were computed")
        while target_ptr < len(sorted_targets):
            target_z = sorted_targets[target_ptr]
            stored_maps[target_z] = prev_kappa
            target_ptr += 1

    convergence_glass_tophat_multi = [stored_maps[z] for z in z_targets]
    convergence_glass_tophat_multi = np.array(convergence_glass_tophat_multi)

    return convergence_glass_tophat_multi, config['z_sources']


def compute_map_statistics(map1, map2):
    """Compute statistical metrics between two convergence maps"""
    # Flatten maps for correlation calculation
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()

    # Remove NaN and infinite values
    valid_mask = np.isfinite(map1_flat) & np.isfinite(map2_flat)
    map1_clean = map1_flat[valid_mask]
    map2_clean = map2_flat[valid_mask]

    # Compute metrics
    mse = np.mean((map1_clean - map2_clean)**2)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(map1_clean, map2_clean)[0, 1]

    return {'mse': mse, 'rmse': rmse, 'correlation': correlation}


def compute_power_spectrum(convergence_map, nside):
    """Compute angular power spectrum (Cl) from HEALPix convergence map"""
    # Ensure map is properly masked and finite
    convergence_clean = np.where(np.isfinite(convergence_map), convergence_map,
                                 0.0)

    # Compute power spectrum using HEALPix
    cl = hp.anafast(convergence_clean, lmax=2 * nside)
    ell = np.arange(len(cl))

    return ell, cl


def compute_cl_statistics(cl1, cl2, low_ell_cutoff=2):
    """Compute statistical metrics between two power spectra"""
    # Apply low-ell cutoff
    cl1_cut = cl1[low_ell_cutoff:]
    cl2_cut = cl2[low_ell_cutoff:]

    # Remove zeros and negative values for ratio calculation
    valid_mask = (cl1_cut > 0) & (
        cl2_cut > 0) & np.isfinite(cl1_cut) & np.isfinite(cl2_cut)
    cl1_valid = cl1_cut[valid_mask]
    cl2_valid = cl2_cut[valid_mask]

    if len(cl1_valid) == 0:
        return {
            'mse': np.inf,
            'rmse': np.inf,
            'mean_ratio': np.inf,
            'correlation': 0.0
        }

    # Compute metrics
    mse = np.mean((cl1_valid - cl2_valid)**2)
    rmse = np.sqrt(mse)
    mean_ratio = np.mean(cl1_valid / cl2_valid)
    correlation = np.corrcoef(cl1_valid, cl2_valid)[0, 1]

    return {
        'mse': mse,
        'rmse': rmse,
        'mean_ratio': mean_ratio,
        'correlation': correlation
    }


@pytest.mark.single_device
def test_convergence_maps_jaxpm_vs_glass_tophat(jaxpm_convergence_maps,
                                                glass_tophat_maps,
                                                convergence_test_config):
    """Test convergence maps: JaxPM vs Glass top-hat windows (target 10% ratio)"""
    convergence_jaxpm, z_sources_jaxpm = jaxpm_convergence_maps
    convergence_tophat, z_sources_tophat = glass_tophat_maps
    config = convergence_test_config

    # Ensure redshift arrays match
    assert np.array_equal(z_sources_jaxpm,
                          z_sources_tophat), "Redshift arrays must match"
    assert len(z_sources_jaxpm) == len(
        config['z_sources']), "Must have all requested redshifts"

    # Test each redshift
    for i, z in enumerate(z_sources_jaxpm):
        jaxpm_map = np.array(convergence_jaxpm[i])
        tophat_map = np.array(convergence_tophat[i])

        # Compute statistics
        stats = compute_map_statistics(jaxpm_map, tophat_map)

        # Compute ratio of map amplitudes
        jaxpm_rms = np.sqrt(np.mean(jaxpm_map**2))
        tophat_rms = np.sqrt(np.mean(tophat_map**2))
        amplitude_ratio = jaxpm_rms / tophat_rms if tophat_rms > 0 else np.inf

        # Print metrics for debugging
        print(
            f"z={z:.1f} - Map MSE: {stats['mse']:.6f}, RMSE: {stats['rmse']:.6f}, Correlation: {stats['correlation']:.4f}, Amplitude ratio: {amplitude_ratio:.3f}"
        )

        # Validate maps are reasonable
        assert np.isfinite(stats['mse']), f"MSE must be finite for z={z}"
        assert np.isfinite(
            stats['correlation']), f"Correlation must be finite for z={z}"
        assert np.isfinite(
            amplitude_ratio), f"Amplitude ratio must be finite for z={z}"

        # JaxPM/top-hat ratio should be around 1.1 (10% higher)
        assert 0.9 < amplitude_ratio < 1.1, f"Amplitude ratio outside expected range for z={z}: {amplitude_ratio}"

        # Correlation should be reasonable
        assert stats[
            'correlation'] > 0.7, f"Correlation too low for z={z}: {stats['correlation']}"


@pytest.mark.single_device
def test_power_spectrum_jaxpm_vs_glass_tophat(jaxpm_convergence_maps,
                                              glass_tophat_maps,
                                              convergence_test_config):
    """Test power spectra (Cl): JaxPM vs Glass top-hat windows (target 10% ratio)"""
    convergence_jaxpm, z_sources_jaxpm = jaxpm_convergence_maps
    convergence_tophat, z_sources_tophat = glass_tophat_maps
    config = convergence_test_config

    # Ensure redshift arrays match
    assert np.array_equal(z_sources_jaxpm,
                          z_sources_tophat), "Redshift arrays must match"

    # Test each redshift
    for i, z in enumerate(z_sources_jaxpm):
        jaxpm_map = np.array(convergence_jaxpm[i])
        tophat_map = np.array(convergence_tophat[i])

        # Compute power spectra
        ell_jaxpm, cl_jaxpm = compute_power_spectrum(jaxpm_map,
                                                     config['nside'])
        ell_tophat, cl_tophat = compute_power_spectrum(tophat_map,
                                                       config['nside'])

        # Ensure ell arrays match
        assert np.array_equal(ell_jaxpm, ell_tophat), "Ell arrays must match"

        # Compute Cl statistics
        cl_stats = compute_cl_statistics(cl_jaxpm, cl_tophat,
                                         config['low_ell_cutoff'])

        # Print metrics for debugging
        print(
            f"z={z:.1f} - Cl MSE: {cl_stats['mse']:.2e}, RMSE: {cl_stats['rmse']:.2e}, Mean ratio: {cl_stats['mean_ratio']:.3f}, Correlation: {cl_stats['correlation']:.4f}"
        )

        # Validate power spectra are reasonable
        assert np.isfinite(cl_stats['mse']), f"Cl MSE must be finite for z={z}"
        assert np.isfinite(
            cl_stats['mean_ratio']), f"Cl mean ratio must be finite for z={z}"
        assert np.isfinite(cl_stats['correlation']
                           ), f"Cl correlation must be finite for z={z}"

        # JaxPM/top-hat Cl ratio should be around 1.1 (10% higher)
        assert 0.9 < cl_stats[
            'mean_ratio'] < 1.1, f"Cl mean ratio outside expected range for z={z}: {cl_stats['mean_ratio']}"

        # Correlation should be high
        assert cl_stats[
            'correlation'] > 0.9, f"Cl correlation too low for z={z}: {cl_stats['correlation']}"
