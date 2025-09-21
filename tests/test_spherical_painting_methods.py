"""
Non-regression tests for JaxPM spherical painting methods.

Requirements from user:
- Use fixed acceptable resolution setup: mesh=256^3, NSIDE=256, paint_nside=512
- Generate LPT particles via fixture
- Fixture for theory C_ell and pixel-window corrected theory
- Fixture for NGP mass reference
- Single test: run 5 other methods, assert mass conservation vs NGP and
  autospectra accuracy vs theory with method-specific ℓ ranges and tolerances

Markers: single_device
"""

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import sys
from pathlib import Path
import numpy as np
import pytest
import healpy as hp
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.tree_util import register_pytree_node_class
from jax_cosmo.redshift import redshift_distribution

from jaxpm.pm import linear_field, pm_forces
from jaxpm.growth import growth_factor as jpm_growth_factor
from jaxpm.distributed import fft3d
from jaxpm.spherical import paint_particles_spherical
from jaxpm.pm import linear_field, lpt


# ----------------------
# Fixed configuration
# ----------------------
BOX_SIZE = (1000.0, 1000.0, 1000.0)
MESH_SHAPE = (256, 256, 256)
NSIDE = 256
PAINT_NSIDE = 512
LMAX = 3 * NSIDE

OBSERVER_POSITION = [500.0, 500.0, 500.0]
R_MIN = 150.0
R_MAX = 400.0

OMEGA_C = 0.25
SIGMA_8 = 0.8
RANDOM_SEED = 42


# ----------------------
# Fixtures
# ----------------------
@pytest.fixture(scope="session")
def positions_lpt():
    """Session fixture that generates LPT particle positions on a 256^3 mesh.

    Uses growth matching to shell-center as in the notebook/script.
    """
    key = jax.random.PRNGKey(RANDOM_SEED)

    # Create cosmology and power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=OMEGA_C, sigma8=SIGMA_8), k)
    pk_fn = lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Generate initial conditions
    initial_conditions = linear_field(MESH_SHAPE, BOX_SIZE, pk_fn, seed=key)

    # Create particle grid
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in MESH_SHAPE], indexing="ij"), axis=-1)

    # Apply LPT
    cosmo = jc.Planck15(Omega_c=OMEGA_C, sigma8=SIGMA_8)
    # Match growth to shell-center
    chi_center = 0.5 * (R_MIN + R_MAX)
    a_use = float(jc.background.a_of_chi(cosmo, chi_center).squeeze())
    dx, p, f = lpt(cosmo, initial_conditions, a=a_use, order=1)
    lpt_positions = particles + dx
    return lpt_positions


@pytest.fixture(scope="session")
def theory_curves():
    """Compute theory C_ell and pixel-window corrected theory for ℓ≥2."""

    @register_pytree_node_class
    class tophat_z(redshift_distribution):
        def pz_fn(self, z):
            zmin, zmax = self.params
            return jnp.where((z >= zmin) & (z <= zmax), 1.0, 0.0)

    cosmo = jc.Planck15(Omega_c=OMEGA_C, sigma8=SIGMA_8)

    # Convert shell distances to redshift bounds
    a_min = jc.background.a_of_chi(cosmo, R_MIN).squeeze()
    a_max = jc.background.a_of_chi(cosmo, R_MAX).squeeze()
    zmin = float(jc.utils.a2z(a_min))
    zmax = float(jc.utils.a2z(a_max))

    nz = tophat_z(zmin, zmax, gals_per_arcmin2=1.0)
    bias = jc.bias.constant_linear_bias(1.0)
    probe = jc.probes.NumberCounts([nz], bias, has_rsd=False)

    ell = jnp.arange(0, LMAX + 1)
    cl_th = jc.angular_cl.angular_cl(cosmo, ell, [probe], nonlinear_fn=jc.power.linear).squeeze()

    # Slice to ℓ>=2
    ell_th = np.array(ell[2:])
    cl_th = np.array(cl_th[2:])

    # Pixel window correction for autospectra
    w = hp.pixwin(NSIDE, lmax=LMAX)
    cl_th_pix = cl_th * (np.asarray(w[2:]) ** 2)

    return ell_th, cl_th, cl_th_pix


@pytest.fixture(scope="session")
def ngp_reference(positions_lpt):
    """NGP raw map and mass reference (total sum and mean)."""
    raw_map = paint_particles_spherical(
        positions_lpt,
        method="ngp",
        nside=NSIDE,
        observer_position=OBSERVER_POSITION,
        R_min=R_MIN,
        R_max=R_MAX,
        box_size=BOX_SIZE,
        mesh_shape=MESH_SHAPE,
    )
    raw_np = np.asarray(raw_map)
    return {
        "raw_map": raw_np,
        "sum": float(np.sum(raw_np)),
        "mean": float(np.mean(raw_np)),
    }


# ----------------------
# Single comprehensive test
# ----------------------

@pytest.mark.single_device
def test_mass_and_spectra_against_theory(positions_lpt, theory_curves, ngp_reference):
    ell_th, cl_th, cl_th_pix = theory_curves

    # Five methods under test (NGP is the reference for mass)
    methods = {
        "Bilinear": {"method": "bilinear"},
        "RBF Neighbors": {"method": "RBF_NEIGHBOR"},
        "NGP + Udgrade": {"method": "ngp", "paint_nside": PAINT_NSIDE, "udgrade_power": 0.0},
        "Bilinear + Udgrade": {"method": "bilinear", "paint_nside": PAINT_NSIDE, "udgrade_power": 0.0},
        "RBF + Udgrade": {"method": "RBF_NEIGHBOR", "paint_nside": PAINT_NSIDE, "udgrade_power": 0.0},
    }

    # Low-ℓ relaxed region and per-method ℓ_max windows (absolute ℓ for NSIDE=256)
    L_RELAX = 20
    L_WINDOWS = {
        # Method-specific main-band limits from the test plan
        "Bilinear": 260,
        "RBF Neighbors": 180,
        "NGP + Udgrade": 300,
        "Bilinear + Udgrade": 300,
        "RBF + Udgrade": 300,
    }
    
    # Prepare common ell grid for data (ℓ>=2)
    ell_data_full = np.arange(LMAX + 1)
    ell_data = ell_data_full[2:]

    # Mass reference from NGP
    mass_ref = ngp_reference["sum"]

    for name, cfg in methods.items():
        # Paint raw map
        raw_map = paint_particles_spherical(
            positions_lpt,
            method=cfg["method"],
            nside=NSIDE,
            observer_position=OBSERVER_POSITION,
            R_min=R_MIN,
            R_max=R_MAX,
            box_size=BOX_SIZE,
            mesh_shape=MESH_SHAPE,
            **{k: v for k, v in cfg.items() if k != "method"},
        )
        raw_np = np.asarray(raw_map)

        # 1) Mass conservation vs NGP
        cur_sum = float(np.sum(raw_np))
        rel_diff = abs(cur_sum - mass_ref) / (mass_ref if mass_ref != 0 else 1.0)
        assert np.isfinite(cur_sum)
        assert rel_diff <= 1e-3, f"Mass not conserved for {name}: rel_diff={rel_diff:.3e}"
        print(f"Mass conserved for {name}: rel_diff={rel_diff:.3e}")

        # 2) Autospectrum on overdensity
        mean = float(np.mean(raw_np))
        assert mean > 0, f"Non-positive mean for {name}"
        print(f"Mean density for {name}: {mean:.3e}")
        delta = raw_np / mean - 1.0
        cl_data_full = hp.anafast(delta, lmax=LMAX)
        cl_data = cl_data_full[2:]

        # 3) Compare to theory × pixel window
        m = min(len(ell_data), len(ell_th))
        ell = ell_data[:m]
        r = cl_data[:m] / cl_th_pix[:m]

        # Relaxed low-ℓ checks (ℓ < 20)
        lo_mask = ell < L_RELAX
        if np.any(lo_mask):
            r_lo = r[lo_mask]
            assert np.all(np.isfinite(r_lo)), f"NaNs at low-ℓ for {name}"
            # Relaxed but robust bounds to avoid flakiness from Limber/variance
            assert np.all((r_lo >= 0.2) & (r_lo <= 1.2)), f"Low-ℓ ratios out of bounds for {name}"
            print(f"Low-ℓ ratios within bounds for {name} within [{r_lo.min():.2f}, {r_lo.max():.2f}]")

        # Main window checks (method-specific ℓ range)
        ell_max = L_WINDOWS[name]
        main_mask = (ell >= L_RELAX) & (ell <= ell_max)
        r_main = r[main_mask]
        assert r_main.size > 10, f"Insufficient ℓ coverage in main window for {name}"
        assert np.all(np.isfinite(r_main)), f"NaNs in main window for {name}"
        print(f"Main window ratios within bounds for {name} within [{r_main.min():.2f}, {r_main.max():.2f}]")
        assert np.all((r_main >= 0.5) & (r_main <= 1.55)), f"Ratios out of bounds in main window for {name}"
