import warnings

import jax.numpy as jnp
import jax_cosmo as jc

from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, compensation_kernel, fftk,
                           gradient_kernel, invlaplace_kernel,
                           longrange_kernel)
from jaxpm.painting import paint, readout


def _pad_rfft_3d(rk, n, m):
    """Pad an rFFT field from resolution n to m in Fourier space.

    Matches disco-dj's convention: Nyquist zeroed in all three axes.
    Scales by (m/n)^3.
    """
    hn = n // 2
    padded = jnp.zeros((m, m, m // 2 + 1), dtype=rk.dtype)
    padded = padded.at[:hn, :hn, :hn].set(rk[:hn, :hn, :hn])
    padded = padded.at[:hn, m - hn + 1:m, :hn].set(rk[:hn, hn + 1:n, :hn])
    padded = padded.at[m - hn + 1:m, :hn, :hn].set(rk[hn + 1:n, :hn, :hn])
    padded = padded.at[m - hn + 1:m, m - hn + 1:m, :hn].set(rk[hn + 1:n,
                                                               hn + 1:n, :hn])
    padded = padded.at[0, 0, 0].set(0)  # Zero DC mode
    return padded * (m / n)**3


def _crop_rfft_3d(rk_ext, n, m):
    """Crop an rFFT field from resolution m back to n. Scales by (n/m)^3."""
    hn = n // 2
    cropped = jnp.zeros((n, n, n // 2 + 1), dtype=rk_ext.dtype)
    cropped = cropped.at[:hn, :hn, :hn].set(rk_ext[:hn, :hn, :hn])
    cropped = cropped.at[:hn, hn + 1:n, :hn].set(rk_ext[:hn,
                                                        m - hn + 1:m, :hn])
    cropped = cropped.at[hn + 1:n, :hn, :hn].set(rk_ext[m - hn +
                                                        1:m, :hn, :hn])
    cropped = cropped.at[hn + 1:n,
                         hn + 1:n, :hn].set(rk_ext[m - hn + 1:m,
                                                   m - hn + 1:m, :hn])
    return cropped * (n / m)**3


def _dealiased_product(a, b, n):
    """Compute a*b with 3/2 dealiasing using rFFT."""
    m = (3 * n) // 2
    a_rk = jnp.fft.rfftn(a)
    b_rk = jnp.fft.rfftn(b)
    a_ext = jnp.fft.irfftn(_pad_rfft_3d(a_rk, n, m), s=(m, m, m))
    b_ext = jnp.fft.irfftn(_pad_rfft_3d(b_rk, n, m), s=(m, m, m))
    product_rk_ext = jnp.fft.rfftn(a_ext * b_ext)
    return jnp.fft.irfftn(_crop_rfft_3d(product_rk_ext, n, m), s=(n, n, n))


def pm_forces(positions,
              mesh_shape=None,
              delta=None,
              r_split=0,
              paint_absolute_pos=None,
              halo_size=0,
              sharding=None,
              order='CIC',
              deconvolution=False,
              initial_particles=None,
              gradient_order=1,
              laplace_fd=False):
    """
    Computes gravitational forces on particles using a PM scheme.

    Parameters
    ----------
    order : int or str
        Mass-assignment order forwarded to ``paint``/``readout``
        (NGP=1, CIC=2, TSC=3, PCS=4).
    deconvolution : bool
        Deconvolve the assignment window from the painted density. Applied in
        Fourier space (we are already in k-space for the PM kernels), which
        avoids the FFT/iFFT round-trip of a real-space ``paint(deconvolution=True)``.
    initial_particles : {None, 'uniform'}
        ``None`` -> ``positions`` are absolute, ``'uniform'`` -> ``positions``
        are displacements from a uniform grid.
    gradient_order : int
        Order of the ``gradient_kernel`` used to differentiate the potential
        (0 = exact ``ik``, 1 = 4th-order finite-difference).
    laplace_fd : bool
        Use the finite-difference inverse-Laplace kernel for the Poisson solve.
    paint_absolute_pos : bool, optional
        Deprecated -- use ``initial_particles`` instead (``True`` -> ``None``,
        ``False`` -> ``'uniform'``).
    """
    if mesh_shape is None:
        assert (delta is not None),\
          "If mesh_shape is not provided, delta should be provided"
        mesh_shape = delta.shape

    if paint_absolute_pos is not None:
        warnings.warn(
            "paint_absolute_pos is deprecated; use initial_particles instead "
            "(None for absolute positions, 'uniform' for displacements).",
            DeprecationWarning,
            stacklevel=2)
        initial_particles = None if paint_absolute_pos else 'uniform'

    # Deconvolution are done opportunistically in Fourier space, so we can avoid the extra FFT/iFFT round-trip
    paint_fn = lambda pos: paint(pos,
                                 initial_particles=initial_particles,
                                 order=order,
                                 halo_size=halo_size,
                                 sharding=sharding,
                                 deconvolution=False)
    read_fn = lambda grid_mesh, pos: readout(grid_mesh,
                                             pos,
                                             initial_particles=
                                             initial_particles,
                                             order=order,
                                             halo_size=halo_size,
                                             sharding=sharding)

    if delta is None:
        field = paint_fn(positions)
        delta_k = fft3d(field)
    elif jnp.isrealobj(delta):
        delta_k = fft3d(delta)
    else:
        delta_k = delta

    kvec = fftk(delta_k)
    # Deconvolve the assignment window directly in Fourier space (one multiply,
    # no extra FFT/iFFT) -- only meaningful for the density we just painted.
    if deconvolution and delta is None:
        delta_k = delta_k * compensation_kernel(kvec, order)
    # Computes gravitational potential
    pot_k = delta_k * invlaplace_kernel(
        kvec, fd=laplace_fd) * longrange_kernel(kvec, r_split=r_split)
    # Computes gravitational forces
    forces = jnp.stack([
        read_fn(ifft3d(-gradient_kernel(kvec, i, order=gradient_order) * pot_k),positions
        ) for i in range(3)], axis=-1) # yapf: disable

    return forces


def lpt(cosmo,
        initial_conditions,
        particles=None,
        a=0.1,
        halo_size=0,
        sharding=None,
        order=1,
        paint_order='CIC',
        deconvolution=False,
        gradient_order=1,
        laplace_fd=False,
        dealiased=False,
        exact_growth=False):
    """
    Computes first and second order LPT displacement and momentum,
    e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)

    Parameters
    ----------
    order : int
        LPT order (1 or 2).
    paint_order : int or str
        Mass-assignment order forwarded to ``pm_forces`` (NGP=1, CIC=2,
        TSC=3, PCS=4). Affects the force read-out interpolation.
    deconvolution : bool
        Forwarded to ``pm_forces``. Inert here because LPT supplies the
        density directly via ``delta=`` (no particle painting to deconvolve).
    dealiased : bool
        Use 3/2 zero-padding to dealias the quadratic 2LPT source term.
    exact_growth : bool
        Use exact second-order growth factor coefficient instead of
        the Einstein-de Sitter approximation 3/7.
    """
    initial_particles = None if particles is not None else 'uniform'
    if particles is None:
        particles = jnp.zeros_like(initial_conditions,
                                   shape=(*initial_conditions.shape, 3))

    a = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a))
    delta_k = fft3d(initial_conditions)
    initial_force = pm_forces(particles,
                              delta=delta_k,
                              initial_particles=initial_particles,
                              halo_size=halo_size,
                              sharding=sharding,
                              order=paint_order,
                              deconvolution=deconvolution,
                              gradient_order=gradient_order,
                              laplace_fd=laplace_fd)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * E * dx
    f = a**2 * E * dGfa(cosmo, a) * initial_force
    if order == 2:
        kvec = fftk(delta_k)
        pot_k = delta_k * invlaplace_kernel(kvec, fd=laplace_fd)
        n = initial_conditions.shape[0]

        if dealiased:
            # Compute shear fields at original resolution
            shear = {}
            for i in range(3):
                nabla_ii = gradient_kernel(kvec, i, order=gradient_order)**2
                shear[(i, i)] = ifft3d(nabla_ii * pot_k)
            for i in range(3):
                for j in range(i + 1, 3):
                    nabla_ij = gradient_kernel(kvec, i, order=gradient_order) * \
                        gradient_kernel(kvec, j, order=gradient_order)
                    shear[(i, j)] = ifft3d(nabla_ij * pot_k)

            # Dealiased products with 3/2 zero-padding
            delta2 = jnp.zeros_like(initial_conditions)
            for i in range(3):
                for j in range(i + 1, 3):
                    delta2 = delta2 + _dealiased_product(
                        shear[(i, i)], shear[(j, j)], n)
                    delta2 = delta2 - _dealiased_product(
                        shear[(i, j)], shear[(i, j)], n)
        else:
            delta2 = 0
            shear_acc = 0
            for i in range(3):
                nabla_i_nabla_i = gradient_kernel(kvec,
                                                  i,
                                                  order=gradient_order)**2
                shear_ii = ifft3d(nabla_i_nabla_i * pot_k)
                delta2 += shear_ii * shear_acc
                shear_acc += shear_ii

                for j in range(i + 1, 3):
                    nabla_i_nabla_j = gradient_kernel(
                        kvec, i, order=gradient_order) * gradient_kernel(
                            kvec, j, order=gradient_order)
                    delta2 -= ifft3d(nabla_i_nabla_j * pot_k)**2

        delta_k2 = fft3d(delta2)
        init_force2 = pm_forces(particles,
                                delta=delta_k2,
                                initial_particles=initial_particles,
                                halo_size=halo_size,
                                sharding=sharding,
                                order=paint_order,
                                deconvolution=deconvolution,
                                gradient_order=gradient_order,
                                laplace_fd=laplace_fd)

        if exact_growth:
            # Compute exact 2LPT coefficient: 3/7 * D1(a_ref)^2 / D2(a_ref)
            # At early times (matter domination), this gives the exact ratio
            # |D2_unnorm(1)| / D1_unnorm(1)^2 instead of EdS approx 3/7
            a_ref = jnp.array(1e-5)
            D1_ref = growth_factor(cosmo, a_ref)
            D2_ref = growth_factor_second(cosmo, a_ref)
            lpt2_coeff = 3. / 7. * D1_ref**2 / D2_ref
        else:
            lpt2_coeff = 3. / 7.

        # NOTE: growth_factor_second is renormalized: - D2 = lpt2_coeff * growth_factor_second
        dx2 = lpt2_coeff * growth_factor_second(cosmo, a) * init_force2
        p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2
        f2 = a**2 * E * dGf2a(cosmo, a) * init_force2

        dx += dx2
        p += p2
        f += f2

    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed, sharding=None):
    """
    Generate initial conditions.
    """
    # Initialize a random field with one slice on each gpu
    field = normal_field(seed=seed, shape=mesh_shape, sharding=sharding)
    field = fft3d(field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    field = field * jnp.sqrt(pkmesh)
    field = ifft3d(field)
    return field


def pgd_correction(pos, mesh_shape, params, order='CIC', deconvolution=False):
    """
    improve the short-range interactions of PM-Nbody simulations with potential gradient descent method,
    based on https://arxiv.org/abs/1804.00671

    args:
      pos: particle positions [npart, 3]
      params: [alpha, kl, ks] pgd parameters
      order: mass-assignment order for paint/readout (NGP/CIC/TSC/PCS)
      deconvolution: deconvolve the assignment window (in Fourier space)
    """
    delta = paint(pos, grid_mesh=jnp.zeros(mesh_shape), order=order)
    delta_k = fft3d(delta)
    kvec = fftk(delta_k)
    if deconvolution:
        delta_k = delta_k * compensation_kernel(kvec, order)
    alpha, kl, ks = params
    PGD_range = PGD_kernel(kvec, kl, ks)

    pot_k_pgd = (delta_k * invlaplace_kernel(kvec)) * PGD_range

    forces_pgd = jnp.stack([
        readout(fft3d(-gradient_kernel(kvec, i) * pot_k_pgd), pos, order=order)
        for i in range(3)
    ],
                           axis=-1)

    dpos_pgd = forces_pgd * alpha

    return dpos_pgd
