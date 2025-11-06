"""
Flat-sky raytracing implementation following the Dorian algorithm.

This adapts the spherical raytracing approach to flat-sky geometry,
using 2D Cartesian grids and FFT-based operations instead of HEALPix
and spherical harmonics.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def raytrace_flatsky(
    density_planes,
    plane_redshifts,
    plane_distances,
    z_source,
    pixel_size,
    omega_m=0.3,
    omega_lambda=0.7,
    interp_order=1,
    parallel_transport=True,
):
    """
    Perform flat-sky raytracing through multiple lens planes.
    
    Parameters
    ----------
    density_planes : list of ndarray, shape [(N, N), ...]
        List of 2D density contrast fields (delta = rho/rho_mean - 1) for each lens plane
    plane_redshifts : ndarray, shape (n_planes,)
        Redshift of each lens plane
    plane_distances : ndarray, shape (n_planes,)
        Comoving distance to each lens plane center (Mpc/h)
    z_source : float
        Source redshift
    pixel_size : float
        Angular pixel size in radians
    omega_m : float
        Matter density parameter
    omega_lambda : float
        Dark energy density parameter
    interp_order : int
        Interpolation order (1=bilinear, 3=cubic)
    parallel_transport : bool
        Whether to apply parallel transport correction
        
    Returns
    -------
    kappa_born : ndarray
        Born approximation convergence map
    A_final : ndarray, shape (2, 2, N, N)
        Final distortion matrix at each ray position
    beta_final : ndarray, shape (2, N, N)
        Final angular positions of rays (theta_x, theta_y)
    theta_init : ndarray, shape (2, N, N)
        Initial angular positions (pixel centers)
    """
    # Constants (in physical units)
    c_km_s = 299792.458  # km/s
    H0 = 100.0  # km/s/Mpc/h
    
    # Lensing prefactor: 3/2 * Omega_m * (H0/c)^2
    constant_factor = 1.5 * omega_m * (H0 / c_km_s) ** 2
    
    # Get geometry
    n_planes = len(density_planes)
    npix = density_planes[0].shape[0]
    assert all(plane.shape == (npix, npix) for plane in density_planes)
    
    # Compute source distance
    d_s = comoving_distance(z_source, omega_m, omega_lambda)
    
    # Initialize ray grid: shoot rays through pixel centers
    # Angular coordinates in radians from field center
    coords_1d = (np.arange(npix) - npix / 2 + 0.5) * pixel_size
    theta_y, theta_x = np.meshgrid(coords_1d, coords_1d, indexing='ij')
    theta_init = np.array([theta_x, theta_y])  # [2, npix, npix]
    
    # Ray angular positions: [k-th plane (previous, current), (x,y), ny, nx]
    beta = np.zeros([2, 2, npix, npix])
    beta[0] = theta_init
    beta[1] = theta_init
    
    # Distortion matrix: [k-th plane (previous, current), row, col, ny, nx]
    A = np.zeros([2, 2, 2, npix, npix])
    # Initialize as identity
    A[0, 0, 0] = 1.0  # A[0][0,0] = 1
    A[0, 1, 1] = 1.0  # A[0][1,1] = 1
    A[1, 0, 0] = 1.0  # A[1][0,0] = 1
    A[1, 1, 1] = 1.0  # A[1][1,1] = 1
    
    # Born approximation accumulator
    kappa_born = np.zeros([npix, npix])
    
    print(f"Starting raytracing with {n_planes} planes")
    print(f"Field size: {npix}x{npix} pixels, pixel_size={pixel_size*60*180/np.pi:.2f} arcmin")
    print(f"Source redshift: {z_source:.3f}, distance: {d_s:.1f} Mpc/h")
    
    # Iterate through lens planes
    for k in range(n_planes):
        print(f"\n--- Plane {k+1}/{n_planes} ---")
        
        z_k = plane_redshifts[k]
        d_k = plane_distances[k]
        delta = density_planes[k]
        a_k = 1.0 / (1.0 + z_k)
        
        print(f"z={z_k:.3f}, d={d_k:.1f} Mpc/h, a={a_k:.4f}")
        
        # ============================================
        # 1. Compute convergence at this plane
        # ============================================
        # delta is already (rho/rho_mean - 1), so convergence is:
        # kappa = constant_factor * d_k/a_k * delta
        kappa = constant_factor * (d_k / a_k) * delta
        
        print(f"Convergence: min={kappa.min():.6f}, max={kappa.max():.6f}, std={kappa.std():.6f}")
        
        # ============================================
        # 2. Compute deflection angle and shear in Fourier space
        # ============================================
        # FFT of convergence
        kappa_fft = np.fft.fft2(kappa)
        
        # k-space grid (including 2*pi factors)
        kx = 2 * np.pi * np.fft.fftfreq(npix, d=pixel_size)
        ky = 2 * np.pi * np.fft.fftfreq(npix, d=pixel_size)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0  # Avoid division by zero at DC mode
        
        # Deflection angle: alpha = -2 * nabla(kappa) / k^2 in Fourier space
        # alpha_x = -i*kx * kappa / k^2, alpha_y = -i*ky * kappa / k^2
        alpha_x_fft = -1j * KX * kappa_fft / K2
        alpha_y_fft = -1j * KY * kappa_fft / K2
        alpha_x_fft[0, 0] = 0.0  # DC mode = 0
        alpha_y_fft[0, 0] = 0.0
        
        alpha_x = np.fft.ifft2(alpha_x_fft).real
        alpha_y = np.fft.ifft2(alpha_y_fft).real
        alpha = np.array([alpha_x, alpha_y])  # [2, npix, npix]
        
        # Shear components: gamma = [d2phi/dx2 - d2phi/dy2, 2*d2phi/dxdy]
        # where phi is lensing potential (kappa = (d2phi/dx2 + d2phi/dy2)/2)
        # gamma1 = (kxx - kyy) / 2, gamma2 = kxy
        gamma1_fft = 0.5 * (KX**2 - KY**2) * kappa_fft / K2
        gamma2_fft = (KX * KY) * kappa_fft / K2
        gamma1_fft[0, 0] = 0.0
        gamma2_fft[0, 0] = 0.0
        
        gamma1 = np.fft.ifft2(gamma1_fft).real
        gamma2 = np.fft.ifft2(gamma2_fft).real
        
        # ============================================
        # 3. Build distortion rate matrix U at current ray positions
        # ============================================
        # U = [[kappa + gamma1, gamma2],
        #      [gamma2, kappa - gamma1]]
        # Need to interpolate these fields at beta[1] positions
        
        # Convert angular positions to pixel coordinates
        # beta is in radians, need to convert to pixel indices
        pixel_coords_x = beta[1, 0] / pixel_size + npix / 2 - 0.5
        pixel_coords_y = beta[1, 1] / pixel_size + npix / 2 - 0.5
        pixel_coords = np.array([pixel_coords_y, pixel_coords_x])  # Note: (y, x) order for map_coordinates
        
        # Interpolate kappa and shear at ray positions
        kappa_interp = map_coordinates(kappa, pixel_coords, order=interp_order, mode='wrap')
        gamma1_interp = map_coordinates(gamma1, pixel_coords, order=interp_order, mode='wrap')
        gamma2_interp = map_coordinates(gamma2, pixel_coords, order=interp_order, mode='wrap')
        alpha_x_interp = map_coordinates(alpha_x, pixel_coords, order=interp_order, mode='wrap')
        alpha_y_interp = map_coordinates(alpha_y, pixel_coords, order=interp_order, mode='wrap')
        alpha_interp = np.array([alpha_x_interp, alpha_y_interp])
        
        # Build U matrix
        U = np.zeros([2, 2, npix, npix])
        U[0, 0] = kappa_interp + gamma1_interp
        U[0, 1] = gamma2_interp
        U[1, 0] = gamma2_interp
        U[1, 1] = kappa_interp - gamma1_interp
        
        # ============================================
        # 4. Propagate ray positions
        # ============================================
        # Compute distance factors
        d_km1 = 0.0 if k == 0 else plane_distances[k - 1]
        d_kp1 = d_s if k == n_planes - 1 else plane_distances[k + 1]
        
        fac1 = d_k / d_kp1 * (d_kp1 - d_km1) / (d_k - d_km1)
        fac2 = (d_kp1 - d_k) / d_kp1
        
        print(f"Distance factors: d_km1={d_km1:.1f}, d_k={d_k:.1f}, d_kp1={d_kp1:.1f}")
        print(f"Propagation factors: fac1={fac1:.4f}, fac2={fac2:.4f}")
        
        # Update beta: beta_new = (1-fac1)*beta_old + fac1*beta_curr - fac2*alpha
        for i in range(2):
            beta[0, i] = (1 - fac1) * beta[0, i] + fac1 * beta[1, i] - fac2 * alpha_interp[i]
        
        # Swap: beta[1] <- beta[0], beta[0] <- beta[1]
        beta[[0, 1]] = beta[[1, 0]]
        
        print(f"Ray deflection: mean_x={alpha_interp[0].mean():.2e}, mean_y={alpha_interp[1].mean():.2e}")
        print(f"Ray spread: std_x={beta[1,0].std()/pixel_size:.2f} pix, std_y={beta[1,1].std()/pixel_size:.2f} pix")
        
        # ============================================
        # 5. Propagate distortion matrix
        # ============================================
        # A_new[i,j] = (1-fac1)*A_old[i,j] + fac1*A_curr[i,j]
        #              - fac2*(U[i,0]*A_curr[0,j] + U[i,1]*A_curr[1,j])
        
        for i in range(2):
            for j in range(2):
                A[0, i, j] = (
                    (1 - fac1) * A[0, i, j]
                    + fac1 * A[1, i, j]
                    - fac2 * (U[i, 0] * A[1, 0, j] + U[i, 1] * A[1, 1, j])
                )
        
        # Swap: A[1] <- A[0], A[0] <- A[1]
        A[[0, 1]] = A[[1, 0]]
        
        # ============================================
        # 6. Parallel transport (flat-sky approximation)
        # ============================================
        if parallel_transport:
            # In flat sky, parallel transport is less critical but we can
            # include a simple rotation correction for large deflections
            # For now, skip as flat-sky tangent space is approximately constant
            pass
        
        # ============================================
        # 7. Accumulate Born approximation
        # ============================================
        lensing_efficiency = (d_s - d_k) / d_s
        kappa_born += lensing_efficiency * kappa
        
        print(f"Born kappa accumulated: {lensing_efficiency:.4f} * plane")
    
    print("\n" + "="*60)
    print("Raytracing complete!")
    print(f"Final Born kappa: mean={kappa_born.mean():.6e}, std={kappa_born.std():.6f}")
    print(f"Final distortion matrix: det(A) mean={np.linalg.det(A[1].transpose(2,3,0,1)).mean():.6f}")
    
    return kappa_born, A[1], beta[1], theta_init


def comoving_distance(z, omega_m=0.3, omega_lambda=0.7, n_steps=1000):
    """
    Compute comoving distance in Mpc/h using simple trapezoidal integration.
    
    Parameters
    ----------
    z : float
        Redshift
    omega_m : float
        Matter density parameter
    omega_lambda : float
        Dark energy density parameter
    n_steps : int
        Number of integration steps
        
    Returns
    -------
    d_c : float
        Comoving distance in Mpc/h
    """
    c_km_s = 299792.458  # km/s
    H0 = 100.0  # km/s/Mpc/h (defines h)
    
    z_grid = np.linspace(0, z, n_steps)
    dz = z / (n_steps - 1)
    
    # E(z) = H(z)/H0 = sqrt(Omega_m*(1+z)^3 + Omega_lambda)
    Ez = np.sqrt(omega_m * (1 + z_grid)**3 + omega_lambda)
    
    # Integrate c/H0 * dz/E(z)
    integrand = 1.0 / Ez
    d_c = (c_km_s / H0) * np.trapz(integrand, dx=dz)
    
    return d_c


def convergence_to_observables(A):
    """
    Convert distortion matrix to observable lensing quantities.
    
    Parameters
    ----------
    A : ndarray, shape (2, 2, N, N)
        Distortion matrix at each position
        
    Returns
    -------
    kappa : ndarray, shape (N, N)
        Convergence (magnification)
    gamma1 : ndarray, shape (N, N)
        Shear component 1
    gamma2 : ndarray, shape (N, N)
        Shear component 2
    mu : ndarray, shape (N, N)
        Magnification factor
    """
    # Convergence: kappa = 1 - (A11 + A22)/2
    kappa = 1.0 - 0.5 * (A[0, 0] + A[1, 1])
    
    # Shear: gamma1 = (A11 - A22)/2, gamma2 = (A12 + A21)/2
    gamma1 = 0.5 * (A[0, 0] - A[1, 1])
    gamma2 = 0.5 * (A[0, 1] + A[1, 0])
    
    # Magnification: mu = 1/det(A) where A_lens = I - distortion matrix
    # det(I - M) = det([[1-A00, -A01], [-A10, 1-A11]])
    #            = (1-A00)*(1-A11) - A01*A10
    # But A is jacobian dtheta/dbeta, so:
    det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    mu = 1.0 / np.abs(det_A)
    
    return kappa, gamma1, gamma2, mu


# ============================================================================
# Example usage
# ============================================================================

def example_raytracing():
    """
    Example demonstrating flat-sky raytracing with synthetic lens planes.
    """
    import matplotlib.pyplot as plt
    
    # Setup
    npix = 256
    n_planes = 10
    pixel_size = 1.0 / 60.0 * np.pi / 180.0  # 1 arcmin in radians
    z_source = 1.0
    
    # Create synthetic density planes (random Gaussian fields)
    print("Generating synthetic density planes...")
    density_planes = []
    plane_redshifts = np.linspace(0.1, 0.9 * z_source, n_planes)
    plane_distances = np.array([comoving_distance(z) for z in plane_redshifts])
    
    for i, z in enumerate(plane_redshifts):
        # Generate smooth random field
        delta = np.random.randn(npix, npix)
        # Smooth in Fourier space
        delta_fft = np.fft.fft2(delta)
        kx = np.fft.fftfreq(npix)
        ky = np.fft.fftfreq(npix)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        # Power spectrum P(k) ~ k^-3 (approximate matter power spectrum)
        K[0, 0] = 1.0
        power = K**-1.5
        delta = np.fft.ifft2(delta_fft * power).real
        # Normalize
        delta = delta / delta.std() * 0.01 * (1 + i)  # Increasing amplitude with z
        density_planes.append(delta)
    
    print(f"\nCreated {n_planes} planes from z={plane_redshifts[0]:.2f} to z={plane_redshifts[-1]:.2f}")
    
    # Run raytracing
    kappa_born, A_final, beta_final, theta_init = raytrace_flatsky(
        density_planes,
        plane_redshifts,
        plane_distances,
        z_source,
        pixel_size,
        interp_order=1,
        parallel_transport=False,
    )
    
    # Convert to observables
    kappa_rt, gamma1, gamma2, mu = convergence_to_observables(A_final)
    
    # Compare Born vs raytraced
    print(f"\nComparison:")
    print(f"Born convergence:     mean={kappa_born.mean():.6e}, std={kappa_born.std():.6f}")
    print(f"Raytraced convergence: mean={kappa_rt.mean():.6e}, std={kappa_rt.std():.6f}")
    print(f"Magnification:         mean={mu.mean():.4f}, std={mu.std():.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    im1 = axes[0, 0].imshow(kappa_born, origin='lower', cmap='RdBu_r')
    axes[0, 0].set_title('Born Convergence')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(kappa_rt, origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('Raytraced Convergence')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(kappa_born - kappa_rt, origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title('Difference (Born - Raytraced)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    im4 = axes[1, 0].imshow(gamma1, origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('Shear γ₁')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(gamma2, origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('Shear γ₂')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(mu, origin='lower', cmap='viridis', vmin=0.8, vmax=1.2)
    axes[1, 2].set_title('Magnification μ')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('raytracing_flatsky_example.png', dpi=150)
    print("\nPlot saved to: raytracing_flatsky_example.png")
    
    return kappa_born, A_final, beta_final, theta_init


if __name__ == '__main__':
    example_raytracing()
