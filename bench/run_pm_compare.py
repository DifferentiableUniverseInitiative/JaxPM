#!/usr/bin/env python
"""End-to-end PM simulation comparing force solvers: FFT vs multigrid vs warm-started multigrid.

Runs an explicit fixed-step leapfrog (lax.scan) from a=0.1 to a=1.0 from IDENTICAL initial
conditions for each solver, so the only difference is the gravity solve. The warm-started MG
threads the potential phi through the scan carry and seeds each solve with the growth-scaled
previous phi (so it converges in ~1 cycle). Saves the final density fields and renders imshow
slice-comparison figures (FFT, MG, MG-warm, and their differences).

  python bench/run_pm_compare.py --mesh 256 --nsteps 20 --outdir fields_cmp
  srun -n1 --gpus=4 python bench/run_pm_compare.py --mesh 512 --pdims 2 2 --nsteps 20
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=256)
    ap.add_argument("--box",
                    type=float,
                    default=256.0,
                    help="box size (Mpc/h) per side")
    ap.add_argument("--pdims", type=int, nargs=2, default=None)
    ap.add_argument("--nsteps", type=int, default=20)
    ap.add_argument("--a0", type=float, default=0.1)
    ap.add_argument("--a1", type=float, default=1.0)
    ap.add_argument("--solvers", nargs="+", default=["fft", "mg", "mgwarm"])
    ap.add_argument("--cycles", type=int, default=3, help="MG V-cycles (cold)")
    ap.add_argument("--warm-cycles",
                    type=int,
                    default=2,
                    help="MG V-cycles when warm-started")
    ap.add_argument(
        "--fft-fd",
        action="store_true",
        default=True,
        help=
        "FFT reference uses the discrete-Laplacian operator (matches MG; fair "
        "solver-only comparison). Default on.")
    ap.add_argument(
        "--fft-continuum",
        dest="fft_fd",
        action="store_false",
        help=
        "FFT reference uses continuum -1/k^2 (JaxPM default); adds the operator gap"
    )
    ap.add_argument("--v", type=int, default=4)
    ap.add_argument("--omega", type=float, default=0.857)
    ap.add_argument("--agg-n", type=int, default=16)
    ap.add_argument("--grad-order", type=int, default=4)
    ap.add_argument(
        "--halo-size",
        type=int,
        default=0,
        help=
        "CIC paint/read halo (cells) on sharded axes; >0 removes multi-device "
        "shard-boundary seams. ~32-64 covers typical PM displacements.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-slab",
        type=int,
        default=0,
        help=
        "save only the central K z-cells of the final field (vs the full cube); "
        "enough for slice/slab plots and far cheaper to gather/store at large N."
    )
    ap.add_argument("--outdir", default="fields_cmp")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--warm-sweep",
        type=int,
        nargs="+",
        default=None,
        help=
        "sweep these warm-cycle counts (e.g. 1 2 3 4); reports accuracy/time tradeoff"
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help=
        "skip the sim; load saved field_{fft,mg,mgwarm}_nN.npy and make the figure (CPU)"
    )
    ap.add_argument(
        "--pk-sweep",
        type=int,
        nargs="+",
        default=None,
        help=
        "run FFT + cold-MG and warm-MG at each given cycle count, then compute "
        "P(k), transfer ratio P/P_fft, and cross-correlation r(k) vs FFT.")
    return ap.parse_args()


def power_cross(d1, d2, box):
    """Auto/cross power and correlation coefficient of two overdensity fields on an N^3 mesh.
    Returns (k_phys, P11, P22, P12, r) binned in integer |k| (fundamental-mode shells)."""
    N = d1.shape[0]
    f1 = np.fft.fftn(d1)
    f2 = np.fft.fftn(d2)
    kk = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    kbin = np.round(np.sqrt(KX**2 + KY**2 + KZ**2)).astype(np.int64).ravel()
    kmax = N // 2
    norm = (box / N**2)**3  # |delta_k|^2 -> P(k) in (box units)^3
    cnt = np.bincount(kbin, minlength=kmax + 1)[:kmax + 1]
    binsum = lambda w: np.bincount(kbin, weights=w.ravel(), minlength=kmax + 1
                                   )[:kmax + 1]
    P11 = binsum((f1 * np.conj(f1)).real) / np.maximum(cnt, 1) * norm
    P22 = binsum((f2 * np.conj(f2)).real) / np.maximum(cnt, 1) * norm
    P12 = binsum((f1 * np.conj(f2)).real) / np.maximum(cnt, 1) * norm
    r = P12 / np.sqrt(np.maximum(P11 * P22, 1e-300))
    kphys = np.arange(kmax + 1) * (2 * np.pi / box)
    return kphys, P11, P22, P12, r


def make_sharding(pdims):
    if pdims is None:
        return None
    from jax.experimental import mesh_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    devs = mesh_utils.create_device_mesh(tuple(pdims))
    mesh = jax.sharding.Mesh(devs, axis_names=("x", "y"))
    return NamedSharding(mesh, P("x", "y"))


def main():
    args = parse_args()

    if args.plot_only:
        # Load fields saved by single-solver runs and render the figure on CPU (no GPU needed).
        N = args.mesh
        results = {}
        for s in ["fft", "mg", "mgwarm"]:
            p = f"{args.outdir}/field_{s}_n{N}.npy"
            if os.path.exists(p):
                tp = f"{args.outdir}/field_{s}_n{N}.time"
                msstep = float(
                    open(tp).read()) if os.path.exists(tp) else float("nan")
                results[s] = (np.load(p), msstep * args.nsteps / 1e3
                              )  # store as "total" so /nsteps recovers ms/step
        if not results:
            raise SystemExit(f"no field_*_n{N}.npy found in {args.outdir}")
        print(f"plot-only: loaded {list(results)} from {args.outdir}",
              flush=True)
        make_figure(results, args)
        return

    is_dist = int(os.environ.get("SLURM_NTASKS", "1")) > 1
    if is_dist:
        jax.distributed.initialize()
    is_coord = (not is_dist) or jax.process_index() == 0

    def cp(*a, **k):
        if is_coord: print(*a, **k, flush=True)

    from jaxpm.distributed import fft3d, ifft3d
    from jaxpm.growth import growth_factor, growth_rate
    from jaxpm.kernels import (fftk, gradient_kernel,
                               interpolate_power_spectrum, invlaplace_kernel)
    from jaxpm.painting import cic_paint_dx
    from jaxpm.pm import linear_field, pm_forces

    N = args.mesh
    mesh_shape = (N, N, N)
    box = (args.box, ) * 3
    sharding = make_sharding(args.pdims)
    cp(f"PM compare: mesh={mesh_shape} box={box} devices={jax.device_count()} "
       f"pdims={args.pdims} nsteps={args.nsteps} solvers={args.solvers}")

    cosmo = jc.Planck15(Omega_c=0.25, sigma8=0.8)
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, sharding)

    a_steps = jnp.linspace(args.a0, args.a1, args.nsteps + 1)
    D = lambda a: growth_factor(cosmo, jnp.atleast_1d(a))[0]
    fr = lambda a: growth_rate(cosmo, jnp.atleast_1d(a))[0]
    E = lambda a: jnp.sqrt(jc.background.Esqr(cosmo, jnp.atleast_1d(a))[0])

    # ---- identical initial conditions for all solvers (Zeldovich/LPT-1 from the same field) ----
    # Build the displacement field straight from the FFT kernels (all sharded) -- avoids
    # materializing an unsharded (N,N,N,3) zeros array, which OOMs at 1024^3.
    # NOTE: call linear_field UNDER jit so XLA shards its internal kmesh/pkmesh broadcast; called
    # eagerly it materializes that intermediate replicated (a full N^3 array) and OOMs at 2048^3.
    ic = jax.block_until_ready(
        jax.jit(lambda key: linear_field(
            mesh_shape, box, pk_fn, seed=key, sharding=sharding))(
                jax.random.PRNGKey(args.seed)))

    @jax.jit
    def zeldovich(ic):  # ic passed as arg (can't close over a sharded global)
        dk = fft3d(ic)
        kv = fftk(dk)
        pot_k = dk * invlaplace_kernel(kv, fd=args.fft_fd)
        return jnp.stack(
            [ifft3d(-gradient_kernel(kv, i) * pot_k) for i in range(3)],
            axis=-1)

    init_force = jax.block_until_ready(
        zeldovich(ic))  # (N,N,N,3) force on the grid, sharded
    dx0 = D(args.a0) * init_force  # Zeldovich displacement
    p0 = args.a0**2 * fr(args.a0) * E(args.a0) * dx0

    mg_kw = dict(v1=args.v,
                 v2=args.v,
                 omega=args.omega,
                 agg_n=args.agg_n,
                 grad_order=args.grad_order)

    def forces_only(dx, a, solver, cycles, u0=None):
        out = pm_forces(
            dx,
            mesh_shape=mesh_shape,
            paint_absolute_pos=False,
            halo_size=args.halo_size,
            sharding=sharding,
            solver=("mg" if solver != "fft" else "fft"),
            fd=args.
            fft_fd,  # discrete operator on both sides => pure solver comparison
            mg_params=dict(cycles=cycles, **mg_kw)
            if solver != "fft" else None,
            u0=u0,
            return_potential=(solver != "fft"))
        return out if solver != "fft" else (out, None)

    def kdk_step(carry, i, solver, cold_cycles, warm_cycles):
        dx, p, phi = carry
        a, a_n = a_steps[i], a_steps[i + 1]
        da = a_n - a
        warm = (solver == "mgwarm")
        u0 = (phi *
              (D(a) / D(a_steps[jnp.maximum(i - 1, 0)]))) if warm else None
        cycles = warm_cycles if warm else cold_cycles
        forces, phi_new = forces_only(dx, a, solver, cycles, u0=u0)
        forces = forces * 1.5 * cosmo.Omega_m
        p = p + da * forces / (a**2 * E(a))  # kick
        dx = dx + da * p / (a**3 * E(a))  # drift
        return (dx, p, phi if phi_new is None else phi_new), None

    @jax.jit
    def fft_potential(dx):
        # One-time real-space potential via FFT, to seed the warm start (represents the single
        # accurate solve a real sim does at init; cheap compile, unlike a second MG compile).
        dk = fft3d(
            cic_paint_dx(dx, halo_size=args.halo_size, sharding=sharding))
        return ifft3d(dk * invlaplace_kernel(fftk(dk), fd=True))

    def run_one(solver, cold_cycles, warm_cycles):
        phi_seed = jnp.zeros(mesh_shape, jnp.float32,
                             device=sharding)  # sharded (else 4GB/GPU replica)
        if solver == "mgwarm":
            phi_seed = jax.block_until_ready(
                fft_potential(dx0))  # one-time FFT seed at the IC
        run = jax.jit(lambda d, p, ph: jax.lax.scan(
            partial(kdk_step,
                    solver=solver,
                    cold_cycles=cold_cycles,
                    warm_cycles=warm_cycles),
            (d, p, ph), jnp.arange(args.nsteps))[0])
        jax.block_until_ready(run(dx0, p0, phi_seed))  # compile
        t0 = time.perf_counter()
        dxf, _, _ = jax.block_until_ready(run(dx0, p0, phi_seed))
        dt = time.perf_counter() - t0
        field = jax.block_until_ready(
            cic_paint_dx(dxf, halo_size=args.halo_size, sharding=sharding))
        if args.save_slab > 0:  # keep only a thin z-slab (sliced on-device, still x/y-sharded)
            zc = field.shape[2] // 2
            s = args.save_slab
            field = field[:, :, zc - s // 2:zc + s // 2]
        if is_dist:  # gather the (now small) field to host (device_get fails across procs)
            from jax.experimental.multihost_utils import process_allgather
            field = process_allgather(field, tiled=True)
        return np.asarray(jax.device_get(field), np.float32), dt

    if is_coord:
        os.makedirs(args.outdir, exist_ok=True)

    # ---------- power-spectrum / cross-correlation sweep ----------
    if args.pk_sweep:
        # Incremental: keep only the FFT reference + the current field (each is N^3; 4.3 GB at 1024).
        dens = lambda f: (f / f.mean() - 1.0).astype(np.float32)  # overdensity
        cp("[fft     ] reference")
        ref = dens(run_one("fft", args.cycles, 1)[0])
        out = None
        if is_coord:
            k0, Pref, _, _, _ = power_cross(ref, ref, args.box)
            out = {"k": k0, "P_fft": Pref}
        for c in args.pk_sweep:
            for kind, solver, cc, wc in [("cold", "mg", c, 1),
                                         ("warm", "mgwarm", args.cycles, c)]:
                cp(f"[{kind} c{c}] ...")
                d = dens(run_one(solver, cc, wc)[0])
                if is_coord:
                    k0, Pxx, _, _, r = power_cross(d, ref, args.box)
                    out[f"P_{kind}{c}"] = Pxx
                    out[f"r_{kind}{c}"] = r
                    j = len(k0) // 2
                    cp(f"  {kind} c{c}: r(kNyq/2)={r[j]:.4f}  "
                       f"P/Pfft(kNyq/2)={Pxx[j]/max(Pref[j],1e-300):.3f}")
                del d
        if is_coord:
            np.savez(f"{args.outdir}/pk_n{N}.npz", **out)
            make_pk_plot(out, args)
            cp(f"saved {args.outdir}/pk_n{N}.npz + plot")
        return

    # ---------- warm-cycles sweep mode ----------
    if args.warm_sweep:
        ref, tf = run_one("fft", args.cycles, 1)
        cp(f"[fft     ] {tf/args.nsteps*1e3:6.2f} ms/step   (reference)")
        sweep = []
        coldf, tc = run_one("mg", args.cycles, 1)
        relc = float(np.linalg.norm(coldf - ref) / np.linalg.norm(ref))
        cp(f"[mg  c{args.cycles}  ] {tc/args.nsteps*1e3:6.2f} ms/step   relL2={relc:.3e}   "
           f"speedup {tf/tc:.2f}x")
        for wc in args.warm_sweep:
            f, dt = run_one("mgwarm", args.cycles, wc)
            rel = float(np.linalg.norm(f - ref) / np.linalg.norm(ref))
            sweep.append((wc, dt / args.nsteps * 1e3, rel, tf / dt))
            cp(f"[warm wc{wc}] {dt/args.nsteps*1e3:6.2f} ms/step   relL2={rel:.3e}   "
               f"speedup {tf/dt:.2f}x" +
               ("  *** beats FFT ***" if dt < tf else ""))
        if is_coord and not args.no_plot:
            make_sweep_plot(sweep, tf / args.nsteps * 1e3,
                            (relc, tc / args.nsteps * 1e3), args)
        return

    # ---------- 3-solver field comparison + figure ----------
    results = {}
    for solver in args.solvers:
        f, dt = run_one(solver, args.cycles, args.warm_cycles)
        results[solver] = (f, dt)
        cp(f"[{solver:7s}] sim {dt*1e3:8.1f} ms total  ({dt/args.nsteps*1e3:6.2f} ms/step)"
           )
    if "fft" in results:
        ref = results["fft"][0]
        for s in args.solvers:
            if s != "fft":
                rel = np.linalg.norm(results[s][0] - ref) / np.linalg.norm(ref)
                cp(f"   final-field rel-L2 {s} vs FFT: {rel:.3e}   speedup {results['fft'][1]/results[s][1]:.2f}x"
                   )
    if is_coord:
        for s, (f, dt) in results.items():
            np.save(f"{args.outdir}/field_{s}_n{N}.npy", f)
            with open(f"{args.outdir}/field_{s}_n{N}.time", "w") as fh:
                fh.write(str(dt / args.nsteps * 1e3))  # ms/step
        if not args.no_plot and len(results) >= 2:
            make_figure(results, args)
        cp(f"saved fields to {args.outdir}/")


def make_pk_plot(out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    k = out["k"]
    Pf = out["P_fft"]
    m = k > 0
    cycles = sorted(
        {int(key.split("cold")[1])
         for key in out if key.startswith("P_cold")})
    cmap = cm.viridis(np.linspace(0.15, 0.85, len(cycles)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    # (a) P(k)
    ax[0].loglog(k[m], Pf[m], "k-", lw=2, label="FFT")
    for c, col in zip(cycles, cmap):
        ax[0].loglog(k[m],
                     out[f"P_warm{c}"][m],
                     "-",
                     color=col,
                     label=f"warm wc{c}")
        ax[0].loglog(k[m],
                     out[f"P_cold{c}"][m],
                     "--",
                     color=col,
                     label=f"cold c{c}")
    ax[0].set_xlabel("k [h/Mpc]")
    ax[0].set_ylabel("P(k)")
    ax[0].set_title("power spectrum")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=.3, which="both")
    # (b) transfer ratio P/P_fft
    for c, col in zip(cycles, cmap):
        ax[1].semilogx(k[m],
                       out[f"P_warm{c}"][m] / np.maximum(Pf[m], 1e-300),
                       "-",
                       color=col,
                       label=f"warm wc{c}")
        ax[1].semilogx(k[m],
                       out[f"P_cold{c}"][m] / np.maximum(Pf[m], 1e-300),
                       "--",
                       color=col)
    ax[1].axhline(1, c="k", lw=.8)
    ax[1].set_ylim(0.8, 1.2)
    ax[1].set_xlabel("k [h/Mpc]")
    ax[1].set_ylabel(r"$P/P_{\rm FFT}$")
    ax[1].set_title("transfer ratio")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")
    # (c) cross-correlation 1-r(k)
    for c, col in zip(cycles, cmap):
        ax[2].loglog(k[m],
                     1 - out[f"r_warm{c}"][m],
                     "-",
                     color=col,
                     label=f"warm wc{c}")
        ax[2].loglog(k[m], 1 - out[f"r_cold{c}"][m], "--", color=col)
    ax[2].set_xlabel("k [h/Mpc]")
    ax[2].set_ylabel(r"$1-r(k)$")
    ax[2].set_title("decorrelation vs FFT")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3, which="both")
    fig.suptitle(
        f"{args.mesh}$^3$, box={args.box:g} Mpc/h, {args.nsteps} steps  "
        f"(solid=warm, dashed=cold)")
    o = f"{args.outdir}/pk_n{args.mesh}.png"
    fig.savefig(o, dpi=150)
    print(f"  wrote {o}", flush=True)


def make_sweep_plot(sweep, t_fft, cold, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wc = [s[0] for s in sweep]
    ts = [s[1] for s in sweep]
    rel = [s[2] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].plot(wc, rel, "o-", label="warm MG")
    ax[0].axhline(cold[0],
                  ls="--",
                  c="gray",
                  label=f"cold MG (c{args.cycles})")
    ax[0].set_xlabel("warm cycles")
    ax[0].set_ylabel("final-field rel-L2 vs FFT")
    ax[0].set_yscale("log")
    ax[0].set_xticks(wc)
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(ts, rel, "o-")
    for w, t, r, _ in sweep:
        ax[1].annotate(f"wc{w}", (t, r),
                       fontsize=8,
                       xytext=(3, 3),
                       textcoords="offset points")
    ax[1].axvline(t_fft, ls="--", c="k", label=f"FFT {t_fft:.1f} ms/step")
    ax[1].scatter([cold[1]], [cold[0]],
                  c="gray",
                  marker="s",
                  label=f"cold MG c{args.cycles}")
    ax[1].set_xlabel("ms / step")
    ax[1].set_ylabel("final-field rel-L2 vs FFT")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.suptitle(
        f"warm-cycles accuracy/speed tradeoff  ({args.mesh}^3, {args.nsteps} steps, "
        f"{jax.device_count()} GPU)")
    out = f"{args.outdir}/warm_sweep_n{args.mesh}.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}", flush=True)


def make_figure(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = args.mesh
    order = [s for s in ["fft", "mg", "mgwarm"] if s in results]
    nz = results[order[0]][0].shape[
        2]  # actual z-extent (full cube, or a pre-saved slab)
    slab = min(nz, max(
        1, N // 16))  # project a z-slab (thin slices are mostly-empty voids ->
    zc = nz // 2  # blocky log-clamped patches); slab = std cosmic-web viz
    z0, z1 = zc - slab // 2, zc + slab // 2
    proj = lambda f: np.log10(
        np.maximum(f[:, :, z0:z1].sum(axis=2) / slab, 1e-3))
    ncol = len(order)
    fig, axes = plt.subplots(2,
                             ncol,
                             figsize=(4.2 * ncol, 8),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    ref = results["fft"][0] if "fft" in results else results[order[0]][0]
    vmin, vmax = np.percentile(proj(ref), [2, 99.5])
    for j, s in enumerate(order):
        f, dt = results[s]
        msstep = dt / args.nsteps * 1e3
        lbl = f"{s}   ({msstep:.1f} ms/step)" if msstep == msstep else s
        im = axes[0, j].imshow(proj(f).T,
                               origin="lower",
                               cmap="magma",
                               vmin=vmin,
                               vmax=vmax)
        axes[0, j].set_title(lbl)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        fig.colorbar(im, ax=axes[0, j], fraction=0.046)
        # difference vs FFT (same slab projection)
        d = (f - ref)[:, :, z0:z1].sum(axis=2) / slab
        dm = np.percentile(np.abs(d), 99) + 1e-30
        im2 = axes[1, j].imshow(d.T,
                                origin="lower",
                                cmap="RdBu_r",
                                vmin=-dm,
                                vmax=dm)
        rel = np.linalg.norm(f - ref) / (np.linalg.norm(ref) + 1e-30)
        axes[1, j].set_title(f"{s} - FFT  (relL2={rel:.1e})")
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        fig.colorbar(im2, ax=axes[1, j], fraction=0.046)
    fig.suptitle(
        f"PM density (log), {slab}-cell z-slab, {N}^3, {args.nsteps} steps",
        fontsize=14)
    out = f"{args.outdir}/compare_n{N}.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
