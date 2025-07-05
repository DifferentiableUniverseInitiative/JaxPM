# Task Bundle for Claude — Density Handling & κ Accuracy

Hand the checklist below to Claude.
Each task is broken into **Step 1 → Step 2 → …** so implementation is straightforward.

---

## 1 · Verify all painting routines return **physical density ρ**

functions are paint_particles_spherical and by extension spherical_density_fn
and density_plane_fn .. make sure that they return density and not overdensity

## 2 · Verify all lensing routines return **physical κ**

Chec the math in convergence_Born It seems not bad but still I have a slight missmatch in the cl 

## 3 · Check notebook 08-Kappa-Comparison-vs-Theory.ipynb

The cell ## Main Result: Theory vs Simulation Comparison

produces a 4 times larger power spectrum .. analyze the cl difference and suggest a fix