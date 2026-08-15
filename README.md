# CFD Solvers — Finite Difference Heat Conduction

A progressive set of 2D CFD solvers written from scratch in C++, built as coursework
for SEM 7 CFD. The codes evolve deliberately: steady diffusion first, then transient
time marching, then a unified θ-family (β) formulation — each step reusing the
matrix-free **Conjugate Gradient** core developed in the previous one.

No external libraries. No linear-algebra package. The sparse operator `A` is never
assembled; it is only ever *applied* as a 5-point stencil.

---

## Repository layout

| File | Grid | Description |
| --- | --- | --- |
| [diffusion.cpp](diffusion.cpp) | 1000 × 1000 | Steady-state 2D diffusion (Laplace). Gauss–Seidel **and** CG implemented; `main()` currently runs the CG path. |
| [transient_heat_conduction.cpp](transient_heat_conduction.cpp) | 500 × 100 | Transient conduction. Compile-time switch between explicit FTCS and fully implicit (backward Euler + CG). |
| [beta_formulation_transient_heat_conduction.cpp](beta_formulation_transient_heat_conduction.cpp) | 500 × 100 | **Current solver.** Unifies the above under a single runtime parameter β ∈ [0,1]. |
| [build/](build/) | — | MSVC build output (gitignored). |
| [results/](results/) | — | Solver output, `.dat` + `.vtk` (gitignored). |

---

## The β formulation

The active solver generalises the time discretisation into one parameter, so the
whole θ-family is reachable without recompiling:

```
T^(n+1) - T^n
------------- = β·∇²T^(n+1) + (1-β)·∇²T^n
     Δt
```

| β | Scheme | Accuracy | Stability |
| --- | --- | --- | --- |
| `0.0` | Explicit / FTCS | O(Δt) | Conditional — `Fo_x + Fo_y ≤ ½` |
| `0.5` | Crank–Nicolson | O(Δt²) | Unconditional |
| `1.0` | Fully implicit / backward Euler | O(Δt) | Unconditional |
| other | Generalised θ | O(Δt) | Conditional if β < 0.5 |

Passed as the first command-line argument:

```powershell
build\beta_formulation_transient_heat_conduction.exe 0.5
```

At β = 0 the update is a direct explicit sweep and CG is bypassed entirely. For any
β > 0 the implicit block is solved by CG each timestep.

### What the solver does for you

- **Automatic Δt selection** — below β = 0.5 the step is set from the conditional
  stability limit with a safety factor (`DT_SAFETY = 0.8`); at or above β = 0.5 it is
  set by accuracy instead (`DT_ACCURACY = 1e-3`).
- **Stability check** — computes `Fo_x`, `Fo_y` against the analytic limit
  `1/(2(1-2β))` and aborts with a diagnostic if violated.
- **Patankar positivity warning** — flags `AP1 < 0`, where the scheme remains stable
  but the solution may oscillate.
- **Warm-started CG** — each solve starts from `T^n`, so the initial residual is often
  already below tolerance and the solve exits in zero iterations. This case is
  explicitly guarded: entering the loop would divide by a vanishing `p·Ap` and push
  NaN into the field.
- **Iteration accounting** — reports total and per-timestep average CG iterations.

---

## Numerics

**Spatial discretisation.** Second-order central differences on a uniform Cartesian
grid, giving the standard 5-point stencil (`AW`, `AE`, `AS`, `AN`, `AP`).

**Boundary conditions.** All four walls use a generalised Robin form

```
c + a·(∂T/∂n) = b·T
```

so Dirichlet (`a = 0`), Neumann (`b = 0`), and mixed conditions all come from the same
code path. The shipped configuration is Dirichlet on west/east/north with a mixed
condition on the south wall (`A_SOUTH = -10.0`).

**Linear solver — Conjugate Gradient.** Matrix-free, with the classical recurrences:

```
r₀ = b - Ax₀,   p₀ = r₀
αₖ = (rₖ·rₖ)/(pₖ·Apₖ)
xₖ₊₁ = xₖ + αₖpₖ
rₖ₊₁ = rₖ - αₖApₖ
βₖ = (rₖ₊₁·rₖ₊₁)/(rₖ·rₖ)
pₖ₊₁ = rₖ₊₁ + βₖpₖ
```

The residual recurrence avoids recomputing `b - Ax` each iteration, saving one full
stencil sweep. `PDIR` is zero on the boundary throughout, which is exactly the
homogeneous-Dirichlet operator CG requires for symmetry.

Convergence is measured in either the L1 or L2 norm, selected at compile time via
`NORM_TYPE`.

---

## Building

The toolchain is **MSVC only** — the project is configured for `cl.exe` from
VS 2026 Build Tools. There is no g++/WSL path here.

### From VS Code

`Ctrl+Shift+B` runs the `C/C++: cl.exe build active file` task in
[.vscode/tasks.json](.vscode/tasks.json). All artefacts (`.exe`, `.obj`, `.pdb`,
`.ilk`) go to `build/`, and the working directory is the project root so `results/`
resolves correctly.

### From a terminal

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl /Zi /EHsc /nologo /Fe:build\beta_formulation_transient_heat_conduction.exe /Fo:build\ /Fd:build\beta_formulation_transient_heat_conduction.pdb beta_formulation_transient_heat_conduction.cpp
```

The `/Fo:` and `/Fd:` flags matter — without them MSVC drops `.obj` and `vc140.pdb`
into the project root.

> **Run from the project root.** Output paths are relative to the *current working
> directory*, not the binary. `build\...exe` from the root writes to `results\`;
> `cd build` first and you get `build\results\` instead.

---

## Output

The β solver writes a numbered time series into `results/`:

- `temperature_%05d.dat` — Tecplot ASCII, with `STRANDID` and `SOLUTIONTIME` set so
  Tecplot assembles the files into a single animating transient zone.
- `temperature_%05d.vtk` — Legacy VTK `STRUCTURED_GRID`, picked up by ParaView as a
  time series automatically.

Write frequency is controlled by `WRITE_INTERVAL` (simulation time between dumps, not
timestep count).

> The two earlier solvers predate the `results/` convention and write
> `temperature_field*.dat` / `.vtk` into the working directory instead.

---

## Configuration

Most knobs are `#define`s near the top of the file.

| Symbol | Meaning |
| --- | --- |
| `NI`, `NJ` | Grid nodes in x and y |
| `LX`, `LY` | Domain extents (set in `main()`) |
| `T_WEST/SOUTH/EAST/NORTH` | Boundary temperatures |
| `A_*`, `B_*`, `C_*` | Robin coefficients per wall |
| `kT`, `rho`, `Cp` | Material properties (α = k/ρCp) |
| `TMAX`, `MAXSTEP` | Stop time and timestep cap |
| `WRITE_INTERVAL` | Simulation time between field dumps |
| `CCSS` | CG convergence tolerance (`1e-13`) |
| `MAXITER_CG` | CG iteration cap (`50000`) |
| `NORM_TYPE` | `NORM_L1` or `NORM_L2` |

> **Domain note.** The β solver ships with `LX = LY = 1.0`, whereas
> `transient_heat_conduction.cpp` uses `LX = 5.0, LY = 1.0`. Comparing probe values
> between the two requires matching the domain first.

---

## Roadmap

The through-line is the linear-solver core: everything below reuses and extends the
same matrix-free Krylov machinery, moving from symmetric to non-symmetric operators as
the physics demands it.

### Phase 1 — FDM advection *(next)*

Introduce the hyperbolic term and the numerical difficulties that come with it.

- Linear advection and the advection–diffusion equation on the existing FDM grid
- Upwind, central, QUICK, and higher-order convective schemes
- False diffusion and dispersion; Péclet-number behaviour
- CFL-based timestep control alongside the existing Fourier constraint
- Implicit advection–diffusion — **the operator stops being symmetric here**, so CG no
  longer applies and **BiCG / BiCGStab** enter the picture

### Phase 2 — Pivot to FVM

Re-derive the same physics in conservative, flux-based form.

- Cell-centred finite volume with explicit face-flux evaluation
- Discrete conservation and the treatment of boundary fluxes as surface integrals
- Non-uniform and stretched grids
- Convective scheme comparison in FVM form
- Cross-validation against the Phase 1 FDM results

### Phase 3 — Full Navier–Stokes

- 2D incompressible Navier–Stokes
- Pressure–velocity coupling: SIMPLE / SIMPLEC / PISO
- Staggered grid, or collocated with Rhie–Chow interpolation
- Momentum predictor solved with **BiCGStab**, pressure-correction Poisson solved with
  **CG** (symmetric positive-definite — the natural home of the Phase 0 solver)
- Preconditioning (Jacobi, ILU) to attack the pressure equation's conditioning
- Validation against lid-driven cavity and backward-facing step benchmarks

### Cross-cutting

- Preconditioner library shared across CG and BiCGStab
- Convergence and iteration-count studies across solver/preconditioner pairs
- Grid-independence and formal order-of-accuracy verification
- Consistent VTK/Tecplot output across every solver in the repo
