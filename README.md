# CFD Solvers

Finite-volume and finite-difference CFD solvers in C++17, written from scratch. The only
dependency is the standard library. No sparse matrix is ever assembled: every operator is
applied as a stencil, and each solver is one self-contained translation unit that writes
Tecplot ASCII and legacy VTK.

They were built in order of difficulty. Steady diffusion first, then transient conduction,
then the beta-family time discretisation, then cell-centred finite volume, and finally the
2D incompressible Navier-Stokes solver in `nv.cpp`. That last one reuses most of what came
before it: the conjugate gradient core, the Robin boundary framework, the FOU/SOU/QUICK
face weights.

None of this is coursework. It overlaps a course I am taking, but nothing here is required
by it, and the code has run well ahead of it for a while.

## Solvers

| File | Governing equation | Method | Linear solver | Status |
| --- | --- | --- | --- | --- |
| [nv.cpp](nv.cpp) | 2D incompressible Navier-Stokes | Staggered MAC finite volume, Chorin projection | Matrix-free CG on the pressure Poisson equation | Validated against Ghia et al. (1982) at Re = 100 |
| [burgers_eqn.cpp](burgers_eqn.cpp) | 2D velocity Burgers (advection-diffusion) | Cell-centred FVM, additive operator split | Gauss-Seidel, four sub-solves per step | Prints a diagonal-dominance margin every step |
| [fvm_beta_formulation_diffusion.cpp](fvm_beta_formulation_diffusion.cpp) | 2D transient diffusion with a volumetric source | Cell-centred FVM, beta time family | Matrix-free CG | Prints a closing flux balance |
| [beta_formulation_transient_heat_conduction.cpp](beta_formulation_transient_heat_conduction.cpp) | 2D transient heat conduction | FDM 5-point, beta time family | Matrix-free CG | Grid fixed at 100x100 by `#define` |
| [beta_formulation_transient_heat_conduction_vector.cpp](beta_formulation_transient_heat_conduction_vector.cpp) | 2D transient heat conduction | FDM 5-point, beta time family | Matrix-free CG | `std::vector` port of the above, grid sized at runtime |
| [transient_heat_conduction.cpp](transient_heat_conduction.cpp) | 2D transient heat conduction | FDM 5-point, FTCS or backward Euler | Matrix-free CG on the implicit path | Scheme picked at compile time |
| [diffusion.cpp](diffusion.cpp) | 2D steady diffusion (Laplace) | FDM 5-point | Matrix-free CG; a Gauss-Seidel path exists but is commented out | Grid fixed at 1000x1000 |
| [beta_formulation_cuda.cu](beta_formulation_cuda.cu) | 2D transient heat conduction | FDM 5-point, beta time family | Device-side CG | CUDA port of the beta formulation, built with `nvcc` |

Only `nv.cpp` has been compared against a reference solution. Two of the others print a
diagnostic at exit. Those diagnostics check internal consistency of the discretisation,
which is a property of the code and says nothing about whether the answer is physically
right.

## The supporting solvers

### burgers_eqn.cpp

Solves the 2D velocity Burgers equations, which are the momentum equations with the
pressure term dropped, on a cell-centred collocated grid.

Each timestep is an additive operator split. The code prints it as
`u^{n+1} = uA + uD - u^n`: the convection operator and the diffusion operator each get a
`VOL/dt` transient diagonal and are solved implicitly for both velocity components, so
there are four Gauss-Seidel sub-solves per step, and `COMBINE` assembles the update.
Convection uses the FOU, SOU or QUICK face weights chosen at runtime, assembled straight
into the coefficient arrays, which puts the far-upwind terms `CWW`, `CEE`, `CSS` and
`CNN` on the stencil. Diffusion uses face conductances built on the cell-to-cell spacings.
Boundaries go through a two-component Robin table `a*dphi/dn + b*phi = c` per wall, and
corner nodes are the average of the two adjacent boundary nodes.

The shipped case is inflow `u = v = 1` on the west and south walls with zero-gradient
outflow on the east and north, started from rest. `RE` is fixed at 1, so `nu = 1.0`. The
timestep is the smaller of a CFL target of 0.20 and a fixed accuracy step of 1e-3. Output
goes to `results/burgers_%05d.dat` (Tecplot ASCII, `DATAPACKING=BLOCK`, cell-centred) and
`results/burgers_%05d.vtk`. The CLI is `burgers_eqn [fou|sou|quick] [NI NJ]`, and it
prompts for the scheme on standard input if you give it no arguments.

`CHECK_DIAG_DOMINANCE` runs every step and evaluates `aP - sum|a_nb|` over the interior.
That is the sufficient condition for Gauss-Seidel to converge. The absolute value matters
because SOU and QUICK produce off-diagonals of both signs; under FOU the test collapses
back to `aP - sum(a_nb)`. The running minimum gets reported at exit next to the total
Gauss-Seidel iteration count.

### fvm_beta_formulation_diffusion.cpp

Solves 2D transient diffusion with a volumetric source in conservative cell-centred
finite-volume form, `rho*Cp*dT/dt = div(k grad T) + q`.

The time discretisation is the beta family, weighting `beta` at the new level and
`1 - beta` at the old, so beta = 0 gives FTCS, beta = 0.5 gives Crank-Nicolson and
beta = 1 gives backward Euler. Face conductances come from the actual cell-to-cell
spacings, not a uniform `DELX`. `CALC_COEFF_TRANSIENT` builds a `1 + beta*Dsum` diagonal
that folds in the Patankar source slope `-QSLOPE*dt/(rho*Cp)`; the slope has to be
non-positive and the code aborts if it is not. A separate warning fires when the
explicit-side diagonal `AP1` goes negative, which leaves the scheme stable but lets the
solution oscillate.

The linear solve is matrix-free conjugate gradient, warm-started from the previous
timestep and skipped completely when beta is zero. Boundaries use the Robin form per wall,
applied to both the current and the previous-level field, with corners averaged.
`SET_DELTAT` takes the step from the conditional stability limit with a safety factor of
0.8 when beta < 0.5, and from `DT_ACCURACY = 1e-3` otherwise. Output is numbered Tecplot
and VTK under `results/`. The CLI is `fvm_beta_formulation_diffusion [beta] [NI NJ]`.

`CHECK_FLUX_BALANCE` closes the run. It integrates conductive flux through all four
boundaries from the wall-adjacent temperature gradients, adds the integrated volumetric
generation, and prints the net along with the net normalised by the sum of the magnitudes.
That residual goes to zero at steady state, so it tests conservation in the assembled
operator. It says nothing about how accurate the solution is.

### beta_formulation_transient_heat_conduction.cpp

The finite-difference ancestor of the solver above: 2D transient heat conduction on a
uniform Cartesian grid, same beta time family, same matrix-free CG, no source term and no
flux balance.

Coefficients are written directly in Fourier-number form, `AW = beta*Fo_x` and
`AP = 1 + 2*beta*(Fo_x + Fo_y)` on the implicit side against
`AP1 = 1 - 2*(1-beta)*(Fo_x + Fo_y)` on the explicit side. `CHECK_STABILITY` compares
`Fo_x + Fo_y` against the analytic limit `1/(2*(1-2*beta))` and aborts if it is exceeded.
A second warning flags `AP1 < 0`, where the scheme stays stable but the solution can
oscillate. At beta = 0 the update is a direct explicit sweep and CG never runs.

The CG solve is warm-started from the previous timestep. The case where the initial
residual already sits below tolerance is guarded explicitly, because entering the loop
would divide by a vanishing `p.Ap` and push NaN into the field.

The grid is fixed at 100x100 by `#define`. West, east and north are Dirichlet, the south
wall is mixed (`A_SOUTH = -10.0`), and all four go through the same Robin form. Output is
`results/temperature_%05d.dat` and `.vtk`, the latter a `STRUCTURED_GRID` with
`POINT_DATA`. The CLI is `beta_formulation_transient_heat_conduction [beta]`. It closes
with the total and per-step CG iteration counts and a wall-clock breakdown.

### beta_formulation_transient_heat_conduction_vector.cpp

A `std::vector` port of the file above. Diffing the two confirms the numerics are
untouched: every `T[j][i]` becomes `T[ID(j,i)]` over a flat `Field = std::vector<double>`,
and the physics, coefficients, boundary treatment and CG recurrences are identical.

What changed is structural. The grid is sized at runtime from `argv[2]` and `argv[3]`
instead of by `#define`, defaulting to 252x252, which gets the static arrays out of the
binary and takes the compile-time grid ceiling with them. `T` and `T_old_time` swap by
`std::swap` on the vector handles instead of copying element by element. The `RESTRICT`
macro arrives, and the hot loops read through raw pointers. Wall-clock accounting splits
I/O from compute. Console output drops from every step to every hundredth. The dead
Gauss-Seidel and steady-state routines the original was still carrying are gone. Default
`BETA` is 1.0 here and 0.5 in the array version. The CLI is `..._vector [beta] [NI NJ]`.

### transient_heat_conduction.cpp

2D transient heat conduction by finite differences, 500x100 grid, domain `LX = 5.0` by
`LY = 1.0`.

`TIME_MARCHING_SCHEME` picks at compile time between explicit FTCS, where the step is set
to `SAFETY_FO = 0.4` of the von Neumann limit, and fully implicit backward Euler at
`DT_IMPLICIT = 1e-3` solved by matrix-free CG. `CHECK_STABILITY_EXPLICIT` derives the
limit from the amplification factor of the FTCS stencil, `Fo_x + Fo_y <= 1/2`. On the
explicit path it aborts when that is violated; on the implicit path it prints the same
number as advisory.

West, east and north are Dirichlet with a mixed south wall, in the same Robin form used
everywhere else here. Output is numbered `temperature_%05d.dat` and `.vtk` written to the
current working directory, which predates the `results/` convention. There is no command
line; everything is a `#define`. On the implicit path it closes with the total and
per-step CG iteration count.

### diffusion.cpp

The oldest file here. Steady 2D diffusion, `lap(T) = 0`, finite differences, grid fixed at
1000x1000.

Both a Gauss-Seidel path and a conjugate gradient path are implemented. `main` runs CG and
the Gauss-Seidel iteration is commented out, left in as a reference point. Coefficients
are the plain 5-point stencil. The boundary condition is applied once before the solve
through the Robin form, mixed on the west wall (`a = 10`, `b = 5`) and homogeneous
Dirichlet on the other three, after which CG iterates over the interior only.

Output is `temperature_field_CG.dat` and `temperature_field_CG.vtk` in the working
directory. The VTK goes out as a `STRUCTURED_GRID` with an explicit `POINTS` list so a
stretched grid would not force a format change later. There is no command line. It closes
by printing the number of CG iterations it took.

### beta_formulation_cuda.cu

A CUDA port of the beta-formulation heat conduction solver, built with `nvcc`. Listed
here for completeness; the detail lives in the file header.

## The Navier-Stokes solver

`nv.cpp` solves the 2D incompressible Navier-Stokes equations in primitive variables on a
uniform Cartesian mesh by finite volume, with the lid-driven cavity as the default case.
It is the main piece of work in this repository and the only solver with measured
validation data behind it.

### Formulation

```
du/dt + div(u u) = -(1/rho) grad(p) + nu lap(u)
div(u) = 0
```

`rho = 1`, lid speed `U_LID = 1`, and `nu = U_LID*LX/RE` from the Reynolds number given on
the command line.

### Staggered MAC layout

The grid is staggered in the Harlow-Welch MAC arrangement: pressure at cell centres, `u`
on vertical faces, `v` on horizontal faces. These live in three separate arrays with three
index helpers, `IDU`, `IDV` and `IDP`, instead of one shared array. For `NX x NY` pressure
cells, `u` is `(NX+1) x (NY+2)`, `v` is `(NX+2) x (NY+1)`, and `p` is `(NX+2) x (NY+2)`.

Staggering is a choice, and it is the whole reason this file is simpler than it could have
been. Store `u` on the face that the pressure gradient drives it across, and both the
discrete gradient and the discrete divergence become exact differences between adjacent
stored values. The gradient at a `u` face is `p[i+1] - p[i]` over `DELX`. The divergence in
a pressure cell is the difference of the two `u` faces bounding it.

A collocated grid admits an odd-even pressure mode, where a checkerboard pressure field
produces zero gradient at every node. On this layout that mode has nowhere to live. So
there is no Rhie-Chow momentum-weighted interpolation anywhere in the file. It would have
nothing to fix.

### The three projection sub-steps

Each timestep is a Chorin projection.

The predictor advances momentum explicitly with the pressure absent,
`U* = U^n + dt*(-conv + nu*lap(U))`, reading the previous-level fields `U_old` and `V_old`
and writing into the separate arrays `US` and `VS`.

The pressure Poisson solve is implicit: `lap(p) = (rho/dt)*div(U*)`. One sign detail
matters here. The assembled operator `AP*P - sum(A_nb*P_nb)` is minus the Laplacian, which
is what makes it positive definite and therefore usable by CG, so the right-hand side is
assembled as `-(rho/dt)*div(U*)` to match. The solve itself is matrix-free conjugate
gradient on the 5-point stencil, warm-started from the previous timestep's pressure, with
relative tolerance `RTOL_CG = 1e-8` measured against the initial residual and
`CCSS = 1e-12` as an absolute floor.

Every pressure wall is Neumann in the cavity case, so the pressure is fixed only up to an
additive constant and the Poisson matrix is singular. One interior cell, `P[1][1]`, gets
pinned to zero by symmetric row-and-column deletion: its residual and search direction are
held at zero and the cell is skipped in every stencil sweep, so the removed degree of
freedom cannot creep back in. `VALIDATE_BC` spots the all-Neumann case and turns the pin
on by itself, and turns it off again if any pressure wall is given a Dirichlet condition.

The corrector then applies the gradient, `U^{n+1} = U* - (dt/rho)*grad(p)`.

### Boundary conditions

Everything goes through one generalized Robin form, `a*dphi/dn + b*phi = c`, tabulated per
wall and per component in `BC_A_W[3]` and its companions, where the three slots are `u`,
`v` and `p`. Dirichlet is `a = 0`. Neumann is `b = 0`. A mixed condition needs no separate
code path at all.

Staggering splits the application in two. A wall-normal velocity component sits exactly on
the wall, so `WALL_VALUE` assigns it directly. A wall-tangential component and the
pressure sit half a cell off the wall, so they go through a ghost cell whose value is the
affine function `GHOST_SLOPE*phi_int + GHOST_OFFSET` of the first interior value.

Pressure ghosts never get evaluated during the solve. `CALC_COEFF_P` folds `GHOST_SLOPE`
into the diagonal `AP` and `GHOST_OFFSET` into a separate boundary source `SPBC`, so the
CG stencil only ever reads interior cells and the operator stays symmetric.

`VALIDATE_BC` throws out degenerate coefficient pairs before the run starts, checking the
direct denominator `a + b*delta` and the ghost denominator `b/2 - a/delta`. It also warns
when a wall is zero-gradient in its normal component, since the domain is permeable there
and the pure-Neumann pressure problem may stop being compatible.

### Convection schemes

One set of three face weights, applied the same way at every face.

| Scheme | Upwind `wU` | Downwind `wD` | Far-upwind `wUU` |
| --- | --- | --- | --- |
| FOU | 1.000 | 0.000 | 0.000 |
| SOU | 1.500 | 0.000 | -0.500 |
| QUICK | 0.750 | 0.375 | -0.125 |

`FACE_VALUE` picks the upwind side from the sign of that face's own transporting velocity,
which on a staggered grid is an average of the two stored components straddling the face.
Upwinding is therefore decided per face, not per cell. Diffusion is central differencing in
face-conductance form and does not care which scheme is selected.

### Time stepping

`SET_DELTAT` recomputes the step every timestep as the smallest of three limits: a fixed
accuracy step `DT_ACCURACY = 0.01`, a convective limit
`CFL_TARGET/max(|u|/DELX + |v|/DELY)` at `CFL_TARGET = 0.5`, and an explicit viscous limit
`VN_SAFETY*0.5/(nu*(1/DELX^2 + 1/DELY^2))` at `VN_SAFETY = 0.8`. The result is floored at
`DT_FLOOR = 1e-8` and clipped so the run lands exactly on `TMAX`. The march stays
time-accurate and never breaks early on a steady-state test.

### Output and command line

Every write interval produces one Tecplot ASCII file and one legacy VTK file,
`results/ldc_%05d.dat` and `results/ldc_%05d.vtk`. The Tecplot file uses
`DATAPACKING=BLOCK` with `VARLOCATION` marking `U`, `V`, `P` and `VMAG` cell-centred, and
carries `STRANDID` and `SOLUTIONTIME` so the series assembles into one animating transient
zone. The VTK file is a `RECTILINEAR_GRID` with `CELL_DATA` holding pressure and velocity
magnitude as scalars and velocity as a vector. All three fields are interpolated to
pressure-cell centres so both formats share one output grid.

```
./ns2d [fou|sou|quick] [NX NY] [RE]
```

Given no arguments it prompts for the scheme on standard input and defaults to 64x64 at
Re = 100. `NX` and `NY` have to be at least 5, and `RE` has to be positive.

It closes with the mass balance and continuity block from `CHECK_MASS_BALANCE`, the total
and per-step CG iteration counts, and a wall-clock breakdown that separates I/O from
compute.

## Lid-driven cavity at Re = 100

### Implementation verification

Closing diagnostics from the headline run, 128x128 with QUICK to t = 30:

```
BALANCE : Qw = 0.0000e+00   Qe = 0.0000e+00   Qs = 0.0000e+00   Qn = 0.0000e+00
          net = 0.0000e+00   normalised = 0.0000e+00
          sum(div*VOL) = 1.4989e-19
CONTINUITY: max|div(U)| final = 2.8422e-13   worst over the run = 9.3657e-08
          RMS ||div|| = 8.3912e-15
Total CG iterations = 17362389   (average 706 per timestep)
WALLTIME: total = 648.155 s   I/O = 33.754 s   compute = 614.401 s
          (24576 steps, 6.554e+05 cell-updates/s compute-only)
```

Net volumetric flow through all four walls is identically zero and `sum(div*VOL)` is at
roundoff, which confirms the discrete divergence telescopes exactly to the wall fluxes the
way the divergence theorem says it should. `max|div|` sits at the CG tolerance, well below
discretisation error, so the projection is enforcing continuity to machine precision on
every step. The 24,576 steps needed to reach t = 30 put `dt` at about 1.22e-3, which is the
viscous stability limit at this resolution and the binding constraint on the step.

These numbers are about the operator. They show the discretisation conserves what it was
built to conserve, and they say nothing about whether the flow is physically right. That
is the next section.

### Validation against Ghia et al. (1982)

Reference: U. Ghia, K. N. Ghia and C. T. Shin, *High-Re Solutions for Incompressible Flow
Using the Navier-Stokes Equations and a Multigrid Method*, Journal of Computational
Physics **48** (1982) 387-411. Tables I and II give centreline velocities on a 129x129
grid, and Table V gives the vortex properties.

Every error metric below covers the 30 interior tabulated stations. The two wall stations
on each profile are left out. ParaView's `CellDatatoPointData` filter builds a wall node by
averaging the adjacent cells, so it cannot reproduce a boundary condition and reports a
wall velocity that is neither the imposed value nor a real solver error. Exporting raw
cell-centred data with the exact wall values appended kills the artefact outright and
leaves the interior errors unchanged, which is how I know the exclusion is dropping a
filter artefact and not an inconvenient result.

Integral quantities from the headline run, against Table V:

| Quantity | Present | Ghia et al. | Deviation |
| --- | --- | --- | --- |
| psi_min, primary vortex | -0.103426 | -0.103423 | 0.003% |
| Primary vortex centre (x, y) | (0.6133, 0.7383) | (0.6172, 0.7344) | within one cell |
| Vorticity at vortex centre | 3.14867 | 3.16646 | 0.56% |
| Bottom-left eddy psi_max | 1.8397e-6 | 1.7488e-6 | 5.2% |
| Bottom-left eddy centre | (0.0352, 0.0352) | (0.0313, 0.0391) | within one cell |
| Bottom-right eddy psi_max | 1.2755e-5 | 1.2537e-5 | 1.7% |
| Bottom-right eddy centre | (0.9414, 0.0586) | (0.9453, 0.0625) | within one cell |

The stream function comes from integrating `psi = int_0^y u dy` up from the floor.
Integrating the other path, `psi = -int_0^x v dx`, agrees to 4.9e-4, which bounds the
precision of the psi_min figure.

Centreline velocities:

| Metric | u | v | Combined |
| --- | --- | --- | --- |
| Max absolute deviation | 0.00497 | 0.00910 | 0.00910 |
| RMS deviation | 0.00243 | 0.00498 | 0.00392 |
| Relative L2 norm | | | 1.27% |
| Mean signed deviation | | | -0.00023 |

Extrema: u_min = -0.21381 at y = 0.4570, against Ghia -0.21090 at 0.4531;
v_min = -0.25362 at x = 0.8086, against Ghia -0.24533 at 0.8047;
v_max = 0.17938 at x = 0.2383, against Ghia 0.17527 at 0.2344.

### Scheme and grid comparison

| Grid | Scheme | u max | u RMS | v max | v RMS | Combined rel. L2 | u_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64x64 | FOU | 0.01124 | 0.00619 | 0.00871 | 0.00426 | | -0.19992 |
| 128x128 | FOU | 0.00623 | 0.00277 | 0.00650 | 0.00322 | 0.97% | -0.20645 |
| 64x64 | QUICK | 0.00441 | 0.00207 | 0.00756 | 0.00388 | 1.01% | -0.21229 |
| 128x128 | QUICK | 0.00497 | 0.00243 | 0.00910 | 0.00498 | 1.27% | -0.21381 |

Ghia u_min is -0.21090.

FOU sits systematically below the reference, mean signed deviation -0.00091 at 128x128.
That is the signature of first-order numerical diffusion damping the primary vortex. QUICK
takes that damping away and lands slightly above the reference instead, mean signed
deviation between -0.00023 and -0.00030, so its bias is far better balanced even at the
grids where its maximum deviation is larger. FOU's slightly smaller aggregate at 128x128 is
a sign flip in v. It is not better accuracy.

## Limitations

1. No order of accuracy has been demonstrated. QUICK's error does not fall from 64x64 to
   128x128; the relative L2 goes from 1.01% to 1.27%, so the observed order is roughly
   zero. A second-order scheme should give a fourfold reduction. The tables support the
   agreement. They do not support a convergence rate.

2. The likely cause is first-order wall treatment. The far-upwind stencil nodes `uWW`,
   `uEE`, `uSS` and `uNN`, and their v counterparts, fall back to the nearest available
   node wherever the stencil would run off the array. SOU and QUICK therefore degrade to
   first order on every face next to a wall, which caps the global order at one no matter
   how accurate the interior is. The largest v deviations sit at x = 0.8594 and 0.9063,
   inside the right-hand boundary layer, which fits. This is unconfirmed. Splitting the
   error into near-wall and core stations, or running 256x256, would settle it.

3. QUICK overshoots v systematically. Every station in the negative lobe comes out 0.003
   to 0.009 too negative, every station in the positive lobe 0.002 to 0.004 too positive,
   at both grids. The vortex is uniformly a little too strong. This is not transient; the
   t = 30 run is converged.

4. Run provenance needs regenerating. Several exported CSVs were renamed after the fact,
   and two 128x128 QUICK exports differ by 1.18e-3, which means they came from different
   runs or different times. Every validation dataset should be regenerated from one build
   with a scripted, recorded command line before these numbers go anywhere.

5. Transcription bugs were found and fixed. A hand-transcribed copy contained an
   unallocated `PDIR` vector, swapped index-helper parameters, the predictor result never
   reaching the pressure solve, `WALL_VALUE` where `GHOST_VALUE` was required on the u
   tangential walls, a nonzero normal-velocity boundary condition at the lid, and several
   loop-bound errors. The committed source still needs confirming as the corrected
   version, and the published numbers still need confirming to reproduce from it.

## Building and running

One source file, either compiler.

```bash
g++ -O2 -std=c++17 -Wall -Wextra nv.cpp -o ns2d
./ns2d quick 128 128 100
```

That build should emit no warnings. Under MSVC:

```
cl /O2 /std:c++17 /EHsc nv.cpp /Fe:ns2d.exe
.\ns2d.exe quick 128 128 100
```

A sanitizer build for regression checking:

```bash
g++ -O1 -g -std=c++17 -fsanitize=address,undefined nv.cpp -o ns2d_dbg
./ns2d_dbg sou 16 12 100
```

Output lands in `results/` relative to the current working directory, not the binary, so
run from the project root. The other solvers build the same way with the source name
swapped. `beta_formulation_cuda.cu` wants `nvcc` and carries its build line in the file
header. Under VS Code, `.vscode/tasks.json` has `cl.exe` tasks that send every artefact to
`build/` and set the working directory to the project root.

### Post-processing

Open the numbered `.vtk` series in ParaView, apply `CellDatatoPointData`, then
`Plot Over Line` from (0.5, 0, 0) to (0.5, 1, 0) for u against y, and from (0, 0.5, 0) to
(1, 0.5, 0) for v against x, and export as CSV.

Better: export the raw `CELL_DATA` and average the two columns straddling the centreline.
That sidesteps the wall artefact described above completely.
