# SENS: Reproducible Experiments for Fourth-Order Structural Mechanics

This repository folder contains the runnable implementation and outputs associated with the revised manuscript **“Sobolev-Enforced Neural Solvers for Fourth-Order Structural Mechanics: A Lax–Richtmyer Convergence Framework.”** The scripts train the SENS model and a conventional strong-form PINN, solve a finite-difference benchmark, calculate error tables, and generate the manuscript figures.

## Problem and benchmark

The primary problem is the dimensionless clamped Euler–Bernoulli beam equation

\[
EI\,u''''(x)=q(x),\qquad x\in(0,1),\qquad EI=1,
\]

with clamped boundary conditions

\[
u(0)=u'(0)=u(1)=u'(1)=0.
\]

For the main sinusoidal load, \(q(x)=\sin(\pi x)\), the exact solution is

\[
u(x)=\frac{\sin(\pi x)}{\pi^4}+\frac{x^2-x}{\pi^3}.
\]

A uniform-load case, \(q(x)=1\), is also included, with exact solution \(u(x)=x^2(1-x)^2/24\). The Gaussian load is used for a controlled load-shape test and is compared with a high-resolution finite-difference reference.

## SENS formulation implemented here

The trial solution enforces the four clamped boundary conditions analytically:

\[
u_\theta(x)=x^2(1-x)^2N_\theta(x).
\]

The weak residual is projected onto clamped polynomial-modal test functions

\[
\phi_k(x)=x^2(1-x)^2P_k(2x-1),
\]

where \(P_k\) are Legendre polynomials. The projected residual coefficients are

\[
r_k=\int_0^1\left(u_\theta''\phi_k''-q\phi_k\right)\,dx.
\]

The minimized objective is the mean normalized squared weak residual plus a small energy-balance consistency term. The exact settings are stored in `config.json`; the current energy-consistency weight is `1e-4`. The test functions satisfy the clamped essential boundary conditions; they are polynomial-modal functions, not sine eigenfunctions of the clamped biharmonic operator.

The conventional PINN uses the pointwise strong residual \(u_\theta''''-q\) and soft penalties on endpoint displacement and slope. Both networks use four hidden tanh layers, width 64, Adam, learning rate 0.001, and 1,200 training epochs for the main comparison. The three comparison seeds are 17, 29, and 41.

## Requirements

Python 3.10 or newer is recommended. Install dependencies in a clean environment:

```bash
python -m pip install -r requirements.txt
```

The solver uses CPU and deterministic PyTorch operations. Exact installed package versions used when generating the committed results are listed in `environment.txt`.

## Reproduce all tables and figures

From this directory, run:

```bash
python reproduce_experiments.py
```

The script writes CSV tables, run metadata, training-loss records, and Figures 1–9 into `results/`. The numerical values are calculated by the solvers; they are not manually entered in the plotting script. Results are generated from the parameter values in `config.json`.

The main files are:

- `src/sens_core.py`: SENS loss, boundary-augmented neural ansatz, PINN baseline, error metrics, and finite-difference solver.
- `reproduce_experiments.py`: end-to-end experiments and figure/table generation.
- `config.json`: architecture, optimizer, seeds, quadrature/test modes, noise levels, and refinement levels.
- `results/Table1_PerSeedComparison.csv`: per-seed main comparison.
- `results/Table1_MainComparison_Summary.csv`: mean and standard deviation across the three specified seeds.
- `results/Table2_Convergence_SENS_vs_FD.csv`: collocation/grid-refinement errors and descriptive log–log slopes.
- `results/Table3_Energy_Ablation.csv`: energy-term ablation diagnostic.
- `results/Table4_Noise_Sensitivity.csv`: controlled Gaussian load-noise sensitivity.
- `results/Table5_Load_Shape_Generalization.csv`: uniform, sinusoidal, and Gaussian load comparisons.

## Interpretation and scope

The manuscript distinguishes the conditional theoretical error framework from finite-resolution experiments. The fixed-architecture SENS refinement results are reported as measured; a monotonic or asymptotic SENS convergence order is not inferred when the computed errors plateau or vary. The finite-difference errors are reported separately and should not be interpreted as evidence that SENS is equally accurate or computationally cheaper.

The noise test perturbs a synthetic load and is not a field-sensor validation. Stiffness identification/inverse mechanics remains preliminary and is not claimed as a quantitatively validated result by this archive. The included scripts do not substantiate generalization to arbitrary real-world structural-health-monitoring data.

## License and citation

The files in this code archive are released under the MIT License. Please cite the manuscript and link to this public code location when reusing the implementation. A DOI is not claimed unless an archival deposit is made separately.