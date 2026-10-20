# SENS: reproducible reference implementation for fourth-order structural mechanics

This public folder contains a runnable implementation of the Sobolev-Enforced Neural Solver (SENS), a conventional strong-form PINN baseline, and a finite-difference reference for a clamped Euler–Bernoulli beam. It includes the core source, experiment script, configuration, random seeds, dependency list, and environment snapshot.

**Provenance and reproducibility note.** The original historical training scripts used to produce earlier manuscript figures were not present among the files available for this revision. `src/sens_core.py` is therefore an independently reconstructed reference implementation based on the mathematical method described in the manuscript; it is not represented as the historical script. The script computes its own tables and figures from solver outputs, but exact agreement with earlier reported numerical values should not be claimed unless those values are regenerated from this version and the manuscript is updated accordingly. This distinction is included for transparent reproducibility.

## Problem and benchmark

The dimensionless clamped Euler–Bernoulli beam equation is

\[ EI\,u''''(x)=q(x),\qquad x\in(0,1),\qquad EI=1, \]

with clamped boundary conditions

\[u(0)=u'(0)=u(1)=u'(1)=0.\]

For the main sinusoidal load, \(q(x)=\sin(\pi x)\), the exact solution is

\[u(x)=\frac{\sin(\pi x)}{\pi^4}+\frac{x^2-x}{\pi^3}.\]

A uniform-load case, \(q(x)=1\), is also included, with exact solution \(u(x)=x^2(1-x)^2/24\). A narrow Gaussian load is used for a controlled load-shape test and is compared with a high-resolution finite-difference reference.

## SENS formulation implemented here

The trial solution enforces the four clamped boundary conditions analytically:

\[u_\theta(x)=x^2(1-x)^2N_\theta(x).\]

The weak residual is projected onto clamped polynomial-modal test functions

\[\phi_k(x)=x^2(1-x)^2P_k(2x-1),\]

where \(P_k\) are Legendre polynomials. The projected residual coefficients are

\[r_k=\int_0^1\left(u_\theta''\phi_k''-q\phi_k\right)\,dx.\]

The minimized objective is the mean normalized squared weak residual plus an energy-balance consistency term. The conventional PINN uses the pointwise strong residual \(u_\theta''''-q\) and soft penalties on endpoint displacement and slope.

The default configuration uses four hidden tanh layers, width 64, Adam with learning rate 0.001, 1,200 epochs, seeds 17, 29, and 41, 10 spectral test modes, and refinement sizes from 25 to 1600. All settings are available in `config.json`.

## Install and run

Python 3.10 or newer is recommended. Install the pinned dependencies in a clean environment:

```bash
python -m pip install -r requirements.txt
```

Then run from this directory:

```bash
python reproduce_experiments.py
```

The script writes CSV tables, run metadata, training-loss records, and Figures 1–9 into `results/`. The numeric values are calculated by the solvers, not hard-coded in the plotting script. Output files are generated on execution rather than committed in this repository snapshot. Package versions recorded in `environment.txt` describe the editing environment; they are not represented as the environment of the historical runs.

Key files:

- `src/sens_core.py`: SENS loss, boundary-augmented ansatz, PINN baseline, error metrics, and finite-difference solver.
- `reproduce_experiments.py`: end-to-end experiments and figure/table generation.
- `config.json`: architecture, optimizer, seeds, modes, noise levels, and refinement levels.
- `requirements.txt`: pinned Python dependencies.
- `environment.txt`: environment snapshot and provenance note.

## Interpretation and scope

Finite-range slopes are descriptive empirical summaries, not formal convergence-order proofs. SENS and finite-difference errors are reported separately; this archive does not claim that SENS is equally accurate or computationally cheaper than finite differences for this one-dimensional benchmark. The noise test perturbs a synthetic load and is not field-sensor validation. Inverse stiffness identification remains preliminary and is not quantitatively validated by this package.

## License and citation

The code is released under the MIT License. Please cite the associated manuscript and link to this public code archive when reusing the implementation. No DOI is claimed unless an archival deposit is made separately.
