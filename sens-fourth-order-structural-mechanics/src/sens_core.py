"""Reference implementation of a Sobolev-Enforced Neural Solver (SENS).

This is an independently reconstructed, runnable reference implementation based
on the revised manuscript's mathematical description. It is not represented as
the unavailable historical training script that generated earlier figures.
"""
from __future__ import annotations
import math, random
import numpy as np
import torch
from torch import nn
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import Legendre

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load(x, kind="sinusoidal", noise=0.0, seed=0):
    """Dimensionless load; noise is Gaussian with sigma relative to peak load."""
    if isinstance(x, torch.Tensor):
        if kind == "sinusoidal":
            q = torch.sin(math.pi*x)
        elif kind == "uniform":
            q = torch.ones_like(x)
        elif kind == "gaussian":
            q = torch.exp(-((x-0.5)/0.08)**2)
        else:
            raise ValueError(f"Unknown load kind: {kind}")
        if noise:
            e = np.random.default_rng(seed).normal(0.0, noise, size=tuple(x.shape))
            q = q + torch.as_tensor(e, dtype=x.dtype, device=x.device)
        return q
    xx = np.asarray(x)
    if kind == "sinusoidal":
        q = np.sin(math.pi*xx)
    elif kind == "uniform":
        q = np.ones_like(xx)
    elif kind == "gaussian":
        q = np.exp(-((xx-0.5)/0.08)**2)
    else:
        raise ValueError(f"Unknown load kind: {kind}")
    if noise:
        q = q + np.random.default_rng(seed).normal(0.0, noise, size=xx.shape)
    return q


def exact_solution(x: np.ndarray, kind: str = "sinusoidal") -> np.ndarray:
    """Exact clamped-beam solutions for EI=1 and two load profiles."""
    x = np.asarray(x, dtype=float)
    if kind == "sinusoidal":
        return np.sin(np.pi*x)/np.pi**4 + (x*x-x)/np.pi**3
    if kind == "uniform":
        return x*x*(1-x)*(1-x)/24.0
    raise ValueError(f"No closed-form exact solution for load kind: {kind}")


def make_mlp(width: int = 64, depth: int = 4, seed: int = 17) -> nn.Module:
    seed_all(seed)
    layers: list[nn.Module] = [nn.Linear(1, width, dtype=DTYPE), nn.Tanh()]
    for _ in range(depth-1):
        layers.extend([nn.Linear(width, width, dtype=DTYPE), nn.Tanh()])
    layers.append(nn.Linear(width, 1, dtype=DTYPE))
    return nn.Sequential(*layers).to(DEVICE)


def d_dx(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    out = y
    for _ in range(order):
        out = torch.autograd.grad(
            out, x, torch.ones_like(out), create_graph=True, retain_graph=True
        )[0]
    return out


def quadrature_and_tests(n: int, n_modes: int):
    """Gauss-Legendre quadrature and clamped polynomial-modal test functions."""
    z, w = np.polynomial.legendre.leggauss(n)
    x = (z+1.0)/2.0
    w = w/2.0
    g = x*x*(1-x)*(1-x)
    gp = 2*x - 6*x*x + 4*x*x*x
    gpp = 2 - 12*x + 12*x*x
    phi, phi2 = [], []
    zz = 2*x-1
    for k in range(n_modes):
        p = Legendre.basis(k)
        pk = p(zz)
        pk1 = 2*p.deriv(1)(zz)
        pk2 = 4*p.deriv(2)(zz)
        phi.append(g*pk)
        phi2.append(gpp*pk + 2*gp*pk1 + g*pk2)
    phi = np.asarray(phi).T
    phi2 = np.asarray(phi2).T
    norm2 = np.sum(w[:, None]*(phi2**2 + phi**2), axis=0)
    return (
        torch.tensor(x[:, None], dtype=DTYPE, device=DEVICE),
        torch.tensor(w[:, None], dtype=DTYPE, device=DEVICE),
        torch.tensor(phi, dtype=DTYPE, device=DEVICE),
        torch.tensor(phi2, dtype=DTYPE, device=DEVICE),
        torch.tensor(norm2, dtype=DTYPE, device=DEVICE),
    )


def train_sens(n=100, seed=17, epochs=1200, width=64, depth=4,
               modes=None, noise=0.0, kind="sinusoidal", lr=1e-3,
               energy_weight=1e-4):
    """Train SENS: exact clamped ansatz, projected weak residual, energy consistency."""
    seed_all(seed)
    nm = min(18, max(6, int(math.sqrt(n)))) if modes is None else int(modes)
    x, w, phi, phi2, norm2 = quadrature_and_tests(n, nm)
    q = load(x, kind=kind, noise=noise, seed=seed+1000)
    model = make_mlp(width=width, depth=depth, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        xx = x.detach().clone().requires_grad_(True)
        g = xx**2*(1-xx)**2
        u = g*model(xx)
        u2 = d_dx(u, xx, 2)
        # Weak form for EI=1: integral(u'' phi_k'' - q phi_k) dx = 0.
        r = torch.sum(w*(u2*phi2 - q*phi), dim=0)
        lweak = torch.mean(r*r/norm2)
        internal = torch.sum(w*u2*u2)
        external = torch.sum(w*q*u)
        escale = torch.clamp(torch.abs(external), min=1e-3)
        lenergy = ((internal-external)/escale)**2
        loss = lweak + energy_weight*lenergy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        opt.step()
        if (epoch+1) % 100 == 0 or epoch == 0:
            history.append((epoch+1, float(loss.detach()),
                            float(lweak.detach()), float(lenergy.detach())))
    return model, np.asarray(history, dtype=float)


def train_pinn(n=100, seed=17, epochs=1200, width=64, depth=4, lr=1e-3,
               boundary_weight=100.0, kind="sinusoidal"):
    """Conventional strong-form PINN with soft clamped-boundary penalties."""
    seed_all(seed)
    model = make_mlp(width=width, depth=depth, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed+2000)
    hist = []
    for epoch in range(epochs):
        x_np = rng.uniform(0.0, 1.0, size=(n, 1))
        x = torch.tensor(x_np, dtype=DTYPE, device=DEVICE, requires_grad=True)
        u = model(x)
        u4 = d_dx(u, x, 4)
        q = load(x, kind=kind)
        lres = torch.mean((u4-q)**2)
        xb = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
        ub = model(xb)
        ub1 = d_dx(ub, xb, 1)
        lbc = torch.mean(ub**2) + torch.mean(ub1**2)
        loss = lres + boundary_weight*lbc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        opt.step()
        if (epoch+1) % 100 == 0 or epoch == 0:
            hist.append((epoch+1, float(loss.detach()),
                         float(lres.detach()), float(lbc.detach())))
    return model, np.asarray(hist, dtype=float)


def errors(model: nn.Module, xeval: np.ndarray, bc_augmented: bool,
           kind: str = "sinusoidal", reference=None, reference_x=None) -> dict[str, float]:
    xt = torch.tensor(np.asarray(xeval)[:, None], dtype=DTYPE, device=DEVICE, requires_grad=True)
    pred = model(xt)
    if bc_augmented:
        pred = xt**2*(1-xt)**2*pred
    p2 = d_dx(pred, xt, 2)
    xx = np.asarray(xeval)
    if kind in ("sinusoidal", "uniform"):
        tru = exact_solution(xx, kind)
        if kind == "sinusoidal":
            tru2 = -np.sin(np.pi*xx)/np.pi**2 + 2/np.pi**3
        else:
            tru2 = (2 - 12*xx + 12*xx*xx)/24.0
    else:
        if reference is None or reference_x is None:
            raise ValueError("A numerical reference is required for the Gaussian load.")
        tru = np.interp(xx, reference_x, reference)
        tru2_ref = np.gradient(np.gradient(reference, reference_x), reference_x)
        tru2 = np.interp(xx, reference_x, tru2_ref)
    pe = pred.detach().cpu().numpy().ravel()
    p2e = p2.detach().cpu().numpy().ravel()
    return {
        "Linf": float(np.max(np.abs(pe-tru))),
        "L2": float(np.sqrt(np.mean((pe-tru)**2))),
        "H2_seminorm_RMS": float(np.sqrt(np.mean((p2e-tru2)**2))),
    }


def finite_difference(n: int, kind: str = "sinusoidal") -> tuple[np.ndarray, np.ndarray]:
    """Clamped 1-D finite-difference reference using 9-point derivative weights."""
    if n < 10:
        raise ValueError("Use at least 10 intervals for the 9-point stencil.")
    x = np.linspace(0.0, 1.0, n+1)
    h = 1.0/n
    A = lil_matrix((n+1, n+1), dtype=float)
    b = np.zeros(n+1)
    # Essential displacement conditions and one-sided fourth-order slope conditions.
    A[0, 0] = 1.0
    A[1, 0:5] = np.array([-25, 48, -36, 16, -3])/(12*h)
    A[n-1, n-4:n+1] = np.array([3, -16, 36, -48, 25])/(12*h)
    A[n, n] = 1.0
    for i in range(2, n-1):
        start = min(max(i-4, 0), n-8)
        inds = np.arange(start, start+9)
        offsets = (x[inds]-x[i])/h
        V = np.vstack([offsets**m for m in range(9)])
        rhs = np.zeros(9)
        rhs[4] = math.factorial(4)
        weights = np.linalg.solve(V, rhs)/(h**4)
        A[i, inds] = weights
        b[i] = load(np.array([x[i]]), kind=kind)[0]
    # Equilibrate rows because derivative and essential-boundary equations have different scales.
    A[1, :] = A[1, :]*h
    A[n-1, :] = A[n-1, :]*h
    for i in range(2, n-1):
        A[i, :] = A[i, :]*(h**4)
        b[i] = b[i]*(h**4)
    u = spsolve(csr_matrix(A), b)
    return x, np.asarray(u)
