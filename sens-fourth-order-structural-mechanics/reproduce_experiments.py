"""Reproduce the computational experiments for the revised SENS manuscript.

Run from this directory:
    python reproduce_experiments.py

All numerical tables and figures are generated from solver outputs. No
reported error values are hard-coded. Settings are recorded in config.json.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from src.sens_core import train_sens, train_pinn, exact_solution, finite_difference, errors

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"results")
os.makedirs(OUT,exist_ok=True)
with open(os.path.join(HERE,"config.json"),encoding="utf-8") as f: CFG=json.load(f)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,name),dpi=240,bbox_inches="tight")
    plt.close()


def fit_slope(n, err):
    n=np.asarray(n,dtype=float); err=np.asarray(err,dtype=float)
    good=np.isfinite(err)&(err>0)
    if good.sum()<2: return float("nan")
    return float(np.polyfit(np.log(n[good]),np.log(err[good]),1)[0])


def main():
    cfg=CFG; seeds=cfg["seeds"]
    n0=int(cfg["comparison_collocation_points"]); e_main=int(cfg["epochs_main"])
    width=int(cfg["network_width"]); depth=int(cfg["network_hidden_layers"])
    evalx=np.linspace(0.,1.,int(cfg["evaluation_points"]))
    exact=exact_solution(evalx,"sinusoidal")
    run_rows=[]; sens_hist_ref=None; pinn_hist_ref=None; sens_model_ref=None; pinn_model_ref=None
    print("[1/6] Matched-architecture SPINN/SENS runs",flush=True)
    for seed in seeds:
        tic=time.time()
        sm,sh=train_sens(n=n0,seed=int(seed),epochs=e_main,width=width,depth=depth,
                         modes=int(cfg["spectral_test_modes"]),kind="sinusoidal",energy_weight=float(cfg["energy_consistency_weight"]))
        se=errors(sm,evalx,True,kind="sinusoidal")
        pd.DataFrame(sh,columns=["epoch","loss","weak_residual_loss","energy_balance_loss"]).to_csv(
            os.path.join(OUT,f"loss_SENS_seed{seed}.csv"),index=False)
        if int(seed)==int(seeds[0]): sens_hist_ref=sh; sens_model_ref=sm
        print(f"  seed {seed}: SENS done in {time.time()-tic:.1f}s",flush=True)
        tic=time.time()
        pm,ph=train_pinn(n=n0,seed=int(seed),epochs=e_main,width=width,depth=depth,
                         boundary_weight=float(cfg["pinn_boundary_weight"]),kind="sinusoidal")
        pe=errors(pm,evalx,False,kind="sinusoidal")
        pd.DataFrame(ph,columns=["epoch","loss","strong_residual_loss","boundary_loss"]).to_csv(
            os.path.join(OUT,f"loss_PINN_seed{seed}.csv"),index=False)
        if int(seed)==int(seeds[0]): pinn_hist_ref=ph; pinn_model_ref=pm
        print(f"  seed {seed}: SPINN done in {time.time()-tic:.1f}s",flush=True)
        for name,metric in [("SENS",se),("SPINN",pe)]:
            run_rows.append({"method":name,"seed":int(seed),"N":n0,**metric})
    per_seed=pd.DataFrame(run_rows)
    per_seed.to_csv(os.path.join(OUT,"Table1_PerSeedComparison.csv"),index=False)
    summary=per_seed.groupby("method").agg({k:["mean","std"] for k in ["Linf","L2","H2_seminorm_RMS"]})
    summary.columns=[f"{a}_{b}" for a,b in summary.columns]
    summary.reset_index().to_csv(os.path.join(OUT,"Table1_MainComparison_Summary.csv"),index=False)

    fd_x,fd_u=finite_difference(int(cfg["fd_reference_intervals"]),"sinusoidal")
    xt=torch.tensor(evalx[:,None],dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        s_pred=(xt**2*(1-xt)**2*sens_model_ref(xt)).cpu().numpy().ravel()
        p_pred=pinn_model_ref(xt).cpu().numpy().ravel()
    fd_ref=np.interp(evalx,fd_x,fd_u)
    plt.figure(figsize=(7.2,4.6))
    plt.plot(evalx,exact,'k-',lw=2,label="Exact solution")
    plt.plot(evalx,p_pred,'--',lw=1.6,label="Standard PINN")
    plt.plot(evalx,s_pred,'-.',lw=1.6,label="SENS")
    plt.plot(evalx,fd_ref,':',lw=1.4,label="FD reference")
    plt.xlabel("x"); plt.ylabel("Deflection u(x)"); plt.title(f"Clamped beam solution comparison (N={n0})"); plt.legend()
    save_fig("Figure1_Solution_Comparison.png")

    plt.figure(figsize=(7.2,4.6))
    plt.semilogy(evalx,np.maximum(np.abs(p_pred-exact),1e-14),label="Standard PINN")
    plt.semilogy(evalx,np.maximum(np.abs(s_pred-exact),1e-14),label="SENS")
    plt.xlabel("x"); plt.ylabel("Absolute pointwise error"); plt.title("Pointwise solution error")
    plt.legend(); save_fig("Figure2_Pointwise_Error.png")

    print("[2/6] Collocation-refinement study",flush=True)
    conv=[]
    for n in cfg["refinement_points"]:
        n=int(n)
        if n==n0:
            sm=sens_model_ref
        else:
            tic=time.time()
            sm,_=train_sens(n=n,seed=int(seeds[0]),epochs=int(cfg["epochs_refinement"]),width=width,depth=depth,
                            kind="sinusoidal",energy_weight=float(cfg["energy_consistency_weight"]))
            print(f"  SENS N={n} done in {time.time()-tic:.1f}s",flush=True)
        met=errors(sm,evalx,True,kind="sinusoidal")
        fx,fu=finite_difference(n,"sinusoidal")
        fd_err=float(np.max(np.abs(fu-exact_solution(fx,"sinusoidal"))))
        conv.append({"N":n,"h":1.0/n,"SENS_Linf":met["Linf"],"SENS_L2":met["L2"],
                     "SENS_H2_seminorm_RMS":met["H2_seminorm_RMS"],"FD_Linf":fd_err})
    conv_df=pd.DataFrame(conv).sort_values("N")
    conv_df["SENS_fitted_loglog_slope"] = fit_slope(conv_df.N,conv_df.SENS_Linf)
    conv_df["FD_fitted_loglog_slope"] = fit_slope(conv_df.N,conv_df.FD_Linf)
    conv_df.to_csv(os.path.join(OUT,"Table2_Convergence_SENS_vs_FD.csv"),index=False)

    # Figure 3: seed-based comparison of two complementary error metrics.
    srow=summary.reset_index().set_index("method")
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.3))
    for ax,metric,title in [(axes[0],"Linf","Maximum pointwise error"),(axes[1],"H2_seminorm_RMS","RMS second-derivative error")]:
        means=[srow.loc["SPINN",f"{metric}_mean"],srow.loc["SENS",f"{metric}_mean"]]
        stds=[srow.loc["SPINN",f"{metric}_std"],srow.loc["SENS",f"{metric}_std"]]
        ax.bar(["Standard PINN","SENS"],means,yerr=stds,capsize=4)
        ax.set_title(title); ax.set_ylabel("Error"); ax.tick_params(axis="x",rotation=12)
    fig.suptitle(f"Three-seed error comparison (N={n0})")
    save_fig("Figure3_Error_Metric_Comparison.png")

    plt.figure(figsize=(7.2,4.6))
    plt.loglog(conv_df.N,conv_df.SENS_Linf,'o-',label="SENS L∞ error")
    plt.xlabel("Collocation points N"); plt.ylabel("Maximum absolute error")
    plt.title("SENS error under collocation refinement"); plt.grid(True,which="both",alpha=.25); plt.legend()
    save_fig("Figure6_Extended_Convergence.png")

    plt.figure(figsize=(7.2,4.6))
    plt.semilogy(sens_hist_ref[:,0],np.maximum(sens_hist_ref[:,1],1e-16),label="SENS total loss")
    plt.semilogy(pinn_hist_ref[:,0],np.maximum(pinn_hist_ref[:,1],1e-16),label="Standard PINN total loss")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Training-loss histories (first seed)"); plt.legend()
    save_fig("Figure4_Training_Loss.png")

    print("[3/6] Energy-term ablation",flush=True)
    ab_model,_=train_sens(n=n0,seed=int(seeds[0]),epochs=e_main,width=width,depth=depth,
                          modes=int(cfg["spectral_test_modes"]),kind="sinusoidal",energy_weight=0.0)
    ab_full=errors(sens_model_ref,evalx,True,kind="sinusoidal")
    ab_removed=errors(ab_model,evalx,True,kind="sinusoidal")
    pd.DataFrame([{"configuration":"Full SENS",**ab_full},{"configuration":"Without energy term",**ab_removed}]).to_csv(
        os.path.join(OUT,"Table3_Energy_Ablation.csv"),index=False)
    plt.figure(figsize=(6.8,4.5))
    xx=np.arange(2); vals=[ab_full["Linf"],ab_removed["Linf"]]
    plt.bar(xx,vals); plt.xticks(xx,["Full SENS","No energy term"]); plt.ylabel("L∞ error")
    plt.title("Ablation of energy-consistency term"); save_fig("Figure5_Energy_Ablation.png")

    print("[4/6] Synthetic load-noise sensitivity",flush=True)
    noise_rows=[]
    for level in cfg["noise_levels"]:
        level=float(level)
        if level==0.0: sm=sens_model_ref
        else:
            sm,_=train_sens(n=n0,seed=int(seeds[0]),epochs=int(cfg["epochs_noise"]),width=width,depth=depth,
                            modes=int(cfg["spectral_test_modes"]),kind="sinusoidal",noise=level,energy_weight=float(cfg["energy_consistency_weight"]))
        met=errors(sm,evalx,True,kind="sinusoidal")
        noise_rows.append({"noise_fraction_of_max_load":level,**met,
                           "relative_Linf_to_peak_displacement":met["Linf"]/max(abs(exact))})
        print(f"  noise={level:.2f} complete",flush=True)
    noise_df=pd.DataFrame(noise_rows)
    noise_df.to_csv(os.path.join(OUT,"Table4_Noise_Sensitivity.csv"),index=False)
    plt.figure(figsize=(7.2,4.6))
    plt.plot(100*noise_df.noise_fraction_of_max_load,100*noise_df.relative_Linf_to_peak_displacement,'o-',label="Relative L∞ error")
    plt.xlabel("Added Gaussian load-noise standard deviation (% of max load)")
    plt.ylabel("L∞ error / peak clean displacement (%)")
    plt.title("Robustness to synthetic load noise"); plt.grid(True,alpha=.25); plt.legend()
    save_fig("Figure9_Noise_Robustness.png")

    print("[5/6] Controlled load-shape generalization",flush=True)
    load_rows=[]
    for kind in ["uniform","gaussian"]:
        tic=time.time()
        sm,_=train_sens(n=n0,seed=int(seeds[0]),epochs=int(cfg["epochs_load_cases"]),width=width,depth=depth,
                        modes=int(cfg["spectral_test_modes"]),kind=kind,energy_weight=float(cfg["energy_consistency_weight"]))
        ref_x,ref_u=finite_difference(int(cfg["fd_reference_intervals"]),kind)
        met=errors(sm,evalx,True,kind=kind,reference=ref_u,reference_x=ref_x)
        load_rows.append({"load_kind":kind,**met})
        print(f"  load {kind} done in {time.time()-tic:.1f}s",flush=True)
    load_rows.append({"load_kind":"sinusoidal",**errors(sens_model_ref,evalx,True,kind="sinusoidal")})
    load_df=pd.DataFrame(load_rows)
    load_df.to_csv(os.path.join(OUT,"Table5_Load_Shape_Generalization.csv"),index=False)
    plt.figure(figsize=(7.2,4.6))
    plt.bar(load_df.load_kind,load_df.Linf)
    plt.ylabel("Maximum absolute error (analytic/high-resolution FD reference)")
    plt.title("SENS under three smooth load profiles")
    save_fig("Figure7_Load_Shape_Generalization.png")

    plt.figure(figsize=(7.2,4.6))
    plt.loglog(conv_df.N,conv_df.SENS_Linf,'o-',label="SENS (collocation refinement)")
    plt.loglog(conv_df.N,conv_df.FD_Linf,'d-',label="Finite differences (grid refinement)")
    plt.xlabel("N"); plt.ylabel("L∞ error"); plt.title("SENS and finite-difference error decay")
    plt.grid(True,which="both",alpha=.25); plt.legend()
    save_fig("Figure8_SENS_vs_FD.png")

    print("[6/6] Writing metadata",flush=True)
    metadata={"configuration":cfg,
              "problem":"EI*u''''=q(x), x in [0,1], u(0)=u'(0)=u(1)=u'(1)=0, EI=1",
              "loads":{"sinusoidal":"q(x)=sin(pi*x); exact u=sin(pi*x)/pi^4 +(x^2-x)/pi^3",
                       "uniform":"q(x)=1; exact u=x^2(1-x)^2/24",
                       "gaussian":"q(x)=exp(-((x-0.5)/0.08)^2); high-resolution FD reference"},
              "notes":["This is a newly executed, self-contained reproduction implementation for the revised manuscript; it does not claim to recover unavailable historical training scripts.",
                       "The synthetic load-noise experiment is controlled and is not field-data validation.",
                       "A finite-resolution plateau in SENS error is reported as observed; no empirical order is inferred from the theorem.",
                       "The inverse stiffness-identification idea is not claimed as a quantitatively validated result."]}
    with open(os.path.join(OUT,"run_metadata.json"),"w",encoding="utf-8") as f: json.dump(metadata,f,indent=2)
    print("Finished. Generated files are in results/",flush=True)
    print("Main table summary:\n",summary.reset_index().to_string(index=False),flush=True)
    print("Convergence table:\n",conv_df[["N","SENS_Linf","FD_Linf"]].to_string(index=False),flush=True)

if __name__=="__main__": main()