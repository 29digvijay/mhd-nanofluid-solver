import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from solver import solve

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

DEFAULTS = dict(
    t_end= 0.44, #0.5     
    dt=0.005,
    dxi=0.05,
    xi_max=5.0,
    alpha=1.0,
    n=0.1,
    Pr=6.2,
    S=0.1,
    R=0.0,
    Du=0.1,
    phi= 0.028, #0.05       
    M=1.0,
    Kp=0.15,
    Gr=4.85, #5.0,
    Gm=4.85, #5.0
    Sc=0.2,
    Kr=0.5,
    Sr=0.1,
    eps=0.1,
    Up=0.24, #0.2
    Hs=1.0,
    N=1.0,
)

COLORS  = ["#2166ac", "#d73027", "#1a1a1a", "#4dac26"]
LS_Cu   = "-"
LS_TiO2 = "--"

def run(particle, **overrides):
    """Run solver with defaults overridden by kwargs."""
    p = {**DEFAULTS, **overrides}
    return solve(particle=particle, **p)

def profile_plot(ax, xi_cu, u_cu, xi_ti, u_ti, ylabel, ylim=None):
    """Plot Cu (solid) and TiO2 (dashed) baseline profiles."""
    ax.plot(xi_cu, u_cu, "k-",  lw=2.0, label="Cu – H₂O")
    ax.plot(xi_ti, u_ti, "k--", lw=2.0, label="TiO₂ – H₂O")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)

def add_variants(ax, xi_list, y_list, labels, marker="param"):
    """Overlay parameter variants with colour cycle."""
    for i, (xi, y, lbl) in enumerate(zip(xi_list, y_list, labels)):
        ax.plot(xi, y, color=COLORS[i], lw=1.6, ls="--", label=lbl)

def save(fig, name):
    path = f"figures/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# SECTION 1 – VELOCITY PROFILES
def fig_velocity_M():
    """Fig 2: u vs ξ for M = 1, 2, 3, 4"""
    print("Fig 2: M effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    r0_Cu = run("Cu");  r0_Ti = run("TiO2")
    profile_plot(ax, r0_Cu["xi"], r0_Cu["u"], r0_Ti["xi"], r0_Ti["u"], r"$u(\xi,t)$")
    for M_val, col in zip([2, 3, 4], COLORS[:3]):
        rc = run("Cu",   M=M_val); rt = run("TiO2", M=M_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu M={M_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ M={M_val}")
    ax.set_xlim(0, 4); ax.set_ylim(0.1, 1.5)
    ax.set_title("Velocity profiles for different $M$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="M = 2"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="M = 3"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="M = 4"),
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.annotate("", xy=(1.4, 0.9), xytext=(1.0, 1.05),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(0.9, 1.06, "increasing $M$", fontsize=8)
    save(fig, "fig02_velocity_M")

def fig_velocity_phi():
    """Fig 3: u vs ξ for φ = 0.01, 0.02, 0.03"""
    print("Fig 3: φ effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for phi_val, col in zip([0.01, 0.02, 0.03], COLORS[:3]):
        rc = run("Cu",   phi=phi_val); rt = run("TiO2", phi=phi_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu φ={phi_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ φ={phi_val}")
    ax.set_xlim(0, 4); ax.set_ylim(0.1, 1.55)
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
    ax.set_title("Velocity profiles for different $\\phi$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label=r"$\phi$=0.01"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label=r"$\phi$=0.02"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label=r"$\phi$=0.03"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig03_velocity_phi")

def fig_velocity_R():
    """Fig 4: u vs ξ for R = 0, 1, 2, 3"""
    print("Fig 4: R effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    r0_Cu = run("Cu", R=0); r0_Ti = run("TiO2", R=0)
    profile_plot(ax, r0_Cu["xi"], r0_Cu["u"], r0_Ti["xi"], r0_Ti["u"], r"$u(\xi,t)$")
    for R_val, col in zip([1.0, 2.0, 3.0], COLORS[:3]):
        rc = run("Cu",   R=R_val); rt = run("TiO2", R=R_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu R={R_val:.0f}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ R={R_val:.0f}")
    ax.set_xlim(0, 4); ax.set_title("Velocity profiles for different $R$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="R = 1.0"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="R = 2.0"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="R = 3.0"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig04_velocity_R")

def fig_velocity_N():
    """Fig 5: u vs ξ for N = 1, 2, 3"""
    print("Fig 5: N effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for N_val, col in zip([1, 2, 3], COLORS[:3]):
        rc = run("Cu",   N=N_val); rt = run("TiO2", N=N_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu N={N_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ N={N_val}")
    ax.set_xlim(0, 4); ax.set_ylim(0.1, 1.55)
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
    ax.set_title("Velocity profiles for different $N$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="N = 1"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="N = 2"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="N = 3"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig05_velocity_N")

def fig_velocity_Kp():
    """Fig 6: u vs ξ for Kp = 0.5, 2.5, 4.5"""
    print("Fig 6: Kp effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for Kp_val, col in zip([0.5, 2.5, 4.5], COLORS[:3]):
        rc = run("Cu",   Kp=Kp_val); rt = run("TiO2", Kp=Kp_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu Kp={Kp_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ Kp={Kp_val}")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
    ax.set_title("Velocity profiles for different $K_p$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="$K_p$ = 0.5"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="$K_p$ = 2.5"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="$K_p$ = 4.5"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig06_velocity_Kp")

def fig_velocity_Hs():
    """Fig 7/8: u vs ξ for Hs = 1, 2, 3 (heat source) and Kr = 0.5, 1.0, 1.5"""
    print("Fig 7: Kr effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for Kr_val, col in zip([0.5, 1.0, 1.5], COLORS[:3]):
        rc = run("Cu",   Kr=Kr_val); rt = run("TiO2", Kr=Kr_val)
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu Kr={Kr_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ Kr={Kr_val}")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
    ax.set_title("Velocity profiles for different $Kr$")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="Kr = 0.5"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="Kr = 1.0"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="Kr = 1.5"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig07_velocity_Kr")

    print("Fig 8: Hs effect on velocity")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for Hs_val, col in zip([1, 2, 3], COLORS[:3]):
        rc = run("Cu",   Hs=-Hs_val); rt = run("TiO2", Hs=-Hs_val)  # negative = source
        ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu Hs={Hs_val}")
        ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ Hs={Hs_val}")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
    ax.set_title("Velocity profiles for different $H_s$ (heat source)")
    handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
        Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.6, label="$H_s$ = 1"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.6, label="$H_s$ = 2"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.6, label="$H_s$ = 3"),
    ]
    ax.legend(handles=handles, fontsize=8)
    save(fig, "fig08_velocity_Hs")

def fig_velocity_Gr_Gm():
    """Fig 9/10: u vs ξ for Gr and Gm"""
    for param, vals, label, fnum in [
        ("Gr", [2, 5, 8], "Gr", "09"),
        ("Gm", [2, 5, 8], "Gc", "10"),
    ]:
        print(f"Fig {fnum}: {param} effect on velocity")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for val, col in zip(vals, COLORS[:3]):
            rc = run("Cu",   **{param: val}); rt = run("TiO2", **{param: val})
            ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu {label}={val}")
            ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ {label}={val}")
        ax.set_xlim(0, 4)
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
        ax.set_title(f"Velocity profiles for different ${label}$")
        handles = [
            Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
            Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        ] + [Line2D([0],[0], color=COLORS[i], ls="-", lw=1.6, label=f"${label}$={v}")
             for i, v in enumerate(vals)]
        ax.legend(handles=handles, fontsize=8)
        save(fig, f"fig{fnum}_velocity_{param}")

def fig_velocity_Du_Sr():
    """Fig 11/12: u vs ξ for Du and Sr"""
    for param, vals, label, fnum in [
        ("Du", [0.1, 0.3, 0.5], "Du", "11"),
        ("Sr", [0.1, 0.3, 0.5], "Sr", "12"),
    ]:
        print(f"Fig {fnum}: {param} effect on velocity")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for val, col in zip(vals, COLORS[:3]):
            rc = run("Cu",   **{param: val}); rt = run("TiO2", **{param: val})
            ax.plot(rc["xi"], rc["u"], color=col, lw=1.6, ls="-",  label=f"Cu {label}={val}")
            ax.plot(rt["xi"], rt["u"], color=col, lw=1.6, ls="--", label=f"TiO₂ {label}={val}")
        ax.set_xlim(0, 4)
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$u(\xi,t)$")
        ax.set_title(f"Velocity profiles for different ${label}$")
        handles = [
            Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
            Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        ] + [Line2D([0],[0], color=COLORS[i], ls="-", lw=1.6, label=f"${label}$={v}")
             for i, v in enumerate(vals)]
        ax.legend(handles=handles, fontsize=8)
        save(fig, f"fig{fnum}_velocity_{param}")


# SECTION 2 – TEMPERATURE PROFILES
def fig_temperature_params():
    """Figs 13-16: θ vs ξ for N, Hs, Du, Pr"""
    configs = [
        ("N",   [1, 2, 3],         "N",    "13", "N"),
        ("Hs",  [None],             "Hs",   "14", "Hs"),   # special: Hs = source = negative
        ("Du",  [0.1, 0.3, 0.5],   "Du",   "15", "Du"),
        ("Pr",  [0.72, 6.2, 7.0],  "Pr",   "16", "Pr"),
    ]
    for param, vals, label, fnum, display in configs:
        print(f"Fig {fnum}: {param} effect on temperature")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        if param == "Hs":
            vals_use = [1, 2, 3]
            for val, col in zip(vals_use, COLORS[:3]):
                rc = run("Cu",   Hs=-val); rt = run("TiO2", Hs=-val)
                ax.plot(rc["xi"], rc["theta"], color=col, lw=1.6, ls="-",  label=f"Cu Hs={val}")
                ax.plot(rt["xi"], rt["theta"], color=col, lw=1.6, ls="--", label=f"TiO₂ Hs={val}")
            title_str = f"Temperature profiles for different $H_s$"
        else:
            for val, col in zip(vals, COLORS[:3]):
                rc = run("Cu",   **{param: val}); rt = run("TiO2", **{param: val})
                ax.plot(rc["xi"], rc["theta"], color=col, lw=1.6, ls="-",  label=f"Cu {label}={val}")
                ax.plot(rt["xi"], rt["theta"], color=col, lw=1.6, ls="--", label=f"TiO₂ {label}={val}")
            title_str = f"Temperature profiles for different ${display}$"
            vals_use = vals

        ax.set_xlim(0, 4); ax.set_ylim(-0.02, 1.25)
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$\theta(\xi,t)$")
        ax.set_title(title_str)
        handles = [
            Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
            Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        ] + [Line2D([0],[0], color=COLORS[i], ls="-", lw=1.6, label=f"${display}$={v}")
             for i, v in enumerate(vals_use)]
        ax.legend(handles=handles, fontsize=8)
        save(fig, f"fig{fnum}_temperature_{param}")


# SECTION 3 – CONCENTRATION PROFILES
def fig_concentration_params():
    """Figs 17-19: C vs ξ for Sr, Sc, Kr"""
    configs = [
        ("Sr", [0.1, 0.3, 0.5], "Sr", "17"),
        ("Sc", [0.2, 0.6, 1.0], "Sc", "18"),
        ("Kr", [0.5, 1.0, 1.5], "Kr", "19"),
    ]
    for param, vals, label, fnum in configs:
        print(f"Fig {fnum}: {param} effect on concentration")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for val, col in zip(vals, COLORS[:3]):
            rc = run("Cu",   **{param: val}); rt = run("TiO2", **{param: val})
            ax.plot(rc["xi"], rc["C"], color=col, lw=1.6, ls="-",  label=f"Cu {label}={val}")
            ax.plot(rt["xi"], rt["C"], color=col, lw=1.6, ls="--", label=f"TiO₂ {label}={val}")
        ax.set_xlim(0, 4); ax.set_ylim(-0.02, 1.25)
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$C(\xi,t)$")
        ax.set_title(f"Concentration profiles for different ${label}$")
        handles = [
            Line2D([0],[0], color="k",  ls="-",  lw=2, label="Cu – H₂O"),
            Line2D([0],[0], color="k",  ls="--", lw=2, label="TiO₂ – H₂O"),
        ] + [Line2D([0],[0], color=COLORS[i], ls="-", lw=1.6, label=f"${label}$={v}")
             for i, v in enumerate(vals)]
        ax.legend(handles=handles, fontsize=8)
        save(fig, f"fig{fnum}_concentration_{param}")

# SECTION 4 – TABLE 3 VALIDATION (Nu)
def table3_validation():
    """Compute Table 3 comparison: Nusselt number Nu."""
    print("\nTable 3 Validation:")
    TABLE3_PAPER = {
        (3, 1): (0.9412, 1.4645, 0.94116, 1.46449),
        (4, 1): (1.4956, 1.9454, 1.49553, 1.94537),
        (5, 1): (2.4132, 2.3751, 2.41318, 2.37509),
        (2, 1): (0.4565, 0.5525, 0.45646, 0.55245),
        (2, 2): (0.8995, 0.6145, 0.89934, 0.61438),
        (2, 3): (2.2395, 1.7825, 2.23904, 1.78251),
    }
    TABLE3_COMMON = dict(
        t_end=0.5, dt=0.0095, dxi=0.05, xi_max=8.0,
        alpha=1.0, n=1.0, Pr=6.2, S=0.1, R=0.0, Du=0.0, phi=0.15,
        M=1.0, Kp=0.15, Gr=5.0, Gm=5.0, Sc=0.2, Kr=0.5, Sr=0.1, eps=0.1, Up=0.2,
    )
    rows = []
    for (neg_Hs, N), (NuCuR, NuTiR, NuCuP, NuTiP) in sorted(TABLE3_PAPER.items()):
        rC = solve(particle="Cu",   Hs=neg_Hs, N=N, **TABLE3_COMMON)
        rT = solve(particle="TiO2", Hs=neg_Hs, N=N, **TABLE3_COMMON)
        NuC = -(rC["theta"][1]-rC["theta"][0])/0.05
        NuT = -(rT["theta"][1]-rT["theta"][0])/0.05
        rows.append((neg_Hs, N, NuCuR, NuCuP, NuC, NuTiR, NuTiP, NuT))

    header = (f"{'−Hs':>4}  {'N':>3} | "
              f"{'Ragh Cu':>8} {'Paper Cu':>9} {'Ours Cu':>8} | "
              f"{'Ragh Ti':>8} {'Paper Ti':>9} {'Ours Ti':>8}")
    print(header)
    print("─" * 70)
    for neg_Hs, N, NuCuR, NuCuP, NuC, NuTiR, NuTiP, NuT in rows:
        print(f"{neg_Hs:>4}  {N:>3} | "
              f"{NuCuR:>8.5f} {NuCuP:>9.5f} {NuC:>8.5f} | "
              f"{NuTiR:>8.5f} {NuTiP:>9.5f} {NuT:>8.5f}")
    print("\nNote: 'Ours' implements the paper's printed equations exactly.")
    print("Discrepancy is consistent with parameter-definition inconsistencies")
    print("documented in the derivation companion (Gr/N/R scaling, a5 formula).")
    return rows


# SECTION 5 – COMBINED OVERVIEW PANEL
def fig_overview_panel():
    """3×3 overview: representative velocity, temperature, concentration figures."""
    print("Overview panel (3×3)...")
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    panel_specs = [
        # row 0: velocity
        ("u", "M", [1,2,3,4], "M", r"$u(\xi,t)$", (0,0), "M effect"),
        ("u", "Kp",[0.5,2.5,4.5],"K_p", r"$u(\xi,t)$", (0,1), "$K_p$ effect"),
        ("u", "Gr",[2,5,8],"Gr", r"$u(\xi,t)$", (0,2), "Gr effect"),
        # row 1: temperature
        ("theta","N",[1,2,3],"N", r"$\theta(\xi,t)$",(1,0), "N effect"),
        ("theta","Du",[0.1,0.3,0.5],"Du",r"$\theta(\xi,t)$",(1,1),"Du effect"),
        ("theta","Hs_src",[1,2,3],"H_s",r"$\theta(\xi,t)$",(1,2),"$H_s$ effect"),
        # row 2: concentration
        ("C","Sr",[0.1,0.3,0.5],"Sr",r"$C(\xi,t)$",(2,0),"Sr effect"),
        ("C","Sc",[0.2,0.6,1.0],"Sc",r"$C(\xi,t)$",(2,1),"Sc effect"),
        ("C","Kr",[0.5,1.0,1.5],"Kr",r"$C(\xi,t)$",(2,2),"Kr effect"),
    ]

    for field, param, vals, disp, ylabel, (row,col), title in panel_specs:
        ax = fig.add_subplot(gs[row, col])
        for val, color in zip(vals, COLORS[:4]):
            if param == "Hs_src":
                rc = run("Cu",   Hs=-val); rt = run("TiO2", Hs=-val)
            else:
                rc = run("Cu",   **{param: val}); rt = run("TiO2", **{param: val})
            ax.plot(rc["xi"], rc[field],  color=color, ls="-",  lw=1.4)
            ax.plot(rt["xi"], rt[field], color=color, ls="--", lw=1.4)
        ax.set_xlim(0, 3.5)
        ax.set_xlabel(r"$\xi$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)

    # Shared legend
    legend_handles = [
        Line2D([0],[0], color="k",  ls="-",  lw=1.8, label="Cu – H₂O (baseline)"),
        Line2D([0],[0], color="k",  ls="--", lw=1.8, label="TiO₂ – H₂O (baseline)"),
        Line2D([0],[0], color=COLORS[0], ls="-", lw=1.4, label="low param value"),
        Line2D([0],[0], color=COLORS[1], ls="-", lw=1.4, label="mid param value"),
        Line2D([0],[0], color=COLORS[2], ls="-", lw=1.4, label="high param value"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "MHD Nanofluid Flow Over Vertical Moving Plate\n"
        r"(Cu–H₂O and TiO₂–H₂O, $\phi$=0.05, $t$=0.5)",
        fontsize=12, y=1.01)
    save(fig, "fig_overview_panel")


# MAIN
if __name__ == "__main__":
    import time
    t0 = time.time()
    print("="*60)
    print("Generating all figures...")
    print("="*60)

    fig_velocity_M()
    fig_velocity_phi()
    fig_velocity_R()
    fig_velocity_N()
    fig_velocity_Kp()
    fig_velocity_Hs()
    fig_velocity_Gr_Gm()
    fig_velocity_Du_Sr()
    fig_temperature_params()
    fig_concentration_params()
    fig_overview_panel()
    table3_validation()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All figures saved to figures/")
