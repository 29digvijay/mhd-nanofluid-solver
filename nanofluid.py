"""
nanofluid.py
-----------
Thermo-physical properties and nanofluid coefficients a1–a5
for Cu–water and TiO2–water systems.

Reference: Reddy, Mng'ang'a & Kanaka Rao (2026), Asia-Pacific J. Chem. Eng.
Table 2 (base fluid + nanoparticle data at 25°C / 298 K)
"""

import numpy as np

# ── Base-fluid (water) properties ────────────────────────────────────────────
WATER = dict(k=0.613, cp=4179, beta_T=21e-5, rho=997.1)

# ── Nanoparticle properties ───────────────────────────────────────────────────
PARTICLES = {
    "Cu":  dict(k=400,    cp=385,   beta_T=1.67e-5, rho=8933),
    "TiO2":dict(k=8.9538, cp=686.2, beta_T=0.9e-5,  rho=4250),
}


def nanofluid_coeffs(phi: float, particle: str = "Cu") -> dict:
    """
    Return the five dimensionless nanofluid coefficients a1–a5 for a given
    nanoparticle volume fraction phi and particle type.

    Definitions (Paper eqs. after (P12)):
        a1 = (1-phi) + phi*(rho_s/rho_f)
        a2 = (1-phi)^{-2.5}                        [Brinkman viscosity]
        a3 = [(1-phi) + phi*(rho*beta_T)_s/(rho*beta_T)_f]
        a4 = [(1-phi) + phi*(rho*cp)_s/(rho*cp)_f]
        a5 = (1+2*phi + (2-2*phi)*kf/ks)           [Maxwell conductivity]
             / (1-phi  + (2+phi )*kf/ks)            ← CORRECTED denominator

    Note: the paper's printed a5 denominator uses (1-2φ)+(2+2φ)kf/ks which
    is inconsistent with the Maxwell formula; the corrected form is used here.
    """
    f = WATER
    s = PARTICLES[particle]

    kf, ks = f["k"], s["k"]

    a1 = (1 - phi) + phi * (s["rho"] / f["rho"])
    a2 = (1 - phi) ** (-2.5)
    a3 = (1 - phi) + phi * (s["rho"] * s["beta_T"]) / (f["rho"] * f["beta_T"])
    a4 = (1 - phi) + phi * (s["rho"] * s["cp"]) / (f["rho"] * f["cp"])
    # Maxwell – corrected denominator
    a5_num = (1 + 2*phi) + (2 - 2*phi) * (kf / ks)
    a5_den = (1 - phi)   + (2 + phi)   * (kf / ks)
    a5 = a5_num / a5_den

    return dict(a1=a1, a2=a2, a3=a3, a4=a4, a5=a5)


if __name__ == "__main__":
    for fluid in ["Cu", "TiO2"]:
        c = nanofluid_coeffs(0.02, fluid)
        print(f"\n{fluid}-water (phi=0.02):")
        for k, v in c.items():
            print(f"  {k} = {v:.6f}")
