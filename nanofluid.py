import numpy as np

# ── Base-fluid (water) properties ────────────────────────────────────────────
WATER = dict(k=0.613, cp=4179, beta_T=21e-5, rho=997.1)

# ── Nanoparticle properties ───────────────────────────────────────────────────
PARTICLES = {
    "Cu":  dict(k=400,    cp=385,   beta_T=1.67e-5, rho=8933),
    "TiO2":dict(k=8.9538, cp=686.2, beta_T=0.9e-5,  rho=4250),
}


def nanofluid_coeffs(phi: float, particle: str = "Cu") -> dict:
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
