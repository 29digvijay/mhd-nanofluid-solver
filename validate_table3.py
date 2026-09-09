import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from solver import solve

TABLE3_PAPER = {
    (3, 1): (0.9412, 1.4645, 0.94116, 1.46449),
    (4, 1): (1.4956, 1.9454, 1.49553, 1.94537),
    (5, 1): (2.4132, 2.3751, 2.41318, 2.37509),
    (2, 1): (0.4565, 0.5525, 0.45646, 0.55245),
    (2, 2): (0.8995, 0.6145, 0.89934, 0.61438),
    (2, 3): (2.2395, 1.7825, 2.23904, 1.78251),
}

COMMON = dict(
    t_end=0.5,
    dt=0.0095,
    dxi=0.05,      
    xi_max=5.0,
    alpha=1.0,
    n=1.0,
    Pr=6.2,
    S=0.1,
    R=0.0,
    Du=0.0,
    phi=0.15,
    M=1.0,
    Kp=0.15,
    Gr=5.0,
    Gm=5.0,
    Sc=0.2,
    Kr=0.5,
    Sr=0.1,
    eps=0.1,
    Up=0.1,
)

print(f"\n{'='*90}")
print(f"Table 3 Validation — Nusselt Number Nu at t=0.5, phi=0.15")
print(f"{'='*90}")
header = (f"{'−Hs':>4}  {'N':>3} │ "
          f"{'Nu_Cu(Ragh)':>12} {'Nu_Cu(paper)':>13} {'Nu_Cu(ours)':>12} {'err%':>6} │ "
          f"{'Nu_TiO2(Ragh)':>14} {'Nu_TiO2(paper)':>15} {'Nu_TiO2(ours)':>14} {'err%':>6}")
print(header)
print(f"{'─'*90}")

for (neg_Hs, N), (Nu_Cu_R, Nu_TiO2_R, Nu_Cu_P, Nu_TiO2_P) in sorted(TABLE3_PAPER.items()):
    Hs = -neg_Hs  

    r_Cu = solve(**COMMON, Hs=Hs, N=N, particle="Cu")
    r_Ti = solve(**COMMON, Hs=Hs, N=N, particle="TiO2")

    Nu_Cu   = r_Cu["Nu"][-1]
    Nu_TiO2 = r_Ti["Nu"][-1]

    err_Cu   = abs(Nu_Cu   - Nu_Cu_P)   / Nu_Cu_P   * 100
    err_TiO2 = abs(Nu_TiO2 - Nu_TiO2_P) / Nu_TiO2_P * 100

    print(f"{neg_Hs:>4}  {N:>3} │ "
          f"{Nu_Cu_R:>12.5f} {Nu_Cu_P:>13.5f} {Nu_Cu:>12.5f} {err_Cu:>5.2f}% │ "
          f"{Nu_TiO2_R:>14.5f} {Nu_TiO2_P:>15.5f} {Nu_TiO2:>14.5f} {err_TiO2:>5.2f}%")

print(f"{'='*90}")
print("\nNote: 'err%' is relative to paper's own 'Present results' column.")
print("The paper's 'Present results' agree with Raghunath to within 0.01–0.1%.")
