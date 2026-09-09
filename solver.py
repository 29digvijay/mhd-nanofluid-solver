import numpy as np
from nanofluid import nanofluid_coeffs


def thomas(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x  = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom

    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    return x


def solve(
    # ── Grid ───────────────────────────────────────────────────────────────
    xi_max  = 5.0,    # domain [0, xi_max]; paper uses ~5 (xi->inf approx)
    dxi     = 0.05,   # spatial step  (paper: 0.35; use finer for accuracy)
    dt      = 0.0095, # time step     (paper value)
    t_end   = 0.5,    # final time    (Table 3 comparison at t=0.5)
    alpha   = 1.0,    # implicitness: 1=backward-Euler, 0.5=Crank–Nicolson
    # ── Flow parameters (paper default set, page 6) ─────────────────────
    M       = 1.0,    # magnetic parameter
    Kp      = 0.15,   # permeability parameter
    Gr      = 5.0,    # thermal Grashof
    Gm      = 5.0,    # solutal Grashof  (Gc in paper)
    Pr      = 6.2,    # Prandtl (water)
    Sc      = 0.2,    # Schmidt
    N       = 1.0,    # radiation parameter
    Hs      = 1.0,    # heat source  (paper default: -Hs=3 in Table 3)
    Du      = 0.1,    # Dufour
    Sr      = 0.1,    # Soret
    Kr      = 0.5,    # chemical reaction
    R       = 0.0,    # radiation absorption  (=0 for Table 3 validation)
    S       = 0.1,    # suction/injection
    Up      = 0.1,    # plate velocity  (epsilon in paper; here Up=eps)
    eps     = 0.1,    # oscillation amplitude
    n       = 0.1,    # oscillation frequency
    # ── Nanofluid ─────────────────────────────────────────────────────────
    phi     = 0.15,   # volume fraction (Table 3 uses phi=0.15)
    particle= "Cu",   # "Cu" or "TiO2"
    # ── Output control ───────────────────────────────────────────────────
    verbose = False,
):
    """
    Run the solver and return a results dict with final profiles and
    time-series of wall quantities Cf, Nu, Sh.
    """
    # ── Nanofluid coefficients ────────────────────────────────────────────
    c = nanofluid_coeffs(phi, particle)
    a1, a2, a3, a4, a5 = c["a1"], c["a2"], c["a3"], c["a4"], c["a5"]

    # ── Grid setup ────────────────────────────────────────────────────────
    J    = int(round(xi_max / dxi)) + 1
    xi   = np.linspace(0, xi_max, J)
    b1   = alpha * dt / dxi**2          # implicit diffusion weight
    lam  = M + 1.0 / Kp                 # combined drag coefficient

    # ── Initial conditions ─────────────────────────────────────────────
    u     = np.zeros(J)
    theta = np.zeros(J)
    C     = np.zeros(J)

    # ── Storage for wall quantities ────────────────────────────────────
    t_arr  = []
    Cf_arr = []
    Nu_arr = []
    Sh_arr = []

    t = 0.0
    step = 0

    while t < t_end - 1e-12:
        dt_use = min(dt, t_end - t)
        t_new  = t + dt_use
        b1_use = alpha * dt_use / dxi**2

        # free-stream + oscillation forcing at t+dt
        exp_new = np.exp(n * t_new)
        exp_old = np.exp(n * t)
        UInf    = 1.0 + eps * exp_new
        TwBC    = 1.0 + eps * exp_new   # theta BC at wall
        CwBC    = 1.0 + eps * exp_new   # C BC at wall

        # ── Interior indices ──────────────────────────────────────────
        idx = slice(1, J-1)

        # ── 1. Momentum tridiagonal ───────────────────────────────────
        #  -a2*b1*u_{j-1} + [a1+2*a2*b1+lam*alpha*dt]*u_j - a2*b1*u_{j+1} = RHS
        #  with upwind convection S*du/dxi handled explicitly
        diag_u  = a1 + 2*a2*b1_use + lam*alpha*dt_use
        lo_u    = -a2 * b1_use
        hi_u    = -a2 * b1_use

        # Explicit convection (upwind, since S>0 → flow toward wall)
        conv_u  = a1 * S * (u[1:-1] - u[0:-2]) / dxi  # backward difference

        # Explicit diffusion (1-alpha) portion
        diff_u_ex = a2 * (1-alpha) * (u[0:-2] - 2*u[1:-1] + u[2:]) / dxi**2

        # Explicit drag
        drag_u_ex = lam * (1-alpha) * (u[1:-1] - 1 - eps*exp_old)

        RHS_u = (a1 * u[1:-1]
                 + dt_use * (-conv_u + diff_u_ex)
                 + dt_use * a3 * (Gr*theta[1:-1] + Gm*C[1:-1])
                 + dt_use * n * eps * exp_new
                 - dt_use * drag_u_ex
                 + dt_use * lam * alpha * (1 + eps*exp_new))

        n_int = J - 2
        a_u   = np.full(n_int, lo_u)
        b_u   = np.full(n_int, diag_u)
        cc_u  = np.full(n_int, hi_u)
        d_u   = RHS_u.copy()

        # BCs folded in
        d_u[0]  -= lo_u * Up       # xi=0: u=Up (plate velocity)
        d_u[-1] -= hi_u * UInf     # xi=inf: u->1+eps*e^{nt}

        u_int = thomas(a_u, b_u, cc_u, d_u)
        u_new = np.empty(J)
        u_new[0]    = Up
        u_new[1:-1] = u_int
        u_new[-1]   = UInf

        # ── 2. Species tridiagonal (uses old theta for Soret) ─────────
        diag_C = 1.0 + 2*b1_use/Sc + Kr*alpha*dt_use
        lo_C   = -b1_use / Sc
        hi_C   = -b1_use / Sc

        conv_C    = S * (C[1:-1] - C[0:-2]) / dxi
        diff_C_ex = (1-alpha) * (C[0:-2] - 2*C[1:-1] + C[2:]) / (Sc*dxi**2)
        soret_ex  = Sr * (theta[0:-2] - 2*theta[1:-1] + theta[2:]) / dxi**2
        drag_C_ex = Kr * (1-alpha) * C[1:-1]

        RHS_C = (C[1:-1]
                 + dt_use * (-conv_C + diff_C_ex + soret_ex - drag_C_ex))

        # add implicit Soret using old theta (one-lag)
        soret_im = Sr * alpha * (theta[0:-2] - 2*theta[1:-1] + theta[2:]) / dxi**2
        RHS_C   += dt_use * soret_im

        a_C  = np.full(n_int, lo_C)
        b_C  = np.full(n_int, diag_C)
        cc_C = np.full(n_int, hi_C)
        d_C  = RHS_C.copy()

        d_C[0]  -= lo_C * CwBC
        d_C[-1] -= hi_C * 0.0       # C->0 at inf

        C_int = thomas(a_C, b_C, cc_C, d_C)
        C_new = np.empty(J)
        C_new[0]    = CwBC
        C_new[1:-1] = C_int
        C_new[-1]   = 0.0

        # ── 3. Energy tridiagonal (Dufour uses new C) ─────────────────
        diag_th = a4 + 2*a5*b1_use/Pr + (N+Hs)*alpha*dt_use/Pr
        lo_th   = -a5*b1_use/Pr
        hi_th   = -a5*b1_use/Pr

        conv_th    = a4 * S * (theta[1:-1] - theta[0:-2]) / dxi
        diff_th_ex = a5*(1-alpha)*(theta[0:-2]-2*theta[1:-1]+theta[2:])/(Pr*dxi**2)
        sink_th_ex = (N+Hs)*(1-alpha)*theta[1:-1]/Pr
        du_ex      = Du*(1-alpha)*(C[0:-2]-2*C[1:-1]+C[2:])/(Pr*dxi**2)
        rad_ex     = R * C[1:-1]

        # Dufour implicit (using new C)
        du_im = Du*alpha*(C_new[0:-2]-2*C_new[1:-1]+C_new[2:])/(Pr*dxi**2)
        rad_im= R * alpha * C_new[1:-1]

        RHS_th = (a4*theta[1:-1]
                  + dt_use*(-conv_th + diff_th_ex - sink_th_ex + du_ex + rad_ex)
                  + dt_use*(du_im + rad_im))

        a_th  = np.full(n_int, lo_th)
        b_th  = np.full(n_int, diag_th)
        cc_th = np.full(n_int, hi_th)
        d_th  = RHS_th.copy()

        d_th[0]  -= lo_th * TwBC
        d_th[-1] -= hi_th * 0.0

        th_int = thomas(a_th, b_th, cc_th, d_th)
        th_new = np.empty(J)
        th_new[0]    = TwBC
        th_new[1:-1] = th_int
        th_new[-1]   = 0.0

        u, theta, C = u_new, th_new, C_new
        t = t_new
        step += 1

        # ── Wall quantities (one-sided second-order stencil) ──────────
        # du/dxi|0 ≈ (-3u0 + 4u1 - u2)/(2*dxi)
        du_wall  = (-3*u[0]     + 4*u[1]     - u[2])     / (2*dxi)
        dth_wall = (-3*theta[0] + 4*theta[1] - theta[2]) / (2*dxi)
        dC_wall  = (-3*C[0]     + 4*C[1]     - C[2])     / (2*dxi)

        Cf_val = a2 * du_wall
        Nu_val = -(a5/Pr) * dth_wall
        Sh_val = -dC_wall

        t_arr.append(t)
        Cf_arr.append(Cf_val)
        Nu_arr.append(Nu_val)
        Sh_arr.append(Sh_val)

        if verbose and step % 20 == 0:
            print(f"  t={t:.4f}  Cf={Cf_val:.5f}  Nu={Nu_val:.5f}  Sh={Sh_val:.5f}")

    return dict(
        xi=xi, u=u, theta=theta, C=C,
        t=np.array(t_arr),
        Cf=np.array(Cf_arr),
        Nu=np.array(Nu_arr),
        Sh=np.array(Sh_arr),
        coeffs=c,
    )
