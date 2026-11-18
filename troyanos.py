"""
troyanos.py -- Modelo de los troyanos de Jupiter (Reto III.1, Mecanica Clasica UIS)
Autores: Paola Andrea Rios Campos, Juan Diego Sarmiento Villarreal

Contiene:
  1. Constantes fisicas (JPL) y parametro de masas mu.
  2. Teoria lineal alrededor de L4 (frecuencias de pequenas oscilaciones).
  3. Modelo promediado de la resonancia 1:1 (pendulo no lineal en phi = lambda - lambda_J).
  4. PR3C circular plano en el marco rotante (adimensional) + constante de Jacobi.
  5. Problema restringido 3D en marco inercial con Sol y Jupiter en orbita
     kepleriana (circular o eliptica), integrado para muchos asteroides a la vez (RK4 vectorizado).
  6. Conversiones elementos orbitales <-> vector de estado.

Unidades del modelo inercial: UA y dias (GM en UA^3/dia^2). Resultados se reportan en anos.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. Constantes (JPL DE440 / Horizons)
# ---------------------------------------------------------------------------
GM_SUN = 2.9591220828559115e-04      # UA^3/dia^2
GM_JUP = 2.8253458420837780e-07      # UA^3/dia^2 (sistema de Jupiter)
GM_TOT = GM_SUN + GM_JUP
MU = GM_JUP / GM_TOT                 # parametro de masas ~ 9.5388e-4
DAY_PER_YR = 365.25

# Jupiter, elementos osculantes heliocentricos (ecliptica J2000) en la epoca de los datos
# JPL Horizons, 2026-Jun-09 00:00 TDB = JD 2461200.5 (misma epoca que el CSV del SBDB)
EPOCH_JD = 2461200.5
JUP_ELEM = dict(a=5.203080405012691, e=4.822534670267895e-02,
                i=np.radians(1.303446053841260), Om=np.radians(1.005154861704070e+02),
                w=np.radians(2.736746433399965e+02), M=np.radians(1.024675747387010e+02))

N_JUP = np.sqrt(GM_TOT / JUP_ELEM['a']**3)          # movimiento medio [rad/dia]
T_JUP_YR = 2*np.pi/N_JUP/DAY_PER_YR                  # periodo orbital de Jupiter [anos]


# ---------------------------------------------------------------------------
# 2. Teoria lineal alrededor de L4
# ---------------------------------------------------------------------------
def linear_frequencies(mu=MU):
    """Raices de la ecuacion caracteristica lambda^4 + lambda^2 + (27/4) mu (1-mu) = 0.
    Devuelve (omega_lenta, omega_rapida) en unidades de n (movimiento medio)."""
    disc = np.sqrt(1 - 27*mu*(1-mu))
    w_slow = np.sqrt((1 - disc)/2)
    w_fast = np.sqrt((1 + disc)/2)
    return w_slow, w_fast


def linear_periods_yr(mu=MU, TJ=None):
    TJ = T_JUP_YR if TJ is None else TJ
    ws, wf = linear_frequencies(mu)
    return TJ/ws, TJ/wf


# ---------------------------------------------------------------------------
# 3. Modelo promediado (pendulo de la resonancia 1:1)
#    phi'' = -3 mu n^2 sin(phi) [1 - (2-2cos phi)^(-3/2)]
#    Energia: (1/2) phi'^2 + 3 mu n^2 V(phi),  V(phi) = -cos(phi) + 1/(2 sin(phi/2))
# ---------------------------------------------------------------------------
def V_avg(phi):
    return -np.cos(phi) + 1.0/(2.0*np.sin(np.abs(phi)/2.0))


def averaged_period(phi_min, mu=MU, npts=4000):
    """Periodo de libracion (en unidades de 2pi/n, i.e. periodos de Jupiter) de la orbita
    renacuajo cuyo punto de retorno mas cercano a Jupiter es phi_min (rad, 0<phi_min<pi/3).
    Usa la cuadratura T = 2 * int dphi / phidot, con phidot = sqrt(6 mu n^2 (V(phi_min) - V(phi)))."""
    from scipy.optimize import brentq
    from scipy.integrate import quad
    E = V_avg(phi_min)
    # punto de retorno lejano: V(phi_max) = E con phi_max en (pi/3, pi]
    if E >= V_avg(np.pi):           # no es renacuajo: herradura o circulacion
        return np.nan, np.nan
    phi_max = brentq(lambda p: V_avg(p) - E, np.pi/3, np.pi)
    # sustitucion phi = c + h sin(theta) elimina las singularidades de los extremos
    c, h = (phi_max + phi_min)/2, (phi_max - phi_min)/2

    def integrand(th):
        p = c + h*np.sin(th)
        dV = E - V_avg(p)
        return h*np.cos(th)/np.sqrt(max(6*mu*dV, 1e-300))
    I, _ = quad(integrand, -np.pi/2, np.pi/2, limit=400)
    T = 2*I/(2*np.pi)               # n = 1 -> periodo en unidades de T_J
    return T, phi_max


# ---------------------------------------------------------------------------
# 4. PR3C circular plano, marco rotante, adimensional (G(m1+m2)=1, distancia=1, n=1)
# ---------------------------------------------------------------------------
def cr3bp_rhs(t, s, mu=MU):
    x, y, vx, vy = s
    r1 = np.sqrt((x+mu)**2 + y**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2)
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    return np.array([vx, vy, ax, ay])


def jacobi(s, mu=MU):
    x, y, vx, vy = s
    r1 = np.sqrt((x+mu)**2 + y**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2)
    U = 0.5*(x**2+y**2) + (1-mu)/r1 + mu/r2
    return 2*U - (vx**2+vy**2)


def L4(mu=MU):
    return np.array([0.5-mu, np.sqrt(3)/2])


# ---------------------------------------------------------------------------
# 5. Elementos <-> estado (vectorizado)
# ---------------------------------------------------------------------------
def kepler_E(M, e, tol=1e-13):
    E = np.where(e < 0.8, M, np.pi*np.ones_like(M))
    for _ in range(50):
        f = E - e*np.sin(E) - M
        dE = -f/(1 - e*np.cos(E))
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def elements_to_state(a, e, i, Om, w, M, gm):
    a, e, i, Om, w, M = map(np.asarray, (a, e, i, Om, w, M))
    E = kepler_E(np.mod(M, 2*np.pi), e)
    cE, sE = np.cos(E), np.sin(E)
    fac = np.sqrt(1 - e**2)
    xp, yp = a*(cE - e), a*fac*sE
    r = a*(1 - e*cE)
    nmot = np.sqrt(gm/a**3)
    vxp, vyp = -a*nmot*sE*a/r, a*nmot*fac*cE*a/r
    cO, sO, cw, sw, ci, si = np.cos(Om), np.sin(Om), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    R11 = cO*cw - sO*sw*ci; R12 = -cO*sw - sO*cw*ci
    R21 = sO*cw + cO*sw*ci; R22 = -sO*sw + cO*cw*ci
    R31 = sw*si;            R32 = cw*si
    pos = np.stack([R11*xp + R12*yp, R21*xp + R22*yp, R31*xp + R32*yp], axis=-1)
    vel = np.stack([R11*vxp + R12*vyp, R21*vxp + R22*vyp, R31*vxp + R32*vyp], axis=-1)
    return pos, vel


def state_to_elements(r, v, gm):
    """Devuelve a, e, i, Om, varpi, lambda (longitud media). r, v de forma (...,3)."""
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v*v, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    a = 1.0/(2.0/rn - v2/gm)
    evec = np.cross(v, h)/gm - r/rn[..., None]
    e = np.linalg.norm(evec, axis=-1)
    i = np.arccos(np.clip(h[..., 2]/hn, -1, 1))
    Om = np.arctan2(h[..., 0], -h[..., 1])
    # longitud verdadera "dog-leg" y longitud del perihelio
    cO, sO, ci = np.cos(Om), np.sin(Om), np.cos(i)
    # argumento de latitud u: angulo desde el nodo ascendente
    xn = r[..., 0]*cO + r[..., 1]*sO
    yn = (-r[..., 0]*sO + r[..., 1]*cO)*ci + r[..., 2]*np.sin(i)
    u = np.arctan2(yn, xn)
    exn = evec[..., 0]*cO + evec[..., 1]*sO
    eyn = (-evec[..., 0]*sO + evec[..., 1]*cO)*ci + evec[..., 2]*np.sin(i)
    w = np.arctan2(eyn, exn)
    f = u - w
    E = 2*np.arctan2(np.sqrt(1-e)*np.sin(f/2), np.sqrt(1+e)*np.cos(f/2))
    M = E - e*np.sin(E)
    varpi = Om + w
    lam = np.mod(varpi + M, 2*np.pi)
    return a, e, i, Om, varpi, lam


# ---------------------------------------------------------------------------
# 6. Problema restringido 3D con primarios keplerianos (marco inercial baricentrico)
# ---------------------------------------------------------------------------
class Primaries:
    """Sol y Jupiter en orbita kepleriana relativa (two-body exacto).
    eccentric=False -> orbita circular de radio a_J en el plano de la orbita de Jupiter,
    con la misma longitud media en la epoca."""
    def __init__(self, eccentric=True, elem=JUP_ELEM):
        el = dict(elem)
        if not eccentric:
            lam0 = el['Om'] + el['w'] + el['M']
            el['e'] = 0.0
            el['w'] = 0.0
            el['M'] = lam0 - el['Om']
        self.el = el
        self.n = np.sqrt(GM_TOT/el['a']**3)

    def rel(self, t):
        """Posicion y velocidad de Jupiter relativas al Sol en t (dias desde la epoca)."""
        el = self.el
        M = el['M'] + self.n*t
        p, v = elements_to_state(el['a'], el['e'], el['i'], el['Om'], el['w'],
                                 np.atleast_1d(M), GM_TOT)
        return p[0], v[0]

    def positions(self, t):
        d, _ = self.rel(t)
        return -MU*d, (1-MU)*d          # Sol, Jupiter (baricentricos)


def accel(r, t, prim):
    rs, rj = prim.positions(t)
    d1 = r - rs
    d2 = r - rj
    r1 = np.sqrt(np.sum(d1*d1, axis=1))[:, None]
    r2 = np.sqrt(np.sum(d2*d2, axis=1))[:, None]
    return -GM_SUN*d1/r1**3 - GM_JUP*d2/r2**3


def integrate_rk4(r0, v0, prim, t_end, dt, n_out):
    """RK4 de paso fijo para N particulas de prueba a la vez.
    Devuelve tiempos de salida, posiciones y velocidades baricentricas (N_out, N, 3)."""
    nsteps = int(np.ceil(t_end/dt))
    dt = t_end/nsteps
    every = max(1, nsteps//n_out)
    r, v = r0.copy(), v0.copy()
    T, R, Vv = [0.0], [r.copy()], [v.copy()]
    t = 0.0
    for k in range(1, nsteps+1):
        a1 = accel(r, t, prim)
        a2 = accel(r + 0.5*dt*v + dt*dt/8*a1, t + 0.5*dt, prim)
        a3 = accel(r + 0.5*dt*v + dt*dt/8*a2, t + 0.5*dt, prim)
        a4 = accel(r + dt*v + dt*dt/2*a3, t + dt, prim)
        # RK4 para ecuaciones de segundo orden (Runge-Kutta-Nystrom clasico)
        r_new = r + dt*v + dt*dt/6*(a1 + a2 + a3)
        v_new = v + dt/6*(a1 + 2*a2 + 2*a3 + a4)
        r, v = r_new, v_new
        t = k*dt
        if k % every == 0:
            T.append(t); R.append(r.copy()); Vv.append(v.copy())
    return np.array(T), np.array(R), np.array(Vv)


def resonant_angle(T, R, Vv, prim):
    """phi = lambda - lambda_J (heliocentrico, longitudes medias) y a heliocentrico.
    Devuelve phi en (-pi, pi], a (UA), e, i para cada salida y particula."""
    phis, As, Es, Is = [], [], [], []
    for t, r, v in zip(T, R, Vv):
        rs, rj = prim.positions(t)
        d, dv = prim.rel(t)
        vs = -MU*dv
        _, _, _, _, _, lamJ = state_to_elements(d[None, :], dv[None, :], GM_TOT)
        a, e, i, _, _, lam = state_to_elements(r - rs, v - vs, GM_SUN)
        phis.append(np.angle(np.exp(1j*(lam - lamJ[0]))))
        As.append(a); Es.append(e); Is.append(i)
    return np.array(phis), np.array(As), np.array(Es), np.array(Is)


def jacobi_inertial(T, R, Vv, prim):
    """Integral de Jacobi en el marco inercial (solo vale para primarios circulares):
    C_J = v^2/2 - GM_s/r1 - GM_J/r2 - n * (h . k_J), con k_J la normal al plano de Jupiter."""
    d0, dv0 = prim.rel(0.0)
    kJ = np.cross(d0, dv0); kJ /= np.linalg.norm(kJ)
    out = []
    for t, r, v in zip(T, R, Vv):
        rs, rj = prim.positions(t)
        r1 = np.linalg.norm(r - rs, axis=1); r2 = np.linalg.norm(r - rj, axis=1)
        E = 0.5*np.sum(v*v, axis=1) - GM_SUN/r1 - GM_JUP/r2
        h = np.cross(r, v)
        out.append(E - prim.n*(h @ kJ))
    return np.array(out)


def libration_stats(T_days, phi):
    """Amplitud (semi-rango, grados), centro, periodo (anos) y bandera de escape.
    Escape: phi cruza +-180 deg (circula) o cambia de enjambre (cruza 0 por Jupiter)."""
    ph = np.degrees(phi)
    sign0 = np.sign(ph[0])
    escaped = np.any(np.sign(ph) != sign0, axis=0)
    amp = 0.5*(ph.max(axis=0) - ph.min(axis=0))
    cen = 0.5*(ph.max(axis=0) + ph.min(axis=0))
    mean = ph.mean(axis=0)
    # periodo: cruces ascendentes por la media movil (filtrada) -> tiempo medio entre cruces
    tyr = T_days/DAY_PER_YR
    per = np.full(ph.shape[1], np.nan)
    x = ph - mean[None, :]
    up = (x[:-1] < 0) & (x[1:] >= 0)
    for j in range(ph.shape[1]):
        idx = np.where(up[:, j])[0]
        if len(idx) >= 3:
            # interpolacion lineal del cruce
            tc = tyr[idx] + (tyr[idx+1]-tyr[idx])*(-x[idx, j])/(x[idx+1, j]-x[idx, j])
            # descartar cruces espurios por oscilacion rapida: usar mediana de diferencias largas
            dts = np.diff(tc)
            dts = dts[dts > 40]           # el periodo de libracion es >> T_J
            if len(dts) >= 2:
                per[j] = np.median(dts)
    return amp, cen, mean, per, escaped


def smooth_phi(T_days, phi, window_yr=12.0):
    """Promedio movil de ancho ~ T_J para quitar el termino de periodo corto."""
    dt = (T_days[1]-T_days[0])/DAY_PER_YR
    k = max(1, int(round(window_yr/dt)))
    ker = np.ones(k)/k
    # promediar en el plano complejo para no romper el corte en +-pi
    z = np.exp(1j*phi)
    out = np.empty_like(phi)
    for j in range(phi.shape[1]):
        zr = np.convolve(z[:, j].real, ker, mode='same')
        zi = np.convolve(z[:, j].imag, ker, mode='same')
        out[:, j] = np.angle(zr + 1j*zi)
    h = k//2 + 1
    return out[h:-h], T_days[h:-h]


def integrate_track(r0, v0, prim, t_end, dt, out_every_days):
    """Igual que integrate_rk4 pero guarda solo phi, a, e, i (float32) en cada salida:
    permite integrar ~15 000 asteroides sin agotar la memoria."""
    nsteps = int(np.ceil(t_end/dt)); dt = t_end/nsteps
    every = max(1, int(round(out_every_days/dt)))
    r, v = r0.copy(), v0.copy()
    T, PHI, A, E, I = [], [], [], [], []

    def record(t):
        ph, a, e, i = resonant_angle([t], [r], [v], prim)
        T.append(t); PHI.append(ph[0].astype(np.float32)); A.append(a[0].astype(np.float32))
        E.append(e[0].astype(np.float32)); I.append(i[0].astype(np.float32))
    record(0.0)
    t = 0.0
    for k in range(1, nsteps+1):
        a1 = accel(r, t, prim)
        a2 = accel(r + 0.5*dt*v + dt*dt/8*a1, t + 0.5*dt, prim)
        a3 = accel(r + 0.5*dt*v + dt*dt/8*a2, t + 0.5*dt, prim)
        a4 = accel(r + dt*v + dt*dt/2*a3, t + dt, prim)
        r = r + dt*v + dt*dt/6*(a1 + a2 + a3)
        v = v + dt/6*(a1 + 2*a2 + 2*a3 + a4)
        t = k*dt
        if k % every == 0:
            record(t)
    return np.array(T), np.array(PHI), np.array(A), np.array(E), np.array(I)


def load_trojans(csv_path, epoch_cal='2026-06-09.0'):
    """Lee el CSV del JPL SBDB y se queda con los asteroides de la epoca mayoritaria."""
    import pandas as pd
    d = pd.read_csv(csv_path, low_memory=False)
    d = d[d['epoch_cal'] == epoch_cal].reset_index(drop=True)
    r, v = elements_to_state(d.a.values, d.e.values, np.radians(d.i.values),
                             np.radians(d.om.values), np.radians(d.w.values),
                             np.radians(d.ma.values), GM_SUN)
    return d, r, v


def helio_to_bary(r_h, v_h, prim):
    """Heliocentrico -> baricentrico del par Sol-Jupiter: r_bar = r_hel - mu * r_J,hel."""
    dJ, dvJ = prim.rel(0.0)
    return r_h - MU*dJ, v_h - MU*dvJ
