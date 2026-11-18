"""Verificaciones independientes:
 V1. Ida y vuelta elementos -> estado -> elementos.
 V2. Conservacion de la constante de Jacobi (primarios circulares) en el integrador RK4.
 V3. Convergencia numerica: paso T_J/300 vs T_J/600 (amplitud y periodo de libracion).
 V4. Dos metodos independientes: RK4 vectorizado vs DOP853 adaptativo de SciPy (modelo eliptico).
 V5. Limite mu -> 0 / amplitud -> 0 ya cubierto en run_barrido.py (T_num -> T_lineal)."""
import numpy as np, json
from scipy.integrate import solve_ivp
from troyanos import *
out = {}
d, r, v = load_trojans('datos/sbdb_query_results2.csv')
a, e, i, Om, vp, lam = state_to_elements(r, v, GM_SUN)
lam0 = np.radians(d.om + d.w + d.ma).values
out['V1_err_a'] = float(np.max(abs(a - d.a.values)))
out['V1_err_lambda'] = float(np.max(abs(np.angle(np.exp(1j*(lam - lam0))))))

rng = np.random.default_rng(1)
idx = rng.choice(len(d), 400, replace=False)
anos = 1500
res = {}
for modelo in ['circular', 'eliptico']:
    prim = Primaries(eccentric=(modelo == 'eliptico'))
    rb, vb = helio_to_bary(r[idx], v[idx], prim)
    for nst in [150, 300, 600]:
        T, PHI, A, E, I = integrate_track(rb, vb, prim, anos*DAY_PER_YR, 2*np.pi/prim.n/nst, DAY_PER_YR)
        amp, cen, mean, per, esc = libration_stats(T, PHI.astype(float))
        res[(modelo, nst)] = (amp, per, esc)
    if modelo == 'circular':
        Tt, R, Vv = integrate_rk4(rb, vb, prim, anos*DAY_PER_YR, 2*np.pi/prim.n/300, 300)
        C = jacobi_inertial(Tt, R, Vv, prim)
        dC = np.max(np.abs(C - C[0])/np.abs(C[0]), axis=0)
        out['V2_jacobi_mediana'] = float(np.median(dC)); out['V2_jacobi_max'] = float(np.max(dC))
for modelo in ['circular', 'eliptico']:
    for nst in [150, 300]:
        a1, p1, e1 = res[(modelo, nst)]; a2, p2, e2 = res[(modelo, 600)]
        ok = ~e1 & ~e2 & np.isfinite(p1) & np.isfinite(p2)
        out[f'V3_{modelo}_{nst}_vs_600_dAmp_max_deg'] = float(np.max(abs(a1 - a2)[ok]))
        out[f'V3_{modelo}_{nst}_vs_600_dT_rel_mediana'] = float(np.median(abs(p1 - p2)[ok]/p2[ok]))

# V4: DOP853 en el modelo eliptico para 6 asteroides
prim = Primaries(eccentric=True)
sub = idx[:6]
rb, vb = helio_to_bary(r[sub], v[sub], prim)
def rhs(t, s):
    rr = s[:3][None, :]
    return np.concatenate([s[3:], accel(rr, t, prim)[0]])
tout = np.arange(0, 1500*DAY_PER_YR + 1, DAY_PER_YR)
amps_dop, pers_dop = [], []
for k in range(len(sub)):
    sol = solve_ivp(rhs, (0, tout[-1]), np.concatenate([rb[k], vb[k]]), method='DOP853',
                    rtol=1e-11, atol=1e-13, t_eval=tout)
    R = sol.y[:3].T[:, None, :]; V = sol.y[3:].T[:, None, :]
    ph, *_ = resonant_angle(sol.t, R, V, prim)
    am, ce, me, pe, es = libration_stats(sol.t, ph)
    amps_dop.append(am[0]); pers_dop.append(pe[0])
a_rk, p_rk, _ = res[('eliptico', 300)]
out['V4_amp_DOP853'] = [float(x) for x in amps_dop]
out['V4_amp_RK4'] = [float(x) for x in a_rk[:6]]
out['V4_T_DOP853'] = [float(x) for x in pers_dop]
out['V4_T_RK4'] = [float(x) for x in p_rk[:6]]
json.dump(out, open('resultados/verificacion.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
