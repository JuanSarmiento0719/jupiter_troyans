"""Barrido en amplitud en el PR3C circular plano (marco rotante, adimensional).
Compara: (1) integracion numerica completa, (2) modelo promediado (pendulo), (3) teoria lineal."""
import numpy as np, json
from scipy.integrate import solve_ivp
from troyanos import *

ws, wf = linear_frequencies()
T_lin = 1/ws                               # en periodos de Jupiter
res = []
phis0 = np.radians(np.concatenate([np.arange(59, 30, -1.0), np.arange(30, 23.0, -0.5)]))
for p0 in phis0:
    s0 = np.array([np.cos(p0) - MU, np.sin(p0), 0.0, 0.0])   # en reposo en el marco rotante, r=1 del Sol
    Tav, pmax = averaged_period(p0)
    tend = 2*np.pi*(8*Tav if np.isfinite(Tav) else 400)
    tend = min(max(tend, 2*np.pi*120), 2*np.pi*2000)
    sol = solve_ivp(cr3bp_rhs, (0, tend), s0, method='DOP853', rtol=1e-12, atol=1e-12,
                    dense_output=True)
    t = np.linspace(0, tend, int(tend/(2*np.pi)*40))
    S = sol.sol(t)
    phi = np.arctan2(S[1], S[0] + MU)
    CJ = jacobi(S); dC = np.max(np.abs(CJ - CJ[0]))/abs(CJ[0])
    tdays = t/(2*np.pi)*T_JUP_YR*DAY_PER_YR
    amp, cen, mean, per, esc = libration_stats(tdays, phi[:, None])
    res.append(dict(phi0=np.degrees(p0), amp=float(amp[0]), cen=float(cen[0]), mean=float(mean[0]),
                    phimin=float(np.degrees(phi.min())), phimax=float(np.degrees(phi.max())),
                    T_num=float(per[0]/T_JUP_YR), T_avg=float(Tav),
                    phimax_avg=float(np.degrees(pmax)) if np.isfinite(pmax) else None,
                    esc=bool(esc[0]), dC=float(dC)))
    r = res[-1]
    print(f"{r['phi0']:5.1f} amp={r['amp']:6.2f} Tnum={r['T_num']:.4f} Tavg={r['T_avg']:.4f} "
          f"pmax={r['phimax']:.1f}/{r['phimax_avg']} esc={r['esc']} dC={dC:.1e}", flush=True)
json.dump(dict(T_lin=T_lin, T_JUP_YR=T_JUP_YR, mu=MU, barrido=res), open('resultados/barrido.json', 'w'), indent=1)
