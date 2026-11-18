"""Genera todas las figuras y la tabla de resultados del informe."""
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from troyanos import *

C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e4'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})
B = json.load(open('resultados/barrido.json'))
Tlin_TJ = B['T_lin']; TLIN = Tlin_TJ*T_JUP_YR
bar = [b for b in B['barrido'] if not b['esc']]
D = np.array([b['amp'] for b in bar]); Tn = np.array([b['T_num'] for b in bar])
Ta = np.array([b['T_avg'] for b in bar])
pmin = np.array([b['phimin'] for b in bar]); pmax = np.array([b['phimax'] for b in bar])
resumen = {'T_lin_yr': TLIN, 'T_fast_yr': linear_periods_yr()[1], 'mu': MU, 'T_J': T_JUP_YR}
# umbrales de validez (interpolacion en el barrido numerico)
for th in [0.01, 0.05, 0.10]:
    resumen[f'D_umbral_{int(th*100)}pct'] = float(np.interp(1+th, Tn/Tlin_TJ, D))
resumen['D_max_renacuajo'] = float(D.max())
resumen['phimin_ultimo_renacuajo'] = float(min(b['phi0'] for b in bar))

# ---- datos reales ----
d, r, v = load_trojans('datos/sbdb_query_results2.csv')
reales = {}
for m in ['circular', 'eliptico']:
    z = np.load(f'resultados/real_{m}.npz'); T = z['T']; P = z['PHI'].astype(float)
    Ps, Ts = smooth_phi(T, P, 12.0)
    amp, cen, mean, per, esc = libration_stats(Ts, Ps)
    esc = esc | libration_stats(T, P)[4]
    reales[m] = dict(amp=amp, cen=cen, per=per, esc=esc, T=T, P=P, inc=np.degrees(z['I'][0]))
R = reales['eliptico']
ok = ~R['esc'] & np.isfinite(R['per'])
amp, per, inc = R['amp'], R['per'], R['inc']
resumen['N_total'] = int(len(amp)); resumen['N_L4'] = int((R['P'][0] > 0).sum())
resumen['N_L5'] = int((R['P'][0] < 0).sum()); resumen['N_escapan'] = int(R['esc'].sum())
resumen['N_escapan_circ'] = int(reales['circular']['esc'].sum())
resumen['N_escapan_ambos'] = int((reales['circular']['esc'] & R['esc']).sum())
resumen['amp_percentiles_5_25_50_75_95'] = np.percentile(amp[~R['esc']], [5, 25, 50, 75, 95]).round(1).tolist()
resumen['per_percentiles_5_25_50_75_95'] = np.percentile(per[ok], [5, 25, 50, 75, 95]).round(1).tolist()
for th in [1, 5, 10]:
    resumen[f'frac_amp_bajo_umbral_{th}pct'] = float(np.mean(amp[~R['esc']] < resumen[f'D_umbral_{th}pct']))
    resumen[f'frac_T_dentro_{th}pct_lineal'] = float(np.mean(np.abs(per[ok]/TLIN - 1) < th/100))
# fase media observada (instantanea, dato puro) vs prediccion lineal 60 deg
ph0 = np.degrees(R['P'][0])
resumen['phi_medio_obs_L4'] = float(ph0[ph0 > 0].mean()); resumen['phi_medio_obs_L5'] = float(-ph0[ph0 < 0].mean())
resumen['phi_medio_obs_L4_err'] = float(ph0[ph0 > 0].std()/np.sqrt((ph0 > 0).sum()))
resumen['phi_medio_obs_L5_err'] = float(ph0[ph0 < 0].std()/np.sqrt((ph0 < 0).sum()))
Pall = np.degrees(R['P'][:, ~R['esc']])
resumen['phi_medio_modelo_L4'] = float(np.mean(Pall[:, Pall[0] > 0]))
resumen['phi_medio_modelo_L5'] = float(-np.mean(Pall[:, Pall[0] < 0]))
# circular vs eliptico
Cc = reales['circular']; both = ok & ~Cc['esc'] & np.isfinite(Cc['per'])
resumen['circ_vs_elip_dAmp_mediana_deg'] = float(np.median(np.abs(Cc['amp'] - amp)[both]))
resumen['circ_vs_elip_dT_mediana_rel'] = float(np.median(np.abs(Cc['per'] - per)[both]/per[both]))
resumen['circ_vs_elip_dAmp_p95_deg'] = float(np.percentile(np.abs(Cc['amp'] - amp)[both], 95))
# inclinacion: periodo mediano a amplitud pequena por rango de inclinacion
resumen['T_amp_lt10_por_inc'] = {}
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 60)]:
    s = ok & (inc >= lo) & (inc < hi) & (amp < 10)
    resumen['T_amp_lt10_por_inc'][f'{lo}-{hi}'] = [int(s.sum()), float(np.median(per[s]))]
# amplitud predicha por la teoria lineal desde la condicion inicial
a0 = d.a.values; dphi0 = np.radians(np.abs(ph0)) - np.pi/3
Dlin = np.degrees(np.sqrt(dphi0**2 + ((a0 - JUP_ELEM['a'])/(JUP_ELEM['a']*np.sqrt(3*MU)))**2))
resumen['Dlin_vs_Dnum_mediana_rel'] = float(np.median(np.abs(Dlin - amp)[~R['esc']]/amp[~R['esc']]))
json.dump(resumen, open('resultados/resumen.json', 'w'), indent=1)
print(json.dumps(resumen, indent=1))

# ---------------- Figura 1: marco rotante con troyanos reales ----------------
fig, ax = plt.subplots(figsize=(5.2, 5.2))
dJ, _ = Primaries(True).rel(0.0)
thJ = np.arctan2(dJ[1], dJ[0])
x = r[:, 0]*np.cos(thJ) + r[:, 1]*np.sin(thJ); y = -r[:, 0]*np.sin(thJ) + r[:, 1]*np.cos(thJ)
ax.scatter(x, y, s=0.6, color=C1, alpha=0.35, lw=0, label='Troyanos reales (JPL, 2026-06-09)')
th = np.linspace(0, 2*np.pi, 400); aJ = np.linalg.norm(dJ)
ax.plot(aJ*np.cos(th), aJ*np.sin(th), color=MUTED, lw=0.8, ls='--')
ax.plot(0, 0, 'o', color=C4, ms=10, label='Sol'); ax.plot(aJ, 0, 'o', color=C2, ms=7, label='Júpiter')
for nm, ang in [('$L_4$', 60), ('$L_5$', -60), ('$L_3$', 180)]:
    ax.plot(aJ*np.cos(np.radians(ang)), aJ*np.sin(np.radians(ang)), 'x', color=INK, ms=8, mew=2)
    ax.annotate(nm, (aJ*np.cos(np.radians(ang)), aJ*np.sin(np.radians(ang))), xytext=(8, 6),
                textcoords='offset points', fontsize=12, color=INK)
ax.set_aspect('equal'); ax.set_xlabel('x [UA] (marco rotante, Júpiter sobre +x)'); ax.set_ylabel('y [UA]')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=C1,ms=4,label='Troyanos reales (JPL, 2026-06-09)'),Line2D([],[],marker='o',ls='',color=C4,ms=8,label='Sol'),Line2D([],[],marker='o',ls='',color=C2,ms=6,label='Júpiter')],loc='lower left', fontsize=8)
fig.savefig('figs/fig1_marco_rotante.png'); plt.close(fig)

# ---------------- Figura 2: periodo vs amplitud ----------------
fig, ax = plt.subplots(figsize=(6, 3.9))
ax.axhspan(0.95, 1.05, color=GRID, alpha=0.7, lw=0)
ax.plot(D, Tn/Tlin_TJ, 'o-', color=C1, ms=4, lw=1.6, label='PR3C completo (numérico, DOP853)')
ax.plot(D, Ta/Tlin_TJ, '-', color=C2, lw=1.6, label='Modelo promediado (cuadratura)')
ax.axhline(1, color=INK, lw=1.2, ls='--', label=f'Teoría lineal: $T$ = {TLIN:.1f} años')
for th, lab in [(5, '5 %'), (10, '10 %')]:
    Dth = resumen[f'D_umbral_{th}pct']
    ax.axvline(Dth, color=MUTED, lw=0.8, ls=':')
    ax.annotate(f'{lab}: $D$ ≈ {Dth:.0f}°', (Dth, 1.30), xytext=(3, 0), textcoords='offset points',
                fontsize=8, color=MUTED, rotation=90, va='bottom')
ax.set_xlabel('Amplitud de libración $D$ [grados]'); ax.set_ylabel('$T_{\\rm lib}/T_{\\rm lineal}$')
ax.set_xlim(0, 75); ax.set_ylim(0.97, 1.65); ax.legend(fontsize=8, loc='upper left')
fig.savefig('figs/fig2_periodo_amplitud.png'); plt.close(fig)

# ---------------- Figura 3: asimetria del renacuajo ----------------
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(D, pmax, 'o-', color=C1, ms=3.5, lw=1.5, label='$\\varphi_{\\max}$ numérico')
ax.plot(D, pmin, 'o-', color=C2, ms=3.5, lw=1.5, label='$\\varphi_{\\min}$ numérico')
ax.plot(D, 60 + D, '--', color=C1, lw=1.2, label='$60°+D$ (lineal)')
ax.plot(D, 60 - D, '--', color=C2, lw=1.2, label='$60°-D$ (lineal)')
ax.axhline(180, color=MUTED, lw=0.8, ls=':'); ax.annotate('$L_3$', (2, 172), color=MUTED)
ax.set_xlabel('Amplitud de libración $D$ [grados]'); ax.set_ylabel('$\\varphi=\\lambda-\\lambda_J$ [grados]')
ax.set_xlim(0, 75); ax.set_ylim(-15, 190); ax.legend(fontsize=8, ncol=2, loc='lower left')
fig.savefig('figs/fig3_asimetria.png'); plt.close(fig)

# ---------------- Figura 4: troyanos reales T vs D, por inclinacion ----------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
bins = [(0, 10, '#9ec5f4'), (10, 20, '#5598e7'), (20, 30, '#256abf'), (30, 60, '#0d366b')]
for lo, hi, col in bins:
    s = ok & (inc >= lo) & (inc < hi)
    ax.scatter(amp[s], per[s], s=1.2, color=col, lw=0, alpha=0.6, label=f'$i$ = {lo}–{hi}° ({s.sum()})')
ax.plot(D, Tn*T_JUP_YR, '-', color=C2, lw=2, label='PR3C plano (barrido)')
ax.axhline(TLIN, color=INK, ls='--', lw=1.2, label='Lineal')
ax.set_xlim(0, 60); ax.set_ylim(135, 215)
ax.set_xlabel('Amplitud de libración $D$ [grados]'); ax.set_ylabel('Período de libración [años]')
ax.legend(fontsize=7.5, markerscale=6, loc='upper left', ncol=2)
fig.savefig('figs/fig4_reales_T_vs_D.png'); plt.close(fig)

# ---------------- Figura 5: distribucion de amplitudes reales ----------------
fig, ax = plt.subplots(figsize=(6, 3.4))
hb = np.arange(0, 62, 2)
ax.hist(amp[~R['esc'] & (R['P'][0] > 0)], bins=hb, color=C1, alpha=0.85, label=f"$L_4$ ({resumen['N_L4']})",
        edgecolor='white', lw=0.5)
ax.hist(amp[~R['esc'] & (R['P'][0] < 0)], bins=hb, color=C2, alpha=0.75, label=f"$L_5$ ({resumen['N_L5']})",
        edgecolor='white', lw=0.5)
for th in [5, 10]:
    Dth = resumen[f'D_umbral_{th}pct']
    ax.axvline(Dth, color=INK, lw=1, ls='--')
    ax.annotate(f'error {th} %\n{100*resumen[f"frac_amp_bajo_umbral_{th}pct"]:.1f} % a la izq.', (Dth, ax.get_ylim()[1]*(0.85 if th==5 else 0.6)),
                xytext=(4, 0), textcoords='offset points', fontsize=8, color=INK)
ax.set_xlabel('Amplitud de libración $D$ [grados] (modelo elíptico, 2000 años)'); ax.set_ylabel('Número de asteroides')
ax.legend(fontsize=8, loc='center right')
fig.savefig('figs/fig5_hist_amplitud.png'); plt.close(fig)

# ---------------- Figura 6: ejemplos phi(t) ----------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
Tyr = R['T']/DAY_PER_YR
s_ok = np.where(ok & (R['P'][0] > 0) & (inc < 10))[0]
j_small = s_ok[np.argmin(np.abs(amp[s_ok] - 5))]; j_big = s_ok[np.argmin(np.abs(amp[s_ok] - 35))]
for j, col in [(j_small, C1), (j_big, C2)]:
    ax.plot(Tyr, np.degrees(R['P'][:, j]), color=col, lw=1.1,
            label=f"{d.full_name.values[j].strip()}  ($D$ = {amp[j]:.0f}°)")
    c = R['cen'][j]
    # solucion lineal con las condiciones iniciales reales: dphi(0) y dphi/dt(0) = -(3/2) n da/a
    w = 2*np.pi/TLIN; nJ = 2*np.pi/T_JUP_YR
    dp0 = np.degrees(R['P'][0, j]) - 60; dpdt0 = np.degrees(-1.5*nJ*(d.a.values[j]-JUP_ELEM['a'])/JUP_ELEM['a'])
    ax.plot(Tyr, 60 + dp0*np.cos(w*Tyr) + dpdt0/w*np.sin(w*Tyr), color=col, lw=1, ls='--', alpha=0.8)
ax.set_xlim(0, 800); ax.set_xlabel('Tiempo [años]'); ax.set_ylabel('$\\varphi$ [grados]')
ax.plot([],[],ls='--',color=MUTED,label='teoría lineal (mismas cond. iniciales)'); ax.set_ylim(25,125); ax.legend(fontsize=8, loc='upper right')
fig.savefig('figs/fig6_ejemplos.png'); plt.close(fig)
json.dump(dict(j_small=int(j_small), j_big=int(j_big), n_small=d.full_name.values[j_small].strip(),
               n_big=d.full_name.values[j_big].strip()), open('resultados/ejemplos.json', 'w'))
print('ok figuras')
