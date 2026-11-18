"""Integra todos los troyanos reales (epoca 2026-06-09) con primarios circulares o elipticos.
Uso: python run_real.py circular|eliptico [anos] [pasos_por_orbita]"""
import sys, time, numpy as np
from troyanos import *
modelo = sys.argv[1]
anos = float(sys.argv[2]) if len(sys.argv) > 2 else 2000
nst = int(sys.argv[3]) if len(sys.argv) > 3 else 300
csv = sys.argv[4] if len(sys.argv) > 4 else 'datos/sbdb_query_results2.csv'
d, r, v = load_trojans(csv)
prim = Primaries(eccentric=(modelo == 'eliptico'))
rb, vb = helio_to_bary(r, v, prim)
t0 = time.time()
T, PHI, A, E, I = integrate_track(rb, vb, prim, anos*DAY_PER_YR, 2*np.pi/prim.n/nst, DAY_PER_YR)
print(modelo, 'tiempo de computo', time.time()-t0, 's')
np.savez_compressed(f'resultados/real_{modelo}.npz', T=T, PHI=PHI, A=A, E=E, I=I,
                    spkid=d.spkid.values)
