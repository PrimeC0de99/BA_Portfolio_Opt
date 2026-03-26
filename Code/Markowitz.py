import pandas as pd 
from cvxopt import matrix, solvers 
import numpy as np 
import Daten as da 
from dateutil.relativedelta import relativedelta 

# ------------------------------------------------------------ 
# 1. Daten Setup 
# ------------------------------------------------------------ 
rendite_tag = da.rendite_tag 
aktien = da.aktien20_98 

# Zeitraum definieren: START am 01.01.2006, ENDE am 31.12.2025
# Ändere freq="YS" zu "QS" für Quartale oder "6MS" für Halbjahre
jahre = pd.date_range(start="2006-01-01", end="2025-12-31", freq="QS")          # Frequenz 

portfolio_features = {} 
rueck = 12  # Lookback-Window in Monaten (z.B. 12 für ein Jahr Historie)

# ------------------------------------------------------------ 
# 2. Feature Schätzung & Konditionszahlen 
# ------------------------------------------------------------ 
stability_results = [] 

for folgejahr in jahre: 
    start_rueck = folgejahr - relativedelta(months=rueck) 
     
    # Filter: Tagesrenditen im Lookback-Zeitraum
    hist_daten = rendite_tag[ 
        (rendite_tag["Date"] >= start_rueck) &  
        (rendite_tag["Date"] < folgejahr) 
    ][aktien] 

    if hist_daten.empty: 
        continue 
     
    mu_log = hist_daten.mean() 
    cov_mat = hist_daten.cov() 

    # Speicherung der Features 
    month_str = str(folgejahr.date()) 
    portfolio_features[month_str] = { 
        "mu": mu_log, 
        "cov": cov_mat 
    } 
     
    # Konditionszahl der rohen Kovarianzmatrix berechnen & speichern 
    cond_no = np.linalg.cond(cov_mat.values) 
    stability_results.append({'Date': folgejahr, 'Condition_Number': cond_no}) 

# Stabilitäts-Daten in DataFrame umwandeln 
df_stability = pd.DataFrame(stability_results).set_index('Date') 

# ------------------------------------------------------------ 
# 3. Markowitz Optimierung (MinVar) 
# ------------------------------------------------------------ 
minvar_portfolios = {} 
solvers.options['show_progress'] = False # Unterdrückt CVXOPT-Output

for jahr in portfolio_features.keys(): 
    mu = portfolio_features[jahr]['mu'].values 
    cov_mat = portfolio_features[jahr]['cov'].values 
    n = len(mu) 

    # Regularisierung (Tikhonov / Ridge), um numerische Stabilität zu garantieren
    epsilon = 1e-6 
    cov_mat_reg = cov_mat + np.eye(n) * epsilon 
    # Falls Matrix noch nicht positiv definit, Epsilon erhöhen
    while np.any(np.linalg.eigvals(cov_mat_reg) <= 0): 
        epsilon *= 10 
        cov_mat_reg = cov_mat + np.eye(n) * epsilon 

    # QP Setup: Zielfunktion min w.T * P * w 
    P = matrix(cov_mat_reg) 
    q = matrix(np.zeros(n)) 
    G = matrix(-np.eye(n))      # Nebenbedingung: w >= 0 (Short-Selling-Verbot)
    h = matrix(np.zeros(n)) 
    A = matrix(np.ones((1, n))) # Nebenbedingung: Summe der Gewichte = 1 
    b = matrix(np.ones(1)) 
     
    try:
        sol = solvers.qp(P, q, G, h, A, b) 
        gewicht = np.array(sol['x']).flatten() 
        gewicht[gewicht < 1e-12] = 0 # Numerisches Rauschen auf 0 setzen
         
        minvar_portfolios[jahr] = pd.Series(gewicht, index=portfolio_features[jahr]['mu'].index) 
    except:
        print(f"Warnung: Optimierung für {jahr} fehlgeschlagen.")

# ------------------------------------------------------------ 
# 4. Zusammenfassung & Stabilitäts-Metriken
# ------------------------------------------------------------ 
print("\n--- Analyse der Kovarianzmatrizen abgeschlossen ---") 
print(f"Max. Konditionszahl: {df_stability['Condition_Number'].max():.2f}") 
print(f"Ø Konditionszahl:    {df_stability['Condition_Number'].mean():.2f}")
