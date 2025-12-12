import pandas as pd
from cvxopt import matrix, solvers
import numpy as np
import Daten as da
from dateutil.relativedelta import relativedelta


rendite_tag = da.rendite_tag
aktien = da.aktien

# ------------------------------------------------------------
#  5. Für jeden Monat Erwartungswert & Kovarianzmatrix (letzte 6 Monate)
# ------------------------------------------------------------
# Erster Folgemonat (Optimierungsziel) ist jetzt 2000-01-01
monate = pd.date_range(start="2000-01-01", end="2024-12-01", freq="MS") # <-- ZURÜCK AUF 2000-01-01
portfolio_features = {}
rueck = 6  # Zeitraum für Schätzung (UNVERÄNDERT)

for folgemonat in monate:
    start_rueck = folgemonat - relativedelta(monate=rueck)
    
    # Filter: Tagesrenditen der letzten 6 Monate vor dem Folgemonat
    hist_daten = rendite_tag[
        (rendite_tag["Date"] >= start_rueck) &
        (rendite_tag["Date"] < folgemonat)
    ][aktien]

    if hist_daten.empty:
        continue
    
    # ... (Rest der Schätzung bleibt unverändert) ...

    # --- Schätzung unter Normalverteilungsannahme ---
    mu_log = hist_daten.mean()
    sigma2_log = hist_daten.var()

    # Erwarteter arithmetischer Return unter lognormaler Annahme
    mu_erwartet = np.exp(mu_log + 0.5 * sigma2_log) - 1

    # Kovarianzmatrix der log-returns
    cov_mat = hist_daten.cov()

    # Speicherung
    portfolio_features[str(folgemonat.date())] = {
        "mu": mu_erwartet,
        "cov": cov_mat
    }

# Beispiel-Ausgabe
first_key = list(portfolio_features.keys())[0]
print(f"Beispielmonat: {first_key}")
print("Erwartete Renditen:\n", portfolio_features[first_key]["mu"].head())
print("\nKovarianzmatrix (Ausschnitt):\n", portfolio_features[first_key]["cov"].iloc[:5, :5])



# -----------------------------------------------------------------------------
#                               Optimierung
# -----------------------------------------------------------------------------

# cvxopt Einstellungen: keine Ausgabe bei der Optimierung
solvers.options['show_progress'] = False

# Liste für Portfolios
minvar_portfolios = {}

# Alle Monate aus portfolio_features (Startet jetzt mit 2000-01-01)
monate = list(portfolio_features.keys())

#  Schleife über ALLE Monate (ab 2000-01-01) - direkte Optimierung
for monat in monate: # <-- ÄNDERUNG HIER: Schleife startet beim ersten verfügbaren Monat
    mu = portfolio_features[monat]['mu'].values       # Erwartete Renditen
    cov_mat = portfolio_features[monat]['cov'].values # Kovarianzmatrix
    n = len(mu)

    # --- Regularisierung falls Kovarianzmatrix nicht positiv definit ---
    epsilon = 1e-6
    cov_mat_reg = cov_mat + np.eye(n) * epsilon
    while np.any(np.linalg.eigvals(cov_mat_reg) <= 0):
        epsilon *= 10
        cov_mat_reg = cov_mat + np.eye(n) * epsilon

    # --- Quadratische Programmierung: min w.T * cov * w, s.t. sum(w)=1, w>=0 ---
    P = matrix(cov_mat_reg)
    q = matrix(np.zeros(n))
    
    # Gleichungen: Summe der Gewichte = 1
    G = matrix(-np.eye(n))  # w >= 0 -> -w <= 0
    h = matrix(np.zeros(n))
    
    A = matrix(np.ones((1, n)))
    b = matrix(np.ones(1))
    
    # Optimierung
    sol = solvers.qp(P, q, G, h, A, b)
    gewicht = np.array(sol['x']).flatten()
    
    # Kleine Rundungsfehler eliminieren
    gewicht[gewicht < 1e-12] = 0
    
    # Speichern als Pandas Series
    minvar_portfolios[monat] = pd.Series(gewicht, index=portfolio_features[monat]['mu'].index)

# Beispielausgabe
example_month = monate[0] # <-- Beispiel zeigt jetzt den ersten optimierten Monat (2000-01-01)
print(f"MinVar Portfolio für {example_month}:")

print(minvar_portfolios[example_month])
