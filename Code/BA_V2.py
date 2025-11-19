import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
#  1. Rohdaten laden (UNVERÄNDERT)
# ------------------------------------------------------------
Daten = pd.read_csv("F:/R/BA/Kursdaten.csv")
Daten["Date"] = pd.to_datetime(Daten["Date"])

# ------------------------------------------------------------
#  2. Auswahl der 20 Aktien ab 1998-11-09 (UNVERÄNDERT)
# ------------------------------------------------------------
stock_cols = [c for c in Daten.columns if c != "Date"]
start_date_all = pd.Timestamp("1998-11-09")
valid_stocks = [col for col in stock_cols if not pd.isna(Daten.loc[Daten["Date"] == start_date_all, col]).values[0]]
Daten_20 = Daten.loc[Daten["Date"] >= start_date_all, ["Date"] + valid_stocks]

# ------------------------------------------------------------
#  3. Daten ab dem 01.07.1999 (für 6 Monate Lookback)
# ------------------------------------------------------------
start_lookback = pd.Timestamp("1999-07-01") # Startet 6 Monate vor 2000-01-01
Daten_20_extended = Daten_20[Daten_20["Date"] >= start_lookback].copy() # <-- ÄNDERUNG HIER

# ------------------------------------------------------------
#  4. Tägliche log-Returns berechnen
# ------------------------------------------------------------
daily_returns = Daten_20_extended.copy() # <-- Basis ist jetzt der erweiterte Datensatz
daily_returns[valid_stocks] = np.log(daily_returns[valid_stocks] / daily_returns[valid_stocks].shift(1))
daily_returns = daily_returns.dropna() # Entfernt die erste Zeile (Rendite vom 02.07.1999 ist die erste)

# ------------------------------------------------------------
#  5. Für jeden Monat Erwartungswert & Kovarianzmatrix (letzte 6 Monate)
# ------------------------------------------------------------
# Erster Folgemonat (Optimierungsziel) ist jetzt 2000-01-01
months = pd.date_range(start="2000-01-01", end="2024-12-01", freq="MS") # <-- ZURÜCK AUF 2000-01-01
portfolio_input = {}
lookback_months = 6  # Zeitraum für Schätzung (UNVERÄNDERT)

for folgemonat in months:
    start_window = folgemonat - relativedelta(months=lookback_months)
    
    # Filter: Tagesrenditen der letzten 6 Monate vor dem Folgemonat
    prev_window_returns = daily_returns[
        (daily_returns["Date"] >= start_window) &
        (daily_returns["Date"] < folgemonat)
    ][valid_stocks]

    if prev_window_returns.empty:
        continue
    
    # ... (Rest der Schätzung bleibt unverändert) ...

    # --- Schätzung unter Normalverteilungsannahme ---
    mu_log = prev_window_returns.mean()
    sigma2_log = prev_window_returns.var()

    # Erwarteter arithmetischer Return unter lognormaler Annahme
    mu_expected = np.exp(mu_log + 0.5 * sigma2_log) - 1

    # Kovarianzmatrix der log-returns
    cov_mat = prev_window_returns.cov()

    # Speicherung
    portfolio_input[str(folgemonat.date())] = {
        "mu": mu_expected,
        "cov": cov_mat
    }

# Beispiel-Ausgabe
first_key = list(portfolio_input.keys())[0]
print(f"Beispielmonat: {first_key}")
print("Erwartete Renditen:\n", portfolio_input[first_key]["mu"].head())
print("\nKovarianzmatrix (Ausschnitt):\n", portfolio_input[first_key]["cov"].iloc[:5, :5])



# -------------------------------------------------------------------------------------

from cvxopt import matrix, solvers

# cvxopt Einstellungen: keine Ausgabe bei der Optimierung
solvers.options['show_progress'] = False

# Liste für Portfolios
minvar_portfolios = {}

# Alle Monate aus portfolio_input (Startet jetzt mit 2000-01-01)
all_months = list(portfolio_input.keys())

#  Schleife über ALLE Monate (ab 2000-01-01) - direkte Optimierung
for month in all_months: # <-- ÄNDERUNG HIER: Schleife startet beim ersten verfügbaren Monat
    mu = portfolio_input[month]['mu'].values       # Erwartete Renditen
    cov_mat = portfolio_input[month]['cov'].values # Kovarianzmatrix
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
    weights = np.array(sol['x']).flatten()
    
    # Kleine Rundungsfehler eliminieren
    weights[weights < 1e-12] = 0
    
    # Speichern als Pandas Series
    minvar_portfolios[month] = pd.Series(weights, index=portfolio_input[month]['mu'].index)

# Beispielausgabe
example_month = all_months[0] # <-- Beispiel zeigt jetzt den ersten optimierten Monat (2000-01-01)
print(f"MinVar Portfolio für {example_month}:")

print(minvar_portfolios[example_month])


# Hier alles selbst!!!!!
# ---------------------------------------------------------
#                       HRP
# ---------------------------------------------------------

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Berechnung der Korrelationsmatrix
df_corr = Daten_20_extended.drop(columns = ['Date'])
corr = Daten_20_extended.corr()
corr = corr.drop(columns = ['Date'], index = ['Date'])

# print('Korrelationsmatrix: ')
# print(f' {corr}')

# Distanzmatrix bzw Metrik die auf die Korrelationsmatrix angewendet wird
dist = 1 - corr
dist = np.sqrt(0.5 * dist)

print('Distanzmatrix: \n', dist)


# Vorbereitungen für das Dendrogramm
dist_array = dist.values
dist_vector = squareform(dist_array)
Z = linkage(dist_vector, method='ward')


# 3. Dendrogramm erstellen
plt.figure(figsize=(20, 7))
dendrogram(
    Z,
    orientation='top',           # Ausrichtung: von oben nach unten
    labels=dist.columns.tolist(), # Namen der Aktien als Labels
    distance_sort='descending',  # Sortierung (optional)
    show_leaf_counts=True
)

plt.title('Hierarchisches Clustering der Aktienrenditen')
plt.xlabel('Aktien')
plt.ylabel('Distanz (Cluster-Ähnlichkeit)')
plt.show()


def getQuasiDiag(Z, asset_names):
    """
    Ermittelt iterativ die quasi-diagonale Reihenfolge der Assets (Seriation).
    """
    Z = Z.astype(int)
    sortIx = pd.Series([Z[-1, 0], Z[-1, 1]])
    numItems = Z[-1, 3] 
    
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) 
        df0 = sortIx[sortIx >= numItems]
        
        i = df0.index
        j = df0.values - numItems
        
        sortIx[i] = Z[j, 0] 
        df0 = pd.Series(Z[j, 1], index=i + 1)
        
        # Wichtig: pd.Series.append ist deprecated. pd.concat verwenden!
        sortIx = pd.concat([sortIx, df0])
        
        sortIx = sortIx.sort_index() 
        sortIx.index = range(sortIx.shape[0])
        
    # Konvertiere die sortierten Indizes in Asset-Namen
    return [asset_names[i] for i in sortIx.tolist()]
