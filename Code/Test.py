import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# 📘 1. Rohdaten laden
# ------------------------------------------------------------
Daten = pd.read_csv("F:/R/BA/Kursdaten.csv")
Daten["Date"] = pd.to_datetime(Daten["Date"])

# ------------------------------------------------------------
# 📘 2. Auswahl der 20 Aktien ab 1998-11-09
# ------------------------------------------------------------
stock_cols = [c for c in Daten.columns if c != "Date"]
start_date = pd.Timestamp("1998-11-09")
valid_stocks = [col for col in stock_cols if not pd.isna(Daten.loc[Daten["Date"] == start_date, col]).values[0]]
Daten_20 = Daten.loc[Daten["Date"] >= start_date, ["Date"] + valid_stocks]

# ------------------------------------------------------------
# 📘 3. Daten ab dem 01.01.2000
# ------------------------------------------------------------
Daten_20_2000 = Daten_20[Daten_20["Date"] >= "2000-01-01"].copy()

# ------------------------------------------------------------
# 📘 4. Tägliche log-Returns berechnen
# ------------------------------------------------------------
daily_returns = Daten_20_2000.copy()
daily_returns[valid_stocks] = np.log(daily_returns[valid_stocks] / daily_returns[valid_stocks].shift(1))
daily_returns = daily_returns.dropna()

# ------------------------------------------------------------
# 📘 5. Für jeden Monat Erwartungswert & Kovarianzmatrix (letzte 6 Monate)
# ------------------------------------------------------------
months = pd.date_range(start="2000-01-01", end="2024-12-01", freq="MS")
portfolio_input = {}
lookback_months = 6  # Zeitraum für Schätzung

for folgemonat in months:
    start_window = folgemonat - relativedelta(months=lookback_months)
    
    # Filter: Tagesrenditen der letzten 6 Monate vor dem Folgemonat
    prev_window_returns = daily_returns[
        (daily_returns["Date"] >= start_window) &
        (daily_returns["Date"] < folgemonat)
    ][valid_stocks]

    if prev_window_returns.empty:
        continue

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


















import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# ⚡ cvxopt Einstellungen: keine Ausgabe bei der Optimierung
solvers.options['show_progress'] = False

# Liste für Portfolios
minvar_portfolios = {}

# Alle Monate aus portfolio_input
all_months = list(portfolio_input.keys())
first_month = all_months[0]

# 1️⃣ Erster Monat: naive Diversifikation
n_assets = len(portfolio_input[first_month]['mu'])
naive_weights = np.repeat(1/n_assets, n_assets)
minvar_portfolios[first_month] = pd.Series(naive_weights, index=portfolio_input[first_month]['mu'].index)

# 2️⃣ Schleife ab dem zweiten Monat
for month in all_months[1:]:
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
example_month = all_months[1]
print(f"MinVar Portfolio für {example_month}:")
print(minvar_portfolios[example_month])
