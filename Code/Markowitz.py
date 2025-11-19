from cvxopt import matrix, solvers
import numpy as np
import Temp_BA as tp


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