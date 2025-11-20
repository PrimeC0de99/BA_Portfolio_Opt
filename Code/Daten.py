import pandas as pd
import numpy as np
# from dateutil.relativedelta import relativedelta


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
