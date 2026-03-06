import pandas as pd
import numpy as np
# from dateutil.relativedelta import relativedelta


# ------------------------------------------------------------
#  1. Rohdaten laden (UNVERÄNDERT)
# ------------------------------------------------------------
Daten = pd.read_csv("~/Dokumente/BA/Daten/Kursdaten.csv")
Daten["Date"] = pd.to_datetime(Daten["Date"])

# ------------------------------------------------------------
#  2. Auswahl der 20 Aktien ab 1998-11-09 (UNVERÄNDERT)
# ------------------------------------------------------------
aktien_df = [c for c in Daten.columns if c != "Date"]
startdatum = pd.Timestamp("1998-11-09")
aktien20_98 = [col for col in aktien_df if not pd.isna(Daten.loc[Daten["Date"] == startdatum, col]).values[0]]
daten_20 = Daten.loc[Daten["Date"] >= startdatum, ["Date"] + aktien20_98]

# ------------------------------------------------------------
#  3. Daten ab dem 01.07.1999 (für 6 Monate Lookback)
# ------------------------------------------------------------
start_lookback = pd.Timestamp("1999-07-01") # Startet 6 Monate vor 2000-01-01
daten_20_erweitert = daten_20[daten_20["Date"] >= start_lookback].copy()

# ------------------------------------------------------------
#  4. Tägliche log-Returns berechnen
# ------------------------------------------------------------
rendite_tag = daten_20_erweitert.copy() # <-- Basis ist jetzt der erweiterte Datensatz
rendite_tag[aktien20_98] = np.log(rendite_tag[aktien20_98] / rendite_tag[aktien20_98].shift(1))
rendite_tag = rendite_tag.dropna()


# Daten_20_erweitert sind die Daten der 20 Aktien ab dem Jahr 2000 (und das halbe Jahr davor für das erste Portfolio)
