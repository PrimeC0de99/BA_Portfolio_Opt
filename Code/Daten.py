import pandas as pd
import numpy as np

# ------------------------------------------------------------
#  1. Rohdaten laden (NEUE DATEI)
# ------------------------------------------------------------
# Wir laden die neue Datei und stellen sicher, dass 'Date' das Datum ist
Daten = pd.read_csv("~/Dokumente/BA/Daten/Kursdaten_International.csv")
Daten["Date"] = pd.to_datetime(Daten["Date"])

# ------------------------------------------------------------
#  2. Auswahl der Aktien (ANPASST AN 2005)
# ------------------------------------------------------------
aktien_df = [c for c in Daten.columns if c != "Date"]

# Wir setzen das Startdatum auf den ersten verfügbaren Tag im neuen Datensatz
startdatum = Daten["Date"].min() 
print(f"Analysestart mit Daten ab: {startdatum}")

# Da wir im Download-Skript schon dropna() gemacht haben, 
# sind zum Startdatum alle verbliebenen Aktien verfügbar.
aktien_liste = aktien_df 
daten_20 = Daten.loc[Daten["Date"] >= startdatum].copy()

# ------------------------------------------------------------
#  3. Daten für Portfolio-Optimierung (Lookback-Anpassung)
# ------------------------------------------------------------
# Wenn dein erstes Portfolio am 01.01.2006 starten soll (12 Mon. Lookback),
# muss start_lookback auf 2005-01-01 gesetzt werden.
start_lookback = pd.Timestamp("2005-01-01") 
daten_20_erweitert = daten_20[daten_20["Date"] >= start_lookback].copy()

# ------------------------------------------------------------
#  4. Tägliche log-Returns berechnen
# ------------------------------------------------------------
rendite_tag = daten_20_erweitert.copy()
rendite_tag[aktien_liste] = np.log(rendite_tag[aktien_liste] / rendite_tag[aktien_liste].shift(1))
rendite_tag = rendite_tag.dropna()

# WICHTIG: Überschreibe die alte Variable für die folgenden Skripte
aktien20_98 = aktien_liste
