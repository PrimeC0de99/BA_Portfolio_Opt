import pandas as pd
import numpy as np

# ------------------------------------------------------------
#  1. Rohdaten laden
# ------------------------------------------------------------
# Wir laden die neue Datei und stellen sicher, dass 'Date' das Datum ist
Daten = pd.read_csv("C:/Users/nicla/Documents/VS_Projekte/BA/BA_Portfolio_Opt/Daten/Kursdaten_International.csv")
Daten["Date"] = pd.to_datetime(Daten["Date"])

# ------------------------------------------------------------
#  2. Auswahl der Aktien
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
start_lookback = pd.Timestamp("2005-01-01") 
daten_20_erweitert = daten_20[daten_20["Date"] >= start_lookback].copy()

# ------------------------------------------------------------
#  4. Tägliche log-Returns berechnen
# ------------------------------------------------------------
rendite_tag = daten_20_erweitert.copy()
rendite_tag[aktien_liste] = np.log(rendite_tag[aktien_liste] / rendite_tag[aktien_liste].shift(1))
rendite_tag = rendite_tag.dropna()

# für Folgedateien
aktien20_98 = aktien_liste
