# HRP
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import Daten as da


Daten_20_extended = da.Daten_20_extended
daily_returns = da.daily_returns
valid_stocks = da.valid_stocks

# Hier alles selbst!!!!!
# ---------------------------------------------------------
#                       HRP
# ---------------------------------------------------------

def hrp(df):
    """
    Generieren der Korrelationsmatrix und der Abstandsmatrtix.

    Parameters
    ----------
    df : Datensatz (Dataframe)

    Returns
    -------
    corr : Korrelationsmatrix
    dist : Abstandsmatrix

    """
    corr = df.corr()
    # corr = corr.drop(columns = ['Date'], index = ['Date'])
    
    dist = 1 - corr
    dist = np.sqrt(0.5 * dist)
    
    return corr, dist
    

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
    
    # Korrelations- und Abstandsmatrix
    corr, dist = hrp(prev_window_returns)

    # Speicherung
    portfolio_input[str(folgemonat.date())] = {
        "Korrelation": corr,
        "Abstand": dist
    }


