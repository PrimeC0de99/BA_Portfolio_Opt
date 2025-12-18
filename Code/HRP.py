# HRP
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import Daten as da


# Daten_20_extended = da.Daten_20_extended
rendite_tag = da.rendite_tag
aktien20_98 = da.aktien20_98

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
    #corr = corr.drop(columns = ['Date'], index = ['Date'])
    
    dist = 1 - corr
    dist = np.sqrt(0.5 * dist)
    
    # Symmetrie der Abstandsmatrix erzwingen
    dist = (dist + dist.T) / 2.0
    
    return corr, dist
    

def getQuasiDiag(Z, aktien):
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
    return [aktien[i] for i in sortIx.tolist()]



# -----------------------------------------------------------------------------



def get_ivp_weights(cov_mat):
    """Berechnet Inverse-Varianz-Gewichte (IVP)."""
    # 1.0 / np.diag(cov_mat) ist die inverse Varianz
    ivp = 1.0 / np.diag(cov_mat) 
    # Normalisierung
    return ivp / np.sum(ivp)

def getClusterVar(cov, cItems):
    """Berechnet die Varianz eines Clusters unter IVP-Gewichtung."""
    
    # Extrahiere die Kovarianzmatrix für den Cluster
    cov_sub = cov.loc[cItems, cItems].values
    
    # Berechne die IVP-Gewichte für diesen Cluster
    w_ivp = get_ivp_weights(cov_sub)
    
    # Berechne die Varianz des Sub-Portfolios: w' * Cov * w
    cVar = (w_ivp @ cov_sub @ w_ivp).item()
    
    return cVar


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1.0, index=sortIx) # Initialisierung als Float
    cItems = [sortIx] # initialize all items in one cluster
    
    while len(cItems) > 0:
        # Korrektur 1: Benutzung der Ganzzahl-Division (//) für die Bisektion
        cItems = [i[j:k] for i in cItems 
                  for j, k in ((0, len(i) // 2), 
                               (len(i) // 2, len(i))) 
                  if len(i) > 1]
        
        # Korrektur 2: Benutzung von range() anstelle von xrange()
        for i in range(0, len(cItems), 2): # parse in pairs
            cItems0 = cItems[i] # cluster 1
            cItems1 = cItems[i + 1] # cluster 2
            
            # getClusterVar muss definiert sein (siehe oben)
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            
            alpha = 1.0 - cVar0 / (cVar0 + cVar1)
            
            w[cItems0] *= alpha # weight 1
            w[cItems1] *= (1.0 - alpha) # weight 2
            
    return w / w.sum() # Finale Normierung (wichtig!)


# -----------------------------------------------------------------------------


# Erster Folgemonat (Optimierungsziel) ist jetzt 2000-01-01
monate = pd.date_range(start="2004-01-01", end="2024-12-01", freq="MS") # <-- ZURÜCK AUF 2000-01-01
portfolio_features = {}
hrp_port = {}
rueck = 6  # Zeitraum für Schätzung (UNVERÄNDERT)

for folgemonat in monate:
    start_window = folgemonat - relativedelta(month=rueck)
    
    # Filter: Tagesrenditen der letzten 6 Monate vor dem Folgemonat
    prev_window_returns = rendite_tag[
        (rendite_tag["Date"] >= start_window) &
        (rendite_tag["Date"] < folgemonat)
    ][aktien20_98]

    if prev_window_returns.empty:
        continue
    
    month_key = str(folgemonat.date())
    aktien = prev_window_returns.columns.tolist()
    
    # Korrelations-, Abstands- und Kovarianzmatrix
    corr, dist = hrp(prev_window_returns)
    cov = prev_window_returns.cov()
    
    # Dendrogramm (Linkage)
    dist_array = dist.values
    dist_vector = squareform(dist_array)
    Z = linkage(dist_vector, method = 'ward')

    # Quasidiagonalisierung
    order = getQuasiDiag(Z, aktien)
    
    # Gewichte für HRP
    w = getRecBipart(cov, order)
    
    # HRP Portfolios speichern
    hrp_port[month_key] = w.reindex(aktien20_98, fill_value=0.0)
    
    
    # Speicherung
    portfolio_features[str(folgemonat.date())] = {
        "Korrelation": corr,
        "Abstand": dist
    }
