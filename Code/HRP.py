# HRP 
from scipy.cluster.hierarchy import linkage 
from scipy.spatial.distance import squareform 
import numpy as np 
import pandas as pd 
from dateutil.relativedelta import relativedelta 
 
import Daten as da 
import Markowitz as mark 
 
# Daten Setup
rendite_tag = da.rendite_tag    # Log-Renditen 
aktien = da.aktien20_98 
portfolio_input = mark.portfolio_features 
 
# --------------------------------------------------------- 
#                       HRP FUNKTIONEN
# --------------------------------------------------------- 
 
def hrp(df): 
    """ Generiert die Korrelations- und Abstandsmatrix. """ 
    corr = df.corr() 
    dist = 1 - corr 
    dist = np.sqrt(0.5 * dist) 
    
    # Symmetrie erzwingen
    dist = (dist + dist.T) / 2.0 
    return corr, dist 
 
def getQuasiDiag(Z, aktien_namen): 
    """ Ermittelt die quasi-diagonale Reihenfolge der Assets (Seriation). """ 
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
         
        # NEU: pd.concat statt deprecated append
        sortIx = pd.concat([sortIx, df0]).sort_index()  
        sortIx.index = range(sortIx.shape[0]) 
         
    return [aktien_namen[i] for i in sortIx.tolist()] 
 
def get_ivp_weights(cov_mat): 
    """ Berechnet Inverse-Varianz-Gewichte. """ 
    ivp = 1.0 / np.diag(cov_mat)  
    return ivp / np.sum(ivp) 
 
def getClusterVar(cov, cItems): 
    """ Berechnet die Varianz eines Clusters unter IVP-Gewichtung. """ 
    cov_sub = cov.loc[cItems, cItems].values 
    w_ivp = get_ivp_weights(cov_sub) 
    cVar = (w_ivp @ cov_sub @ w_ivp).item() 
    return cVar 
 
def getRecBipart(cov, sortIx): 
    """ Rekursive Bisektion zur Gewichtsallokation. """ 
    w = pd.Series(1.0, index=sortIx) 
    cItems = [sortIx] 
     
    while len(cItems) > 0: 
        cItems = [i[j:k] for i in cItems  
                  for j, k in ((0, len(i) // 2),  
                               (len(i) // 2, len(i)))  
                  if len(i) > 1] 
         
        for i in range(0, len(cItems), 2): 
            cItems0 = cItems[i] 
            cItems1 = cItems[i + 1] 
             
            cVar0 = getClusterVar(cov, cItems0) 
            cVar1 = getClusterVar(cov, cItems1) 
             
            alpha = 1.0 - cVar0 / (cVar0 + cVar1) 
             
            w[cItems0] *= alpha 
            w[cItems1] *= (1.0 - alpha) 
             
    return w / w.sum() 
 
# ----------------------------------------------------------------------------- 
#                       HRP LOOP (OPTIMIERUNG)
# ----------------------------------------------------------------------------- 

# FESTES ENDDATUM: 31.12.2025 (identisch zu Markowitz)
# Ändere freq="YS" zu "QS" für Quartale, falls gewünscht
jahre = pd.date_range(start="2006-01-01", end="2025-12-31", freq="QS")          # Frequenz YS, QS, 6ME

hrp_port = {} 
portfolio_features = {} 
rueck = 12  # Lookback in Monaten

for folgejahr in jahre: 
    start_window = folgejahr - relativedelta(months=rueck) 
     
    # Filter: Historische Renditen für die Baum-Struktur
    prev_window_returns = rendite_tag[ 
        (rendite_tag["Date"] >= start_window) & 
        (rendite_tag["Date"] < folgejahr) 
    ][aktien] 

    if prev_window_returns.empty: 
        continue 
     
    year_key = str(folgejahr.date()) 
     
    # 1. Korrelations- und Abstandsmatrix
    corr, dist = hrp(prev_window_returns) 
    
    # 2. Kovarianzmatrix aus dem Markowitz-Modul ziehen (Konsistenz!)
    cov = portfolio_input[year_key]['cov'] 
              
    # 3. Clustering (Linkage) 
    dist_array = dist.values 
    dist_vector = squareform(dist_array) 
    Z = linkage(dist_vector, method='ward') 
 
    # 4. Quasidiagonalisierung & Gewichte
    # Wir übergeben die Spaltennamen von 'dist', damit die Indizes passen
    order = getQuasiDiag(Z, dist.columns.tolist()) 
    w = getRecBipart(cov, order) 
     
    # 5. Speichern
    hrp_port[year_key] = w.reindex(aktien, fill_value=0.0) 
    portfolio_features[year_key] = { 
        "Korrelation": corr, 
        "Abstand": dist 
    } 

print("\n--- HRP-Optimierung abgeschlossen (Zeitraum bis 2025) ---")
