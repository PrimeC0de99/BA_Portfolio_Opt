# HRP
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# Hier alles selbst!!!!!
# ---------------------------------------------------------
#                       HRP
# ---------------------------------------------------------

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
