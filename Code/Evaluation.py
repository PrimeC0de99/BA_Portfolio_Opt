import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import HRP as HRP
import Markowitz as mark
import Daten as da
import re


# 1. Daten laden
hrp_port = HRP.hrp_port
minvar_portfolios = mark.minvar_portfolios
rendite_tag = da.rendite_tag

# Definition der Aktienliste direkt aus da
aktien_liste = da.aktien20_98 


# =============================================================================
# MARKTKAPITALISIERUNG LADEN & MARKTGEWICHTES PORTFOLIO ERZEUGEN
# =============================================================================

# 1. Bereinigte CSV direkt in Pandas laden
df_raw = pd.read_csv(r"C:/Users/nicla/Documents/VS_Projekte/BA/BA_Portfolio_Opt/Daten/Marktkapitalisierung_2_clean.csv")

# 2. Robuste Funktion zur Umwandlung von Text (z.B. "$5.271 T" oder "$238.18 B") in Floats
def clean_marketcap(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace('$', '').replace(',', '').strip()
    if 'T' in val_str:
        # Billionen (Trillions) in Milliarden umrechnen, damit die Relation stimmt
        return float(val_str.replace('T', '').strip()) * 1000.0
    elif 'B' in val_str:
        return float(val_str.replace('B', '').strip())
    else:
        try:
            return float(val_str)
        except ValueError:
            return 0.0

df_raw['Marketcap'] = df_raw['Marketcap'].apply(clean_marketcap)

# 3. Pivot-Tabelle erstellen: Jahre als Index, Ticker als Spalten
mcap_df = df_raw.pivot(index='Year', columns='Ticker', values='Marketcap').fillna(0.0)

# DEBUG
print("\n--- Namensabgleich ---")
print("Spalten in der mcap_df (erste 5):", mcap_df.columns.tolist()[:5])
print("Namen in der Kurs-CSV (erste 5):", aktien_liste[:5])
print("----------------------\n")

# Marktgewichtetes Portfolio für deine Rebalancing-Stichtage aufbauen
mkt_port = {}
for key in hrp_port.keys():
    jahr = pd.to_datetime(key).year
    if jahr in mcap_df.index:
        mcap_t = mcap_df.loc[jahr].reindex(aktien_liste, fill_value=0.0)
    else:
        mcap_t = pd.Series(0.0, index=aktien_liste)
    
    sum_mcap = mcap_t.sum()
    if sum_mcap > 0:
        w_mkt = mcap_t / sum_mcap
    else:
        w_mkt = pd.Series(1.0 / len(aktien_liste), index=aktien_liste)
        
    mkt_port[key] = w_mkt


# -----------------------------------------------------------------------------
# Vorbereitung
# -----------------------------------------------------------------------------
end_date = pd.Timestamp("2025-12-31")
rendite_tag = rendite_tag[rendite_tag['Date'] <= end_date].copy()

def evaluate_portfolios(hrp_weights_dict, mv_weights_dict, mkt_weights_dict, daily_returns, risk_free_rate=0.0):
    """
    Berechnet Performance-Metriken für JEDE Rebalancing-Frequenz.
    Garantierte chronologische Konsistenz durch explizite Datetime-Sortierung.
    """
    # 1. Daten in DataFrames konvertieren und den Index sauber in Datetime transformieren
    df_hrp = pd.DataFrame(hrp_weights_dict).T
    df_mv = pd.DataFrame(mv_weights_dict).T
    df_mkt = pd.DataFrame(mkt_weights_dict).T 
    
    df_hrp.index = pd.to_datetime(df_hrp.index)
    df_mv.index = pd.to_datetime(df_mv.index)
    df_mkt.index = pd.to_datetime(df_mkt.index) 
    
    # Sortieren, um chronologische Konsistenz zu garantieren
    df_hrp = df_hrp.sort_index()
    df_mv = df_mv.sort_index()
    df_mkt = df_mkt.sort_index() 

    num_assets = len(df_hrp.columns)
    df_naive = pd.DataFrame(1.0 / num_assets, index=df_hrp.index, columns=df_hrp.columns)

    hrp_rets, mv_rets, mkt_rets, naive_rets = [], [], [], [] 
    daily_returns['Date'] = pd.to_datetime(daily_returns['Date'])
    
    # Echte, harmonisierte Rebalancing-Termine nutzen
    rebal_dates = df_hrp.index

    for i in range(len(rebal_dates)):
        current_date = rebal_dates[i]
        
        # Fenster exakt bestimmen
        if i < len(rebal_dates) - 1:
            next_date = rebal_dates[i+1]
        else:
            next_date = daily_returns['Date'].max() + pd.Timedelta(days=1)

        mask = (daily_returns['Date'] >= current_date) & (daily_returns['Date'] < next_date)
        window_returns = daily_returns.loc[mask].set_index('Date')
        
        if window_returns.empty:
            continue
            
        # Gewichte mathematisch exakt über loc[current_date] zuordnen
        w_hrp = df_hrp.loc[current_date].reindex(window_returns.columns).fillna(0)
        w_mv = df_mv.loc[current_date].reindex(window_returns.columns).fillna(0)
        w_mkt = df_mkt.loc[current_date].reindex(window_returns.columns).fillna(0) 
        w_naive = df_naive.loc[current_date].reindex(window_returns.columns).fillna(0)

        hrp_rets.append(window_returns @ w_hrp)
        mv_rets.append(window_returns @ w_mv)
        mkt_rets.append(window_returns @ w_mkt) 
        naive_rets.append(window_returns @ w_naive)

    # Zusammenführen der täglichen Renditen
    portfolio_results = pd.DataFrame({
        'HRP': pd.concat(hrp_rets),
        'Markowitz (MinVar)': pd.concat(mv_rets),
        'Market Cap': pd.concat(mkt_rets), 
        'Naive (1/n)': pd.concat(naive_rets)
    }).sort_index()

    # Performance-Kennzahlen (arithmetisch kumuliert aus Log-Returns)
    cum_returns = np.exp(portfolio_results.cumsum()) - 1
    ann_return = portfolio_results.mean() * 252
    ann_vol = portfolio_results.std() * np.sqrt(252)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol
    
    # Max Drawdown (Korrekt über Wealth Index)
    wealth_index = (1 + cum_returns)
    peak = wealth_index.expanding().max()
    drawdown = (wealth_index / peak) - 1
    max_drawdown = drawdown.min()

    metrics = pd.DataFrame({
        'Ann. Rendite': ann_return,
        'Ann. Volatilität': ann_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }).T
    
    return cum_returns, metrics, portfolio_results

def calculate_turnover(weights_dict):
    df_weights = pd.DataFrame(weights_dict).T
    diff = df_weights.diff().abs().sum(axis=1)
    return diff.mean()

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

try:
    cum_returns, metrics, p_rets = evaluate_portfolios(hrp_port, minvar_portfolios, mkt_port, rendite_tag)

    # -------------------------------------------------------------------------
    # 1. Performance Plots
    # -------------------------------------------------------------------------
    
    # Basis-Wertentwicklung über den gesamten Zeitraum (Startwert = 1)
    wertentwicklung = 1 + cum_returns

    # A) Gesamt-Plot (2006 – 2025) für den Hauptteil
    plt.figure(figsize=(12, 6))
    for col in wertentwicklung.columns:
        plt.plot(wertentwicklung[col], label=col)
    plt.title("Kumulierte Wertentwicklung (Out-of-Sample: 2006 – 2025)", fontsize=14)
    plt.ylabel("Portfolio-Wert (Basis 1)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_gesamt.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # B) Teil 1: 2006 – 2015 (Re-Basiert auf 1)
    we_p1 = wertentwicklung.loc['2006-01-01':'2015-12-31']
    we_p1 = we_p1 / we_p1.iloc[0]  # Auf 1 normieren

    plt.figure(figsize=(10, 5))
    for col in we_p1.columns:
        plt.plot(we_p1[col], label=col)
    plt.title("Kumulierte Wertentwicklung (2006 – 2015)", fontsize=13)
    plt.ylabel("Portfolio-Wert (Basis 1)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_2006_2015.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # C) Teil 2: 2016 – 2025 (Re-Basiert auf 1)
    we_p2 = wertentwicklung.loc['2016-01-01':'2025-12-31']
    we_p2 = we_p2 / we_p2.iloc[0]  # Auf 1 normieren

    plt.figure(figsize=(10, 5))
    for col in we_p2.columns:
        plt.plot(we_p2[col], label=col)
    plt.title("Kumulierte Wertentwicklung (2016 – 2025)", fontsize=13)
    plt.ylabel("Portfolio-Wert (Basis 1)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_2016_2025.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    # 2. Metriken & Turnover
    turnover_hrp = calculate_turnover(hrp_port)
    turnover_mv = calculate_turnover(minvar_portfolios)
    turnover_mkt = calculate_turnover(mkt_port) 
    
    metrics.loc['Avg. Turnover'] = [turnover_hrp, turnover_mv, turnover_mkt, 0.0] 

    print("\n--- Final Metrics ---")
    print(metrics.round(4))

    # 3. Gewichts-Plots
    def plot_weights(weights_dict, title):
        df_w = pd.DataFrame(weights_dict).T
        df_w.index = pd.to_datetime(df_w.index)
        df_w.plot.area(figsize=(12, 5), stacked=True, cmap='tab20', alpha=0.8)
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.pdf", format='pdf', bbox_inches='tight') 
        plt.show()

    plot_weights(hrp_port, "HRP Gewichtsverteilung")
    plot_weights(minvar_portfolios, "Markowitz Gewichtsverteilung")           
    plot_weights(mkt_port, "Market Cap Gewichtsverteilung") 

    # 4. Stabilitäts-Plot (Konditionszahl der Kovarianzmatrix)
    if hasattr(mark, 'df_stability'):
        plt.figure(figsize=(12, 5))
        plt.semilogy(mark.df_stability.index, mark.df_stability['Condition_Number'], 
                     color='#d62728', marker='o', markersize=4, label='Konditionszahl $\kappa(\Sigma)$')
        
        plt.title("Numerische Stabilität der Schätzung (Konditionszahl)", fontsize=14)
        plt.ylabel("Konditionszahl (Log-Skala)", fontsize=12)
        plt.xlabel("Rebalancing-Zeitpunkt", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"kondition.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    else:
        print("\n[Hinweis] df_stability wurde im Markowitz-Modul nicht gefunden.")
    
except Exception as e:
    print(f"Fehler: {e}")