#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import HRP as HRP
import Markowitz as mark
import Daten as da

# Daten aus den Modulen laden
hrp_port = HRP.hrp_port
minvar_portfolios = mark.minvar_portfolios
rendite_tag = da.rendite_tag

def evaluate_portfolios(hrp_weights_dict, mv_weights_dict, daily_returns, risk_free_rate=0.0):
    """
    Berechnet Performance-Metriken für JÄHRLICHES Rebalancing.
    """
    # 1. Gewichte in DataFrames umwandeln
    df_hrp = pd.DataFrame(hrp_weights_dict).T
    df_mv = pd.DataFrame(mv_weights_dict).T
    
    df_hrp.index = pd.to_datetime(df_hrp.index)
    df_mv.index = pd.to_datetime(df_mv.index)

    # 2. Naive Diversifikation (1/n) Benchmark
    num_assets = len(df_hrp.columns)
    df_naive = pd.DataFrame(1.0 / num_assets, index=df_hrp.index, columns=df_hrp.columns)

    # 3. Portfolio-Renditen berechnen (Out-of-Sample für das gesamte Jahr)
    hrp_rets, mv_rets, naive_rets = [], [], []
    daily_returns['Date'] = pd.to_datetime(daily_returns['Date'])
    
    # Wir iterieren über die Jahre, an denen optimiert wurde
    for date in df_hrp.index:
        # Maske für das gesamte Kalenderjahr der Haltedauer
        # (Gewicht vom 01.01. wird auf das Jahr angewendet)
        mask = (daily_returns['Date'].dt.year == date.year)
        year_returns = daily_returns.loc[mask].set_index('Date')
        
        if year_returns.empty:
            continue
            
        # Gewichte für dieses Jahr ziehen
        w_hrp = df_hrp.loc[date].reindex(year_returns.columns).fillna(0)
        w_mv = df_mv.loc[date].reindex(year_returns.columns).fillna(0)
        w_naive = df_naive.loc[date].reindex(year_returns.columns).fillna(0)

        # Tägliche Portfolio-Returns für dieses Jahr berechnen
        hrp_rets.append(year_returns @ w_hrp)
        mv_rets.append(year_returns @ w_mv)
        naive_rets.append(year_returns @ w_naive)

    # Ergebnisse zu Zeitreihen zusammenführen
    portfolio_results = pd.DataFrame({
        'HRP': pd.concat(hrp_rets),
        'Markowitz (MinVar)': pd.concat(mv_rets),
        'Naive (1/n)': pd.concat(naive_rets)
    }).sort_index()

    # 4. Performance-Kennzahlen berechnen
    # Kumulierte Renditen (aus Log-Renditen)
    cum_returns = np.exp(portfolio_results.cumsum()) - 1
    
    # Metriken (annualisiert)
    ann_return = portfolio_results.mean() * 252
    ann_vol = portfolio_results.std() * np.sqrt(252)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol
    
    # Max Drawdown
    peak = (1 + cum_returns).expanding().max()
    drawdown = ((1 + cum_returns) / peak) - 1
    max_drawdown = drawdown.min()

    # Tabelle erstellen
    metrics = pd.DataFrame({
        'Ann. Rendite': ann_return,
        'Ann. Volatilität': ann_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }).T
    
    return cum_returns, metrics, portfolio_results

def calculate_turnover(weights_dict):
    """Berechnet den durchschnittlichen Turnover pro Rebalancing-Ereignis."""
    df_weights = pd.DataFrame(weights_dict).T
    # Die absolute Differenz der Gewichte zwischen den Jahren
    diff = df_weights.diff().abs().sum(axis=1)
    return diff.mean()

# =============================================================================
# HAUPTPROGRAMM / AUSFÜHRUNG
# =============================================================================

try:
    # 1. Performance-Auswertung starten
    cum_returns, metrics, p_rets = evaluate_portfolios(hrp_port, minvar_portfolios, rendite_tag)

    # 2. Visualisierung
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns[col], label=col)
    
    plt.title("Kumulierte Performance - Jährliches Rebalancing (Out-of-Sample)", fontsize=14)
    plt.ylabel("Kumulierte Rendite", fontsize=12)
    plt.xlabel("Datum", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Metriken und Turnover anzeigen
    print("\n--- Performance Metriken (Jährliches Rebalancing) ---")
    print(metrics.round(4))

    turnover_hrp = calculate_turnover(hrp_port)
    turnover_mv = calculate_turnover(minvar_portfolios)
    
    print("\n--- Durchschnittlicher jährlicher Turnover ---")
    print(f"HRP:                {turnover_hrp:.4f}")
    print(f"Markowitz (MinVar): {turnover_mv:.4f}")
    print(f"Naive (1/n):        0.0000")

    # 4. LaTeX Export
    print("\n--- LaTeX Code für die Tabelle ---")
    # Wir fügen den Turnover der Tabelle hinzu für den Export
    export_metrics = metrics.copy()
    export_metrics.loc['Annu. Turnover'] = [turnover_hrp, turnover_mv, 0.0]
    print(export_metrics.round(4).to_latex())

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")

