import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import HRP as HRP
import Markowitz as mark
import Daten as da

# 1. Daten laden
hrp_port = HRP.hrp_port
minvar_portfolios = mark.minvar_portfolios
rendite_tag = da.rendite_tag

# -----------------------------------------------------------------------------
# DATEN-SCHNITT & VORBEREITUNG
# -----------------------------------------------------------------------------
end_date = pd.Timestamp("2025-12-31")
rendite_tag = rendite_tag[rendite_tag['Date'] <= end_date].copy()

def evaluate_portfolios(hrp_weights_dict, mv_weights_dict, daily_returns, risk_free_rate=0.0):
    """
    Berechnet Performance-Metriken für JEDE Rebalancing-Frequenz.
    """
    df_hrp = pd.DataFrame(hrp_weights_dict).T
    df_mv = pd.DataFrame(mv_weights_dict).T
    df_hrp.index = pd.to_datetime(df_hrp.index)
    df_mv.index = pd.to_datetime(df_mv.index)

    num_assets = len(df_hrp.columns)
    df_naive = pd.DataFrame(1.0 / num_assets, index=df_hrp.index, columns=df_hrp.columns)

    hrp_rets, mv_rets, naive_rets = [], [], []
    daily_returns['Date'] = pd.to_datetime(daily_returns['Date'])
    
    # Termine sortieren für korrekte Intervallbildung
    rebal_dates = df_hrp.index.sort_values()

    for i in range(len(rebal_dates)):
        current_date = rebal_dates[i]
        
        # Fenster bestimmen (vom aktuellen zum nächsten Rebalancing)
        if i < len(rebal_dates) - 1:
            next_date = rebal_dates[i+1]
        else:
            next_date = daily_returns['Date'].max() + pd.Timedelta(days=1)

        mask = (daily_returns['Date'] >= current_date) & (daily_returns['Date'] < next_date)
        window_returns = daily_returns.loc[mask].set_index('Date')
        
        if window_returns.empty:
            continue
            
        # Gewichte anwenden
        w_hrp = df_hrp.loc[current_date].reindex(window_returns.columns).fillna(0)
        w_mv = df_mv.loc[current_date].reindex(window_returns.columns).fillna(0)
        w_naive = df_naive.loc[current_date].reindex(window_returns.columns).fillna(0)

        hrp_rets.append(window_returns @ w_hrp)
        mv_rets.append(window_returns @ w_mv)
        naive_rets.append(window_returns @ w_naive)

    # Zusammenführen
    portfolio_results = pd.DataFrame({
        'HRP': pd.concat(hrp_rets),
        'Markowitz (MinVar)': pd.concat(mv_rets),
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

# =============================================================================
# AUSFÜHRUNG & PLOTS
# =============================================================================

try:
    cum_returns, metrics, p_rets = evaluate_portfolios(hrp_port, minvar_portfolios, rendite_tag)

    # 1. Performance Plot
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns[col], label=col)
    plt.title("Kumulierte Performance (Out-of-Sample)", fontsize=14)
    plt.ylabel("Rendite", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"performance.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # 2. Metriken & Turnover
    turnover_hrp = calculate_turnover(hrp_port)
    turnover_mv = calculate_turnover(minvar_portfolios)
    metrics.loc['Avg. Turnover'] = [turnover_hrp, turnover_mv, 0.0]

    print("\n--- Final Metrics ---")
    print(metrics.round(4))

    # 3. Gewichts-Plots (Area Charts)
    def plot_weights(weights_dict, title):
        df_w = pd.DataFrame(weights_dict).T
        df_w.index = pd.to_datetime(df_w.index)
        df_w.plot.area(figsize=(12, 5), stacked=True, cmap='tab20', alpha=0.8)
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(f"gewichte.pdf", format='pdf', bbox_inches='tight')
        plt.show()

    plot_weights(hrp_port, "HRP Gewichtsverteilung")
    plot_weights(minvar_portfolios, "Markowitz Gewichtsverteilung")             # Überschreibt aktuell den HRP Plot
                                                                                # mit dem Markowitz Plot
    # 4. Stabilitäts-Plot (Konditionszahl der Kovarianzmatrix)
    # Wir nehmen die Daten direkt aus dem mark-Modul (df_stability)
    if hasattr(mark, 'df_stability'):
        plt.figure(figsize=(12, 5))
        # Semilogarithmische Darstellung, da Konditionszahlen oft Größenordnungen springen
        plt.semilogy(mark.df_stability.index, mark.df_stability['Condition_Number'], 
                     color='#d62728', marker='o', markersize=4, label='Konditionszahl $\kappa(\Sigma)$')
        
        # Optische Hilfslinie für "kritische" Instabilität (Faustregel 10^3 bis 10^4)
        plt.axhline(y=1000, color='black', linestyle='--', alpha=0.3, label='Schwelle zur Instabilität ($10^3$)')
        
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
    
