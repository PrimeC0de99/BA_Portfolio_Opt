#import numpy as np
import Daten as da
import Markowitz as ma
import HRP as hrp

# Daten laden
hrp_gewichte = hrp.hrp_port
mark_gewichte = ma.minvar_portfolios
#daily_returns = da.daily_returns
Daten_20_extended = da.daten_20_erweitert
aktien = da.aktien20_98

# Bestimmen der tatsächlichen täglichen Renditen
daily_returns = Daten_20_extended.copy()    # <-- Basis ist jetzt der erweiterte Datensatz
daily_returns[aktien] = (daily_returns[aktien] / daily_returns[aktien].shift(1)) - 1
daily_returns = daily_returns.dropna()      # Entfernt die erste Zeile (Rendite vom 02.07.1999 ist die erste)

# tägliche Renditen ab dem Jahr 2004
daily_returns2004 = daily_returns.loc[daily_returns['Date'] >= '2004-01-01', ['Date'] + aktien]
print('\n Tägliche Renditen ab 2004: \n', daily_returns2004.head(10))

# tägliche Renditefaktoren
daily_returns2004_faktoren = daily_returns2004[aktien] + 1
daily_returns2004_faktoren.insert(0, 'Date', daily_returns2004['Date'].values) # Datumsspalte einfügen

print('\n Tägliche Renditefaktoren (2004): \n', daily_returns2004_faktoren)

# Fügen Sie dem Faktoren-DataFrame einen Monats-Key hinzu
daily_returns2004_faktoren['Month_Key'] = daily_returns2004_faktoren['Date'].dt.to_period('M')

print('\n Faktoren-DF mit Month_Key: \n', daily_returns2004_faktoren.head(10))

# Gruppiert nach Monat und multipliziert alle täglichen Faktoren jeder Aktie
monthly_factors = daily_returns2004_faktoren.groupby('Month_Key')[aktien].prod()

print('\n Monatliche Faktoren (kumuliert): \n', monthly_factors.head(5))

# monatliche Renditen
return_month = monthly_factors - 1

print('\n Monatliche Renditen: \n', return_month.head(10))


performance_monatlich = 1





'''
Plots für monatliche Varianz machen, da wir monatlich das Portfolio neu bestimmen.
So kann man prüfen welches Portfolio in einem bestimmten Zeitraum besser performt.
Aber natürlich auch die Varianz über den gesamten Zeitraum bestimmen (für die generelle Performance).

Ggf. von monatlicher Allokation auf halbjährlich wechseln?
'''
