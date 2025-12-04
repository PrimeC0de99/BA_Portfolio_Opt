# import numpy as np

import Daten as da
import Markowitz as ma
import HRP as hrp

# Daten laden
hrp = hrp.hrp_port
mark = ma.minvar_portfolios
# daily_returns = da.daily_returns
Daten_20_extended = da.Daten_20_extended
valid_stocks = da.valid_stocks



# Bestimmen der tatsächlichen täglichen Renditen
daily_returns = Daten_20_extended.copy() # <-- Basis ist jetzt der erweiterte Datensatz
daily_returns[valid_stocks] = (daily_returns[valid_stocks] / daily_returns[valid_stocks].shift(1)) - 1
daily_returns = daily_returns.dropna() # Entfernt die erste Zeile (Rendite vom 02.07.1999 ist die erste)

# tägliche Renditen ab dem Jahr 2004
daily_returns2004 = daily_returns.loc[daily_returns['Date'] >= '2004-01-01', ['Date'] + valid_stocks]

print('\n Tägliche Renditen ab 2004: \n', daily_returns2004.head(10))





'''
Plots für monatliche Varianz machen, da wir monatlich das Portfolio neu bestimmen.
So kann man prüfen welches Portfolio in einem bestimmten Zeitraum besser performt.
Aber natürlich auch die Varianz über den gesamten Zeitraum bestimmen (für die generelle Performance).

Ggf. von monatlicher Allokation auf halbjährlich wechseln?
'''

