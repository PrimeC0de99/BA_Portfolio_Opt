import pandas as pd
import re

# Mapping-Dictionary für den Abgleich zwischen Textdatei und Kurs-CSV
ticker_map = {
    "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE", 
    "Münchener Rück": "MUV2.DE", "Telekom": "DTE.DE", "DHL": "DHL.DE", 
    "Rheinmetall": "RHM.DE", "E.ON": "EOAN.DE", "Infineon": "IFX.DE", 
    "Beiersdorf": "BEI.DE", "NVIDIA": "NVDA", "Microsoft": "MSFT", 
    "Alphabet": "GOOGL", "Amazon": "AMZN", "UnitedHealth": "UNH", 
    "Visa": "V", "Procter & Gamble": "PG", "Costco": "COST", 
    "NextEra": "NEE", "Lockheed Martin": "LMT", "Toyota": "7203.T", 
    "Sony": "6758.T", "Tokyo Electron": "8035.T", "Keyence": "6861.T", 
    "Mitsubishi UFJ": "8306.T", "Nintendo": "7974.T", "Shin-Etsu": "4063.T", 
    "Takedea Pharmaceutical": "4502.T", "Fast Retailing": "9983.T", "Itochu": "8001.T"
}

def convert_txt_to_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data = []
    company = None
    
    for line in lines:
        line = line.strip()
        # Leere Zeilen und Tabellenköpfe überspringen
        if not line or line.startswith("Year"):
            continue
            
        # Unternehmensnamen extrahieren
        if line.endswith(":"):
            company = line.replace(":", "").strip()
        else:
            # Werte anhand der Tabulatoren trennen
            parts = re.split(r'\t+', line)
            if len(parts) >= 2:
                year = parts[0]
                mcap = parts[1]
                change = parts[2] if len(parts) > 2 else ""
                ticker = ticker_map.get(company, company)
                
                data.append([company, ticker, year, mcap, change])
                
    df = pd.DataFrame(data, columns=['Company', 'Ticker', 'Year', 'Marketcap', 'Change'])
    df.to_csv("Marktkapitalisierung_2_clean.csv", index=False)
    return df

# Ausführung:
df_clean = convert_txt_to_csv("C:/Users/nicla/Desktop/Marktkapitalisierung.txt")