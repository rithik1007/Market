"""
NSE Stock Universe — Curated list of liquid, institutional-grade NSE stocks
organized by sector for sector rotation analysis.
"""

SECTOR_STOCKS = {
    "Banking": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB",
        "BANDHANBNK", "AUBANK", "RBLBANK", "CANBK", "UNIONBANK",
    ],
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LTTS",
        "TATAELXSI", "HAPPSTMNDS", "ROUTE", "KPITTECH", "SONATSOFTW",
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
        "LUPIN", "BIOCON", "TORNTPHARM", "ALKEM", "IPCALAB",
        "LAURUSLABS", "GLENMARK", "NATCOPHARMA", "GRANULES", "STAR",
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BHARATFORG", "MOTHERSON",
        "BALKRISIND", "MRF", "EXIDEIND", "AMARAJABAT", "APOLLOTYRE",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "VBL",
        "EMAMILTD", "BIKAJI", "RADICO", "UBL", "PGHH",
    ],
    "Energy": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN",
        "TATAPOWER", "ADANIENSOL", "NHPC", "SJVN", "IREDA",
        "TORNTPOWER", "CESC", "JSW ENERGY", "IEX",
    ],
    "Metals & Mining": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
        "SAIL", "NMDC", "NATIONALUM", "HINDCOPPER", "APLAPOLLO",
        "RATNAMANI", "WELCORP", "JINDALSAW",
    ],
    "Infrastructure & Construction": [
        "LARSENTOUB", "ADANIENT", "ADANIPORTS", "DLF", "GODREJPROP",
        "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA", "PHOENIXLTD",
        "IRCON", "NBCC", "RVNL", "BEL", "HAL",
    ],
    "Financial Services": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCAMC", "SBILIFE", "HDFCLIFE",
        "ICICIPRULI", "MUTHOOTFIN", "MANAPPURAM", "CHOLAFIN", "SHRIRAMFIN",
        "M&MFIN", "LICHSGFIN", "PFC", "RECLTD", "CANFINHOME",
    ],
    "Chemicals": [
        "PIDILITIND", "SRF", "AARTI", "DEEPAKNTR", "CLEAN",
        "ATUL", "NAVINFLUOR", "FLUOROCHEM", "TATACHEM", "UPL",
        "PIIND", "SUMICHEM", "GNFC", "GSFC",
    ],
    "Telecom & Media": [
        "BHARTIARTL", "IDEA", "TTML", "NAZARA", "ZEEL",
        "NETWORK18", "TV18BRDCST", "PVRINOX",
    ],
    "Capital Goods": [
        "SIEMENS", "ABB", "HAVELLS", "VOLTAS", "BLUESTARLT",
        "CROMPTON", "KAYNES", "DIXON", "AFFLE", "CUMMINSIND",
        "THERMAX", "GRINDWELL", "TIMKEN", "SCHAEFFLER",
    ],
}

# Flatten to get full stock list with .NS suffix for yfinance
def get_all_tickers():
    """Return all NSE tickers with .NS suffix for yfinance."""
    tickers = []
    for stocks in SECTOR_STOCKS.values():
        for stock in stocks:
            tickers.append(f"{stock}.NS")
    return tickers

def get_stock_sector(ticker):
    """Return the sector for a given ticker."""
    clean = ticker.replace(".NS", "")
    for sector, stocks in SECTOR_STOCKS.items():
        if clean in stocks:
            return sector
    return "Unknown"

def get_sector_stocks(sector):
    """Return all stocks in a sector."""
    return SECTOR_STOCKS.get(sector, [])

# Nifty and Bank Nifty indices
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
}
