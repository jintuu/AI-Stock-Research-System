"""
Stock universe management.
Provides S&P 500, NASDAQ-100, Dow-30 and custom lists — all from free sources.
"""

import logging
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Hard-coded fallbacks (work offline / if Wikipedia is blocked) ──────────────

_DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]

_NASDAQ100_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "ASML", "COST", "AZN", "NFLX", "AMD", "ADBE", "QCOM", "PEP", "CSCO",
    "TMUS", "INTU", "AMAT", "TXN", "BKNG", "ISRG", "AMGN", "HON", "VRTX",
    "LRCX", "MU", "PANW", "GILD", "MELI", "ADI", "KLAC", "SNPS", "REGN",
    "CDNS", "MDLZ", "SBUX", "ABNB", "MAR", "ORLY", "FTNT", "CEG", "NXPI",
    "CTAS", "PCAR", "WDAY", "MCHP", "MNST", "PYPL", "PAYX", "ADP", "IDXX",
    "KDP", "EXC", "CPRT", "BIIB", "DLTR", "ON", "FAST", "ODFL", "ZS",
    "VRSK", "CTSH", "GEHC", "DDOG", "GFS", "TEAM", "TTD", "ANSS",
    "EA", "ALGN", "FANG", "KHC", "WBA", "ILMN", "ENPH", "ZM", "CRWD",
    "DXCM", "WBD", "CMCSA", "CHTR", "EBAY", "LULU", "SIRI", "PDD",
]

# Full S&P 500 fallback — used when Wikipedia fetch fails
_SP500_FULL_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","AVGO",
    "JPM","LLY","V","UNH","XOM","MA","JNJ","PG","COST","HD","MRK","ABBV",
    "BAC","CVX","NFLX","CRM","AMD","PEP","KO","WMT","TMO","ADBE","ACN",
    "CSCO","MCD","ABT","ORCL","DIS","LIN","DHR","PM","QCOM","GE","VZ","TXN",
    "AMGN","IBM","RTX","CAT","SPGI","GS","INTU","LOW","ISRG","AXP","T","NOW",
    "HON","BLK","BKNG","VRTX","ELV","MDT","GILD","SYK","CI","ADP","REGN",
    "MMC","PLD","CB","MDLZ","SO","DE","TJX","SBUX","BDX","AMT","C","MS","MO",
    "SCHW","CME","FI","NOC","AMAT","ITW","PGR","BMY","ZTS","CL","LRCX","GD",
    "MU","WM","PANW","APD","USB","MMM","EOG","PSA","MCO","TGT","FCX","AON",
    "NSC","EMR","ICE","MCK","WFC","F","GM","EW","HCA","SHW","COP","MPC",
    "PSX","VLO","OXY","HAL","SLB","BKR","FANG","DVN","PXD","APA","MRO",
    "CXO","HES","COG","RRC","AR","EQT","CNX","SM","CTRA","OVV","MTDR",
    "PDCE","ROCC","GPOR","REI","ESTE","KLAC","MCHP","SNPS","CDNS","ANSS",
    "CTSH","EPAM","GLOB","INFY","WIT","IQVIA","LDOS","SAIC","BAH","CACI",
    "MANW","KEYW","DXC","HPE","HPQ","DELL","STX","WDC","NTAP","PSTG","PRGS",
    "VRNS","DDOG","SNOW","MDB","ESTC","SPLK","SUMO","DT","NEWR","APPN",
    "HWM","TDG","HII","L3H","TXT","CW","HEICO","SPR","AIR","AAL","DAL",
    "UAL","LUV","ALK","SAVE","JBLU","HA","ALGT","SKY","ULTA","ROST","TJX",
    "GPS","AEO","ANF","URBN","ZUMZ","EXPR","CATO","PLCE","BIG","BURL",
    "WSM","RH","PRGS","FIVE","OLLI","NDC","DLTR","DG","KR","SFM","WMB",
    "JKHY","FISV","FIS","GPN","PAYX","ADP","WEX","BR","EVTC","PRFT","EPAY",
    "NCR","FSTR","CORE","BOK","PPBI","CVBF","SFNC","HTLF","IBCP","BPOP",
    "HBAN","RF","CFG","FITB","KEY","MTB","ZION","CMA","FHN","SNV","SYBT",
    "CBTX","BNCC","BANF","FMBH","FBIZ","FXNC","PFBX","BFIN","BFC","LKFN",
    "MBWM","MCBC","NWIN","OLBK","PBHC","PFIS","RBCAA","SBSI","TBNK","TCBK",
    "ZBK","AGBA","AHH","AIV","AKR","ALEX","ALX","AMH","ARE","BDN","BPR",
    "BRX","BXP","CBL","CIO","CLI","CLPR","COLD","CPT","CUZ","DEA","DEI",
    "DLR","DOC","DRH","EQC","EQR","ESS","EXR","FPH","FR","GEO","HIW",
    "HPP","HR","INN","INVH","IRM","IRT","JBGS","KIM","KRC","KRG","LTC",
    "LXP","MAC","MDC","MPW","NNN","NSA","NVR","NXRT","OFC","OHI","OPEN",
    "OUT","PEB","PK","PLYM","QTS","REG","REXR","RHP","RLJ","RPT","SAFE",
    "SBRA","SLG","SPG","SRC","STAG","STAR","STOR","SUI","TCO","TCI","TI",
    "TPH","TPR","UDR","UE","VNO","VNQ","WELL","WPT","WRE","WSR","XHR",
    "YOD","ABM","ACM","ADC","ADT","AEE","AEP","AES","AFG","AFL","AGCO",
    "AGR","AIG","AIN","AIZ","AJRD","AJG","ALE","ALG","ALGN","ALK","ALL",
    "ALLE","ALTR","ALXN","AMAT","AMCR","AME","AMGN","AMP","AMT","AMTM",
    "AN","ANF","AON","AOS","APA","APD","APH","APTV","ARE","ARL","ARNC",
    "ARW","ASH","ASMB","ASO","ATI","ATO","ATVI","AVB","AVGO","AVY","AWK",
    "AXP","AYI","AZO","BA","BAC","BAX","BBWI","BBY","BC","BDX","BEN","BF-B",
    "BIDU","BIIB","BIO","BK","BKNG","BKR","BLK","BLL","BMY","BR","BRK-B",
    "BSX","BWA","BXP","BYD","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE",
    "CCI","CCK","CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR",
    "CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COO","COP","COST","CPB","CPRT","CRL","CRM","CRWD","CSX","CTAS",
    "CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DD","DE",
    "DECK","DFS","DG","DGX","DHI","DHR","DIS","DISH","DLR","DLTR","DOV",
    "DOW","DPZ","DRE","DRI","DTE","DUK","DVA","DVN","DXC","DXCM","EA",
    "EBAY","ECL","ED","EFX","EIX","EL","EMN","EMR","ENPH","EOG","EPAM",
    "EQIX","EQR","EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC",
    "EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV",
    "FI","FICO","FIS","FISV","FITB","FLT","FMC","FOX","FOXA","FRT","FTNT",
    "FTV","GD","GE","GEHC","GEN","GILD","GIS","GL","GLW","GM","GNRC","GOOG",
    "GOOGL","GPC","GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD",
    "HES","HIG","HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC","HST",
    "HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","ILMN","INCY",
    "INTC","INTU","INVH","IP","IPG","IQV","IRM","ISRG","IT","ITW","IVZ",
    "J","JBHT","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS",
    "KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR","L","LDOS","LEN","LH",
    "LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX","LULU","LUV",
    "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO",
    "MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM",
    "MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRO","MS","MSCI","MSFT",
    "MSI","MTB","MTD","MU","NCLH","NDAQ","NEE","NEM","NFLX","NI","NKE",
    "NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWL","NWS",
    "NWSA","NXPI","O","ODFL","OGN","OKE","OMC","ON","OPEN","ORCL","ORLY",
    "OTIS","OXY","PAYC","PAYX","PCAR","PCG","PEAK","PEG","PEP","PFE","PFG",
    "PG","PGR","PH","PHM","PKG","PLD","PM","PNC","PNR","PNW","POOL","PPG",
    "PPL","PRU","PSA","PSX","PTC","PWR","PXD","PYPL","QCOM","QRVO","RCL",
    "RE","REG","REGN","RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST",
    "RSG","RTX","SBAC","SBUX","SCHW","SHW","SIVB","SJM","SLB","SNA","SNPS",
    "SO","SPG","SPGI","SRE","STE","STT","STX","STZ","SWK","SWKS","SYF",
    "SYK","SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT",
    "TJX","TMO","TMUS","TPR","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT",
    "TTWO","TXN","TXT","TYL","UA","UAA","UAL","UDR","UHS","ULTA","UNH",
    "UNP","UPS","URI","USB","V","VFC","VICI","VLO","VMC","VNO","VRSK",
    "VRSN","VRTX","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC",
    "WELL","WFC","WHR","WM","WMB","WMT","WRB","WRK","WST","WTW","WY","WYNN",
    "XEL","XOM","XRAY","XYL","YUM","ZBH","ZBRA","ZION","ZTS",
]

# Keep the old name as an alias for backward compatibility
_SP500_TOP100 = _SP500_FULL_FALLBACK[:100]


def get_sp500_tickers() -> List[str]:
    """Fetch full S&P 500 list — tries two Wikipedia strategies before falling back."""

    # Strategy 1: pd.read_html (fast)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, header=0)
        df = tables[0]
        if "Symbol" in df.columns:
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            if len(tickers) >= 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia (strategy 1).")
                return tickers
    except Exception as e:
        logger.debug(f"S&P 500 Wikipedia strategy 1 failed: {e}")

    # Strategy 2: requests + BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockResearch/1.0)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=10,
        )
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"id": "constituents"})
        if table:
            tickers = []
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if cells:
                    sym = cells[0].get_text(strip=True).replace(".", "-")
                    if sym:
                        tickers.append(sym)
            if len(tickers) >= 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers via BeautifulSoup.")
                return tickers
    except Exception as e:
        logger.debug(f"S&P 500 Wikipedia strategy 2 failed: {e}")

    # Strategy 3: hardcoded full fallback
    logger.warning("All Wikipedia fetches failed — using hardcoded S&P 500 fallback list.")
    return _SP500_FULL_FALLBACK


def get_nasdaq100_tickers() -> List[str]:
    """Fetch NASDAQ-100 from Wikipedia (falls back to sample list)."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url, header=0)
        for t in tables:
            if "Ticker" in t.columns:
                return t["Ticker"].tolist()
            if "Symbol" in t.columns:
                return t["Symbol"].tolist()
        raise ValueError("Ticker column not found in Wikipedia table.")
    except Exception as e:
        logger.warning(f"NASDAQ-100 fetch failed ({e}), using fallback sample.")
        return _NASDAQ100_SAMPLE


def get_universe(config: dict) -> List[str]:
    """Return the list of tickers to scan according to config."""
    cfg          = config.get("stock_universe", {})
    universe_type = cfg.get("default", "sp500_top100")
    custom       = cfg.get("custom_tickers", [])
    max_stocks   = int(cfg.get("max_stocks", 50))   # cast to int — Streamlit can return float

    if custom:
        tickers = [t.upper().strip() for t in custom if t.strip()]
    elif universe_type == "sp500":
        tickers = get_sp500_tickers()
    elif universe_type == "nasdaq100":
        tickers = get_nasdaq100_tickers()
    elif universe_type == "dow30":
        tickers = _DOW30
    else:  # sp500_top100 — auto-upgrade to full list if user wants more than 100
        tickers = get_sp500_tickers() if max_stocks > 100 else _SP500_TOP100

    # Deduplicate preserving order, then cap
    seen, unique = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:max_stocks]
