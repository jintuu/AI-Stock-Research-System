"""
India Stock Universe
────────────────────
Provides NSE stock lists: NIFTY 50, 100, 200, 500 and custom.
All tickers use the .NS suffix (NSE).

Sources (free):
  • Wikipedia — NIFTY 50, NIFTY 100 constituent pages
  • Hardcoded fallback lists for reliability
"""

import logging
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Hardcoded fallbacks ───────────────────────────────────────────────────────

# NIFTY 50 constituents (as of 2024-25)
_NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "TITAN.NS", "NTPC.NS", "POWERGRID.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "TECHM.NS", "ONGC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS",
    "ADANIENT.NS", "GRASIM.NS", "BPCL.NS", "TATACONSUM.NS", "SBILIFE.NS",
    "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "HCLTECH.NS", "TATASTEEL.NS", "HINDALCO.NS", "LTIM.NS", "ADANIPORTS.NS",
    "BRITANNIA.NS", "HDFCLIFE.NS", "UPL.NS", "M&M.NS", "BAJAJ-AUTO.NS",
]

# NIFTY NEXT 50 constituents
_NIFTY_NEXT50 = [
    "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANDHANBNK.NS", "BANKBARODA.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "CUMMINSIND.NS", "DLF.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "ICICIGI.NS", "INDUSTOWER.NS", "INDIGO.NS", "JUBLFOOD.NS",
    "LUPIN.NS", "MFSL.NS", "MOTHERSON.NS", "MPHASIS.NS",
    "NAUKRI.NS", "NMDC.NS", "OFSS.NS", "PAGEIND.NS",
    "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS",
    "RECLTD.NS", "SAIL.NS", "SHREECEM.NS", "SIEMENS.NS",
    "SRTRANSFIN.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
    "VEDL.NS", "VOLTAS.NS", "ZEEL.NS", "ZOMATO.NS",
    "PAYTM.NS", "DMART.NS", "MARICO.NS", "DABUR.NS",
    "PGHH.NS", "MCDOWELL-N.NS",
]

# Broader NIFTY 200 (additional stocks beyond top 100)
_NIFTY_200_EXTRA = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ALKEM.NS",
    "AMARAJABAT.NS", "APLAPOLLO.NS", "ATGL.NS", "AUBANK.NS", "BALKRISIND.NS",
    "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS", "BIOCON.NS",
    "CANFINHOME.NS", "CANBK.NS", "CDSL.NS", "COFORGE.NS", "CONCOR.NS",
    "CROMPTON.NS", "CUB.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", "DIXON.NS",
    "DLPL.NS", "EDELWEISS.NS", "EMAMILTD.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    "FORTIS.NS", "GAIL.NS", "GICRE.NS", "GLAXO.NS", "GMRINFRA.NS",
    "GNFC.NS", "GRANULES.NS", "GSFC.NS", "GUJGASLTD.NS", "HDFC.NS",
    "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HONAUT.NS", "IBREALEST.NS",
    "IDBI.NS", "IDFCFIRSTB.NS", "IIFL.NS", "INDIANB.NS", "INDHOTEL.NS",
    "IOB.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS",
    "JKCEMENT.NS", "JSWENERGY.NS", "JUBILANT.NS", "KAJARIACER.NS", "KEC.NS",
    "KPITTECH.NS", "KRBL.NS", "L&TFH.NS", "LALPATHLAB.NS", "LICHSGFIN.NS",
    "LINDEINDIA.NS", "LTTS.NS", "LUXIND.NS", "M&MFIN.NS", "MANAPPURAM.NS",
    "MAXHEALTH.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", "MPHASIS.NS",
    "MRF.NS", "MUTHOOTFIN.NS", "NAM-INDIA.NS", "NATCOPHARM.NS", "NAVINFLUOR.NS",
    "NIACL.NS", "NLCINDIA.NS", "NSLNISP.NS", "OBEROIRLTY.NS", "OIL.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PHOENIXLTD.NS", "PVR.NS",
    "RADICO.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RBLBANK.NS", "ROUTE.NS",
    "SCHAEFFLER.NS", "SCI.NS", "SHYAMMETL.NS", "SJVN.NS", "SKFINDIA.NS",
    "SONACOMS.NS", "STARHEALTH.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACOMM.NS",
    "TATACHEM.NS", "TATAELXSI.NS", "TTML.NS", "TVSMOTORS.NS", "UCOBANK.NS",
    "UJJIVANSFB.NS", "UNIONBANK.NS", "UNOMINDA.NS", "VBL.NS", "VGUARD.NS",
]

# NIFTY 500 extension (additional midcap/smallcap)
_NIFTY_500_EXTRA = [
    "3MINDIA.NS", "AAVAS.NS", "ABSL.NS", "ACE.NS", "AEGISCHEM.NS",
    "AJANTPHARM.NS", "AKMR.NS", "AKZOINDIA.NS", "ALCHEM.NS", "ALEMBICLTD.NS",
    "ANGELONE.NS", "ANURAS.NS", "APARINDS.NS", "APOLLOTYRE.NS", "APTUS.NS",
    "ARVINDFASN.NS", "ASTRAL.NS", "ATUL.NS", "AVANTIFEED.NS", "BALRAMCHIN.NS",
    "BASF.NS", "BAYERCROP.NS", "BCG.NS", "BEML.NS", "BFUTILITIE.NS",
    "BORORENEW.NS", "BSE.NS", "CAMS.NS", "CAPACITE.NS", "CAPLIPOINT.NS",
    "CARBORUNIV.NS", "CASTROLIND.NS", "CAVALECORP.NS", "CCL.NS", "CESC.NS",
    "CGPOWER.NS", "CHALET.NS", "CHEMCON.NS", "CHENNPETRO.NS", "CLEAN.NS",
    "CRAFTSMAN.NS", "DATAMATICS.NS", "DBREALTY.NS", "DCB.NS", "DCMSHRIRAM.NS",
    "DELTACORP.NS", "DHANI.NS", "DHANUKA.NS", "DIAMONDYD.NS", "DODLA.NS",
    "DOMS.NS", "ELGIEQUIP.NS", "EPL.NS", "EQUITASBNK.NS", "ESABINDIA.NS",
    "ESCORTS.NS", "EVEREADY.NS", "FINPIPE.NS", "FINEORG.NS", "FINOLEX.NS",
    "GAEL.NS", "GESHIP.NS", "GHCL.NS", "GILLETTE.NS", "GLOBALHEALT.NS",
    "GLODYNE.NS", "GNBFINANCE.NS", "GODFRYPHLP.NS", "GPIL.NS", "GRINDWELL.NS",
    "GROBTEA.NS", "GUJALKALI.NS", "GUJFLUORO.NS", "GULFOILLUB.NS", "HAPPSTMNDS.NS",
    "HARIOMPIPE.NS", "HEG.NS", "HEMIPROP.NS", "HERBAGE.NS", "HIKAL.NS",
    "HOMEFIRST.NS", "HUDCO.NS", "ICHEMCONS.NS", "IDEALREAL.NS", "IFBIND.NS",
    "IIFLSEC.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INNOVACAP.NS",
    "INTELLECT.NS", "ISGEC.NS", "ITDCEM.NS", "JBCHEPHARM.NS", "JKLAKSHMI.NS",
    "JKPAPER.NS", "JMFINANCIL.NS", "JSL.NS", "JTEKIND.NS", "KANSAINER.NS",
    "KDDL.NS", "KFINTECH.NS", "KINETIC.NS", "KITEX.NS", "KNRCON.NS",
]

# Build cumulative lists
_NIFTY100  = _NIFTY50 + _NIFTY_NEXT50
_NIFTY200  = _NIFTY100 + _NIFTY_200_EXTRA
_NIFTY500  = _NIFTY200 + _NIFTY_500_EXTRA


# ── Wikipedia fetch ───────────────────────────────────────────────────────────

def _get_nifty50_from_wiki() -> List[str]:
    """Try to fetch live NIFTY 50 constituents from Wikipedia."""
    try:
        url    = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url, header=0)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            sym_col = next((c for c in t.columns if "symbol" in c.lower()), None)
            if sym_col and len(t) >= 40:
                tickers = [str(s).strip() + ".NS" for s in t[sym_col] if str(s).strip()]
                if len(tickers) >= 40:
                    logger.info(f"Fetched {len(tickers)} NIFTY 50 tickers from Wikipedia.")
                    return tickers
    except Exception as e:
        logger.debug(f"NIFTY 50 Wikipedia fetch failed: {e}")

    # BeautifulSoup fallback
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockResearch/1.0)"}
        resp    = requests.get("https://en.wikipedia.org/wiki/NIFTY_50",
                               headers=headers, timeout=10)
        soup    = BeautifulSoup(resp.text, "lxml")
        tables  = soup.find_all("table", {"class": "wikitable"})
        for table in tables:
            rows = table.find_all("tr")[1:]
            syms = []
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    s = cells[0].get_text(strip=True)
                    if s:
                        syms.append(s + ".NS")
            if len(syms) >= 40:
                logger.info(f"Fetched {len(syms)} NIFTY 50 tickers via BS4.")
                return syms
    except Exception as e:
        logger.debug(f"NIFTY 50 BS4 fallback failed: {e}")

    logger.warning("Using hardcoded NIFTY 50 fallback.")
    return _NIFTY50


# ── Public API ────────────────────────────────────────────────────────────────

def get_nifty50_tickers() -> List[str]:
    return _get_nifty50_from_wiki()


def get_nifty100_tickers() -> List[str]:
    return list(_NIFTY100)


def get_nifty200_tickers() -> List[str]:
    return list(_NIFTY200)


def get_nifty500_tickers() -> List[str]:
    return list(_NIFTY500)


def get_india_universe(config: dict) -> List[str]:
    """Return the Indian stock ticker list according to config."""
    india_cfg = config.get("india", {})
    univ_cfg  = india_cfg.get("universe", {})

    universe_type = univ_cfg.get("default", "nifty50")
    custom        = univ_cfg.get("custom_tickers", [])
    max_stocks    = int(univ_cfg.get("max_stocks", 50))

    if custom:
        tickers = [
            t.strip().upper() + (".NS" if not t.strip().endswith(".NS") and not t.strip().endswith(".BO") else "")
            for t in custom if t.strip()
        ]
    elif universe_type == "nifty50":
        tickers = get_nifty50_tickers()
    elif universe_type == "nifty100":
        tickers = get_nifty100_tickers()
    elif universe_type == "nifty200":
        tickers = get_nifty200_tickers()
    elif universe_type == "nifty500":
        tickers = get_nifty500_tickers()
    else:
        tickers = get_nifty50_tickers()

    # Deduplicate preserving order
    seen, unique = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:max_stocks]
