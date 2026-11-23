"""
dashboard.py - market dashboard (Dash + yfinance + FRED)

Install:  pip install dash yfinance pandas tzdata requests lxml
Run:      python dashboard.py      then open http://127.0.0.1:8050
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

ET = ZoneInfo("America/New_York")
REFRESH_SECONDS = 60
PANEL, GRID = "#131920", "#232b36"
GREEN, RED, GOLD = "#3fb950", "#f0524f", "#e3b341"


# --------------------------------------------------------------------------
# Symbols (from TradingViewWidget.js). Each entry is (label, data-source symbol).
# --------------------------------------------------------------------------
def _same(symbols):
    return [(s, s) for s in symbols.split()]


GROUPS = {
    "Stocks": dict(kind="yahoo", rth=True, options=True, symbols=_same(
        "ABNB ADI AMC AMD ANT AON BA BABA BAC BB BBBY BLK CACC CAT CLF "
        "CLNE CLOV COIN COKE COST CVM DD DE DIS DNUT DT GERN GME "
        "HOOD INTC JNJ JPM KMI KR LMND LOVE LUMN LYV MAC MRNA NKE NOK "
        "NVDA OMEX PTLO PTON RBLX RKT SIRI SNAP SNDL SOFI SPCE TAL TSLA "
        "UL UWMC WKHS WOOF YRD Z")),
    "Food Stocks": dict(kind="yahoo", rth=True, options=True,
                        symbols=_same("GIS HSY MKC SJM TR")),
    "High Dividends": dict(kind="yahoo", rth=True, options=True,
                           symbols=_same("ENB EPD IBM MO NHI O PFE PM")),
    "Fast Food": dict(kind="yahoo", rth=True, options=True,
                      symbols=_same("CMG MCD QSR SBUX WEN")),
    "US Market": dict(kind="yahoo", rth=True, options=True, symbols=[
        ("SPX", "^GSPC"), ("NDX", "^NDX"), ("DJI", "^DJI"), ("SPY", "SPY"),
        ("VIX", "^VIX"), ("VOO", "VOO"), ("VTI", "VTI")]),
    "Tech": dict(kind="yahoo", rth=True, options=True,
                 symbols=_same("AAPL AMZN GOOGL META MSFT NFLX")),
    "Futures": dict(kind="yahoo", symbols=[
        ("CORN", "ZC=F"), ("GOLD", "GC=F"), ("SILVER", "SI=F"),
        ("SOY", "ZS=F"), ("USOIL", "CL=F"), ("WHEAT", "ZW=F")]),
    "Forex": dict(kind="yahoo", symbols=[
        ("DXY", "DX-Y.NYB"), ("EURUSD", "EURUSD=X"),
        ("GBPUSD", "GBPUSD=X"), ("USDJPY", "JPY=X")]),
    # BINANCE:LUNAUSDT is left out on purpose - see notes.
    "Crypto": dict(kind="yahoo", symbols=[
        ("BTCUSD", "BTC-USD"), ("DOGEUSD", "DOGE-USD"), ("ETHUSD", "ETH-USD")]),
    "Fed": dict(kind="fred", symbols=_same(
        "CCSA CP CPIAUCSL DGORDER DGS10 DGS2 DGS30 DGS5 DPRIME ECOMSA FEDFUNDS "
        "FYFSD FYFSGDA1 GDP GFDEBTN GFDEGDQ HOUST PAYEMS PERMIT PI POP PSAVERT "
        "TDSP TOTALSL UNRATE USGOVT WM1NS WM2NS")),
    # Finviz-style S&P 500 heat map. Its symbols are loaded at runtime.
    "Map": dict(kind="map", symbols=[]),
}
DEFAULT_GROUP, DEFAULT_SYMBOL = "Stocks", "NVDA"

# Scrolling ticker tape: every Yahoo symbol from GROUPS, in tab order, no repeats.
# (Fed is left out because those are FRED economic series, not prices.)
TAPE_SYMBOLS = list(dict.fromkeys(
    pair for g in GROUPS.values() if g["kind"] == "yahoo" for pair in g["symbols"]))

# --------------------------------------------------------------------------
# Small TTL cache so a refresh does not re-download everything
# --------------------------------------------------------------------------
_cache = {}
_cache_lock = threading.RLock()
_yf_lock = threading.Lock()   # precaution: run only one yf.download at a time
FRED_SERIES = {}              # series id -> pd.Series, filled by fred_quotes()


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = fn()
        _cache[key] = (time.time(), value)
        return value


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def yahoo_quotes(group, pairs=None):
    pairs = pairs or GROUPS[group]["symbols"]
    with _yf_lock:
        raw = yf.download([y for _, y in pairs], period="5d", interval="1d",
                          group_by="ticker", auto_adjust=False,
                          progress=False, threads=True)
    rows, missing = [], []
    if raw is None or raw.empty:
        return rows
    for label, y in pairs:
        try:
            df = raw[y].dropna(subset=["Close"])
        except KeyError:
            df = pd.DataFrame()
        if len(df) < 2:
            missing.append(label)
            continue
        last, prev = df.iloc[-1], df.iloc[-2]
        price, prev_close = float(last["Close"]), float(prev["Close"])
        hi, lo, vol = float(last["High"]), float(last["Low"]), last["Volume"]
        rows.append(dict(
            label=label, y=y, price=price, chg=price - prev_close,
            pct=(price / prev_close - 1) * 100,
            vol=None if pd.isna(vol) else float(vol), hi=hi, lo=lo,
            pos=(price - lo) / (hi - lo) * 100 if hi > lo else None,
            asof=str(df.index[-1].date())))
    if missing:
        print(f"[yahoo] no data, skipped: {', '.join(missing)}")
    return rows


def fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, na_values=["."])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def fred_quotes():
    ids = [y for _, y in GROUPS["Fed"]["symbols"]]

    def safe(series_id):
        try:
            return fred_series(series_id)
        except Exception as exc:
            print(f"[FRED] {series_id} skipped: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(safe, ids))
    rows = []
    for series_id, s in zip(ids, results):
        if s is None or len(s) < 2:
            continue
        FRED_SERIES[series_id] = s
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        rows.append(dict(
            label=series_id, y=series_id, price=last, chg=last - prev,
            pct=(last - prev) / abs(prev) * 100 if prev else 0.0,
            vol=None, hi=None, lo=None, pos=None,
            asof=str(s.index[-1].date())))
    return rows


def quotes(group):
    if GROUPS[group]["kind"] == "fred":
        return cached(("q", group), 6 * 3600, fred_quotes)
    return cached(("q", group), REFRESH_SECONDS - 5, lambda: yahoo_quotes(group))


def history(group, y):
    if GROUPS[group]["kind"] == "fred":
        s = FRED_SERIES.get(y)
        if s is None:
            s = fred_series(y)
        return s[s.index >= "2000-01-01"].to_frame("Close")
    df = yf.Ticker(y).history(period="5d", interval="5m", auto_adjust=False)
    if not df.empty and df.index.tz is not None:
        # Convert to Eastern, then drop the timezone so Plotly shows ET wall time.
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def options(y, n_expirations=3, top_n=8):
    """Most-traded contracts over the nearest expirations + put/call volume ratio."""
    try:
        ticker = yf.Ticker(y)
        frames = []
        for exp in ticker.options[:n_expirations]:
            chain = ticker.option_chain(exp)
            for side, df in (("C", chain.calls), ("P", chain.puts)):
                part = df[["strike", "lastPrice", "volume"]].copy()
                part["side"], part["exp"] = side, exp
                frames.append(part)
        if not frames:
            return None
        allc = pd.concat(frames, ignore_index=True)
        allc["volume"] = allc["volume"].fillna(0)
        call_vol = allc.loc[allc["side"] == "C", "volume"].sum()
        put_vol = allc.loc[allc["side"] == "P", "volume"].sum()
        return dict(top=allc.nlargest(top_n, "volume").to_dict("records"),
                    pc=put_vol / call_vol if call_vol else None)
    except Exception as exc:
        print(f"[options] {y}: {exc}")
        return None


# --------------------------------------------------------------------------
# S&P 500 map (Finviz-style treemap)
#   Constituent list:            Wikipedia "List of S&P 500 companies"
#   Sector / industry / mkt cap: yfinance Ticker.info, saved to sp500_meta.json
#                                and refreshed in the background once a day
#   Daily % change:              yf.download, same as the other tabs
# --------------------------------------------------------------------------
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
META_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "sp500_meta.json")
META_MAX_AGE = 24 * 3600
MAP_BG, MAP_HEADER = "#262931", "#3a3d47"
# Color scale clamps at +/-3%, like the Finviz legend (colors are approximate).
MAP_STOPS = [(-3, "#f63538"), (-2, "#bf4045"), (-1, "#8b444e"), (0, "#414554"),
             (1, "#35764e"), (2, "#2f9e4f"), (3, "#30cc5a")]
# Wikipedia uses GICS sector names. These are only used for a ticker when
# Yahoo returns no sector, so it lands in the matching Yahoo-named sector.
GICS_TO_YAHOO = {
    "Information Technology": "Technology",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
}

_meta = dict(rows=None, saved=0.0, loading=False, done=0, total=0,
             error=None, error_at=0.0)
_meta_lock = threading.Lock()


def sp500_constituents():
    # Wikipedia rejects requests without a User-Agent, so fetch with requests.
    resp = requests.get(SP500_URL, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0 (market-dashboard)"})
    resp.raise_for_status()
    for table in pd.read_html(StringIO(resp.text), match="Symbol"):
        if {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"} <= set(table.columns):
            return table
    raise ValueError("S&P 500 table not found on the Wikipedia page")


def _ticker_meta(sym):
    for attempt in range(3):
        try:
            info = yf.Ticker(sym).info or {}
            return dict(name=info.get("shortName") or info.get("longName"),
                        sector=info.get("sector"), industry=info.get("industry"),
                        cap=info.get("marketCap"),
                        ref=info.get("regularMarketPrice") or info.get("currentPrice"))
        except Exception as exc:
            if attempt == 2:
                print(f"[map] {sym}: info failed: {exc}")
            else:
                time.sleep(2 * (attempt + 1))
    return {}


def _build_meta():
    try:
        table = sp500_constituents()
        # Yahoo writes class shares with a dash: BRK.B -> BRK-B
        syms = [str(s).strip().replace(".", "-") for s in table["Symbol"]]
        with _meta_lock:
            _meta.update(total=len(syms), done=0)

        def one(sym):
            m = _ticker_meta(sym)
            with _meta_lock:
                _meta["done"] += 1
            return m

        with ThreadPoolExecutor(max_workers=4) as pool:
            infos = list(pool.map(one, syms))

        rows = {}
        for sym, rec, info in zip(syms, table.to_dict("records"), infos):
            gics = str(rec["GICS Sector"])
            rows[sym] = dict(
                name=info.get("name") or str(rec["Security"]),
                sector=info.get("sector") or GICS_TO_YAHOO.get(gics, gics),
                industry=info.get("industry") or str(rec["GICS Sub-Industry"]),
                cap=info.get("cap"), ref=info.get("ref"))
        no_cap = [s for s, r in rows.items() if not r["cap"]]
        if no_cap:
            print(f"[map] no market cap, left off the map: {', '.join(no_cap)}")
        saved = time.time()
        with open(META_FILE, "w") as f:
            json.dump(dict(saved=saved, rows=rows), f)
        with _meta_lock:
            _meta.update(rows=rows, saved=saved, error=None)
    except Exception as exc:
        print(f"[map] loading S&P 500 data failed: {exc}")
        with _meta_lock:
            _meta.update(error=str(exc), error_at=time.time())
    finally:
        with _meta_lock:
            _meta["loading"] = False


def map_meta():
    """Return (rows, state). Starts a background refresh when data is missing or old."""
    with _meta_lock:
        if _meta["rows"] is None and os.path.exists(META_FILE):
            try:
                with open(META_FILE) as f:
                    saved = json.load(f)
                _meta.update(rows=saved["rows"], saved=float(saved["saved"]))
            except Exception as exc:
                print(f"[map] could not read {META_FILE}: {exc}")
        stale = time.time() - _meta["saved"] > META_MAX_AGE
        retry_ok = time.time() - _meta["error_at"] > 300   # wait 5 min after a failure
        if stale and retry_ok and not _meta["loading"]:
            _meta.update(loading=True, done=0, total=0)
            threading.Thread(target=_build_meta, daemon=True).start()
        return _meta["rows"], dict(_meta)


def map_color(pct):
    p = max(MAP_STOPS[0][0], min(MAP_STOPS[-1][0], pct))
    for (x0, c0), (x1, c1) in zip(MAP_STOPS, MAP_STOPS[1:]):
        if p <= x1:
            t = (p - x0) / (x1 - x0)
            a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{round(u + (v - u) * t):02x}" for u, v in zip(a, b))
    return MAP_STOPS[-1][1]


def fmt_cap(v):
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= size:
            return f"${v / size:.2f}{suffix}"
    return f"${v:,.0f}"


def map_figure(meta_rows, quote_rows):
    q = {r["y"]: r for r in quote_rows}
    tiles = []
    for sym, m in meta_rows.items():
        r = q.get(sym)
        if r is None or not m.get("cap"):
            continue
        cap = float(m["cap"])
        if m.get("ref"):
            # Scale the saved market cap by today's price move so sizes stay current.
            cap *= r["price"] / float(m["ref"])
        tiles.append(dict(sym=sym, name=m["name"], sector=m["sector"],
                          industry=m["industry"], cap=cap, pct=r["pct"],
                          price=r["price"], chg=r["chg"]))
    if not tiles:
        return None, 0
    df = pd.DataFrame(tiles)
    total = df["cap"].sum()

    ids, labels, parents, values = [], [], [], []
    colors, text, hover, sizes = [], [], [], []

    def add(id_, label, parent, value, color_, txt, hov, size):
        ids.append(id_)
        labels.append(label)
        parents.append(parent)
        values.append(value)
        colors.append(color_)
        text.append(txt)
        hover.append(hov)
        sizes.append(size)

    def weighted(part):
        return (part["cap"] * part["pct"]).sum() / part["cap"].sum()

    for sector, sdf in df.groupby("sector"):
        sid = f"s|{sector}"
        add(sid, sector, "", 0, MAP_HEADER, sector.upper(),
            f"<b>{sector}</b><br>Cap-weighted change: {weighted(sdf):+.2f}%"
            f"<br>Market cap: {fmt_cap(sdf['cap'].sum())}<br>Stocks: {len(sdf)}", 12)
        for industry, idf in sdf.groupby("industry"):
            iid = f"i|{sector}|{industry}"
            add(iid, industry, sid, 0, MAP_HEADER, industry.upper(),
                f"<b>{industry}</b><br>Cap-weighted change: {weighted(idf):+.2f}%"
                f"<br>Market cap: {fmt_cap(idf['cap'].sum())}<br>Stocks: {len(idf)}", 9)
            for t in idf.to_dict("records"):
                share = t["cap"] / total
                add(f"t|{t['sym']}", t["sym"], iid, t["cap"], map_color(t["pct"]),
                    f"<b>{t['sym']}</b><br>{t['pct']:+.2f}%",
                    f"<b>{t['sym']}</b> {t['name']}<br>{industry}"
                    f"<br>Price: {fmt_num(t['price'])} ({t['chg']:+,.2f})"
                    f"<br>Change: {t['pct']:+.2f}%<br>Market cap: {fmt_cap(t['cap'])}",
                    max(8, min(40, 6 + 150 * share ** 0.5)))

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="remainder", sort=True,
        marker=dict(colors=colors, line=dict(width=1, color=MAP_BG),
                    pad=dict(t=16, l=1, r=1, b=1)),
        text=text, textinfo="text", textposition="middle center",
        textfont=dict(size=sizes, color="#ffffff"),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        pathbar=dict(visible=True), root_color=MAP_BG,
    ))
    fig.update_layout(paper_bgcolor=MAP_BG, margin=dict(l=4, r=4, t=4, b=4),
                      uirevision="map", font=dict(color="#e6edf3"))
    return fig, len(df)


def message_figure(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#7d8896"))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def fmt_num(v):
    if v is None or pd.isna(v):
        return "--"
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"


def fmt_vol(v):
    if v is None or pd.isna(v) or v == 0:
        return "--"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= size:
            return f"{v / size:.1f}{suffix}"
    return f"{v:.0f}"


def color(v):
    return "fd-up" if v >= 0 else "fd-dn"


def row(panel, r, cells, cols, selected):
    cls = f"fd-row {cols}" + (" on" if r["y"] == selected else "")
    return html.Button(cells, className=cls,
                       id={"type": "row", "panel": panel, "sym": r["y"]})


def stat(label, value):
    return html.Div([html.Div(label, className="fd-mut", style={"fontSize": "10px"}),
                     html.Div(value, className="fd-b")])


def make_figure(group, y, df):
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="No chart data", showarrow=False,
                           font=dict(color="#7d8896"))
    elif "Open" in df.columns:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
            close=df["Close"], increasing_line_color=GREEN,
            decreasing_line_color=RED, showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                                 line=dict(color=GOLD, width=1.5)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL,
                      plot_bgcolor=PANEL, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_rangeslider_visible=False, uirevision=y)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, side="right")
    if GROUPS[group].get("rth"):
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),              # hide weekends
            dict(bounds=[16, 9.5], pattern="hour"),   # hide 4:00pm-9:30am ET
        ])
    return fig


# --------------------------------------------------------------------------
# App + layout
# --------------------------------------------------------------------------
app = Dash(__name__, title="Market Dashboard")

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:13px/1.35 -apple-system,"Segoe UI",Roboto,sans-serif}
.fd-top{display:flex;align-items:center;gap:16px;padding:8px 12px;border-bottom:1px solid #232b36;white-space:nowrap}
.fd-tape-wrap{flex:1;overflow:hidden}
.fd-tape{display:inline-flex;white-space:nowrap;animation:fd-scroll 300s linear infinite}
.fd-tape:hover{animation-play-state:paused}
.fd-tape>span{padding-right:22px}
@keyframes fd-scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.fd-groups{display:flex;flex-wrap:wrap;gap:4px;padding:6px 12px;border-bottom:1px solid #232b36}
.fd-grp{background:none;border:0;color:#7d8896;font:inherit;padding:3px 8px;border-radius:4px;cursor:pointer}
.fd-grp.on{background:#3a3115;color:#e3b341;font-weight:700}
.fd-grid3{display:grid;grid-template-columns:repeat(3,1fr)}
.fd-grid2{display:grid;grid-template-columns:1fr 2fr}
.fd-panel{border-right:1px solid #232b36;border-bottom:1px solid #232b36;min-width:0;background:#131920}
.fd-title{padding:6px 10px;font-size:10px;letter-spacing:.08em;color:#7d8896;text-transform:uppercase;border-bottom:1px solid #232b36}
.fd-body{height:260px;overflow-y:auto}
.fd-tall{height:400px}
.fd-row{display:grid;align-items:center;gap:8px;width:100%;box-sizing:border-box;padding:5px 10px;background:none;border:0;color:inherit;font:inherit;text-align:left;cursor:pointer}
.fd-row:hover,.fd-row.on{background:#1b222c}
.fd-static{cursor:default}
.fd-c4{grid-template-columns:1.2fr 1fr 1fr .8fr}
.fd-c3{grid-template-columns:1fr 1.6fr .7fr}
.fd-c2{grid-template-columns:1fr 1fr}
.fd-c2l{grid-template-columns:auto 1fr}
.fd-split{display:grid;grid-template-columns:1fr 1fr}
.fd-split>*+*{border-left:1px solid #232b36}
.fd-title .fd-split{margin:0 -10px}
.fd-title .fd-split>span{padding:0 10px}
.fd-r{text-align:right}
.fd-b{font-weight:700}
.fd-mut{color:#7d8896}
.fd-up{color:#3fb950}
.fd-dn{color:#f0524f}
.fd-bar{height:6px;background:#2a3340;border-radius:3px;overflow:hidden}
.fd-head{display:flex;align-items:baseline;gap:12px;padding:8px 12px}
.fd-stats{margin-left:auto;display:flex;gap:18px;text-align:right}
.fd-msg{padding:12px;color:#7d8896}
.fd-legend{display:flex;justify-content:flex-end;gap:2px;padding:6px 12px}
.fd-legend span{min-width:44px;padding:3px 6px;text-align:center;font-size:11px;color:#fff}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


def panel(title_id, body_id, tall=False):
    return html.Div(className="fd-panel", children=[
        html.Div(id=title_id, className="fd-title"),
        html.Div(id=body_id, className="fd-body" + (" fd-tall" if tall else "")),
    ])


app.layout = html.Div([
    dcc.Store(id="group", data=DEFAULT_GROUP),
    dcc.Store(id="selected", data=DEFAULT_SYMBOL),
    dcc.Interval(id="tick", interval=REFRESH_SECONDS * 1000),
    # Fast poll used only while the map's first-run data is loading.
    dcc.Interval(id="map-tick", interval=3000, disabled=True),
    html.Div(className="fd-top", children=[
        html.Div(id="status", className="fd-b"),
        html.Div(className="fd-tape-wrap",
                 children=html.Div(id="tape", className="fd-tape")),
    ]),
    html.Div(id="groups", className="fd-groups"),
    html.Div(id="main-view", children=[
        html.Div(className="fd-grid3", children=[
            panel("watch-title", "watch"),
            panel("mid-title", "mid"),
            panel("opt-title", "opts"),
        ]),
        html.Div(className="fd-grid2", children=[
            panel("movers-title", "movers", tall=True),
            html.Div(className="fd-panel", children=[
                html.Div(id="head", className="fd-head"),
                dcc.Graph(id="chart", config={"displayModeBar": False},
                          style={"height": "360px"}),
            ]),
        ]),
    ]),
    html.Div(id="map-view", style={"display": "none"}, children=[
        html.Div(className="fd-head", children=[
            html.Span("S&P 500 Map", className="fd-b"),
            html.Span("Size = market cap, color = % change from previous close. "
                      "Click a sector or industry to zoom in.", className="fd-mut"),
            html.Div(id="map-status", className="fd-mut",
                     style={"marginLeft": "auto"}),
        ]),
        dcc.Graph(id="map", config={"displayModeBar": False},
                  style={"height": "calc(100vh - 150px)", "minHeight": "600px"}),
        html.Div(className="fd-legend", children=[
            html.Span(f"{v:+d}%" if v else "0%", style={"background": c})
            for v, c in MAP_STOPS]),
    ]),
])


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
@app.callback(
    Output("group", "data"), Output("selected", "data"),
    Input({"type": "grp", "name": ALL}, "n_clicks"),
    Input({"type": "row", "panel": ALL, "sym": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_click(_group_clicks, _row_clicks):
    tid = ctx.triggered_id
    # Re-rendered buttons fire with n_clicks=None; only react to real clicks.
    if not isinstance(tid, dict) or not any(t["value"] for t in ctx.triggered):
        raise PreventUpdate
    if tid["type"] == "grp":
        symbols = GROUPS[tid["name"]]["symbols"]
        return tid["name"], symbols[0][1] if symbols else no_update
    return no_update, tid["sym"]


@app.callback(
    Output("status", "children"), Output("tape", "children"),
    Output("groups", "children"),
    Output("watch-title", "children"), Output("watch", "children"),
    Output("mid-title", "children"), Output("mid", "children"),
    Output("movers-title", "children"), Output("movers", "children"),
    Input("group", "data"), Input("selected", "data"),
    Input("tick", "n_intervals"),
)
def render_panels(group, selected, _n):
    now = datetime.now(ET)
    is_open = now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)
    status = html.Span(
        f"● Market {'Open' if is_open else 'Closed'}  {now:%I:%M %p} ET",
        className="fd-up" if is_open else "fd-mut")

    try:
        tape_rows = cached(("q", "tape"), REFRESH_SECONDS - 5,
                           lambda: yahoo_quotes("tape", TAPE_SYMBOLS))
        items = [html.Span([html.Span(r["label"] + " ", className="fd-b"),
                            fmt_num(r["price"]) + " ",
                            html.Span(f"{r['pct']:+.2f}%", className=color(r["pct"]))])
                 for r in tape_rows]
        # Two copies back to back so the scroll loops without a gap.
        tape = items + items
    except Exception as exc:
        print(f"[tape] {exc}")
        tape = []

    group_buttons = [
        html.Button(name, id={"type": "grp", "name": name},
                    className="fd-grp" + (" on" if name == group else ""))
        for name in GROUPS]

    if GROUPS[group]["kind"] == "map":
        # The panels are hidden while the map is shown; leave them as they are.
        return (status, tape, group_buttons) + (no_update,) * 6

    try:
        rows = quotes(group)
    except Exception as exc:
        print(f"[quotes] {group}: {exc}")
        rows = []
    if not rows:
        msg = html.Div("No data returned. Check the console.", className="fd-msg")
        return (status, tape, group_buttons, "Watchlist", msg,
                "Day range position", msg, "Top losers / Top gainers", msg)

    watch = [row("watch", r, [
        html.Span(r["label"], className="fd-b"),
        html.Span(fmt_num(r["price"]), className="fd-r"),
        html.Span(f"{r['pct']:+.2f}%", className="fd-r " + color(r["pct"])),
        html.Span(fmt_vol(r["vol"]), className="fd-r fd-mut"),
    ], "fd-c4", selected) for r in rows]

    if GROUPS[group]["kind"] == "fred":
        mid_title = "Last observation date"
        mid = [row("mid", r, [
            html.Span(r["label"], className="fd-b"),
            html.Span(r["asof"], className="fd-r fd-mut"),
        ], "fd-c2", selected) for r in rows]
    else:
        mid_title = "Day range position (0% = low, 100% = high)"
        ranged = sorted((r for r in rows if r["pos"] is not None),
                        key=lambda r: r["pos"], reverse=True)
        mid = [row("mid", r, [
            html.Span(r["label"], className="fd-b"),
            html.Div(className="fd-bar", children=html.Div(style={
                "width": f"{r['pos']:.0f}%", "height": "100%",
                "background": GREEN if r["pos"] >= 50 else RED})),
            html.Span(f"{r['pos']:.0f}%",
                      className="fd-r " + color(r["pos"] - 50)),
        ], "fd-c3", selected) for r in ranged]

    def mover_rows(panel_name, picked):
        if not picked:
            return html.Div("None today", className="fd-msg")
        return [row(panel_name, r, [
            html.Span(r["label"], className="fd-b"),
            html.Span(f"{'▲' if r['pct'] >= 0 else '▼'} {r['pct']:+.2f}%",
                      className=color(r["pct"])),
        ], "fd-c2l", selected) for r in picked]

    losers = sorted((r for r in rows if r["pct"] < 0),
                    key=lambda r: r["pct"])[:15]
    gainers = sorted((r for r in rows if r["pct"] > 0),
                     key=lambda r: r["pct"], reverse=True)[:15]
    movers_title = html.Div(className="fd-split", children=[
        html.Span("Top losers"), html.Span("Top gainers")])
    movers = html.Div(className="fd-split", children=[
        html.Div(mover_rows("losers", losers)),
        html.Div(mover_rows("gainers", gainers))])

    return (status, tape, group_buttons, f"Watchlist - {group}", watch,
            mid_title, mid, movers_title, movers)


@app.callback(
    Output("head", "children"), Output("chart", "figure"),
    Output("opt-title", "children"), Output("opts", "children"),
    Input("selected", "data"), Input("tick", "n_intervals"),
    State("group", "data"),
)
def render_chart(selected, _n, group):
    if GROUPS[group]["kind"] == "map":
        raise PreventUpdate
    label = next((l for l, y in GROUPS[group]["symbols"] if y == selected), selected)
    try:
        q = next((r for r in quotes(group) if r["y"] == selected), None)
    except Exception:
        q = None
    try:
        df = cached(("hist", group, selected), REFRESH_SECONDS - 5,
                    lambda: history(group, selected))
    except Exception as exc:
        print(f"[history] {selected}: {exc}")
        df = pd.DataFrame()

    opt = None
    if GROUPS[group].get("options") and not selected.startswith("^"):
        opt = cached(("opt", selected), 120, lambda: options(selected))

    if q is None:
        head = [html.Span(label, className="fd-b", style={"fontSize": "16px"}),
                html.Span("no quote", className="fd-mut")]
    else:
        pc = f"{opt['pc']:.2f}" if opt and opt["pc"] is not None else "--"
        head = [
            html.Span(label, className="fd-b", style={"fontSize": "16px"}),
            html.Span(fmt_num(q["price"]), className="fd-b",
                      style={"fontSize": "20px"}),
            html.Span(f"{q['chg']:+,.2f}  {q['pct']:+.2f}%",
                      className=color(q["pct"])),
            html.Div(className="fd-stats", children=[
                stat("VOLUME", fmt_vol(q["vol"])),
                stat("DAY HIGH", fmt_num(q["hi"])),
                stat("DAY LOW", fmt_num(q["lo"])),
                stat("PUT/CALL VOL", pc),
                stat("AS OF", q["asof"]),
            ]),
        ]

    if opt:
        opt_rows = [html.Div(className="fd-row fd-c3 fd-static", children=[
            html.Span([
                html.Span(label + " ", className="fd-b"),
                f"{c['strike']:g}{c['side']} "
                f"{datetime.strptime(c['exp'], '%Y-%m-%d'):%m/%d}"]),
            html.Span(f"vol {fmt_vol(c['volume'])}", className="fd-r"),
            html.Span(fmt_num(c["lastPrice"]),
                      className="fd-r " + ("fd-up" if c["side"] == "C" else "fd-dn")),
        ]) for c in opt["top"]]
    else:
        opt_rows = html.Div(f"No options data for {label}.", className="fd-msg")

    return (head, make_figure(group, selected, df),
            f"Top options by volume - {label}", opt_rows)


@app.callback(
    Output("main-view", "style"), Output("map-view", "style"),
    Output("map", "figure"), Output("map-status", "children"),
    Output("map-tick", "disabled"),
    Input("group", "data"), Input("tick", "n_intervals"),
    Input("map-tick", "n_intervals"),
)
def render_map(group, _n, _m):
    if GROUPS[group]["kind"] != "map":
        return {}, {"display": "none"}, no_update, no_update, True
    hidden, shown = {"display": "none"}, {}

    rows, state = map_meta()
    if rows is None:
        if state["error"] and not state["loading"]:
            msg = f"Could not load S&P 500 data: {state['error']} (retrying in 5 min)"
        else:
            msg = (f"First run: loading sector and market cap data from Yahoo "
                   f"({state['done']}/{state['total'] or '?'}). "
                   f"It is saved to sp500_meta.json, so this only happens once a day.")
        return hidden, shown, message_figure(msg), msg, False

    try:
        qrows = cached(("q", "Map"), REFRESH_SECONDS - 5,
                       lambda: yahoo_quotes("Map", [(s, s) for s in rows]))
    except Exception as exc:
        print(f"[map] quotes: {exc}")
        qrows = []
    fig, n = map_figure(rows, qrows)
    if fig is None:
        msg = "No price data returned for the S&P 500. Check the console."
        return hidden, shown, message_figure(msg), msg, True

    asof = max(r["asof"] for r in qrows)
    status = f"{n} stocks · prices as of {asof}"
    if state["loading"]:
        status += f" · refreshing sector data {state['done']}/{state['total'] or '?'}"
    return hidden, shown, fig, status, True


if __name__ == "__main__":
    app.run(debug=True)
