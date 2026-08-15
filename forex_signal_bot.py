"""
Forex Confluence Signal Bot
----------------------------
Strategy: Multi-indicator confluence (trend + momentum + volatility + mean-reversion)
Risk model: Fixed 100 pip target / 30 pip stop (~3.3:1 reward:risk)

IMPORTANT REALITY CHECK (read before trading real money):
A 3.3:1 reward:risk system only needs to win ~24% of trades to break even
(before spread/slippage). That sounds easy, but wide fixed targets on major
pairs can take days to hit and often get stopped out first because price
chops around before trending. This script gives you an HONEST backtest of
win rate / expectancy on real data -- trust that number, not the words
"insane" or "high accuracy". Paper trade for at least 100 signals before
risking real capital.

Data source: Twelve Data (https://twelvedata.com) - free tier gives you
800 API credits/day, which is plenty for checking a few pairs periodically.
Sign up for a free API key and drop it in TWELVE_DATA_API_KEY below (or set
it as an environment variable).

Setup:
    pip install requests pandas numpy --break-system-packages

Usage:
    python forex_signal_bot.py backtest XAU/USD          # backtest last ~5000 bars (H1)
    python forex_signal_bot.py live XAU/USD               # check current signal once
    python forex_signal_bot.py watch XAU/USD EUR/USD ...  # loop + Telegram alerts
    python forex_signal_bot.py sweep XAU/USD               # empirically test filter strictness
                                                             # combos and rank by real win rate/expectancy
    python forex_signal_bot.py backtest                   # no pair given -> defaults to DEFAULT_PAIRS (XAU/USD)
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "YOUR_API_KEY_HERE")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")   # from @BotFather
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")       # your chat id

TP_PIPS = 100
SL_PIPS = 30

# All internal calculation (candles, session filter, backtest) runs in UTC --
# that's what SESSION_START_HOUR/SESSION_END_HOUR below are defined in, and
# what the API is now explicitly told to return. DISPLAY_TIMEZONE only affects
# what's shown in the live/watch Telegram message, so timestamps make sense
# for wherever you actually are. Use any IANA name, e.g. "Africa/Accra"
# (Ghana, UTC+0), "Australia/Brisbane" (UTC+10), "America/New_York", etc.
DISPLAY_TIMEZONE = "Africa/Accra"

INTERVAL = "15min"      # timeframe: 1min,5min,15min,1h,4h,1day
OUTPUT_SIZE = 5000       # how many bars to pull (max ~5000 on free tier per call)
CHECK_EVERY_SECONDS = 300  # 5 min, used by "watch" mode -- tighter since 15min candles close faster

# ATR-adaptive targets: instead of a fixed pip TP/SL, scale the target to
# each pair's actual recent volatility (ATR). This matters a lot on faster
# timeframes like 15min, where a fixed 100-pip target is far too wide for
# the size of moves that timeframe normally produces -- trades either
# never resolve or take absurdly long. Fixed pips still make sense on
# slower timeframes (1h/4h/1day on majors) where 100 pips is a "normal"
# sized move for the horizon.
USE_ATR_TARGETS = True    # on by default now that 15min is the default timeframe
ATR_TP_MULT = 3.0         # TP = entry +/- ATR_TP_MULT * ATR   (keeps ~3.3:1 R:R like 100/30)
ATR_SL_MULT = 1.0         # SL = entry +/- ATR_SL_MULT * ATR

# Minimum-volatility filter: on fast timeframes (15min especially), a lot
# of candles happen during dead/quiet sessions where confluence conditions
# can technically align but there's no real momentum behind the move.
# Requiring current ATR to be at least this fraction of its own 50-bar
# average filters those out. Set to 0 to disable.
MIN_VOLATILITY_RATIO = 0.6

# Smart Money Concepts (SMC) filter: this is now the PRIMARY signal driver.
# A trade requires market structure + an active order block/FVG to align --
# that's the anchor. The 4 indicator votes (trend/momentum/RSI/BB) become a
# lighter secondary confirmation layer on top, only needing MIN_CONFLUENCE_VOTES
# out of 4 to agree rather than all 4. This trades some win-rate for a lot
# more signal frequency -- expect noticeably more trades, expect a lower
# per-trade win rate too. Backtest before trusting it live.
USE_SMC_FILTER = True
MIN_CONFLUENCE_VOTES = 3  # out of 4 indicator votes required, on top of SMC alignment
SMC_SWING_LOOKBACK = 2     # bars on each side to confirm a fractal swing point
SMC_ZONE_VALID_BARS = 50   # how many bars an order block / FVG stays "active" before going stale
SMC_REQUIRE_REJECTION = True  # require a wick-into-zone + confirming close, not just "price is inside the zone"

# Higher-timeframe bias filter: only take signals that agree with the trend
# on a slower timeframe. A very common real cause of bad win rates on fast
# timeframes is taking setups that look fine on M15 but are fighting the H1
# trend. Off by default cost: fewer signals. Benefit: each one has a bigger
# structural tailwind behind it.
USE_HTF_FILTER = True
HTF_INTERVAL = "1h"

# Session filter: skip low-liquidity hours (Asian session chop is a common
# source of fake SMC signals -- thin volume produces structure breaks and
# "order blocks" that don't hold up once London/NY volume shows up).
# Hours are UTC. Default window covers London open through NY close.
# This default is tuned for standard forex pairs (EUR/USD etc.), where the
# Asian session really is thin. Gold doesn't follow that pattern the same
# way -- it's a global asset traded actively out of Asia too (see
# PAIR_OVERRIDES below).
USE_SESSION_FILTER = True
SESSION_START_HOUR = 7   # 07:00 UTC (London open)
SESSION_END_HOUR = 16    # 16:00 UTC (NY afternoon)

# Pairs to use when you run the CLI with only a mode and no pair specified,
# e.g. `python forex_signal_bot.py backtest`.
DEFAULT_PAIRS = ["XAU/USD"]

# Per-pair overrides on top of the global defaults above. Use this instead of
# changing the global filters, so other pairs you might test later keep the
# more conservative forex-tuned defaults.
#
# For XAU/USD:
#   - session_start_hour/session_end_hour widened to effectively all-day.
#     This isn't "loosening for more signals" -- it's a correctness fix.
#     Gold has real, tradeable structure during the Asian session (your
#     backtested Aug 5-7 signals all fired 01:45-05:45 UTC, which the
#     default 7-16 window would have silently discarded as "off session"
#     even though they were valid setups on a genuinely global asset).
#   - min_confluence_votes deliberately NOT lowered here. More signals is
#     not the goal -- higher accuracy is. Use `sweep` mode (see CLI section)
#     to empirically find which vote threshold actually produces the best
#     win rate / expectancy on XAU/USD's real data, rather than guessing.
PAIR_OVERRIDES = {
    "XAU/USD": {
        "session_start_hour": 0,
        "session_end_hour": 24,
    },
}


def resolve_pair_settings(pair: str) -> dict:
    """Merge global defaults with any per-pair override for this instrument."""
    settings = {
        "min_confluence_votes": MIN_CONFLUENCE_VOTES,
        "session_start_hour": SESSION_START_HOUR,
        "session_end_hour": SESSION_END_HOUR,
    }
    settings.update(PAIR_OVERRIDES.get(pair.upper(), {}))
    return settings

# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------
def fetch_ohlc(pair: str, interval: str = INTERVAL, outputsize: int = OUTPUT_SIZE) -> pd.DataFrame:
    """Pull OHLC candles for a forex pair from Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
        "timezone": "UTC",  # explicit -- without this, Twelve Data returns
                            # "Exchange" local time by default, which silently
                            # broke the UTC-hour assumption in the session
                            # filter below and made every signal timestamp
                            # come out several hours off from real UTC.
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()

    if "values" not in data:
        raise RuntimeError(f"API error for {pair}: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def pip_size(pair: str) -> float:
    """Pip convention varies by instrument:
    - JPY pairs: 0.01
    - Gold (XAU): 0.01 (adjust to 0.1 if your broker quotes gold with 1 decimal)
    - Silver (XAG): 0.001 (adjust to your broker's convention)
    - Everything else (standard forex): 0.0001
    """
    p = pair.upper()
    if "JPY" in p:
        return 0.01
    if "XAU" in p:
        return 0.01
    if "XAG" in p:
        return 0.001
    return 0.0001


def to_display_tz(ts) -> str:
    """Convert a UTC timestamp to DISPLAY_TIMEZONE for human-readable output,
    with the zone shown explicitly so it's never ambiguous which clock a
    message timestamp is on."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(ZoneInfo(DISPLAY_TIMEZONE))
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger(df["close"])
    df["atr14"] = atr(df, 14)
    df["atr_avg50"] = df["atr14"].rolling(50).mean()
    return df


# ---------------------------------------------------------------------------
# SMART MONEY CONCEPTS (SMC)
# Market structure (break of structure), order blocks, and fair value gaps.
# Kept simple/readable rather than matching every edge case a dedicated SMC
# indicator would -- good enough to meaningfully filter signals by context.
# ---------------------------------------------------------------------------
def find_swings(df: pd.DataFrame, lookback: int = SMC_SWING_LOOKBACK):
    """Fractal swing highs/lows: a bar is a swing point if it's the highest
    high (or lowest low) among `lookback` bars on each side of it."""
    n = len(df)
    swing_high = [False] * n
    swing_low = [False] * n
    highs = df["high"].values
    lows = df["low"].values

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            swing_high[i] = True
        if lows[i] == window_l.min():
            swing_low[i] = True

    return swing_high, swing_low


def add_smc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, per bar:
      structure_trend            'bullish' / 'bearish' / None, from break of structure (BOS)
      bull_ob_low / bull_ob_high  most recent active bullish order block zone
      bear_ob_low / bear_ob_high  most recent active bearish order block zone
      bull_fvg_low / bull_fvg_high most recent unfilled bullish fair value gap
      bear_fvg_low / bear_fvg_high most recent unfilled bearish fair value gap
    """
    df = df.copy()
    n = len(df)
    swing_high, swing_low = find_swings(df)

    opens, closes = df["open"].values, df["close"].values
    highs, lows = df["high"].values, df["low"].values

    structure_trend = [None] * n
    bull_ob_low, bull_ob_high = [np.nan] * n, [np.nan] * n
    bear_ob_low, bear_ob_high = [np.nan] * n, [np.nan] * n

    last_swing_high, last_swing_low = None, None
    trend = None
    active_bull_ob, active_bear_ob = None, None  # (low, high, created_at_index)

    for i in range(n):
        if swing_high[i]:
            last_swing_high = highs[i]
        if swing_low[i]:
            last_swing_low = lows[i]

        # Break of structure up -> bullish. Order block = the last down-close
        # candle before the impulsive move that broke structure.
        if last_swing_high is not None and closes[i] > last_swing_high and trend != "bullish":
            trend = "bullish"
            for k in range(i - 1, max(i - 30, 0), -1):
                if closes[k] < opens[k]:
                    active_bull_ob = (lows[k], highs[k], i)
                    break

        # Break of structure down -> bearish. Order block = last up-close candle.
        if last_swing_low is not None and closes[i] < last_swing_low and trend != "bearish":
            trend = "bearish"
            for k in range(i - 1, max(i - 30, 0), -1):
                if closes[k] > opens[k]:
                    active_bear_ob = (lows[k], highs[k], i)
                    break

        # Expire order blocks: too old, or price has already closed through them
        if active_bull_ob and (i - active_bull_ob[2] > SMC_ZONE_VALID_BARS or lows[i] < active_bull_ob[0]):
            active_bull_ob = None
        if active_bear_ob and (i - active_bear_ob[2] > SMC_ZONE_VALID_BARS or highs[i] > active_bear_ob[1]):
            active_bear_ob = None

        structure_trend[i] = trend
        if active_bull_ob:
            bull_ob_low[i], bull_ob_high[i] = active_bull_ob[0], active_bull_ob[1]
        if active_bear_ob:
            bear_ob_low[i], bear_ob_high[i] = active_bear_ob[0], active_bear_ob[1]

    df["structure_trend"] = structure_trend
    df["bull_ob_low"], df["bull_ob_high"] = bull_ob_low, bull_ob_high
    df["bear_ob_low"], df["bear_ob_high"] = bear_ob_low, bear_ob_high

    # Fair Value Gaps: 3-candle imbalance (a gap between candle[i-2] and candle[i])
    bull_fvg_low, bull_fvg_high = [np.nan] * n, [np.nan] * n
    bear_fvg_low, bear_fvg_high = [np.nan] * n, [np.nan] * n
    active_bull_fvg, active_bear_fvg = None, None

    for i in range(2, n):
        if highs[i - 2] < lows[i]:            # gap up
            active_bull_fvg = (highs[i - 2], lows[i], i)
        if lows[i - 2] > highs[i]:             # gap down
            active_bear_fvg = (highs[i], lows[i - 2], i)

        if active_bull_fvg and (i - active_bull_fvg[2] > SMC_ZONE_VALID_BARS or lows[i] < active_bull_fvg[0]):
            active_bull_fvg = None
        if active_bear_fvg and (i - active_bear_fvg[2] > SMC_ZONE_VALID_BARS or highs[i] > active_bear_fvg[1]):
            active_bear_fvg = None

        if active_bull_fvg:
            bull_fvg_low[i], bull_fvg_high[i] = active_bull_fvg[0], active_bull_fvg[1]
        if active_bear_fvg:
            bear_fvg_low[i], bear_fvg_high[i] = active_bear_fvg[0], active_bear_fvg[1]

    df["bull_fvg_low"], df["bull_fvg_high"] = bull_fvg_low, bull_fvg_high
    df["bear_fvg_low"], df["bear_fvg_high"] = bear_fvg_low, bear_fvg_high

    return df


def get_htf_trend(pair: str, htf_interval: str = None, bars: int = 300) -> pd.DataFrame:
    """Fetch a higher timeframe and compute a simple EMA50/EMA200 trend label
    per bar, for aligning against the faster timeframe used for entries."""
    if htf_interval is None:
        htf_interval = HTF_INTERVAL
    htf_df = fetch_ohlc(pair, interval=htf_interval, outputsize=bars)
    htf_df["ema50"] = ema(htf_df["close"], 50)
    htf_df["ema200"] = ema(htf_df["close"], 200)
    htf_df["htf_trend"] = np.where(
        htf_df["ema50"] > htf_df["ema200"], "bullish",
        np.where(htf_df["ema50"] < htf_df["ema200"], "bearish", None)
    )
    return htf_df[["datetime", "htf_trend"]]


def attach_htf_trend(df: pd.DataFrame, pair: str, htf_interval: str = None) -> pd.DataFrame:
    """Merge each bar with the most recently CLOSED higher-timeframe trend
    at that point in time (no lookahead -- uses backward merge_asof)."""
    htf = get_htf_trend(pair, htf_interval)
    df = df.copy()
    df = pd.merge_asof(df.sort_values("datetime"), htf.sort_values("datetime"),
                        on="datetime", direction="backward")
    return df


# ---------------------------------------------------------------------------
# CONFLUENCE STRATEGY
# Signal fires only when trend + momentum + volatility-position agree.
# This is deliberately conservative -- fewer signals, higher quality.
# ---------------------------------------------------------------------------
def generate_signal(row, use_volatility_filter: bool = False, use_smc: bool = None, min_votes: int = None,
                     use_session_filter: bool = None, use_htf_filter: bool = None,
                     session_start_hour: int = None, session_end_hour: int = None) -> str:
    """Return 'BUY', 'SELL', or None for a single row of indicator data."""
    if use_smc is None:
        use_smc = USE_SMC_FILTER
    if min_votes is None:
        min_votes = MIN_CONFLUENCE_VOTES
    if use_session_filter is None:
        use_session_filter = USE_SESSION_FILTER
    if use_htf_filter is None:
        use_htf_filter = USE_HTF_FILTER
    if session_start_hour is None:
        session_start_hour = SESSION_START_HOUR
    if session_end_hour is None:
        session_end_hour = SESSION_END_HOUR

    if pd.isna(row["ema200"]) or pd.isna(row["bb_lower"]):
        return None

    if use_volatility_filter and MIN_VOLATILITY_RATIO > 0:
        if pd.isna(row.get("atr_avg50")) or row["atr_avg50"] == 0:
            return None
        if row["atr14"] < MIN_VOLATILITY_RATIO * row["atr_avg50"]:
            return None  # market too quiet right now -- skip, likely noise

    # --- Session filter: skip thin-liquidity hours ---
    if use_session_filter:
        hour = row["datetime"].hour if hasattr(row["datetime"], "hour") else pd.Timestamp(row["datetime"]).hour
        if not (session_start_hour <= hour < session_end_hour):
            return None

    # --- Higher-timeframe bias gate ---
    htf_buy_ok, htf_sell_ok = True, True
    if use_htf_filter:
        htf_trend = row.get("htf_trend")
        if htf_trend is None or pd.isna(htf_trend):
            return None  # no HTF data yet (warm-up period) -- skip rather than guess
        htf_buy_ok = htf_trend == "bullish"
        htf_sell_ok = htf_trend == "bearish"
        if not htf_buy_ok and not htf_sell_ok:
            return None

    # --- SMC gate (primary): no trade at all unless structure + zone align ---
    smc_buy_ok, smc_sell_ok = True, True
    if use_smc:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        bull_ob_active = not pd.isna(row.get("bull_ob_low"))
        bull_fvg_active = not pd.isna(row.get("bull_fvg_low"))
        bear_ob_active = not pd.isna(row.get("bear_ob_low"))
        bear_fvg_active = not pd.isna(row.get("bear_fvg_low"))

        if SMC_REQUIRE_REJECTION:
            # Wick INTO the zone, but close back out with a confirming candle --
            # this is "mitigation + rejection", not just "price happens to be here".
            in_bull_ob = bull_ob_active and l <= row["bull_ob_high"] and c >= row["bull_ob_low"] and c > o
            in_bull_fvg = bull_fvg_active and l <= row["bull_fvg_high"] and c >= row["bull_fvg_low"] and c > o
            in_bear_ob = bear_ob_active and h >= row["bear_ob_low"] and c <= row["bear_ob_high"] and c < o
            in_bear_fvg = bear_fvg_active and h >= row["bear_fvg_low"] and c <= row["bear_fvg_high"] and c < o
        else:
            in_bull_ob = bull_ob_active and row["bull_ob_low"] <= c <= row["bull_ob_high"]
            in_bull_fvg = bull_fvg_active and row["bull_fvg_low"] <= c <= row["bull_fvg_high"]
            in_bear_ob = bear_ob_active and row["bear_ob_low"] <= c <= row["bear_ob_high"]
            in_bear_fvg = bear_fvg_active and row["bear_fvg_low"] <= c <= row["bear_fvg_high"]

        smc_buy_ok = row.get("structure_trend") == "bullish" and (in_bull_ob or in_bull_fvg)
        smc_sell_ok = row.get("structure_trend") == "bearish" and (in_bear_ob or in_bear_fvg)

        if not smc_buy_ok and not smc_sell_ok:
            return None  # no institutional zone lined up -- skip regardless of indicators

    # --- indicator votes (secondary confirmation) ---
    trend_up = row["ema50"] > row["ema200"]
    trend_down = row["ema50"] < row["ema200"]

    momentum_up = row["macd_hist"] > 0 and row["macd"] > row["macd_signal"]
    momentum_down = row["macd_hist"] < 0 and row["macd"] < row["macd_signal"]

    rsi_ok_buy = 40 < row["rsi14"] < 70          # trending up, not overbought
    rsi_ok_sell = 30 < row["rsi14"] < 60          # trending down, not oversold

    near_lower_band = row["close"] <= row["bb_mid"]   # buying below/at mean, not chasing top
    near_upper_band = row["close"] >= row["bb_mid"]

    buy_votes = sum([trend_up, momentum_up, rsi_ok_buy, near_lower_band])
    sell_votes = sum([trend_down, momentum_down, rsi_ok_sell, near_upper_band])

    if smc_buy_ok and htf_buy_ok and buy_votes >= min_votes:
        return "BUY"
    if smc_sell_ok and htf_sell_ok and sell_votes >= min_votes:
        return "SELL"
    return None


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# Walk forward bar by bar. Once a signal fires, watch subsequent bars until
# TP or SL is hit (whichever comes first, checked with highs/lows).
# ---------------------------------------------------------------------------
def backtest(df: pd.DataFrame, pair: str, use_atr: bool = None, use_volatility_filter: bool = None,
             use_smc: bool = None, min_votes: int = None, use_htf: bool = None, use_session: bool = None,
             use_pair_overrides: bool = True) -> dict:
    # Defaults follow the module-level config, but can be overridden per call
    # (used by the /backtest?atr=true&smc=true&min_votes=3&htf=true&session=true web endpoint).
    # use_pair_overrides=True (default) applies PAIR_OVERRIDES for this pair
    # (e.g. XAU/USD's widened session + lower vote threshold) unless an
    # explicit min_votes/session was already passed in above.
    pair_settings = resolve_pair_settings(pair) if use_pair_overrides else {
        "min_confluence_votes": MIN_CONFLUENCE_VOTES,
        "session_start_hour": SESSION_START_HOUR,
        "session_end_hour": SESSION_END_HOUR,
    }
    if use_atr is None:
        use_atr = USE_ATR_TARGETS
    if use_volatility_filter is None:
        use_volatility_filter = USE_ATR_TARGETS  # ATR mode implies the filter too, by default
    if use_smc is None:
        use_smc = USE_SMC_FILTER
    if min_votes is None:
        min_votes = pair_settings["min_confluence_votes"]
    if use_htf is None:
        use_htf = USE_HTF_FILTER
    if use_session is None:
        use_session = USE_SESSION_FILTER
    session_start_hour = pair_settings["session_start_hour"]
    session_end_hour = pair_settings["session_end_hour"]

    df = add_indicators(df)
    if use_smc:
        df = add_smc_indicators(df)
    if use_htf:
        df = attach_htf_trend(df, pair)
    pip = pip_size(pair)

    trades = []
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        signal = generate_signal(row, use_volatility_filter=use_volatility_filter, use_smc=use_smc,
                                  min_votes=min_votes, use_session_filter=use_session, use_htf_filter=use_htf,
                                  session_start_hour=session_start_hour, session_end_hour=session_end_hour)

        if signal:
            entry = df.iloc[i]["close"]

            if use_atr:
                if pd.isna(row["atr14"]) or row["atr14"] == 0:
                    i += 1
                    continue
                tp_dist = ATR_TP_MULT * row["atr14"]
                sl_dist = ATR_SL_MULT * row["atr14"]
            else:
                tp_dist = TP_PIPS * pip
                sl_dist = SL_PIPS * pip

            tp_pips = tp_dist / pip
            sl_pips = sl_dist / pip

            if signal == "BUY":
                tp_price, sl_price = entry + tp_dist, entry - sl_dist
            else:
                tp_price, sl_price = entry - tp_dist, entry + sl_dist

            outcome, exit_i = None, None
            for j in range(i + 1, n):
                bar = df.iloc[j]
                if signal == "BUY":
                    hit_tp = bar["high"] >= tp_price
                    hit_sl = bar["low"] <= sl_price
                else:
                    hit_tp = bar["low"] <= tp_price
                    hit_sl = bar["high"] >= sl_price

                if hit_tp and hit_sl:
                    outcome = "SL"  # conservative: assume worst case if both hit same bar
                    exit_i = j
                    break
                elif hit_tp:
                    outcome = "TP"
                    exit_i = j
                    break
                elif hit_sl:
                    outcome = "SL"
                    exit_i = j
                    break

            if outcome:
                trades.append({
                    "entry_time": df.iloc[i]["datetime"],
                    "exit_time": df.iloc[exit_i]["datetime"],
                    "direction": signal,
                    "entry": entry,
                    "outcome": outcome,
                    "pips": round(tp_pips if outcome == "TP" else -sl_pips, 1),
                    "tp_pips_target": round(tp_pips, 1),
                    "sl_pips_target": round(sl_pips, 1),
                })
                i = exit_i + 1
                continue
        i += 1

    return summarize(trades, pair, use_atr, use_smc)


def summarize(trades: list, pair: str, use_atr: bool = False, use_smc: bool = False) -> dict:
    n = len(trades)
    if n == 0:
        return {"pair": pair, "trades": 0, "smc_filter": use_smc,
                "message": "No signals generated in this data window."}

    wins = sum(1 for t in trades if t["outcome"] == "TP")
    losses = n - wins
    win_rate = wins / n * 100
    total_pips = sum(t["pips"] for t in trades)
    expectancy = total_pips / n

    # Breakeven win rate needed: with fixed pips this is one number; with
    # ATR-adaptive targets it varies per trade, so average the per-trade
    # reward:risk ratios and derive breakeven from that.
    avg_rr = sum(t["tp_pips_target"] / t["sl_pips_target"] for t in trades) / n
    breakeven_wr = 1 / (1 + avg_rr) * 100

    return {
        "pair": pair,
        "target_mode": "ATR-adaptive" if use_atr else "fixed-pips",
        "smc_filter": use_smc,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "avg_reward_risk_ratio": round(avg_rr, 2),
        "breakeven_win_rate_needed_pct": round(breakeven_wr, 2),
        "total_pips": round(total_pips, 1),
        "expectancy_pips_per_trade": round(expectancy, 2),
        "trade_log": trades,
    }


# ---------------------------------------------------------------------------
# TELEGRAM ALERTS (optional -- works great for phone-only setups)
# ---------------------------------------------------------------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram not configured -- skipping push, see setup notes]")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# LIVE CHECK
# ---------------------------------------------------------------------------
def check_live(pair: str, notify: bool = False, use_atr: bool = None, use_volatility_filter: bool = None,
                use_smc: bool = None, min_votes: int = None, use_htf: bool = None, use_session: bool = None,
                use_pair_overrides: bool = True):
    pair_settings = resolve_pair_settings(pair) if use_pair_overrides else {
        "min_confluence_votes": MIN_CONFLUENCE_VOTES,
        "session_start_hour": SESSION_START_HOUR,
        "session_end_hour": SESSION_END_HOUR,
    }
    if use_atr is None:
        use_atr = USE_ATR_TARGETS
    if use_volatility_filter is None:
        use_volatility_filter = USE_ATR_TARGETS
    if use_smc is None:
        use_smc = USE_SMC_FILTER
    if min_votes is None:
        min_votes = pair_settings["min_confluence_votes"]
    if use_htf is None:
        use_htf = USE_HTF_FILTER
    if use_session is None:
        use_session = USE_SESSION_FILTER
    session_start_hour = pair_settings["session_start_hour"]
    session_end_hour = pair_settings["session_end_hour"]

    df = fetch_ohlc(pair, outputsize=300)  # only need enough bars to warm up indicators
    df = add_indicators(df)
    if use_smc:
        df = add_smc_indicators(df)
    if use_htf:
        df = attach_htf_trend(df, pair)
    last = df.iloc[-1]
    signal = generate_signal(last, use_volatility_filter=use_volatility_filter, use_smc=use_smc, min_votes=min_votes,
                              use_session_filter=use_session, use_htf_filter=use_htf,
                              session_start_hour=session_start_hour, session_end_hour=session_end_hour)
    pip = pip_size(pair)

    ts = last["datetime"]
    price = last["close"]

    if signal:
        if use_atr and not pd.isna(last["atr14"]) and last["atr14"] > 0:
            tp_dist = ATR_TP_MULT * last["atr14"]
            sl_dist = ATR_SL_MULT * last["atr14"]
        else:
            tp_dist = TP_PIPS * pip
            sl_dist = SL_PIPS * pip

        tp_pips = round(tp_dist / pip, 1)
        sl_pips = round(sl_dist / pip, 1)
        tp = price + tp_dist if signal == "BUY" else price - tp_dist
        sl = price - sl_dist if signal == "BUY" else price + sl_dist
        msg = (f"[{pair}] {signal} SIGNAL @ {price:.5f} ({to_display_tz(ts)})\n"
               f"TP: {tp:.5f} ({tp_pips} pips) | SL: {sl:.5f} ({sl_pips} pips)")
    else:
        msg = f"[{pair}] No signal @ {price:.5f} ({to_display_tz(ts)}) -- confluence conditions not met."

    print(msg)
    if notify and signal:
        send_telegram(msg)
    return signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    mode = sys.argv[1]
    pairs = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_PAIRS
    if len(sys.argv) == 2:
        print(f"No pair given -- defaulting to {DEFAULT_PAIRS} (set DEFAULT_PAIRS to change this)")

    if mode == "backtest":
        for pair in pairs:
            settings = resolve_pair_settings(pair)
            print(f"\n=== Backtesting {pair} ({INTERVAL}, ATR={USE_ATR_TARGETS}, SMC={USE_SMC_FILTER}, "
                  f"min_votes={settings['min_confluence_votes']}, "
                  f"session={settings['session_start_hour']}-{settings['session_end_hour']}h UTC, "
                  f"last {OUTPUT_SIZE} bars) ===")
            df = fetch_ohlc(pair)
            result = backtest(df, pair)
            for k, v in result.items():
                if k != "trade_log":
                    print(f"  {k}: {v}")

    elif mode == "live":
        for pair in pairs:
            check_live(pair, notify=False)

    elif mode == "watch":
        print(f"Watching {pairs} every {CHECK_EVERY_SECONDS//60} min. Ctrl+C to stop.")
        while True:
            for pair in pairs:
                try:
                    check_live(pair, notify=True)
                except Exception as e:
                    print(f"Error checking {pair}: {e}")
            time.sleep(CHECK_EVERY_SECONDS)

    elif mode == "sweep":
        # Empirically tests filter-strictness combinations against real data
        # and ranks them -- this is how you find a genuinely more accurate
        # config, instead of guessing at which knob to turn.
        for pair in pairs:
            print(f"\n=== Sweep for {pair} ({INTERVAL}, last {OUTPUT_SIZE} bars) ===")
            df = fetch_ohlc(pair)
            results = []
            for min_votes in [2, 3, 4]:
                for use_htf in [True, False]:
                    for use_smc in [True, False]:
                        r = backtest(df, pair, min_votes=min_votes, use_htf=use_htf, use_smc=use_smc)
                        if r.get("trades", 0) == 0:
                            continue
                        results.append({
                            "min_votes": min_votes,
                            "htf_filter": use_htf,
                            "smc_filter": use_smc,
                            "trades": r["trades"],
                            "win_rate_pct": r["win_rate_pct"],
                            "breakeven_needed_pct": r["breakeven_win_rate_needed_pct"],
                            "expectancy_pips": r["expectancy_pips_per_trade"],
                        })
            if not results:
                print("  No combination produced any trades in this data window.")
                continue
            # Rank by expectancy first -- a higher win rate with worse expectancy
            # (e.g. from fewer, choppier trades) is not actually the better config.
            results.sort(key=lambda r: r["expectancy_pips"], reverse=True)
            print(f"  {'votes':<6}{'htf':<6}{'smc':<6}{'trades':<8}{'win%':<8}{'breakeven%':<12}{'expectancy(pips)'}")
            for r in results:
                print(f"  {r['min_votes']:<6}{str(r['htf_filter']):<6}{str(r['smc_filter']):<6}"
                      f"{r['trades']:<8}{r['win_rate_pct']:<8}{r['breakeven_needed_pct']:<12}{r['expectancy_pips']}")
            print("  Top row = best expectancy found. A config only beats breakeven if "
                  "win% is meaningfully above breakeven%, not just above it -- small samples "
                  "can look profitable by chance. More bars (--outputsize) or a longer "
                  "history reduces that risk.")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
