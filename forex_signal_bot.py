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
    python forex_signal_bot.py backtest EUR/USD          # backtest last ~5000 bars (H1)
    python forex_signal_bot.py live EUR/USD               # check current signal once
    python forex_signal_bot.py watch EUR/USD GBP/USD ...  # loop + Telegram alerts
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "YOUR_API_KEY_HERE")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")   # from @BotFather
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")       # your chat id

TP_PIPS = 100
SL_PIPS = 30
INTERVAL = "1h"          # timeframe: 1min,5min,15min,1h,4h,1day
OUTPUT_SIZE = 5000       # how many bars to pull (max ~5000 on free tier per call)
CHECK_EVERY_SECONDS = 900  # 15 min, used by "watch" mode

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
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()

    if "values" not in data:
        raise RuntimeError(f"API error for {pair}: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def pip_size(pair: str) -> float:
    """JPY pairs use 2 decimal pip convention, everything else uses 4."""
    return 0.01 if "JPY" in pair.upper() else 0.0001


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
    return df


# ---------------------------------------------------------------------------
# CONFLUENCE STRATEGY
# Signal fires only when trend + momentum + volatility-position agree.
# This is deliberately conservative -- fewer signals, higher quality.
# ---------------------------------------------------------------------------
def generate_signal(row) -> str:
    """Return 'BUY', 'SELL', or None for a single row of indicator data."""
    if pd.isna(row["ema200"]) or pd.isna(row["bb_lower"]):
        return None

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

    if buy_votes == 4:
        return "BUY"
    if sell_votes == 4:
        return "SELL"
    return None


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# Walk forward bar by bar. Once a signal fires, watch subsequent bars until
# TP or SL is hit (whichever comes first, checked with highs/lows).
# ---------------------------------------------------------------------------
def backtest(df: pd.DataFrame, pair: str) -> dict:
    df = add_indicators(df)
    pip = pip_size(pair)
    tp_dist = TP_PIPS * pip
    sl_dist = SL_PIPS * pip

    trades = []
    i = 0
    n = len(df)

    while i < n - 1:
        row = df.iloc[i]
        signal = generate_signal(row)

        if signal:
            entry = df.iloc[i]["close"]
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
                    "pips": TP_PIPS if outcome == "TP" else -SL_PIPS,
                })
                i = exit_i + 1
                continue
        i += 1

    return summarize(trades, pair)


def summarize(trades: list, pair: str) -> dict:
    n = len(trades)
    if n == 0:
        return {"pair": pair, "trades": 0, "message": "No signals generated in this data window."}

    wins = sum(1 for t in trades if t["outcome"] == "TP")
    losses = n - wins
    win_rate = wins / n * 100
    total_pips = sum(t["pips"] for t in trades)
    expectancy = total_pips / n
    breakeven_wr = SL_PIPS / (TP_PIPS + SL_PIPS) * 100

    return {
        "pair": pair,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "breakeven_win_rate_needed_pct": round(breakeven_wr, 2),
        "total_pips": total_pips,
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
def check_live(pair: str, notify: bool = False):
    df = fetch_ohlc(pair, outputsize=300)  # only need enough bars to warm up indicators
    df = add_indicators(df)
    last = df.iloc[-1]
    signal = generate_signal(last)
    pip = pip_size(pair)

    ts = last["datetime"]
    price = last["close"]

    if signal:
        tp = price + TP_PIPS * pip if signal == "BUY" else price - TP_PIPS * pip
        sl = price - SL_PIPS * pip if signal == "BUY" else price + SL_PIPS * pip
        msg = (f"[{pair}] {signal} SIGNAL @ {price:.5f} ({ts})\n"
               f"TP: {tp:.5f} ({TP_PIPS} pips) | SL: {sl:.5f} ({SL_PIPS} pips)")
    else:
        msg = f"[{pair}] No signal @ {price:.5f} ({ts}) -- confluence conditions not met."

    print(msg)
    if notify and signal:
        send_telegram(msg)
    return signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return

    mode = sys.argv[1]
    pairs = sys.argv[2:]

    if mode == "backtest":
        for pair in pairs:
            print(f"\n=== Backtesting {pair} ({INTERVAL}, last {OUTPUT_SIZE} bars) ===")
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

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
