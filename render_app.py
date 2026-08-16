"""
render_app.py
--------------
Wraps forex_signal_bot.py so it can run as a FREE Render "Web Service"
instead of a paid Background Worker.

Why this exists: Render's free tier only gives you a web service (which
spins down after 15 min with no incoming requests). This file starts the
watch loop in a background thread AND exposes a tiny "/" endpoint that
returns 200 OK. Point a free uptime pinger (e.g. UptimeRobot) at that
endpoint every 5 minutes, and Render sees regular traffic -> never spins
down -> your watch loop runs continuously for $0/month.

Deploy on Render:
    Build command:  pip install -r requirements.txt
    Start command:  python render_app.py
    Plan:           Free

Set these as Environment Variables in the Render dashboard (not in code):
    TWELVE_DATA_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

This bot trades XAU/USD only -- SYMBOL is hardcoded in forex_signal_bot.py
and every route below (health check, /backtest, /sweep) uses it exclusively.
There is no pair parameter anywhere in this app anymore.
"""

import os
import threading
import time
from flask import Flask, request

import forex_signal_bot as bot  # reuses everything from the existing script

app = Flask(__name__)

_status = {"last_check": None, "last_signal": None}


def watch_loop():
    interval = int(os.environ.get("CHECK_EVERY_SECONDS", bot.CHECK_EVERY_SECONDS))

    # Tracks the last (bar_time, signal) already alerted on, so a signal
    # that's still active on the same candle doesn't get re-sent every
    # 5 minutes until the candle closes.
    last_alerted = None

    while True:
        try:
            # check_live() applies every filter (ATR, SMC, HTF, session,
            # min-votes) using the module's current config -- calling it
            # directly here guarantees the watch loop can never drift out
            # of sync with whatever generate_signal() actually does.
            df = bot.fetch_ohlc(bot.SYMBOL, outputsize=300)
            bar_time = str(df.iloc[-1]["datetime"])

            signal = bot.check_live(bot.SYMBOL, notify=False)  # dry run: get signal + build message, no send yet
            _status["last_signal"] = signal or "none"

            if signal and last_alerted != (bar_time, signal):
                bot.check_live(bot.SYMBOL, notify=True)  # re-run with notify=True to actually send
                last_alerted = (bar_time, signal)

        except Exception as e:
            print(f"Error checking {bot.SYMBOL}: {e}")
            _status["last_signal"] = f"error: {e}"
        _status["last_check"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        time.sleep(interval)


@app.route("/")
def health():
    # This is what UptimeRobot pings every 5 min to keep the service awake.
    return {
        "status": "running",
        "pair": bot.SYMBOL,
        "last_check": _status["last_check"],
        "last_signal": _status["last_signal"],
    }


@app.route("/backtest")
def run_backtest():
    # Visit e.g. /backtest?interval=15min&atr=true&smc=true&htf=true&session=true&trendline=false&min_votes=3&bars=1500
    # Valid intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week
    # atr=true switches from fixed 100/30 pips to ATR-adaptive targets.
    # smc=true requires market structure + order block/FVG rejection as the
    # primary trigger, with only min_votes of the 4 indicators as backup.
    # htf=true additionally requires the 1h trend to agree with the direction.
    # trendline=true additionally requires a classic support/resistance
    # trendline break or bounce to agree with the direction (off by default
    # -- new and unbacktested, see USE_TRENDLINE_FILTER in forex_signal_bot.py).
    # session=true skips low-liquidity hours (widened to all-day for gold --
    # see SESSION_START_HOUR/SESSION_END_HOUR in forex_signal_bot.py).
    # strategy=mean_reversion switches to a completely different, often
    # opposite-direction strategy (trades against short-term extremes rather
    # than with the trend) -- see generate_mean_reversion_signal() for why
    # this isn't blended into the trend-following filters above. When set,
    # smc/htf/trendline/min_votes params are ignored (session filter still
    # applies). zscore threshold controls how extreme a stretch from the
    # rolling mean is required before it counts (default 2.0 std devs).
    # All default to the module's own config rather than silently overriding
    # it -- pass explicit =false to turn any of them off.
    pair = bot.SYMBOL
    interval = request.args.get("interval", bot.INTERVAL)
    bars = int(request.args.get("bars", 1500))
    strategy = request.args.get("strategy", "trend")
    zscore_threshold = float(request.args.get("zscore", 2.0))
    use_atr = request.args.get("atr", str(bot.USE_ATR_TARGETS)).lower() == "true"
    use_smc = request.args.get("smc", str(bot.USE_SMC_FILTER)).lower() == "true"
    use_htf = request.args.get("htf", str(bot.USE_HTF_FILTER)).lower() == "true"
    use_session = request.args.get("session", str(bot.USE_SESSION_FILTER)).lower() == "true"
    use_trendline = request.args.get("trendline", str(bot.USE_TRENDLINE_FILTER)).lower() == "true"
    min_votes = int(request.args.get("min_votes", bot.MIN_CONFLUENCE_VOTES))
    try:
        df = bot.fetch_ohlc(pair, interval=interval, outputsize=bars)
        result = bot.backtest(df, pair, use_atr=use_atr, use_smc=use_smc, min_votes=min_votes,
                               use_htf=use_htf, use_session=use_session, use_trendline=use_trendline,
                               strategy=strategy, zscore_threshold=zscore_threshold)
        result.pop("trade_log", None)  # keep the response small/readable
        result["pair"] = pair
        result["interval"] = interval
        result["strategy"] = strategy
        result["min_votes"] = min_votes
        result["htf_filter"] = use_htf
        result["session_filter"] = use_session
        result["trendline_filter"] = use_trendline
        return result
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/sweep")
def run_sweep():
    # Visit e.g. /sweep?bars=500              (fast: 12 combos)
    #        or  /sweep?bars=500&include_trendline=true   (slower: 24 combos)
    #
    # Plain-English version: instead of guessing which filter settings are
    # "better", this tests a grid of combinations against the SAME real price
    # history and ranks them by expectancy_pips_per_trade -- the number that
    # actually reflects profitability, not just win rate. A higher win rate
    # with worse expectancy is not a better config.
    #
    # Render's free tier gives this app a fraction of a CPU core, and the
    # backtest engine loops row-by-row in Python rather than using vectorized
    # pandas operations -- fine for one backtest, but 24 of them back-to-back
    # on a large bar count was timing out before finishing. Defaults here are
    # tuned to actually complete: bars=500 and a 12-combo grid (min_votes x
    # htf x smc) by default. Pass include_trendline=true to also test the
    # trendline filter (24 combos) once you've confirmed the smaller sweep
    # works -- and/or raise bars back up gradually to see how far this
    # instance can go before it times out again.
    #
    # API cost: 2 calls total (base data + one HTF fetch), regardless of grid
    # size -- the trendline filter costs no extra API calls at all, since
    # it's computed from data already fetched.
    pair = bot.SYMBOL
    interval = request.args.get("interval", bot.INTERVAL)
    bars = int(request.args.get("bars", 500))
    include_trendline = request.args.get("include_trendline", "false").lower() == "true"
    trendline_options = [True, False] if include_trendline else [False]
    try:
        df_raw = bot.fetch_ohlc(pair, interval=interval, outputsize=bars)
        df_with_htf = bot.attach_htf_trend(df_raw.copy(), pair)  # fetched once, reused below

        results = []
        for min_votes in [2, 3, 4]:
            for use_htf in [True, False]:
                for use_smc in [True, False]:
                    for use_trendline in trendline_options:
                        source_df = df_with_htf if use_htf else df_raw
                        r = bot.backtest(source_df.copy(), pair, min_votes=min_votes, use_htf=use_htf,
                                          use_smc=use_smc, use_trendline=use_trendline,
                                          htf_already_attached=use_htf)
                        if r.get("trades", 0) == 0:
                            continue
                        results.append({
                            "min_votes": min_votes,
                            "htf_filter": use_htf,
                            "smc_filter": use_smc,
                            "trendline_filter": use_trendline,
                            "trades": r["trades"],
                            "win_rate_pct": r["win_rate_pct"],
                            "breakeven_needed_pct": r["breakeven_win_rate_needed_pct"],
                            "expectancy_pips_per_trade": r["expectancy_pips_per_trade"],
                            "total_pips": r["total_pips"],
                        })

        # Best expectancy first -- this is the config actually worth trusting,
        # not necessarily the one with the highest win rate.
        results.sort(key=lambda r: r["expectancy_pips_per_trade"], reverse=True)
        return {
            "pair": pair,
            "interval": interval,
            "bars": bars,
            "note": "Ranked by expectancy_pips_per_trade (best first), not win_rate_pct. "
                    "A config only beats breakeven if win_rate_pct is meaningfully above "
                    "breakeven_needed_pct -- with small trade counts that gap can be noise.",
            "results": results,
        }
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    threading.Thread(target=watch_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port, threaded=True)
