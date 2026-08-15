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
    WATCH_PAIRS          e.g. "EUR/USD,GBP/USD,USD/JPY" -- if unset, now
                          defaults to XAU/USD only (see watch_loop() below).
                          Setting WATCH_PAIRS in the dashboard still overrides
                          this, so you can add pairs back any time without a
                          code change/redeploy.
"""

import os
import threading
import time
from flask import Flask, request

import forex_signal_bot as bot  # reuses everything from the existing script

app = Flask(__name__)

_status = {"last_check": None, "last_signals": {}}


def watch_loop():
    # Defaults to XAU/USD only. Twelve Data's free tier caps you at 8 API
    # credits/minute -- with htf checks enabled each pair costs 2 credits per
    # cycle, so watching XAU/USD alone leaves headroom for on-demand
    # /backtest calls to land in the same minute without hitting the limit
    # (4 pairs at 2 credits each was landing right at the ceiling before).
    pairs = os.environ.get("WATCH_PAIRS", "XAU/USD").split(",")
    pairs = [p.strip() for p in pairs if p.strip()]
    interval = int(os.environ.get("CHECK_EVERY_SECONDS", bot.CHECK_EVERY_SECONDS))

    # Tracks the last (bar_time, signal) we already alerted on, per pair,
    # so a signal that's still active on the same candle doesn't get
    # re-sent every 15 minutes until the candle closes.
    last_alerted = {}

    while True:
        for pair in pairs:
            try:
                # check_live() applies every filter (ATR, SMC, HTF, session,
                # min-votes) using the module's current config -- calling it
                # directly here guarantees the watch loop can never drift out
                # of sync with whatever generate_signal() actually does.
                df = bot.fetch_ohlc(pair, outputsize=300)
                bar_time = str(df.iloc[-1]["datetime"])

                signal = bot.check_live(pair, notify=False)  # dry run: get signal + build message, no send yet
                _status["last_signals"][pair] = signal or "none"

                if signal and last_alerted.get(pair) != (bar_time, signal):
                    bot.check_live(pair, notify=True)  # re-run with notify=True to actually send
                    last_alerted[pair] = (bar_time, signal)

            except Exception as e:
                print(f"Error checking {pair}: {e}")
                _status["last_signals"][pair] = f"error: {e}"
        _status["last_check"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        time.sleep(interval)


@app.route("/")
def health():
    # This is what UptimeRobot pings every 5 min to keep the service awake.
    return {
        "status": "running",
        "last_check": _status["last_check"],
        "last_signals": _status["last_signals"],
    }


@app.route("/backtest")
def run_backtest():
    # Visit e.g. /backtest?pair=XAU/USD&interval=15min&atr=true&smc=true&htf=true&session=true&min_votes=3&bars=1500
    # Valid intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week
    # atr=true switches from fixed 100/30 pips to ATR-adaptive targets.
    # smc=true requires market structure + order block/FVG rejection as the
    # primary trigger, with only min_votes of the 4 indicators as backup.
    # htf=true additionally requires the 1h trend to agree with the direction.
    # session=true skips low-liquidity hours outside London/NY (XAU/USD gets
    # its own widened all-day session window automatically -- see
    # PAIR_OVERRIDES in forex_signal_bot.py).
    # All default to the module's own config rather than silently overriding
    # it -- pass explicit =false to turn any of them off.
    pair = request.args.get("pair", "XAU/USD")
    interval = request.args.get("interval", bot.INTERVAL)
    bars = int(request.args.get("bars", 1500))
    use_atr = request.args.get("atr", str(bot.USE_ATR_TARGETS)).lower() == "true"
    use_smc = request.args.get("smc", str(bot.USE_SMC_FILTER)).lower() == "true"
    use_htf = request.args.get("htf", str(bot.USE_HTF_FILTER)).lower() == "true"
    use_session = request.args.get("session", str(bot.USE_SESSION_FILTER)).lower() == "true"
    min_votes = int(request.args.get("min_votes", bot.MIN_CONFLUENCE_VOTES))
    try:
        df = bot.fetch_ohlc(pair, interval=interval, outputsize=bars)
        result = bot.backtest(df, pair, use_atr=use_atr, use_smc=use_smc, min_votes=min_votes,
                               use_htf=use_htf, use_session=use_session)
        result.pop("trade_log", None)  # keep the response small/readable
        result["interval"] = interval
        result["min_votes"] = min_votes
        result["htf_filter"] = use_htf
        result["session_filter"] = use_session
        return result
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/sweep")
def run_sweep():
    # Visit e.g. /sweep?pair=XAU/USD&interval=15min&bars=1500
    #
    # Plain-English version: instead of guessing which filter settings are
    # "better", this tests a grid of combinations (vote threshold x HTF
    # filter x SMC filter) against the SAME real price history and ranks
    # them by expectancy_pips_per_trade -- the number that actually reflects
    # profitability, not just win rate. A higher win rate with worse
    # expectancy is not a better config.
    #
    # Only costs 2 API calls total (one for the base data, one for the
    # higher-timeframe data used by the htf=true combos), no matter how many
    # combinations are tested -- see htf_already_attached in backtest().
    pair = request.args.get("pair", "XAU/USD")
    interval = request.args.get("interval", bot.INTERVAL)
    bars = int(request.args.get("bars", 1500))
    try:
        df_raw = bot.fetch_ohlc(pair, interval=interval, outputsize=bars)
        df_with_htf = bot.attach_htf_trend(df_raw.copy(), pair)  # fetched once, reused below

        results = []
        for min_votes in [2, 3, 4]:
            for use_htf in [True, False]:
                for use_smc in [True, False]:
                    source_df = df_with_htf if use_htf else df_raw
                    r = bot.backtest(source_df.copy(), pair, min_votes=min_votes, use_htf=use_htf,
                                      use_smc=use_smc, htf_already_attached=use_htf)
                    if r.get("trades", 0) == 0:
                        continue
                    results.append({
                        "min_votes": min_votes,
                        "htf_filter": use_htf,
                        "smc_filter": use_smc,
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
    app.run(host="0.0.0.0", port=port)
