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
    WATCH_PAIRS          e.g. "EUR/USD,GBP/USD,USD/JPY"
"""

import os
import threading
import time
from flask import Flask, request

import forex_signal_bot as bot  # reuses everything from the existing script

app = Flask(__name__)

_status = {"last_check": None, "last_signals": {}}


def watch_loop():
    pairs = os.environ.get("WATCH_PAIRS", "EUR/USD").split(",")
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
    # Visit e.g. /backtest?pair=EUR/USD&interval=15min&atr=true&smc=true&htf=true&session=true&min_votes=3&bars=1500
    # Valid intervals: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week
    # atr=true switches from fixed 100/30 pips to ATR-adaptive targets.
    # smc=true requires market structure + order block/FVG rejection as the
    # primary trigger, with only min_votes of the 4 indicators as backup.
    # htf=true additionally requires the 1h trend to agree with the direction.
    # session=true skips low-liquidity hours outside London/NY.
    # All default to the module's own config rather than silently overriding
    # it -- pass explicit =false to turn any of them off.
    pair = request.args.get("pair", "EUR/USD")
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


if __name__ == "__main__":
    threading.Thread(target=watch_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
