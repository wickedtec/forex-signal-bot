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
                df = bot.fetch_ohlc(pair, outputsize=300)
                df = bot.add_indicators(df)
                last_row = df.iloc[-1]
                signal = bot.generate_signal(last_row)
                bar_time = str(last_row["datetime"])

                _status["last_signals"][pair] = signal or "none"

                if signal and last_alerted.get(pair) != (bar_time, signal):
                    bot.check_live(pair, notify=True)  # sends the Telegram message
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
    # Visit e.g. /backtest?pair=EUR/USD&bars=1500 in your browser.
    # Kept smaller than the CLI default (1500 vs 5000 bars) so it finishes
    # comfortably inside Render's free-tier request timeout.
    pair = request.args.get("pair", "EUR/USD")
    bars = int(request.args.get("bars", 1500))
    try:
        df = bot.fetch_ohlc(pair, outputsize=bars)
        result = bot.backtest(df, pair)
        result.pop("trade_log", None)  # keep the response small/readable
        return result
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    threading.Thread(target=watch_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
    
