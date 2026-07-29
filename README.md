# Confluence Forex Signal Bot

Two versions of the **same strategy** so results should roughly line up:
- `forex_signal_bot.py` — backtest anywhere, get live/watch alerts via Telegram
- `ConfluenceEA.mq5` — runs inside MT5, can auto-trade or just alert, works 24/5 on a VPS

**Strategy logic:** signal only fires when all 4 agree —
trend (EMA50 vs EMA200), momentum (MACD), RSI not overbought/oversold, and price at/past the Bollinger midline.
Fixed 100 pip take-profit / 30 pip stop-loss (~3.3:1 reward:risk).

## ⚠️ Before you risk real money
Run `backtest` mode on several pairs first. The breakeven win rate for this
risk ratio is ~23% — the script prints your *actual* win rate so you can see
if the edge is real on that pair/timeframe, or if it's just noise. Fixed
100-pip targets behave very differently on trending vs ranging pairs, and on
JPY vs non-JPY crosses — test each pair you plan to use.

## Python bot (works entirely from your phone)

1. Get a free API key: https://twelvedata.com (sign up, free tier = 800 credits/day)
2. Best way to run this from a phone: use **Termux** (Android) or a free
   cloud VM (Railway, Render, PythonAnywhere, Replit) — paste the script in,
   it runs 24/7 without your phone needing to stay on.
3. Install deps:
   ```
   pip install requests pandas numpy --break-system-packages
   ```
4. Set your keys (or edit the CONFIG section directly in the file):
   ```
   export TWELVE_DATA_API_KEY="your_key"
   export TELEGRAM_BOT_TOKEN="your_bot_token"   # optional, for phone alerts
   export TELEGRAM_CHAT_ID="your_chat_id"       # optional
   ```
5. Run it:
   ```
   python forex_signal_bot.py backtest EUR/USD
   python forex_signal_bot.py live EUR/USD
   python forex_signal_bot.py watch EUR/USD GBP/USD USD/JPY
   ```

### Telegram alerts setup (2 min, so signals land on your phone)
1. Message **@BotFather** on Telegram → `/newbot` → copy the token it gives you
2. Message **@userinfobot** to get your chat ID
3. Plug both into the env vars above → `watch` mode will now push you a message every time a signal fires

## MT5 EA

1. Copy `ConfluenceEA.mq5` into your MT5 `MQL5/Experts/` folder (via MetaEditor: File > Open Data Folder)
2. Compile in MetaEditor (F7)
3. For phone alerts without auto-trading: leave it attached to a chart, it'll `SendNotification()` on every signal — enable Push Notifications in MT5 mobile app under Options
4. For a live-running bot without keeping a desktop on: rent a cheap VPS (many brokers offer free/discounted VPS if you trade with them), install MT5 there, attach the EA — it'll run 24/5 and push to your phone
5. **Backtest first** in Strategy Tester (Ctrl+R) before going live — same warning as above applies

## Tuning ideas once you've seen real backtest numbers
- Tighten the RSI bands or require 3-of-4 confluence for more signals (lower quality)
- Swap fixed pips for ATR-based TP/SL so targets scale with each pair's volatility
- Add a session filter (e.g. only trade London/NY overlap) — cuts noise on some pairs
- Add spread/slippage cost into the Python backtest for a more realistic number
