"""
Gold (XAU/USD) Telegram Signal Bot
-----------------------------------
Price source: gold-api.com public spot price endpoint (XAU, no API key required)
Signal logic: MA5 / MA10 / MA30 crossover + RSI(15)
Features: 3-hour rolling BUY/SELL/HOLD breakdown, inline Refresh/Start/Stop buttons,
          /status, /price, /start, /stop commands, Flask self-ping server for Render free tier.

IMPORTANT — this bot behaves differently from your crypto bots:
XAU/USD is a forex-style spot gold price, not a 24/7 crypto market. Prices from
gold-api.com will stay flat (or stop updating) outside standard trading hours —
roughly Sun 5PM ET to Fri 5PM ET, with brief low-liquidity pauses around the daily
NY close. During those flat stretches, MA spread will naturally shrink and the bot
will mostly emit HOLD — that's expected, not a bug. If you want 24/7 gold-like price
action (e.g. to catch weekend moves), that's what your PAXG bot is for.

Same core structure/logic as your crypto bots:
  - MA spread threshold: 0.08%
  - RSI bands: 75 (buy ceiling) / 25 (sell floor)
  - Confirmation streak: 3 cycles (at 1-min polling = 3 min sustained trend)
  - No reversal/exit warning system — BUY/SELL alerts only, one consistent message template
  - Manual /start and /stop commands (+ inline buttons) control proactive alerts
  - 1-minute polling interval
"""

import os
import sys
import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from threading import Thread

import requests
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gold_bot")

# ---------------------------------------------------------------------------
# Startup environment variable validation
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS = ["BOT_TOKEN", "CHAT_ID"]

missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing:
    logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
    logger.error("Set these in Render's Environment tab before redeploying.")
    sys.exit(1)

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOLD_API_URL = "https://api.gold-api.com/price/XAU"

POLL_INTERVAL_SECONDS = 1800   # 30 minutes
HISTORY_MAXLEN = 60
RSI_PERIOD = 15
MA_SHORT = 5
MA_MED = 10
MA_LONG = 30
ROLLING_WINDOW_HOURS = 3

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
price_history = deque(maxlen=HISTORY_MAXLEN)
signal_log = deque()
alerts_enabled = True         # controlled by /start and /stop
last_alerted_signal = None    # tracks last signal actually sent, to avoid repeats

# --- Signal confidence filtering ---
MIN_MA_SPREAD_PCT = 0.08
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 3
recent_raw_signals = deque(maxlen=CONFIRMATION_STREAK)

# ---------------------------------------------------------------------------
# Flask keep-alive server (Render free tier)
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "Gold (XAU/USD) bot is alive.", 200


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
def fetch_price():
    """Fetch the latest XAU/USD spot price from gold-api.com. Returns float or None.
    No API key required — free public endpoint."""
    try:
        resp = requests.get(GOLD_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "price" not in data:
            logger.warning(f"Unexpected gold-api.com response: {data}")
            return None

        return float(data["price"])

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching price: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing price data: {e}")
        return None


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def moving_average(data, period):
    if len(data) < period:
        return None
    return sum(list(data)[-period:]) / period


def calculate_rsi(data, period=RSI_PERIOD):
    """Returns RSI value or None if insufficient data."""
    if len(data) < period + 1:
        return None

    prices = list(data)[-(period + 1):]
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def generate_signal():
    """
    Returns one of 'BUY', 'SELL', 'HOLD' based on MA5/MA10/MA30 crossover + RSI(15),
    filtered for confidence: requires meaningful MA separation and RSI clearly
    past the midline. Returns None if not enough data yet.
    """
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    rsi = calculate_rsi(price_history)

    if ma5 is None or ma10 is None or ma30 is None:
        return None

    bullish_alignment = ma5 > ma10 > ma30
    bearish_alignment = ma5 < ma10 < ma30

    ma_spread_pct = abs(ma5 - ma30) / ma30 * 100 if ma30 else 0
    strong_spread = ma_spread_pct >= MIN_MA_SPREAD_PCT

    if bullish_alignment and strong_spread and (rsi is None or rsi < RSI_BUY_CEILING):
        return "BUY"
    elif bearish_alignment and strong_spread and (rsi is None or rsi > RSI_SELL_FLOOR):
        return "SELL"
    else:
        return "HOLD"


def confirmed_signal(raw_signal):
    """
    Tracks the last few raw signals and only returns BUY/SELL once the same
    signal has held for CONFIRMATION_STREAK consecutive cycles. Otherwise
    returns 'HOLD' so a single noisy tick can't trigger a false alert.
    """
    recent_raw_signals.append(raw_signal)

    if len(recent_raw_signals) < CONFIRMATION_STREAK:
        return "HOLD"

    if all(s == "BUY" for s in recent_raw_signals):
        return "BUY"
    elif all(s == "SELL" for s in recent_raw_signals):
        return "SELL"
    else:
        return "HOLD"


def record_signal(signal):
    now = datetime.utcnow()
    signal_log.append((now, signal))
    cutoff = now - timedelta(hours=ROLLING_WINDOW_HOURS)
    while signal_log and signal_log[0][0] < cutoff:
        signal_log.popleft()


def get_rolling_breakdown():
    """Returns dict with BUY/SELL/HOLD percentages over the rolling window."""
    if not signal_log:
        return {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    total = len(signal_log)
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for _, sig in signal_log:
        counts[sig] += 1

    return {k: round((v / total) * 100, 1) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Message formatting — ONE consistent template used everywhere a signal is shown
# ---------------------------------------------------------------------------
def build_status_message(price, signal, rsi):
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    breakdown = get_rolling_breakdown()

    rsi_display = f"{rsi}" if rsi is not None else "N/A (gathering data)"
    ma5_display = f"{ma5:.2f}" if ma5 is not None else "N/A"
    ma10_display = f"{ma10:.2f}" if ma10 is not None else "N/A"
    ma30_display = f"{ma30:.2f}" if ma30 is not None else "N/A"
    signal_display = signal if signal is not None else "Gathering data..."

    msg = (
        f"🪙 *XAU/USD (Gold) Signal*\n\n"
        f"💰 Price: ${price:,.2f}\n"
        f"📊 Signal: *{signal_display}*\n\n"
        f"MA5: {ma5_display} | MA10: {ma10_display} | MA30: {ma30_display}\n"
        f"RSI(15): {rsi_display}\n\n"
        f"📈 Last {ROLLING_WINDOW_HOURS}h breakdown:\n"
        f"  BUY: {breakdown['BUY']}% | SELL: {breakdown['SELL']}% | HOLD: {breakdown['HOLD']}%\n\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    return msg


def build_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
        ],
        [
            InlineKeyboardButton("▶️ Start", callback_data="start"),
            InlineKeyboardButton("⏹ Stop", callback_data="stop"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()

    if price is not None:
        price_history.append(price)
        signal = generate_signal()
        if signal:
            record_signal(signal)
        rsi = calculate_rsi(price_history)
        msg = build_status_message(price, signal, rsi)
        await update.message.reply_text(
            msg, parse_mode="Markdown", reply_markup=build_keyboard()
        )
    else:
        await update.message.reply_text(
            "⚠️ Couldn't fetch the current Gold price. Try again shortly."
        )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()
    if price is not None:
        await update.message.reply_text(f"🪙 XAU/USD: ${price:,.2f}")
    else:
        await update.message.reply_text("⚠️ Couldn't fetch the current Gold price.")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    alerts_enabled = True
    await update.message.reply_text(
        "▶️ Alerts turned ON. You'll get a message whenever a strict BUY/SELL signal confirms."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    alerts_enabled = False
    await update.message.reply_text(
        "⏹ Alerts turned OFF. Use /start to resume, or /status anytime to check manually."
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    query = update.callback_query
    await query.answer()

    if query.data == "refresh":
        price = fetch_price()
        if price is not None:
            price_history.append(price)
            signal = generate_signal()
            if signal:
                record_signal(signal)
            rsi = calculate_rsi(price_history)
            msg = build_status_message(price, signal, rsi)
            await query.edit_message_text(
                msg, parse_mode="Markdown", reply_markup=build_keyboard()
            )
        else:
            await query.edit_message_text("⚠️ Couldn't fetch the current Gold price.")

    elif query.data == "start":
        alerts_enabled = True
        await query.answer("Alerts turned ON", show_alert=False)

    elif query.data == "stop":
        alerts_enabled = False
        await query.answer("Alerts turned OFF", show_alert=False)


# ---------------------------------------------------------------------------
# Background polling loop (sends proactive signal updates)
# ---------------------------------------------------------------------------
async def poll_and_alert(context: ContextTypes.DEFAULT_TYPE):
    global last_alerted_signal

    if not alerts_enabled:
        return

    price = fetch_price()
    if price is None:
        logger.warning("Skipping this poll cycle — no price returned.")
        return

    price_history.append(price)
    raw_signal = generate_signal()
    signal = confirmed_signal(raw_signal) if raw_signal is not None else None

    if signal:
        record_signal(signal)

    rsi = calculate_rsi(price_history)

    # Only alert when the confirmed signal is different from the last one we
    # actually sent — stops repeat BUY/BUY/BUY spam every single cycle.
    if signal in ("BUY", "SELL") and signal != last_alerted_signal:
        msg = build_status_message(price, signal, rsi)
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=msg,
                parse_mode="Markdown",
                reply_markup=build_keyboard(),
            )
            last_alerted_signal = signal
        except Exception as e:
            logger.error(f"Failed to send proactive alert: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.job_queue.run_repeating(
        poll_and_alert, interval=POLL_INTERVAL_SECONDS, first=10
    )

    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logger.info("Gold (XAU/USD) bot starting...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
