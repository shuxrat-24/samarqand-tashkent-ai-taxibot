"""
Samarqand ↔ Toshkent AI Taxi Bot
Separate bot project. Does not modify the existing bot.

Environment variables required:
  TELEGRAM_BOT_TOKEN  - token from @BotFather for the NEW bot
  OPENAI_API_KEY      - OpenAI API key
  TELEGRAM_GROUP_ID   - destination Telegram group ID (for example: -1001234567890)

Install:
  pip install -r requirements.txt

Run:
  python bot.py
"""

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("taxi-ai-bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROUP_ID_RAW = os.getenv("TELEGRAM_GROUP_ID", "").strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in the environment.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment.")
if not GROUP_ID_RAW:
    raise RuntimeError("Set TELEGRAM_GROUP_ID in the environment.")

try:
    GROUP_ID = int(GROUP_ID_RAW)
except ValueError as exc:
    raise RuntimeError("TELEGRAM_GROUP_ID must be an integer.") from exc

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Lightweight in-memory sessions. They reset if the process restarts.
SESSIONS: dict[int, dict[str, Any]] = {}

SYSTEM_PROMPT = """
You are a helpful taxi booking assistant for trips between Samarqand and Toshkent.
Talk naturally and briefly. Support Russian, Uzbek, and English; reply in the language
the passenger uses. No buttons. Do not claim a booking is confirmed until you have
collected the required details and the bot has sent the order to the dispatch group.

Collect these details conversationally:
- route/from and to (normally Samarqand ↔ Toshkent)
- travel date and approximate departure time
- number of passengers/seats
- passenger phone number
- price/budget, if the passenger knows it (optional)

Never invent details. If something required is missing, ask only for the most important
missing detail in your reply. Be clear that this is a request for a driver, not a confirmed
reservation.

Return ONLY valid JSON with exactly these keys:
{
  "reply": "message to send to the passenger",
  "order_complete": false,
  "order": {
    "route": "",
    "date_time": "",
    "passengers": "",
    "phone": "",
    "price": ""
  },
  "new_order": false
}

Set order_complete=true only when route, date/time, passenger count, and phone are all
clearly provided in the conversation. price is optional; if unknown, use "Не указана"
(or the equivalent in the user's language).
Set new_order=true when the user clearly starts a separate new booking. Do not set it
for ordinary follow-up details.
"""

def session_for(user_id: int) -> dict[str, Any]:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            "history": [],
            "order_sent": False,
            "last_order": None,
        }
    return SESSIONS[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        "Здравствуйте! Я помогу оформить поездку Самарканд ↔ Ташкент.\n"
        "Напишите обычным сообщением, откуда и куда едете, дату/время, "
        "сколько пассажиров и ваш номер телефона. Можно писать по-русски, "
        "ўзбекча ёки English."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_message:
        return
    SESSIONS[update.effective_user.id] = {
        "history": [],
        "order_sent": False,
        "last_order": None,
    }
    await update.effective_message.reply_text(
        "Начнём новый заказ. Напишите маршрут, дату и время поездки."
    )


def safe_order(order: Any) -> dict[str, str]:
    if not isinstance(order, dict):
        order = {}
    keys = ("route", "date_time", "passengers", "phone", "price")
    return {key: str(order.get(key, "") or "").strip() for key in keys}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if not message or not user or not message.text:
        return

    # Only handle private chats so group chatter is not sent to the AI.
    if update.effective_chat and update.effective_chat.type != ChatType.PRIVATE:
        return

    session = session_for(user.id)
    user_text = message.text.strip()
    if not user_text:
        return

    # Keep the model context bounded.
    history = session["history"]
    history.append({"role": "user", "content": user_text})
    history[:] = history[-12:]

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        reply = str(data.get("reply", "")).strip()
        order_complete = bool(data.get("order_complete", False))
        order = safe_order(data.get("order"))
        is_new_order = bool(data.get("new_order", False))

        if is_new_order:
            session["order_sent"] = False
            session["last_order"] = None

        if not reply:
            reply = "Напишите, пожалуйста, детали поездки обычным сообщением."

        # Save the assistant response in context.
        history.append({"role": "assistant", "content": raw})
        history[:] = history[-12:]

        if order_complete and not session["order_sent"]:
            username = f"@{user.username}" if user.username else "не указан"
            full_name = " ".join(
                part for part in [user.first_name, user.last_name] if part
            ) or "не указано"

            order_text = (
                "🚕 НОВЫЙ ЗАПРОС НА ПОЕЗДКУ\n\n"
                f"Маршрут: {order['route']}\n"
                f"Дата и время: {order['date_time']}\n"
                f"Пассажиров: {order['passengers']}\n"
                f"Телефон: {order['phone']}\n"
                f"Цена/бюджет: {order['price'] or 'Не указана'}\n\n"
                f"Имя: {full_name}\n"
                f"Telegram: {username}\n"
                f"User ID: {user.id}\n"
                "Статус: ожидает подтверждения диспетчером"
            )
            await context.bot.send_message(chat_id=GROUP_ID, text=order_text)
            session["order_sent"] = True
            session["last_order"] = order
            reply += (
                "\n\nВаш запрос отправлен диспетчеру. Это ещё не подтверждение "
                "поездки: дождитесь ответа."
            )

        await message.reply_text(reply)

    except Exception:
        logger.exception("Failed to process message")
        await message.reply_text(
            "Извините, сейчас не получилось обработать сообщение. "
            "Попробуйте ещё раз чуть позже."
        )


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Taxi AI bot is starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
