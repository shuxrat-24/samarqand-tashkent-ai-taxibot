"""
Samarqand ↔ Toshkent AI Taxi Bot
Separate bot project. Does not modify the existing bot.

Environment variables required:
  TELEGRAM_BOT_TOKEN  - token from @BotFather for the NEW bot
  GEMINI_API_KEY      - Google Gemini API key (Google AI Studio)
  TELEGRAM_GROUP_ID   - destination Telegram group ID (for example: -1001234567890)

Install:
  pip install -r requirements.txt

Run:
  python taxi_ai_bot.py
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROUP_ID_RAW = os.getenv("TELEGRAM_GROUP_ID", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in the environment.")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in the environment.")
if not GROUP_ID_RAW:
    raise RuntimeError("Set TELEGRAM_GROUP_ID in the environment.")

try:
    GROUP_ID = int(GROUP_ID_RAW)
except ValueError as exc:
    raise RuntimeError("TELEGRAM_GROUP_ID must be an integer.") from exc

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Lightweight in-memory sessions. They reset if the process restarts.
SESSIONS: dict[int, dict[str, Any]] = {}

SYSTEM_PROMPT = """
You are the AI dispatcher for shared and private taxi trips between Samarqand and Toshkent.
Be friendly, natural, concise, and reply in the passenger's language (Russian, Uzbek, or English).
Ask exactly ONE clear question per reply. Do not repeat questions already answered.
Do not say a trip is confirmed. It is only a request until a dispatcher confirms it.

Collect these fields in a natural conversation:
1. Route: Samarqand to Toshkent or Toshkent to Samarqand.
2. Pickup point: city district, neighborhood, street, landmark, or exact pickup address in the departure city.
3. Destination point: district, neighborhood, street, landmark, or address in the arrival city.
4. Date and approximate departure time.
5. Number of passengers.
6. Ride type: one seat in a shared car, or the whole car privately.
7. If one shared seat: preferred seat, front or back, or "no preference". This is optional if they do not care.
8. Desired price/budget in sums. Ask explicitly how much they are willing to pay. If they don't know or want the driver to quote, record "Не указана".
9. Phone number for contact.

Conversation guidance:
- If the passenger says only "I want to go to Toshkent", infer the likely route from the current route context only if clear; still ask where in Samarqand to pick them up, then where in Toshkent to drop them off.
- If the passenger says "I'm alone", ask whether they want one seat in a shared car or the whole car privately.
- If they choose one seat, ask front or back (they may say no preference).
- Ask the price/budget explicitly before asking for the phone number.
- Ask for the phone number after the trip details are collected.
- Ask only the next most important missing question. Do not ask for several details in one message.
- Never invent pickup points, destination points, dates, prices, phone numbers, or other details.
- Interpret relative dates like "tomorrow" using the current conversation context; if the exact date cannot be established confidently, ask for clarification.
- If the user sends a phone number before all details are collected, retain it and continue asking for the missing details.
- If the user corrects a previously supplied detail, use the corrected detail.
- Do not send an order to the group until all required fields are collected.

Return ONLY valid JSON with exactly these keys:
{
  "reply": "message to send to the passenger",
  "order_complete": false,
  "order": {
    "route": "",
    "pickup_point": "",
    "destination_point": "",
    "date_time": "",
    "passengers": "",
    "ride_type": "",
    "seat_preference": "",
    "phone": "",
    "price": ""
  },
  "new_order": false
}

Set order_complete=true only when route, pickup point, destination point, date/time, passenger count, ride type, price question has been addressed (a stated amount or "Не указана"), and phone number are all clearly available.
For a shared seat, seat_preference may be "Без предпочтений" if the passenger says they do not care. If the whole car is booked privately, use "Не применимо".
If price is unknown, use "Не указана".
Set new_order=true only when the user clearly starts a separate new booking, not for ordinary follow-up details.
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
    keys = (
        "route", "pickup_point", "destination_point", "date_time",
        "passengers", "ride_type", "seat_preference", "phone", "price"
    )
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
    history[:] = history[-10:]

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ],
            temperature=0.2,
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
        history[:] = history[-10:]

        if order_complete and not session["order_sent"]:
            username = f"@{user.username}" if user.username else "не указан"
            full_name = " ".join(
                part for part in [user.first_name, user.last_name] if part
            ) or "не указано"

            order_text = (
                "🚕 НОВЫЙ ЗАПРОС НА ПОЕЗДКУ\\n\\n"
                f"Маршрут: {order['route']}\\n"
                f"Откуда забрать: {order['pickup_point']}\\n"
                f"Куда доставить: {order['destination_point']}\\n"
                f"Дата и время: {order['date_time']}\\n"
                f"Пассажиров: {order['passengers']}\\n"
                f"Формат поездки: {order['ride_type']}\\n"
                f"Место: {order['seat_preference'] or 'Не применимо'}\\n"
                f"Цена/бюджет: {order['price'] or 'Не указана'} сум\\n"
                f"Телефон: {order['phone']}\\n\\n"
                f"Имя: {full_name}\\n"
                f"Telegram: {username}\\n"
                f"User ID: {user.id}\\n"
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
    logger.info("Taxi AI bot is starting with Gemini model %s...", MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
