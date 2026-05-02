import os
import requests
import logging

logger = logging.getLogger("aeco-notifier")

def send_flood_alert(area_ha: float, lat: float, lon: float, timestamp: str, pdf_link: str = "https://dashboard.geoesg.local/report"):
    """
    Sends a formatted flood alert to a configured Telegram chat.
    Reads credentials from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.warning("Telegram EWS: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return

    message = f"🚨 [GEO-A.E.C.O ALERT]\n" \
              f"Flood Detected near Batu Hijau Operations.\n" \
              f"Estimated Area: {area_ha} Ha\n" \
              f"Coords: {lat:.5f}, {lon:.5f}\n" \
              f"Time: {timestamp} WITA\n" \
              f"Report: {pdf_link}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully.")
        else:
            logger.error(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        logger.error(f"Telegram API request failed: {e}")
