import os
import requests
import logging

def send_telegram_alert(message):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("Telegram Bot Token or Chat ID not found in environment variables. Skipping notification.")
        return False
        
    # Telegram sendMessage API URL
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logging.error(f"Telegram API Error: {response.status_code} - {response.text}")
            return False
            
        logging.info("Telegram notification sent successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return False

if __name__ == "__main__":
    # 測試程式碼
    from dotenv import load_dotenv
    load_dotenv()
    test_msg = "🚨 <b>Polymarket 模擬系統測試</b>\nTelegram 通知模組已成功連線！"
    send_telegram_alert(test_msg)
