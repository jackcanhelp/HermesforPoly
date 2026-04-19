import requests
import json
import logging
import sqlite3
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HermesAgent:
    def __init__(self, model_name="hermes", api_url="http://127.0.0.1:11434/api/chat"):
        """
        初始化 Agent，預設指向本機端 Ollama API
        若您使用的模型名稱不同 (例如: nous-hermes2)，可在實例化時覆蓋 model_name
        """
        self.model_name = model_name
        self.api_url = api_url

    def get_lessons(self):
        try:
            conn = sqlite3.connect("paper_trading.db")
            cursor = conn.cursor()
            cursor.execute("SELECT lesson FROM lessons_learned ORDER BY id DESC LIMIT 5")
            records = cursor.fetchall()
            conn.close()
            if not records:
                return "No past lessons available yet."
            return "\n".join([f"- {r[0]}" for r in records])
        except Exception:
            return "No past lessons available yet."

    def build_judge_prompt(self, question, category, context, bull_arg, bear_arg, sentiment_report):
        past_lessons = self.get_lessons()
        
        system_prompt = (
            "You are Hermes, the Supreme Judge of prediction markets. "
            "Your task is to review the Bull and Bear arguments regarding the event and declare the final true probability of it occurring. "
            "You must consider current factors based on the context.\n\n"
            "--- PAST LESSONS LEARNED ---\n"
            f"{past_lessons}\n"
            "----------------------------\n\n"
            "You MUST output your response ONLY in valid JSON format. Do not use Markdown JSON blocks. "
            "The JSON must have exact two keys: 'reasoning' (a brief explanation of your final verdict resolving the debate) and "
            "'probability' (a float between 0.00 and 1.00)."
        )
        
        judge_prompt = (
            f"Question: {question}\n"
            f"Market Category: {category}\n\n"
            f"News/Facts Context:\n{context}\n\n"
            f"Reddit/Social Crowd Sentiment:\n{sentiment_report}\n\n"
            f"Past Lessons Learned (AVOID THESE MISTAKES):\n{past_lessons}\n\n"
            f"Bull Agent (Argues FOR Yes):\n{bull_arg}\n\n"
            f"Bear Agent (Argues AGAINST Yes):\n{bear_arg}\n\n"
            f"As the final Judge, synthesize fact-based contexts and social momentum. Beware of getting swayed by pure social hype if facts contradict it, but DO leverage social momentum (FOMO/FUD) if the event is popularity-based.\n"
            f"End your judgment with the Exact format: 'PROBABILITY: X%'"
        )
        
        return system_prompt, judge_prompt

    def _call_llm(self, sys_p, usr_p, json_mode=False):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        if json_mode:
            payload["format"] = "json"
            
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling LLM: {e}")
            return None

    def analyze_social_sentiment(self, question, social_context):
        """讓群眾心理學特工 (Sentiment Analyst) 評估社交網路的情緒動能"""
        logging.info("Sentiment Analyst is evaluating the crowd hype...")
        
        sys_prompt = (
            "You are a Quantitative Sentiment Analyst for a prediction market trading firm. "
            "Your job is to read Reddit or Twitter threads regarding a specific event and gauge the crowd's momentum and emotion. "
            "Classify the sentiment into one of the following exact tags: [EXTREME FOMO], [MILDLY POSITIVE], [NEUTRAL/MIXED], [MILDLY NEGATIVE], [EXTREME FUD/PANIC]. "
            "Provide a highly concise 2-sentence summary of what the crowd believes, and conclude with your exact sentiment tag."
        )
        
        prompt = (
            f"Event/Market Question: {question}\n\n"
            f"Raw Reddit Threads:\n{social_context}\n\n"
            f"What is the crowd's momentum?"
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            result_str = response.json().get('message', {}).get('content', '').strip()
            return result_str
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return "[NEUTRAL/MIXED] Failed to analyze sentiment."

    def analyze_event_debate(self, question, category, context, sentiment_report=""):
        logging.info(f"Initiating Debate for: {question}")
        
        # 1. Bull Agent
        logging.info(">> Bull Agent is arguing for YES...")
        bull_sys = "You are a bullish analyst. Provide exactly one strong paragraph arguing WHY this event WILL happen. Do not argue both sides."
        bull_usr = f"Context:\n{context}\n\nEvent: {question}"
        bull_arg = self._call_llm(bull_sys, bull_usr) or "No strong bullish argument generated."
        
        # 2. Bear Agent
        logging.info(">> Bear Agent is arguing for NO...")
        bear_sys = "You are a bearish analyst. Provide exactly one strong paragraph arguing WHY this event WILL NOT happen. Do not argue both sides."
        bear_usr = f"Context:\n{context}\n\nEvent: {question}"
        bear_arg = self._call_llm(bear_sys, bear_usr) or "No strong bearish argument generated."
        
        # 3. Judge Agent
        logging.info(">> Judge Agent is evaluating the debate...")
        sys_p, usr_p = self.build_judge_prompt(question, category, context, bull_arg, bear_arg, sentiment_report)
        
        content = self._call_llm(sys_p, usr_p, json_mode=True)
        if not content: return None
        
        try:
            parsed_data = json.loads(content)
            if "reasoning" not in parsed_data or "probability" not in parsed_data:
                 logging.warning(f"Unexpected JSON structure: {content}")
                 return None
            return parsed_data
        except json.JSONDecodeError as e:
            logging.error(f"Judge did not return valid JSON: {content} \nException: {e}")
            return None

if __name__ == "__main__":
    # 簡單獨立測試
    # 請確保背景已執行 `ollama run <model_name>`
    agent = HermesAgent(model_name="gemma") # 使用您實際裝備的輕量模型名稱，如 'llama3', 'gemma:7b', 'nous-hermes2'
    
    test_q = "Will Apple acquire Tesla before 2026?"
    res = agent.analyze_event(test_q, category="Business/Tech")
    
    if res:
        print("\nAgent Analysis Result:")
        print(f"Reasoning:\n{res['reasoning']}")
        print(f"Estimated Probability: {res['probability']*100:.1f}%")
