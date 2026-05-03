import requests
import json
import logging
import sqlite3
import os
import time
from dotenv import load_dotenv

_DATA_DIR = os.getenv("DATA_DIR", ".")
_DB_PATH = os.path.join(_DATA_DIR, "paper_trading.db")
_RULEBOOK_PATH = os.path.join(_DATA_DIR, "master_rulebook.md")

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HermesAgent:
    def __init__(self, model_name="hermes", api_url=None):
        self.model_name = model_name

        def get_keys(env_var):
            k = os.getenv(env_var, "")
            return [x.strip() for x in k.split(",") if x.strip()]

        self.keys = {
            "ollama": get_keys("OLLAMA_API_KEYS"),
            "nvidia": get_keys("NVIDIA_API_KEYS"),
            "groq": get_keys("GROQ_API_KEYS"),
            "cerebras": get_keys("CEREBRAS_API_KEYS")
        }

        self.urls = {
            "ollama": "http://127.0.0.1:11434/v1/chat/completions",
            "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "cerebras": "https://api.cerebras.ai/v1/chat/completions"
        }

        self.api_url = api_url
        # Round-robin index per provider to evenly distribute across API keys
        self.key_indices = {}

    def get_lessons(self, market_category=None):
        rulebook_content = ""
        rulebook_path = _RULEBOOK_PATH
        if os.path.exists(rulebook_path):
            with open(rulebook_path, "r", encoding="utf-8") as f:
                rulebook_content = f.read()

        recent_lessons = []
        try:
            conn = sqlite3.connect(_DB_PATH)
            cursor = conn.cursor()
            if market_category:
                # 取得同分類的最近3筆教訓
                cursor.execute("SELECT lesson FROM lessons_learned WHERE market_category = ? ORDER BY id DESC LIMIT 3", (market_category,))
            else:
                cursor.execute("SELECT lesson FROM lessons_learned ORDER BY id DESC LIMIT 3")
            records = cursor.fetchall()
            conn.close()
            if records:
                recent_lessons = [f"- {r[0]}" for r in records]
        except Exception as e:
            logging.error(f"Error fetching lessons: {e}")

        final_context = ""
        if rulebook_content:
            final_context += f"[Hermes Master Rulebook]\n{rulebook_content}\n\n"
        if recent_lessons:
            final_context += "[Recent Contextual Lessons]\n" + "\n".join(recent_lessons)
            
        if not final_context:
            return "No past lessons or rulebook available yet."
            
        return final_context

    def build_judge_prompt(self, question, category, context, bull_arg, bear_arg, sentiment_report, current_date=None, days_left=None):
        past_lessons = self.get_lessons(market_category=category)

        # Inject live calibration stats so the Judge can self-correct for known biases
        calibration_block = ""
        try:
            from tracker import PaperTracker
            stats = PaperTracker().get_calibration_stats(category=category, lookback_n=50)
            if stats.get("sample_size", 0) >= 5:
                calibration_block = (
                    "\n--- YOUR CALIBRATION HISTORY ---\n"
                    f"Brier Score (last {stats['sample_size']} resolved trades): {stats['brier_score']:.3f} "
                    f"(0.00=perfect, 0.25=coin flip)\n"
                    f"Your mean predicted prob: {stats['mean_pred']:.1%} | Actual outcome rate: {stats['mean_actual']:.1%}\n"
                    f"Systematic bias: {stats['bias_direction']} by {stats['bias_amount']:.1%}\n"
                    "CRITICAL: Adjust your probability to correct for this bias before outputting.\n"
                    "-----------------------------------\n"
                )
        except Exception:
            pass

        system_prompt = (
            "You are Hermes, the Supreme Judge of prediction markets. "
            "Your task is to review the Bull and Bear arguments regarding the event and declare the final true probability of it occurring. "
            "You must consider current factors based on the context.\n\n"
            "--- PAST LESSONS LEARNED ---\n"
            f"{past_lessons}\n"
            "----------------------------\n"
            f"{calibration_block}\n"
            "You MUST output your response ONLY in valid JSON format. Do not use Markdown JSON blocks. "
            "The JSON must have exact two keys: 'reasoning' (a brief explanation of your final verdict resolving the debate) and "
            "'probability' (a float between 0.00 and 1.00)."
        )

        if current_date and days_left is not None:
            time_context = f"Current Date: {current_date}\nDays until resolution: {days_left} days\n"
        else:
            time_context = ""

        judge_prompt = (
            f"Question: {question}\n"
            f"Market Category: {category}\n\n"
            f"{time_context}"
            f"News/Facts Context:\n{context}\n\n"
            f"Reddit/Social Crowd Sentiment:\n{sentiment_report}\n\n"
            f"Past Lessons Learned (AVOID THESE MISTAKES):\n{past_lessons}\n\n"
            f"Bull Agent (Argues FOR Yes):\n{bull_arg}\n\n"
            f"Bear Agent (Argues AGAINST Yes):\n{bear_arg}\n\n"
            f"As the final Judge, synthesize all evidence above. Beware of getting swayed by pure social hype if facts contradict it, but DO leverage social momentum if the event is popularity-based. "
            f"Output your verdict as valid JSON only."
        )

        return system_prompt, judge_prompt

    def _call_llm_with_provider(self, sys_p, usr_p, json_mode=False, provider="ollama", override_model=None):
        url = self.urls.get(provider, self.urls["ollama"])
        keys = self.keys.get(provider, [])
        target_model = override_model if override_model else self.model_name

        is_openai = "v1" in url

        if is_openai:
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": usr_p}
                ],
                "temperature": 0.2
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
        else:
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": usr_p}
                ],
                "stream": False,
                "options": {"temperature": 0.2}
            }
            if json_mode:
                payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        if keys:
            # Round-robin across API keys to evenly distribute load
            if provider not in self.key_indices:
                self.key_indices[provider] = 0
            idx = self.key_indices[provider] % len(keys)
            self.key_indices[provider] = idx + 1
            headers["Authorization"] = f"Bearer {keys[idx]}"

        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429:
                    wait = (2 ** attempt) * 5  # 5s → 10s → 20s
                    logging.warning(f"Rate limited by {provider}. Waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                if is_openai:
                    return data["choices"][0]["message"]["content"].strip()
                return data.get("message", {}).get("content", "").strip()
            except requests.Timeout:
                logging.warning(f"Timeout on {provider} attempt {attempt+1}/3")
                time.sleep(2 ** attempt)
            except Exception as e:
                logging.error(f"Error calling {provider} LLM: {e}")
                return None

        return None

    def _call_llm_with_fallback(self, sys_p, usr_p, json_mode=False, providers=[]):
        # Fallback mechanism iterating through a list of (provider, model) tuples
        for provider, model in providers:
            if provider != "ollama" and not self.keys.get(provider):
                continue # Skip if no keys available for cloud provider
                
            logging.info(f"Attempting API call via {provider.upper()} with model {model}...")
            result = self._call_llm_with_provider(sys_p, usr_p, json_mode, provider, model)
            if result:
                return result
            logging.warning(f"{provider.upper()} failed. Trying next fallback...")
            time.sleep(1)
            
        logging.error("All providers in fallback chain failed!")
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
        
        # Routing strategy for Sentiment (Speed-focused 8B)
        providers_chain = [
            ("cerebras", "llama3.1-8b"),
            ("groq", "llama-3.1-8b-instant"),
            ("nvidia", "meta/llama-3.1-8b-instruct"),
            ("ollama", self.model_name)
        ]
        
        result = self._call_llm_with_fallback(sys_prompt, prompt, json_mode=False, providers=providers_chain)
        return result if result else "[NEUTRAL/MIXED] Failed to analyze sentiment."

    def analyze_event_debate(self, question, category, context, sentiment_report="", current_date=None, days_left=None):
        logging.info(f"Initiating Debate for: {question}")
        
        # Routing strategy for Bull/Bear (Intelligence-focused 70B)
        debate_chain = [
            ("groq", "llama-3.3-70b-versatile"),
            ("nvidia", "meta/llama-3.1-70b-instruct"),
            ("cerebras", "llama3.1-8b"), # Cerebras currently doesn't offer 70b on free tier
            ("ollama", self.model_name)
        ]
        
        # 1. Bull Agent
        logging.info(">> Bull Agent is arguing for YES...")
        bull_sys = "You are a bullish analyst. Provide exactly one strong paragraph arguing WHY this event WILL happen. Do not argue both sides."
        bull_usr = f"Current Date: {current_date}\nDays until resolution: {days_left}\n\nContext:\n{context}\n\nSocial Sentiment:\n{sentiment_report}\n\nEvent: {question}"
        bull_arg = self._call_llm_with_fallback(bull_sys, bull_usr, json_mode=False, providers=debate_chain) or "No strong bullish argument generated."
        
        # 2. Bear Agent
        logging.info(">> Bear Agent is arguing for NO...")
        bear_sys = "You are a bearish analyst. Provide exactly one strong paragraph arguing WHY this event WILL NOT happen. Do not argue both sides."
        bear_usr = f"Current Date: {current_date}\nDays until resolution: {days_left}\n\nContext:\n{context}\n\nSocial Sentiment:\n{sentiment_report}\n\nEvent: {question}"
        bear_arg = self._call_llm_with_fallback(bear_sys, bear_usr, json_mode=False, providers=debate_chain) or "No strong bearish argument generated."
        
        # Routing strategy for Judge (Supreme-focused 405B)
        judge_chain = [
            ("nvidia", "meta/llama-3.1-405b-instruct"),
            ("groq", "llama-3.3-70b-versatile"), # Groq fallback if 405b fails
            ("ollama", self.model_name)
        ]
        
        # 3. Judge Agent
        logging.info(">> Judge Agent is evaluating the debate...")
        sys_p, usr_p = self.build_judge_prompt(question, category, context, bull_arg, bear_arg, sentiment_report, current_date, days_left)
        
        content = self._call_llm_with_fallback(sys_p, usr_p, json_mode=True, providers=judge_chain)
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
    agent = HermesAgent(model_name="llama3.1")
    test_q = "Will Apple acquire Tesla before 2026?"
    res = agent.analyze_event_debate(test_q, "Business/Tech", "Tesla stocks are booming, Apple has lots of cash.", "[MILDLY POSITIVE]", "2026-01-01", 365)
    if res:
        print("\nAgent Analysis Result:")
        print(f"Reasoning:\n{res['reasoning']}")
        print(f"Estimated Probability: {res['probability']*100:.1f}%")
