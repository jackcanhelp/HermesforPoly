import requests
import json
import logging
from ddgs import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import os
import time
from dotenv import load_dotenv
from agent import HermesAgent

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PolyResearcher:
    def __init__(self, llm_url=None, llm_model="gemma"):
        self.ddgs = DDGS()
        # 我們將使用 HermesAgent 進行推理，捨棄舊的硬編碼 API 請求

    def _fetch_url_text(self, url):
        try:
            # 加上 User-Agent 騙過基礎擋爬蟲
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # 拔除無用 tag
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.extract()
                # 抓取正文段落
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 50]
                text = " ".join(paragraphs)
                return text[:1500]  # 限制長度避免爆 Token
        except:
            pass
        return ""

    def _scrape_duckduckgo(self, query):
        logging.info(f"🔎 Action [search_web]: {query}")
        results_text = ""
        try:
            # DDG 搜尋 URL
            search_results = self.ddgs.text(query, max_results=3)
            if not search_results: return ""
            
            for doc in search_results:
                url = doc.get("href", "")
                snippet = doc.get("body", "")
                
                if "twitter.com" in url or "x.com" in url:
                    results_text += f"\n[Tweet Snippet]: {snippet}"
                    continue
                    
                full_text = self._fetch_url_text(url)
                if len(full_text) > 100:
                    results_text += f"\n[Source URL]: {url}\n[Content]: {full_text}"
                else:
                    results_text += f"\n[Source URL]: {url}\n[Snippet]: {snippet}"
                    
        except Exception as e:
            logging.error(f"DDGS failed for {query}: {e}")
        return results_text

    def _fetch_crypto_price(self, ticker):
        """Fetch current crypto price from CoinGecko API"""
        logging.info(f"💰 Action [fetch_crypto_price]: {ticker}")
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ticker.lower()}&vs_currencies=usd"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if ticker.lower() in data:
                    price = data[ticker.lower()]["usd"]
                    return f"The current live price of {ticker.upper()} is ${price} USD."
            return f"Failed to fetch price for {ticker}. Try searching the web instead."
        except Exception as e:
            return f"Error fetching price: {e}"

    def gather_social_sentiment(self, query):
        """利用搜尋引擎的 site:reddit.com 語法，免 API 捕捉社群論壇的最新討論串"""
        logging.info(f"🗣️ Action [search_reddit]: {query}")
        
        core_keywords = " ".join(query.replace('?', '').replace(',', '').split()[:5])
        social_q = f"{core_keywords} site:reddit.com"
        
        try:
            results = self.ddgs.text(social_q, max_results=5)
            social_snippets = []
            if results:
                for r in list(results):
                    title = r.get('title', '')
                    body = r.get('body', '')
                    social_snippets.append(f"[Reddit Thread] Title: {title} | Snippet: {body}")
            
            time.sleep(2) # 友善延遲
            
            if not social_snippets:
                return "No significant trending discussions found on Reddit regarding this topic."
                
            return "\n".join(social_snippets)
        except Exception as e:
            logging.error(f"Social sentiment scrape failed: {e}")
            return "Failed to retrieve social data due to rate limits."

    def gather_intelligence(self, query):
        """Auto-Researcher ReAct 主迴圈"""
        logging.info(f"🧠 Researcher entering ReAct Loop for: {query}")
        
        # 使用 70B 模型作為情報員的大腦，因為需要極強的 Tool Calling 邏輯能力
        agent = HermesAgent(model_name="llama3.1")
        providers_chain = [
            ("groq", "llama-3.3-70b-versatile"),
            ("nvidia", "meta/llama-3.1-70b-instruct"),
            ("ollama", agent.model_name)
        ]
        
        tools_schema = '''
You have access to the following tools:
1. `search_web`: Search the internet for general news and facts. Input argument should be a specific search query.
2. `scrape_website`: Read the full text of a specific URL. Input argument should be the exact URL.
3. `search_reddit`: Search Reddit for social sentiment and rumors. Input argument should be the topic.
4. `fetch_crypto_price`: Get the live USD price of a cryptocurrency (e.g. 'bitcoin', 'ethereum'). Input argument should be the coin name.

You must output EXACTLY a valid JSON object in the following format at each step. Do NOT output anything outside the JSON.
{
  "thought": "Your reasoning about what to do next based on the evidence.",
  "action": "The exact name of the tool to use (search_web, scrape_website, search_reddit, fetch_crypto_price), OR 'FINAL_ANSWER' if you have gathered enough facts.",
  "action_input": "The string argument for the tool, OR your final comprehensive intelligence digest if action is FINAL_ANSWER."
}
'''
        
        memory = f"Event Question to Resolve: {query}\nCurrent year: {datetime.now().year}\n\n[Action History]\n"
        
        max_steps = 4
        final_digest = ""
        
        for step in range(max_steps):
            sys_p = "You are an elite autonomous research agent. Your goal is to gather undeniable facts for a prediction market question. Think step-by-step. " + tools_schema
            usr_p = memory + "\nWhat is your next step?"
            
            resp = agent._call_llm_with_fallback(sys_p, usr_p, json_mode=True, providers=providers_chain)
            if not resp:
                logging.error("ReAct LLM call failed.")
                break
                
            try:
                parsed = json.loads(resp)
                thought = parsed.get("thought", "")
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", "")
                
                logging.info(f"🤔 Thought: {thought}")
                
                if action == "FINAL_ANSWER":
                    logging.info("🎯 Researcher found the final answer.")
                    final_digest = action_input
                    break
                    
                observation = ""
                if action == "search_web":
                    observation = self._scrape_duckduckgo(action_input)
                elif action == "scrape_website":
                    observation = self._fetch_url_text(action_input)
                elif action == "search_reddit":
                    observation = self.gather_social_sentiment(action_input)
                elif action == "fetch_crypto_price":
                    observation = self._fetch_crypto_price(action_input)
                else:
                    observation = "Unknown tool. Please use a valid tool name or FINAL_ANSWER."
                
                if not observation or len(observation.strip()) < 5:
                    observation = "No useful results found from this action. Try another tool or different keywords."
                    
                memory += f"\n- Thought: {thought}\n- Action: {action}({action_input})\n- Observation: {observation[:2000]}\n"
                time.sleep(2) # 避免 API 過載
                
            except json.JSONDecodeError:
                logging.error(f"Failed to parse ReAct JSON: {resp}")
                memory += "\n- Observation: Your last output was not valid JSON. Please try again.\n"
                
        # 如果經過 4 步還沒給出最終答案，強制總結
        if not final_digest:
            logging.info("⚠️ Max steps reached. Forcing summarization.")
            sys_p = "You are a research analyst. Summarize the raw observations into a concise intelligence digest."
            usr_p = f"Event: {query}\n\n[Observations]:\n{memory}\n\nWrite the comprehensive digest."
            final_digest = agent._call_llm_with_fallback(sys_p, usr_p, json_mode=False, providers=providers_chain) or memory
            
        return final_digest
