import requests
import json
import logging
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from datetime import datetime
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PolyResearcher:
    def __init__(self, llm_url="http://127.0.0.1:11434/api/chat", llm_model="gemma"):
        self.ddgs = DDGS()
        self.llm_url = llm_url
        self.llm_model = llm_model

    def _call_llm(self, sys_p, usr_p, json_mode=False):
        payload = {
            "model": self.llm_model,
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
            response = requests.post(self.llm_url, json=payload, timeout=120)
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logging.error(f"Researcher LLM error: {e}")
            return None

    def _generate_sub_queries(self, question):
        logging.info(f"Generating sub-queries for: {question}")
        sys_p = "You are a research analyst. Given a prediction market question, generate exactly 3 specific Google search queries to gather the most crucial facts needed to resolve it. Output only valid JSON in this format: {'queries': ['q1', 'q2', 'q3']}"
        usr_p = f"Event Question: {question}\nCurrent year: {datetime.now().year}\nGenerate 3 search queries."
        
        resp = self._call_llm(sys_p, usr_p, json_mode=True)
        if resp:
            try:
                queries = json.loads(resp).get("queries", [])
                if isinstance(queries, list) and len(queries) > 0:
                    return queries
            except Exception as e:
                logging.error(f"Failed to parse LLM sub-queries: {e}")
                
        # Fallback
        year = datetime.now().year
        return [f"{question} news {year}", f"{question} official statement {year}", f"{question} site:x.com"]

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
        logging.info(f"🔎 Searching: {query}")
        results_text = ""
        try:
            # DDG 搜尋 URL
            search_results = self.ddgs.text(query, max_results=3)
            if not search_results: return ""
            
            for doc in search_results:
                url = doc.get("href", "")
                snippet = doc.get("body", "")
                
                # 若是 Twitter 則直接用 Snippet，因為 Twitter 會擋 request
                if "twitter.com" in url or "x.com" in url:
                    results_text += f"\n[Tweet Snippet]: {snippet}"
                    continue
                    
                # 嘗試爬取內文
                full_text = self._fetch_url_text(url)
                if len(full_text) > 100:
                    results_text += f"\n[Full-Text Source]: {full_text}"
                else:
                    results_text += f"\n[Snippet Source]: {snippet}"
                    
        except Exception as e:
            logging.error(f"DDGS failed for {query}: {e}")
        return results_text

    def _summarize_research(self, raw_knowledge, question):
        logging.info("🧠 Summarizing research into a Digest...")
        sys_p = "You are a senior intelligence officer. Synthesize the raw, messy scraped web data into a crisp, bulleted 'Intelligence Digest' specifically aimed at helping analysts predict the likelihood of the event resolving as Yes. Drop irrelevant info."
        usr_p = f"Target Event: {question}\n\n[RAW SCRAPED DATA]:\n{raw_knowledge[:10000]}\n\nWrite the comprehensive digest."
        
        digest = self._call_llm(sys_p, usr_p)
        return digest if digest else raw_knowledge[:2000]

    def gather_intelligence(self, query):
        """Auto-Researcher 主迴圈"""
        # 1. 拆解子任務
        sub_queries = self._generate_sub_queries(query)
        logging.info(f"Sub-Queries: {sub_queries}")
        
        import time
        # 2. 循序爬網頁 (放棄多線程以免觸發 DuckDuckGo 的 403 防爬蟲封鎖機制)
        raw_knowledge = ""
        for q in sub_queries:
            res = self._scrape_duckduckgo(q)
            raw_knowledge += res + "\n"
            time.sleep(2) # 友善爬蟲暫停
                
        if not raw_knowledge.strip():
            return "No real-time context found online."
            
        # 3. 濃縮報告
        digest = self._summarize_research(raw_knowledge, query)
        logging.info(">> Research Digest Generated.")
        return digest
