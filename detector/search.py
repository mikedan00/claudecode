"""원본 출처 후보 탐색(웹 검색 + 위키백과) 및 LLM 학습 데이터 코퍼스 분류.

검색 제공자는 키가 설정된 것을 순서대로 사용한다: SerpAPI → Google CSE → Tavily → DuckDuckGo(키 불필요).
위키백과는 대부분의 LLM 학습 데이터에 포함되므로 항상 함께 조회한다.
"""
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from .extractors import UA, TIMEOUT
from .textutil import sentences, tokenize

# 공개적으로 알려진 LLM 사전학습 데이터셋 구성(The Pile, RedPajama, C4, Dolma, FineWeb 등)에 근거한 분류
CORPUS_DOMAINS = [
    (r"(^|\.)wikipedia\.org$", "Wikipedia (거의 모든 LLM 학습셋 포함)"),
    (r"(^|\.)wikimedia\.org$|(^|\.)wiktionary\.org$|(^|\.)wikisource\.org$", "Wikimedia 프로젝트"),
    (r"(^|\.)github\.com$|(^|\.)githubusercontent\.com$", "GitHub (The Stack/RedPajama 코드)"),
    (r"(^|\.)arxiv\.org$", "arXiv (The Pile/RedPajama 논문)"),
    (r"(^|\.)stackexchange\.com$|(^|\.)stackoverflow\.com$", "StackExchange (The Pile/RedPajama)"),
    (r"(^|\.)reddit\.com$", "Reddit (WebText/Dolma)"),
    (r"(^|\.)gutenberg\.org$", "Project Gutenberg (PG-19/Books)"),
    (r"(^|\.)ncbi\.nlm\.nih\.gov$|(^|\.)pubmed", "PubMed (The Pile)"),
    (r"(^|\.)namu\.wiki$", "나무위키 (한국어 웹 코퍼스 빈출)"),
    (r"(^|\.)tistory\.com$|(^|\.)blog\.naver\.com$|(^|\.)velog\.io$|(^|\.)brunch\.co\.kr$|(^|\.)medium\.com$",
     "블로그 (Common Crawl 포함 가능성 높음)"),
    (r"(^|\.)youtube\.com$|(^|\.)youtu\.be$", "YouTube 자막 (일부 학습셋 포함 보고)"),
]


def corpus_category(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0].removeprefix("www.")
    for pattern, label in CORPUS_DOMAINS:
        if re.search(pattern, host):
            return label
    return "공개 웹 (Common Crawl 계열 포함 가능)"


@dataclass
class Candidate:
    url: str
    title: str
    snippet: str
    provider: str


def build_queries(text: str, k: int = 4) -> list[str]:
    """검색 질의: 고유 단어 비중이 높은 긴 문장 일부를 인용 검색에 사용."""
    sents = sentences(text, min_tokens=8)
    scored = sorted(sents, key=lambda s: len(set(tokenize(s))), reverse=True)
    picks, seen = [], set()
    for s in scored:
        q = " ".join(s.split()[:16])
        if q not in seen:
            picks.append(q)
            seen.add(q)
        if len(picks) >= k:
            break
    if not picks and text.strip():
        picks = [" ".join(text.split()[:16])]
    return picks


def _secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name) or None
    except Exception:
        return None


def _serpapi(q: str, key: str) -> list[Candidate]:
    r = requests.get("https://serpapi.com/search.json", params={"q": q, "api_key": key, "num": 8}, timeout=TIMEOUT)
    r.raise_for_status()
    return [Candidate(i["link"], i.get("title", ""), i.get("snippet", ""), "SerpAPI")
            for i in r.json().get("organic_results", []) if i.get("link")]


def _google_cse(q: str, key: str, cx: str) -> list[Candidate]:
    r = requests.get("https://www.googleapis.com/customsearch/v1", params={"q": q, "key": key, "cx": cx}, timeout=TIMEOUT)
    r.raise_for_status()
    return [Candidate(i["link"], i.get("title", ""), i.get("snippet", ""), "Google CSE") for i in r.json().get("items", [])]


def _tavily(q: str, key: str) -> list[Candidate]:
    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": q, "max_results": 8}, timeout=TIMEOUT)
    r.raise_for_status()
    return [Candidate(i["url"], i.get("title", ""), i.get("content", "")[:300], "Tavily") for i in r.json().get("results", [])]


def _duckduckgo(q: str) -> list[Candidate]:
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": q}, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for a in BeautifulSoup(r.text, "html.parser").select("a.result__a")[:8]:
        href = a.get("href", "")
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        if href.startswith("http"):
            out.append(Candidate(href, a.get_text(strip=True), "", "DuckDuckGo"))
    return out


def _wikipedia(q: str, lang: str) -> list[Candidate]:
    r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=UA, timeout=TIMEOUT, params={
        "action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": 3})
    r.raise_for_status()
    return [Candidate(f"https://{lang}.wikipedia.org/wiki/{i['title'].replace(' ', '_')}", i["title"],
                      re.sub(r"<[^>]+>", "", i.get("snippet", "")), "Wikipedia")
            for i in r.json().get("query", {}).get("search", [])]


def web_search(q: str) -> tuple[list[Candidate], str | None]:
    """설정된 제공자로 검색. (결과, 오류메시지) 반환."""
    try:
        if key := _secret("SERPAPI_KEY"):
            return _serpapi(q, key), None
        if (key := _secret("GOOGLE_API_KEY")) and (cx := _secret("GOOGLE_CSE_ID")):
            return _google_cse(q, key, cx), None
        if key := _secret("TAVILY_API_KEY"):
            return _tavily(q, key), None
        return _duckduckgo(q), None
    except Exception as e:
        return [], f"웹 검색 실패({type(e).__name__}): {e}"


def discover(text: str, max_queries: int = 4, use_web: bool = True, use_wiki: bool = True) -> tuple[list[Candidate], list[str]]:
    errors, found = [], {}
    lang = "ko" if re.search(r"[가-힣]", text) else "en"
    for q in build_queries(text, max_queries):
        if use_web:
            res, err = web_search(f'"{q}"')
            if err:
                errors.append(err)
            elif not res:  # 인용 검색 결과 없으면 일반 검색
                res, err = web_search(q)
                if err:
                    errors.append(err)
            for c in res:
                found.setdefault(c.url, c)
        if use_wiki:
            try:
                for c in _wikipedia(" ".join(q.split()[:8]), lang):
                    found.setdefault(c.url, c)
            except Exception as e:
                errors.append(f"위키백과 검색 실패({type(e).__name__})")
    return list(found.values()), sorted(set(errors))


def reverse_image_search(image_url: str) -> tuple[list[Candidate], str | None]:
    key = _secret("SERPAPI_KEY")
    if not key:
        return [], "역이미지 검색은 SERPAPI_KEY 가 필요합니다(Google Lens)."
    try:
        r = requests.get("https://serpapi.com/search.json", timeout=TIMEOUT,
                         params={"engine": "google_lens", "url": image_url, "api_key": key})
        r.raise_for_status()
        return [Candidate(i["link"], i.get("title", ""), i.get("source", ""), "Google Lens")
                for i in r.json().get("visual_matches", [])[:15] if i.get("link")], None
    except Exception as e:
        return [], f"역이미지 검색 실패({type(e).__name__}): {e}"
