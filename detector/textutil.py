"""텍스트 정규화, 토큰화, 문장 분리, n-gram 생성."""
import re
from collections import Counter

_WORD = re.compile(r"[0-9A-Za-zÀ-ɏ]+|[가-힣]+|[぀-ヿ一-鿿]")
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=다\.)\s*|\n{1,}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def sentences(text: str, min_tokens: int = 4) -> list[str]:
    parts = [normalize(p) for p in _SENT_SPLIT.split(text or "")]
    return [p for p in parts if len(tokenize(p)) >= min_tokens]


def shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def term_freq(tokens: list[str]) -> Counter:
    return Counter(tokens)
