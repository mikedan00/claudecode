"""원문과 후보 출처 사이의 복사/유사도 측정.

- containment: 분석 대상의 word n-gram 중 출처에도 그대로 존재하는 비율 (축자 복사율)
- cosine: TF 벡터 코사인 유사도 (어휘 수준 유사도, 의역 탐지 보조)
- passages: 문장 단위로 출처의 가장 비슷한 문장과 짝지은 일치 구간
"""
import math
from dataclasses import dataclass, field

from .textutil import sentences, shingles, term_freq, tokenize

NGRAM = 5
PASSAGE_THRESHOLD = 0.5


@dataclass
class Match:
    query_sentence: str
    source_sentence: str
    score: float


@dataclass
class Comparison:
    containment: float
    cosine: float
    matched_shingles: set = field(default_factory=set, repr=False)
    passages: list[Match] = field(default_factory=list)


def cosine(a_tokens: list[str], b_tokens: list[str]) -> float:
    a, b = term_freq(a_tokens), term_freq(b_tokens)
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def _sent_sim(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b)) if min(len(a), len(b)) >= 3 else len(a & b) / max(len(a), len(b))


def compare(query: str, source: str, n: int = NGRAM) -> Comparison:
    q_tok, s_tok = tokenize(query), tokenize(source)
    q_sh, s_sh = shingles(q_tok, n), shingles(s_tok, n)
    matched = q_sh & s_sh
    containment = len(matched) / len(q_sh) if q_sh else 0.0

    src_sents = [(s, shingles(tokenize(s), 3)) for s in sentences(source)]
    passages = []
    for qs in sentences(query):
        q3 = shingles(tokenize(qs), 3)
        best, best_score = None, 0.0
        for ss, s3 in src_sents:
            sc = _sent_sim(q3, s3)
            if sc > best_score:
                best, best_score = ss, sc
        if best and best_score >= PASSAGE_THRESHOLD:
            passages.append(Match(qs, best, round(best_score, 3)))

    return Comparison(containment, cosine(q_tok, s_tok), matched, passages)


def query_shingles(query: str, n: int = NGRAM) -> set:
    return shingles(tokenize(query), n)
