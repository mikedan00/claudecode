"""AI(LLM) 생성 가능성 휴리스틱 추정.

학습된 분류기가 아닌 통계적 신호의 조합이므로 '확률'이 아닌 '지표'로 다뤄야 한다.
신호: 문장 길이 균일성(낮은 burstiness), 어휘 다양성, LLM 상투어 빈도, 문장 시작어 반복.
"""
import re
import statistics

from .textutil import sentences, tokenize

LLM_PHRASES = [
    # English
    "delve", "it's important to note", "it is important to note", "in conclusion", "furthermore",
    "moreover", "additionally", "in today's", "ever-evolving", "landscape", "tapestry", "testament to",
    "navigate the", "unlock the", "harness the", "plays a crucial role", "a myriad of", "overall,",
    "in summary", "seamless", "robust", "leverage", "foster", "crucial", "pivotal", "embark",
    # Korean
    "결론적으로", "요약하자면", "요약하면", "중요한 점은", "다양한", "핵심적인", "필수적인",
    "살펴보겠습니다", "알아보겠습니다", "에 대해 알아보", "도움이 되길", "도움이 되었기를",
    "기억하세요", "마무리하자면", "뿐만 아니라", "또한,", "이처럼", "효과적으로", "중요합니다",
]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def text_ai_score(text: str) -> dict:
    sents = sentences(text, min_tokens=3)
    tokens = tokenize(text)
    if len(tokens) < 40 or len(sents) < 3:
        return {"score": None, "reason": "텍스트가 너무 짧아 판단 불가(40단어·3문장 이상 필요)", "signals": {}}

    lens = [len(tokenize(s)) for s in sents]
    cv = statistics.pstdev(lens) / statistics.mean(lens)
    uniformity = _clamp((0.6 - cv) / 0.45)  # 사람 글은 보통 CV 0.5~0.8, LLM은 0.2~0.4

    window = 50
    ttrs = [len(set(tokens[i:i + window])) / window for i in range(0, max(1, len(tokens) - window + 1), 10)]
    mattr = statistics.mean(ttrs) if len(tokens) >= window else len(set(tokens)) / len(tokens)
    diversity_signal = _clamp((0.82 - mattr) / 0.25)

    low = text.lower()
    hits = {p: low.count(p) for p in LLM_PHRASES if p in low}
    per_100 = sum(hits.values()) / len(tokens) * 100
    phrase_signal = _clamp(per_100 / 2.0)

    starts = [tokenize(s)[0] for s in sents if tokenize(s)]
    repeat_signal = _clamp((1 - len(set(starts)) / len(starts)) / 0.5)

    list_lines = len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", text, re.M))
    structure_signal = _clamp(list_lines / max(1, len(sents)) / 0.4)

    weights = {"uniformity": 0.3, "phrases": 0.3, "diversity": 0.15, "repetition": 0.15, "structure": 0.1}
    signals = {
        "uniformity": uniformity, "phrases": phrase_signal, "diversity": diversity_signal,
        "repetition": repeat_signal, "structure": structure_signal,
    }
    score = sum(signals[k] * w for k, w in weights.items())
    return {
        "score": round(score * 100, 1),
        "signals": {k: round(v * 100, 1) for k, v in signals.items()},
        "details": {
            "sentence_length_cv": round(cv, 3), "mattr": round(mattr, 3),
            "llm_phrase_hits": hits, "sentences": len(sents), "tokens": len(tokens),
        },
    }


# 이미지/오디오/영상 파일 안에서 발견되는 생성 AI 도구의 흔적
GENAI_MARKERS = {
    b"c2pa": "C2PA 콘텐츠 자격증명(Content Credentials)",
    b"trainedAlgorithmicMedia": "IPTC DigitalSourceType = AI 생성",
    b"compositeWithTrainedAlgorithmicMedia": "IPTC: AI 합성 포함",
    b"Stable Diffusion": "Stable Diffusion", b"stable-diffusion": "Stable Diffusion",
    b"Midjourney": "Midjourney", b"DALL-E": "DALL·E", b"DALL\xc2\xb7E": "DALL·E", b"OpenAI": "OpenAI",
    b"Adobe Firefly": "Adobe Firefly", b"Imagen": "Google Imagen", b"SynthID": "Google SynthID",
    b"ComfyUI": "ComfyUI", b"NovelAI": "NovelAI", b"Leonardo": "Leonardo.Ai", b"Ideogram": "Ideogram",
    b"Suno": "Suno(음악)", b"Udio": "Udio(음악)", b"ElevenLabs": "ElevenLabs(음성)",
    b"Runway": "Runway(영상)", b"Sora": "OpenAI Sora(영상)", b"Pika": "Pika(영상)", b"Kling": "Kling(영상)",
    b"AI generated": "AI generated 태그", b"ai_generated": "AI generated 태그",
}


def scan_genai_markers(data: bytes) -> list[str]:
    head = data[:4_000_000]  # 메타데이터는 대부분 파일 앞부분
    tail = data[-500_000:]
    found = []
    for marker, label in GENAI_MARKERS.items():
        if (marker in head or marker in tail) and label not in found:
            found.append(label)
    return found
