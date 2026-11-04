"""분석 파이프라인: 콘텐츠 → 출처 탐색 → 비교 → 점수화 → 보고서(dict)."""
from concurrent.futures import ThreadPoolExecutor

from . import ai_detect, images, search, similarity
from .extractors import Content, from_url

MIN_CONTAINMENT = 0.03
MIN_COSINE = 0.6
MAX_FETCH = 12

FORMULA = "LLM 재활용 지수 = 0.6 × max(축자 복사율, 학습데이터 추정 중복률) + 0.4 × AI 생성 가능성"


def _fetch_text(c: search.Candidate) -> str:
    try:
        return from_url(c.url).text or c.snippet
    except Exception:
        return c.snippet


def _source_row(name, url, provider, category, comp: similarity.Comparison) -> dict:
    return {
        "title": name, "url": url, "provider": provider, "category": category,
        "containment_pct": round(comp.containment * 100, 1), "cosine_pct": round(comp.cosine * 100, 1),
        "matched_passages": [p.__dict__ for p in comp.passages],
    }


def analyze_text(text: str, references: list[Content], use_web: bool, use_wiki: bool, max_queries: int) -> dict:
    q_sh = similarity.query_shingles(text)
    ai = ai_detect.text_ai_score(text)
    errors, sources = [], []
    copied, training = set(), set()

    for ref in references:
        if not ref.text.strip():
            continue
        comp = similarity.compare(text, ref.text)
        if comp.containment >= MIN_CONTAINMENT or comp.cosine >= MIN_COSINE or comp.passages:
            sources.append(_source_row(ref.name, ref.url, "참조 파일", "사용자 제공 원본", comp))
            copied |= comp.matched_shingles

    if use_web or use_wiki:
        cands, errors = search.discover(text, max_queries, use_web, use_wiki)
        cands = cands[:MAX_FETCH]
        with ThreadPoolExecutor(max_workers=6) as ex:
            texts = list(ex.map(_fetch_text, cands))
        for c, t in zip(cands, texts):
            if not t:
                continue
            comp = similarity.compare(text, t)
            if comp.containment >= MIN_CONTAINMENT or comp.cosine >= MIN_COSINE or comp.passages:
                sources.append(_source_row(c.title or c.url, c.url, c.provider, search.corpus_category(c.url), comp))
                copied |= comp.matched_shingles
                training |= comp.matched_shingles

    sources.sort(key=lambda s: (s["containment_pct"], s["cosine_pct"]), reverse=True)
    total = len(q_sh) or 1
    copy_rate = len(copied & q_sh) / total * 100
    training_rate = len(training & q_sh) / total * 100
    max_cos = max((s["cosine_pct"] for s in sources), default=0.0)
    return {
        "copy_rate": round(copy_rate, 1),
        "training_overlap": round(training_rate, 1),
        "paraphrase_similarity": max_cos,
        "originality": round(100 - copy_rate, 1),
        "ai": ai,
        "sources": sources,
        "errors": errors,
    }


def analyze_image(content: Content, references: list[Content], use_web: bool) -> dict:
    markers = ai_detect.scan_genai_markers(content.raw)
    meta = content.meta
    chunks = meta.get("text_chunks", {})
    reasons = list(markers)
    if any(k.lower() in ("parameters", "prompt", "workflow", "dream", "sd-metadata") for k in chunks):
        reasons.append("PNG 텍스트 청크에 생성 프롬프트/파라미터 존재")
    has_camera = any(k in meta.get("exif", {}) for k in ("Make", "Model"))
    if reasons:
        score, basis = 95.0, "생성 AI 메타데이터/워터마크 표식 발견"
    elif has_camera:
        score, basis = 20.0, "카메라 EXIF(제조사/모델) 존재 — 실제 촬영 가능성"
    else:
        score, basis = 50.0, "판단 근거 부족(메타데이터 제거됨) — 중립값"

    sources, errors = [], []
    try:
        h = images.dhash(content.raw)
    except Exception:
        h = None
    for ref in references:
        if ref.kind != "image" or h is None:
            continue
        try:
            sim = images.hash_similarity(h, images.dhash(ref.raw))
        except Exception:
            continue
        if sim >= 0.8:
            sources.append({"title": ref.name, "url": ref.url, "provider": "참조 이미지", "category": "사용자 제공 원본",
                            "containment_pct": round(sim * 100, 1), "cosine_pct": round(sim * 100, 1), "matched_passages": []})
    if use_web and content.url:
        cands, err = search.reverse_image_search(content.url)
        if err:
            errors.append(err)
        for c in cands:
            sources.append({"title": c.title, "url": c.url, "provider": c.provider, "category": search.corpus_category(c.url),
                            "containment_pct": None, "cosine_pct": None, "matched_passages": []})
    elif use_web:
        errors.append("업로드 이미지는 역이미지 검색 URL이 없어 참조 이미지 비교만 수행합니다. 이미지 URL을 입력하면 역검색합니다.")

    best = max((s["containment_pct"] or 0 for s in sources), default=0.0)
    return {
        "copy_rate": best, "training_overlap": best, "paraphrase_similarity": best,
        "web_appearances": sum(s["provider"] == "Google Lens" for s in sources), "originality": round(100 - best, 1),
        "ai": {"score": score, "basis": basis, "markers": reasons, "has_camera_exif": has_camera},
        "sources": sources, "errors": errors,
    }


def recycle_index(copy_rate: float, training: float, ai_score: float | None) -> float:
    return round(0.6 * max(copy_rate, training) + 0.4 * (ai_score or 0.0), 1)


def analyze(content: Content, references: list[Content] | None = None, use_web: bool = True,
            use_wiki: bool = True, max_queries: int = 4) -> dict:
    references = references or []
    if content.kind == "image":
        res = analyze_image(content, references, use_web)
    elif content.text.strip():
        res = analyze_text(content.text, references, use_web, use_wiki, max_queries)
        if content.kind in ("audio", "video"):
            res["media_markers"] = ai_detect.scan_genai_markers(content.raw) if content.raw else []
    else:
        markers = ai_detect.scan_genai_markers(content.raw) if content.raw else []
        res = {"copy_rate": 0.0, "training_overlap": 0.0, "paraphrase_similarity": 0.0, "originality": None,
               "ai": {"score": 95.0 if markers else None, "markers": markers,
                      "basis": "생성 AI 표식 발견" if markers else "분석 가능한 텍스트/표식 없음"},
               "sources": [], "errors": ["분석 가능한 텍스트가 없습니다."]}

    ai_score = res["ai"].get("score")
    res.update(
        name=content.name, kind=content.kind, url=content.url, meta=content.meta, notes=content.notes,
        recycle_index=recycle_index(res["copy_rate"], res["training_overlap"], ai_score),
        formula=FORMULA, text_length=len(content.text),
    )
    return res
