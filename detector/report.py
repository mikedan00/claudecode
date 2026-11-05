"""보고서 내보내기(JSON / Markdown / CSV)."""
import csv
import io
import json


def to_json(r: dict) -> str:
    return json.dumps(r, ensure_ascii=False, indent=2, default=str)


def to_csv(r: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["순위", "제목", "URL", "검색경로", "코퍼스 분류", "축자 복사율(%)", "어휘 유사도(%)", "일치 문장 수"])
    for i, s in enumerate(r["sources"], 1):
        w.writerow([i, s["title"], s["url"] or "", s["provider"], s["category"],
                    s["containment_pct"], s["cosine_pct"], len(s["matched_passages"])])
    return buf.getvalue()


def _pct(v):
    return "N/A" if v is None else f"{v}%"


def to_markdown(r: dict) -> str:
    ai = r["ai"]
    lines = [
        f"# 콘텐츠 출처 분석 보고서 — {r['name']}", "",
        f"- 유형: {r['kind']}" + (f" / URL: {r['url']}" if r.get("url") else ""),
        f"- **LLM 재활용 지수: {r['recycle_index']}%**  ({r['formula']})",
        f"- 축자 복사율: {_pct(r['copy_rate'])}",
        f"- 학습데이터 추정 중복률: {_pct(r['training_overlap'])}",
        f"- 최대 어휘 유사도(의역 지표): {_pct(r['paraphrase_similarity'])}",
        f"- 독창성: {_pct(r['originality'])}",
        f"- AI 생성 가능성: {_pct(ai.get('score'))}" + (f" — {ai['basis']}" if ai.get("basis") else ""),
        "", "## 역추적된 원본 출처", "",
        "| # | 출처 | 분류 | 복사율 | 유사도 | 일치 문장 |", "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(r["sources"], 1):
        link = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
        lines.append(f"| {i} | {link} | {s['category']} | {_pct(s['containment_pct'])} | {_pct(s['cosine_pct'])} | {len(s['matched_passages'])} |")
    if not r["sources"]:
        lines.append("| - | 발견된 출처 없음 | | | | |")
    for s in r["sources"][:5]:
        if s["matched_passages"]:
            lines += ["", f"### 일치 구간 — {s['title']}"]
            for p in s["matched_passages"][:5]:
                lines += [f"- 분석 대상: “{p['query_sentence']}”", f"  - 출처: “{p['source_sentence']}” (일치도 {p['score']:.0%})"]
    lines += ["", "> 본 수치는 공개 출처 대비 통계적 추정치이며, 실제 LLM 학습 데이터 목록은 비공개이므로 확정적 판정이 아닙니다."]
    return "\n".join(lines)
