"""Streamlit UI — LLM 생성 콘텐츠 출처·복사율 분석기.

실행: streamlit run app.py
"""
import streamlit as st

from detector import analyze, from_bytes, from_url, report

st.set_page_config(page_title="콘텐츠 출처 분석기", page_icon="🔎", layout="wide")

st.title("🔎 LLM 콘텐츠 출처·복사율 분석기")
st.caption("AI가 생성한 콘텐츠가 어떤 원본을 얼마나 가져다 썼는지 수치로 평가하고 출처를 역추적합니다.")

with st.sidebar:
    st.header("분석 옵션")
    use_web = st.toggle("웹 검색으로 출처 탐색", value=True)
    use_wiki = st.toggle("위키백과(학습데이터 대표 코퍼스) 조회", value=True)
    max_q = st.slider("검색 질의 수", 1, 8, 4, help="많을수록 정확하지만 느려지고 API 사용량이 늘어납니다.")
    st.divider()
    st.subheader("참조 원본(선택)")
    ref_files = st.file_uploader("비교할 원본 파일을 올리면 직접 대조합니다", accept_multiple_files=True, key="refs")
    st.divider()
    st.markdown(
        "**검색 API 키(선택)** — `.streamlit/secrets.toml` 또는 환경변수\n\n"
        "`SERPAPI_KEY` · `GOOGLE_API_KEY`+`GOOGLE_CSE_ID` · `TAVILY_API_KEY`\n\n"
        "키가 없으면 DuckDuckGo 로 검색합니다. 이미지 역검색은 `SERPAPI_KEY` 필요."
    )

tab_file, tab_url, tab_text = st.tabs(["📁 파일 업로드", "🔗 URL", "📝 텍스트 붙여넣기"])
content = None
with tab_file:
    up = st.file_uploader("텍스트·워드·PDF·파워포인트·이미지·음악·동영상", key="main")
    if up and st.button("파일 분석", type="primary"):
        content = from_bytes(up.name, up.getvalue(), content_type=up.type or "")
with tab_url:
    url = st.text_input("뉴스·블로그·유튜브(쇼츠 포함)·SNS 게시글·이미지 URL", placeholder="https://")
    if url and st.button("URL 분석", type="primary"):
        try:
            content = from_url(url.strip())
        except Exception as e:
            st.error(f"URL을 가져오지 못했습니다: {e}")
with tab_text:
    pasted = st.text_area("분석할 텍스트", height=200)
    if pasted.strip() and st.button("텍스트 분석", type="primary"):
        content = from_bytes("붙여넣은 텍스트.txt", pasted.encode("utf-8"))

if content is not None:
    refs = []
    for f in ref_files or []:
        try:
            refs.append(from_bytes(f.name, f.getvalue(), content_type=f.type or ""))
        except Exception as e:
            st.warning(f"참조 파일 {f.name} 읽기 실패: {e}")
    with st.spinner("출처 탐색 및 비교 중…"):
        st.session_state["result"] = analyze(content, refs, use_web, use_wiki, max_q)

r = st.session_state.get("result")
if r:
    st.divider()
    st.subheader(f"분석 결과 — {r['name']}")
    for n in r.get("notes", []):
        st.info(n)

    def pct(v):
        return "N/A" if v is None else f"{v}%"

    c = st.columns(5)
    c[0].metric("LLM 재활용 지수", pct(r["recycle_index"]), help=r["formula"])
    c[1].metric("축자 복사율", pct(r["copy_rate"]), help="5단어 연속 구문이 출처와 그대로 일치하는 비율")
    c[2].metric("학습데이터 추정 중복률", pct(r["training_overlap"]), help="공개 웹/위키 등 LLM 학습 코퍼스로 알려진 출처와 겹치는 비율")
    c[3].metric("AI 생성 가능성", pct(r["ai"].get("score")), help="문체 통계·메타데이터 기반 휴리스틱")
    c[4].metric("독창성", pct(r["originality"]))
    if r["recycle_index"] is not None:
        st.progress(min(1.0, r["recycle_index"] / 100), text=f"LLM 재활용 지수 {r['recycle_index']}%")

    with st.expander("AI 생성 가능성 근거"):
        ai = r["ai"]
        if ai.get("basis"):
            st.write(ai["basis"])
        if ai.get("signals"):
            labels = {"uniformity": "문장 길이 균일성", "phrases": "LLM 상투어 빈도", "diversity": "어휘 다양성 부족",
                      "repetition": "문장 시작어 반복", "structure": "목록형 구조"}
            st.bar_chart({labels[k]: v for k, v in ai["signals"].items()}, horizontal=True)
            st.json(ai.get("details", {}), expanded=False)
        if ai.get("markers"):
            st.warning("발견된 생성 AI 표식: " + ", ".join(ai["markers"]))
        if ai.get("reason"):
            st.write(ai["reason"])
    if r.get("media_markers"):
        st.warning("미디어 파일 생성 AI 표식: " + ", ".join(r["media_markers"]))

    st.subheader(f"역추적된 원본 출처 ({len(r['sources'])}건)")
    if r["sources"]:
        st.dataframe(
            [{"순위": i, "출처": s["title"], "URL": s["url"], "분류": s["category"], "검색경로": s["provider"],
              "축자 복사율(%)": s["containment_pct"], "어휘 유사도(%)": s["cosine_pct"],
              "일치 문장": len(s["matched_passages"])} for i, s in enumerate(r["sources"], 1)],
            column_config={"URL": st.column_config.LinkColumn("URL")}, hide_index=True, use_container_width=True)
        for s in r["sources"][:10]:
            if s["matched_passages"]:
                with st.expander(f"일치 구간: {s['title']} ({len(s['matched_passages'])}문장)"):
                    for p in s["matched_passages"]:
                        st.markdown(f"**분석 대상** — {p['query_sentence']}\n\n**출처** — {p['source_sentence']}\n\n`일치도 {p['score']:.0%}`")
                        st.divider()
    else:
        st.write("일치하는 출처를 찾지 못했습니다.")
    if r.get("web_appearances"):
        st.write(f"역이미지 검색으로 웹상 동일·유사 이미지 {r['web_appearances']}건을 찾았습니다.")
    for e in r.get("errors", []):
        st.caption(f"⚠️ {e}")

    if r.get("meta"):
        with st.expander("메타데이터"):
            st.json(r["meta"], expanded=False)

    d = st.columns(3)
    d[0].download_button("📄 Markdown 보고서", report.to_markdown(r), "report.md", "text/markdown")
    d[1].download_button("📊 출처 CSV", report.to_csv(r), "sources.csv", "text/csv")
    d[2].download_button("🧾 JSON 원본", report.to_json(r), "report.json", "application/json")

st.divider()
st.caption("⚖️ 본 결과는 공개 출처와의 통계적 비교 및 휴리스틱에 기반한 추정치입니다. 상용 LLM의 실제 학습 데이터 목록은 "
           "공개되지 않으므로, '학습데이터 추정 중복률'은 공개적으로 알려진 학습 코퍼스(위키백과·Common Crawl 등)와의 중복을 의미합니다.")
