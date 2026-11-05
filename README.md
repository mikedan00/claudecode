# 🔎 LLM 콘텐츠 출처·복사율 분석기

AI(LLM)가 생성한 콘텐츠(텍스트·워드·PDF·파워포인트·이미지·음악·동영상) 또는 웹 URL(뉴스·블로그·유튜브/쇼츠·SNS 게시글)을 넣으면,
**원본을 얼마나 그대로 가져다 썼는지 / 공개 학습 코퍼스와 얼마나 겹치는지**를 수치(%)로 평가하고 **원본 출처를 역추적해 목록화**합니다.

## 빠른 시작 (VS Code)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
streamlit run app.py
```

VS Code 에서는 `F5` → **Streamlit: app.py** (디버깅) 또는 `Terminal > Run Task` → *Run Streamlit* / *Run tests*.

## Streamlit Community Cloud 배포

1. 이 저장소를 GitHub 에 push
2. https://share.streamlit.io → *New app* → 저장소/브랜치 선택, Main file: `app.py`
3. (선택) *Advanced settings → Secrets* 에 `.streamlit/secrets.toml.example` 형식으로 API 키 입력

## 측정 지표

| 지표 | 의미 | 계산 방법 |
|---|---|---|
| **축자 복사율** | 원문을 그대로 옮긴 비율 | 분석 대상의 5단어 n-gram 중 출처(웹·위키·참조파일)에 존재하는 비율(합집합) |
| **학습데이터 추정 중복률** | 공개 LLM 학습 코퍼스와 겹치는 비율 | 공개 웹/위키백과 출처와 일치한 n-gram 비율. 출처는 The Pile·C4·RedPajama·Dolma 등 공개 구성에 따라 분류 |
| **어휘 유사도** | 의역 가능성 | 출처와의 TF 코사인 유사도 최댓값 |
| **AI 생성 가능성** | LLM 작성 여부 지표 | 텍스트: 문장 길이 균일성·LLM 상투어·어휘 다양성·시작어 반복·목록 구조 / 이미지·미디어: C2PA·IPTC·SD 파라미터·생성도구 표식, 카메라 EXIF |
| **LLM 재활용 지수** | 종합 | `0.6 × max(축자 복사율, 학습데이터 중복률) + 0.4 × AI 생성 가능성` |

출처 목록에는 URL, 코퍼스 분류, 출처별 복사율/유사도, **문장 단위 일치 구간**이 표시되며 Markdown·CSV·JSON 으로 내려받을 수 있습니다.

## 입력 유형별 처리

| 입력 | 처리 |
|---|---|
| txt/md/html, docx, pdf, pptx | 텍스트 추출 → 출처 검색·비교 |
| 이미지 | 생성 AI 메타데이터 탐지, 참조 이미지와 지각해시(dHash) 비교, 이미지 URL은 Google Lens 역검색(SerpAPI) |
| 음악/동영상 파일 | 태그·생성도구 표식 탐지, `faster-whisper` 설치 시 음성→텍스트 후 텍스트 분석 |
| YouTube/Shorts URL | 자막(스크립트) 추출 후 텍스트 분석 |
| 뉴스·블로그·SNS URL | 본문 추출(로그인 필요한 SNS는 공개 미리보기 텍스트만) |

## 검색 API (선택)

`SERPAPI_KEY` → `GOOGLE_API_KEY`+`GOOGLE_CSE_ID` → `TAVILY_API_KEY` → (키 없음) DuckDuckGo 순으로 사용. 위키백과 API 는 항상 무료 조회.
환경변수 또는 `.streamlit/secrets.toml` 로 설정합니다.

## 구조

```
app.py                  Streamlit UI
detector/
  extractors.py         파일·URL·YouTube 콘텐츠 추출
  search.py             웹/위키 출처 탐색, 학습 코퍼스 분류, 역이미지 검색
  similarity.py         n-gram 복사율, 코사인 유사도, 문장 일치 구간
  ai_detect.py          AI 생성 가능성 휴리스틱, 생성 AI 표식 탐지
  images.py             지각 해시(dHash)
  analyzer.py           파이프라인·점수화
  report.py             Markdown/CSV/JSON 보고서
tests/                  pytest (오프라인, 네트워크 mock)
```

## 한계 (반드시 읽어주세요)

- 상용 LLM 의 실제 학습 데이터 목록은 비공개입니다. "학습데이터 추정 중복률"은 **공개적으로 알려진 코퍼스에 포함될 가능성이 높은 공개 출처**와의 중복이며, 확정 판정이 아닙니다.
- AI 생성 가능성은 휴리스틱 지표로, 짧은 글·편집된 글에서 오판할 수 있습니다. 법적·징계 판단의 단독 근거로 사용하지 마세요.
- 의역이 심한 경우 축자 복사율은 낮게 나오며, 어휘 유사도로 보조 판단합니다.
- 메타데이터가 제거된 이미지/오디오는 생성 여부를 판단하기 어렵습니다.

## 테스트

```bash
pytest -q
```
