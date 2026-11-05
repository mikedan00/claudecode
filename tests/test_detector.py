import io

from PIL import Image

from detector import ai_detect, analyzer, images, report, search, similarity
from detector.extractors import from_bytes, youtube_id

ORIGINAL = (
    "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs. "
    "It stretches for over 2,300 kilometres off the coast of Queensland, Australia. "
    "The reef is located in the Coral Sea and can be seen from outer space. "
    "It was selected as a World Heritage Site in 1981 because of its outstanding universal value."
)
COPIED = (
    "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs. "
    "It stretches for over 2,300 kilometres off the coast of Queensland, Australia. "
    "My uncle visited it last summer and said the boat ride was bumpy but worth every minute of it."
)
UNRELATED = "Bread rises because yeast eats sugar and releases carbon dioxide gas into the dough while it rests overnight."


def test_compare_detects_verbatim_copy():
    c = similarity.compare(COPIED, ORIGINAL)
    assert 0.5 < c.containment < 1.0
    assert len(c.passages) == 2
    assert similarity.compare(UNRELATED, ORIGINAL).containment == 0


def test_analyze_with_reference_offline():
    ref = from_bytes("orig.txt", ORIGINAL.encode())
    r = analyzer.analyze(from_bytes("q.txt", COPIED.encode()), [ref], use_web=False, use_wiki=False)
    assert r["sources"][0]["title"] == "orig.txt"
    assert 50 < r["copy_rate"] < 100
    assert r["originality"] == round(100 - r["copy_rate"], 1)
    md = report.to_markdown(r)
    assert "orig.txt" in md and "LLM 재활용 지수" in md
    assert "orig.txt" in report.to_csv(r)


def test_analyze_web_sources_mocked(monkeypatch):
    cand = search.Candidate("https://en.wikipedia.org/wiki/Great_Barrier_Reef", "Great Barrier Reef", "", "Wikipedia")
    monkeypatch.setattr(search, "discover", lambda *a, **k: ([cand], []))
    monkeypatch.setattr(analyzer, "_fetch_text", lambda c: ORIGINAL)
    r = analyzer.analyze(from_bytes("q.txt", COPIED.encode()), use_web=True, use_wiki=True)
    assert r["training_overlap"] == r["copy_rate"] > 50
    assert r["sources"][0]["category"].startswith("Wikipedia")


def test_ai_score_ranks_llm_style_higher():
    llm = ("In today's ever-evolving landscape, it is important to note that technology plays a crucial role. "
           "Furthermore, businesses must leverage robust tools to foster seamless growth. "
           "Moreover, teams should harness the power of data to unlock the full potential. "
           "Additionally, leaders must navigate the complex landscape with pivotal strategies. "
           "In conclusion, embracing innovation is a testament to a company's crucial vision.")
    human = ("I missed the bus again. Ugh. So I walked, which took forty minutes because the bridge on Elm was closed "
             "for some reason nobody explained, and by the time I got to work Dave had already eaten the last donut. "
             "Typical. Anyway the meeting got cancelled. Whatever, I'll catch up on email tomorrow, maybe, if the coffee machine works.")
    assert ai_detect.text_ai_score(llm)["score"] > ai_detect.text_ai_score(human)["score"] + 20
    assert ai_detect.text_ai_score("too short")["score"] is None


def _png(color, info=None):
    from PIL.PngImagePlugin import PngInfo
    img = Image.new("RGB", (64, 64), color)
    for x in range(32):
        img.putpixel((x, x), (255, 255, 255))
    meta = PngInfo()
    for k, v in (info or {}).items():
        meta.add_text(k, v)
    b = io.BytesIO()
    img.save(b, "PNG", pnginfo=meta)
    return b.getvalue()


def test_image_genai_metadata_and_hash():
    sd = from_bytes("sd.png", _png((10, 20, 30), {"parameters": "a cat, Steps: 20, Sampler: Euler"}))
    r = analyzer.analyze(sd, [from_bytes("ref.png", _png((10, 20, 30)))], use_web=False)
    assert r["ai"]["score"] == 95.0
    assert r["sources"][0]["containment_pct"] == 100.0
    a, b = images.dhash(_png((0, 0, 0))), images.dhash(_png((0, 0, 0)))
    assert images.hash_similarity(a, b) == 1.0


def test_docx_pptx_extraction():
    import docx
    from pptx import Presentation
    d = docx.Document(); d.add_paragraph("hello docx world")
    b = io.BytesIO(); d.save(b)
    assert "hello docx world" in from_bytes("a.docx", b.getvalue()).text
    p = Presentation(); s = p.slides.add_slide(p.slide_layouts[1]); s.shapes.title.text = "slide title here"
    b = io.BytesIO(); p.save(b)
    assert "slide title here" in from_bytes("a.pptx", b.getvalue()).text


def test_youtube_id_and_corpus():
    assert youtube_id("https://www.youtube.com/watch?v=abc123") == "abc123"
    assert youtube_id("https://youtu.be/xyz") == "xyz"
    assert youtube_id("https://youtube.com/shorts/SHORT1?feature=share") == "SHORT1"
    assert youtube_id("https://example.com/watch?v=1") is None
    assert search.corpus_category("https://ko.wikipedia.org/wiki/x").startswith("Wikipedia")
    assert search.corpus_category("https://foo.tistory.com/1").startswith("블로그")


def test_korean_text():
    ko = "인공지능은 인간의 학습 능력과 추론 능력을 컴퓨터 프로그램으로 실현한 기술이다. 이 기술은 다양한 산업 분야에서 활용되고 있다."
    c = similarity.compare(ko, ko + " 추가 문장도 여기에 있습니다 정말로요.")
    assert c.containment == 1.0
    assert search.build_queries(ko)
