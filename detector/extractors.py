"""업로드 파일/URL에서 분석 가능한 콘텐츠(텍스트, 이미지, 메타데이터) 추출."""
import io
import mimetypes
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; ContentOriginAnalyzer/1.0)"}
TIMEOUT = 15

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".rtf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass
class Content:
    name: str
    kind: str  # text | image | audio | video | unknown
    text: str = ""
    raw: bytes = b""
    url: str | None = None
    meta: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _ext(name: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", name or "")
    return m.group(1).lower() if m else ""


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = [el.get_text(" ", strip=True) for el in root.find_all(["p", "h1", "h2", "h3", "li", "blockquote"])]
    text = "\n".join(b for b in blocks if len(b) > 20)
    if len(text) < 200:
        text = root.get_text("\n", strip=True)
    return title, text


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)


def _docx(data: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _pptx(data: bytes) -> str:
    from pptx import Presentation
    out = []
    for i, slide in enumerate(Presentation(io.BytesIO(data)).slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                out.append(shape.text_frame.text)
        if slide.has_notes_slide:
            out.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(out)


def _image_meta(data: bytes) -> dict:
    from PIL import Image
    from PIL.ExifTags import TAGS
    img = Image.open(io.BytesIO(data))
    meta = {"format": img.format, "size": f"{img.width}x{img.height}", "mode": img.mode}
    exif = {TAGS.get(k, k): str(v)[:200] for k, v in (img.getexif() or {}).items()}
    if exif:
        meta["exif"] = exif
    info = {k: str(v)[:500] for k, v in img.info.items() if isinstance(v, (str, bytes)) and k not in ("icc_profile", "exif")}
    if info:
        meta["text_chunks"] = info
    return meta


def _media_meta(data: bytes, name: str) -> dict:
    try:
        import mutagen
        f = mutagen.File(io.BytesIO(data))
    except Exception:
        f = None
    if not f:
        return {}
    meta = {}
    if getattr(f, "info", None) is not None and hasattr(f.info, "length"):
        meta["duration_sec"] = round(f.info.length, 1)
    if f.tags:
        meta["tags"] = {str(k): str(v)[:200] for k, v in list(f.tags.items())[:40]}
    return meta


def transcribe(data: bytes, name: str) -> str | None:
    """faster-whisper 가 설치돼 있으면 음성을 텍스트로 변환. 없으면 None."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=_ext(name) or ".bin") as tmp:
        tmp.write(data)
        tmp.flush()
        segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(tmp.name)
        return " ".join(s.text for s in segs)


def from_bytes(name: str, data: bytes, url: str | None = None, content_type: str = "") -> Content:
    ext = _ext(name)
    ct = (content_type or mimetypes.guess_type(name)[0] or "").lower()
    c = Content(name=name, kind="unknown", raw=data, url=url)

    if ext == ".pdf" or "pdf" in ct:
        c.kind, c.text = "text", _pdf(data)
    elif ext == ".docx" or "wordprocessingml" in ct:
        c.kind, c.text = "text", _docx(data)
    elif ext == ".pptx" or "presentationml" in ct:
        c.kind, c.text = "text", _pptx(data)
    elif ext in IMAGE_EXT or ct.startswith("image/"):
        c.kind, c.meta = "image", _image_meta(data)
    elif ext in AUDIO_EXT or ct.startswith("audio/") or ext in VIDEO_EXT or ct.startswith("video/"):
        c.kind = "video" if (ext in VIDEO_EXT or ct.startswith("video/")) else "audio"
        c.meta = _media_meta(data, name)
        tr = transcribe(data, name)
        if tr:
            c.text = tr
            c.notes.append("음성을 텍스트로 변환(faster-whisper)하여 텍스트 분석에 사용했습니다.")
        else:
            c.notes.append("음성 텍스트 변환 모듈(faster-whisper)이 없어 메타데이터만 분석합니다.")
    elif ext in {".html", ".htm"} or "html" in ct:
        title, text = html_to_text(data.decode("utf-8", "ignore"))
        c.kind, c.text, c.meta = "text", text, {"title": title}
    elif ext in TEXT_EXT or ct.startswith("text/") or not ext:
        c.kind, c.text = "text", data.decode("utf-8", "ignore")
    return c


def youtube_id(url: str) -> str | None:
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return p.path.strip("/").split("/")[0] or None
    if host.endswith("youtube.com"):
        if p.path == "/watch":
            return parse_qs(p.query).get("v", [None])[0]
        m = re.match(r"^/(shorts|embed|live)/([^/?]+)", p.path)
        if m:
            return m.group(2)
    return None


def _youtube(url: str, vid: str) -> Content:
    c = Content(name=url, kind="video", url=url, meta={"youtube_id": vid})
    try:
        r = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", headers=UA, timeout=TIMEOUT)
        if r.ok:
            j = r.json()
            c.meta.update(title=j.get("title"), author=j.get("author_name"))
    except requests.RequestException:
        pass
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        tr = YouTubeTranscriptApi().fetch(vid, languages=["ko", "en", "ja"])
        c.text = " ".join(s.text for s in tr)
        c.notes.append("YouTube 자막(스크립트)을 텍스트로 분석했습니다.")
    except Exception as e:  # 자막 없음, 차단 등
        c.notes.append(f"자막을 가져오지 못했습니다: {type(e).__name__}. 제목/메타데이터만 분석합니다.")
        c.text = c.meta.get("title") or ""
    return c


def from_url(url: str) -> Content:
    vid = youtube_id(url)
    if vid:
        return _youtube(url, vid)
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    name = urlparse(url).path.rsplit("/", 1)[-1] or urlparse(url).netloc
    if "html" in ct:
        title, text = html_to_text(r.text)
        og = BeautifulSoup(r.text, "html.parser").find("meta", property="og:description")
        if og and og.get("content") and len(text) < 200:
            text = f"{og['content']}\n{text}"
        return Content(name=title or url, kind="text", text=text, url=url, meta={"title": title})
    return from_bytes(name, r.content, url=url, content_type=ct)
