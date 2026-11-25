"""Đọc nội dung văn bản từ các định dạng file phổ biến.

Hỗ trợ: .txt .md .csv .json .pdf .docx .doc .odt .rtf .html/.htm .pptx .xlsx .epub

Thư viện cần cho từng định dạng (chỉ import khi dùng tới):
    pdf  -> pypdf (hoặc PyMuPDF / pdfminer.six nếu có)
    docx -> python-docx
    odt  -> odfpy
    pptx -> python-pptx
    xlsx -> openpyxl
    html, epub -> beautifulsoup4
    doc  -> lệnh `antiword` hoặc `libreoffice`/`soffice` trên máy chủ
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Callable


class DocumentReadError(Exception):
    """File không đọc được (sai định dạng, hỏng, thiếu thư viện...)."""


@dataclass
class Document:
    text: str
    filename: str
    file_type: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "n_chars": len(self.text),
            "n_words": len(self.text.split()),
            **self.meta,
        }


# --------------------------------------------------------------------- helpers
def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.lstrip("\ufeff")
    # bỏ khoảng trắng cuối dòng, gộp > 2 dòng trống
    text = re.sub(r"[ \t\f\v]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _missing(lib: str, ext: str) -> DocumentReadError:
    return DocumentReadError(f"Cần cài thư viện '{lib}' để đọc file {ext} (pip install {lib}).")


# --------------------------------------------------------------------- readers
def read_plain(data: bytes) -> tuple[str, dict]:
    return _decode(data), {}


def read_pdf(data: bytes) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:  # noqa: BLE001
                    raise DocumentReadError("File PDF được bảo vệ bằng mật khẩu.") from e
            pages = [(p.extract_text() or "") for p in reader.pages]
        except DocumentReadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DocumentReadError(f"Không đọc được PDF: {e}") from e
        text = "\n\n".join(pages)
        meta = {"n_pages": len(pages)}
    else:
        try:
            import fitz  # PyMuPDF
            with fitz.open(stream=data, filetype="pdf") as doc:
                pages = [page.get_text() for page in doc]
            text, meta = "\n\n".join(pages), {"n_pages": len(pages)}
        except ImportError:
            try:
                from pdfminer.high_level import extract_text
                text, meta = extract_text(io.BytesIO(data)), {}
            except ImportError as e:
                raise _missing("pypdf", ".pdf") from e

    if not text.strip():
        raise DocumentReadError("PDF không có lớp văn bản (có thể là bản scan ảnh), cần OCR trước khi phân tích.")
    # nối các dòng bị ngắt giữa câu trong PDF
    text = re.sub(r"(?<=[^\s.!?:;\n])\n(?=[^\s\n])", " ", text)
    return text, meta


def read_docx(data: bytes) -> tuple[str, dict]:
    try:
        import docx
    except ImportError as e:
        raise _missing("python-docx", ".docx") from e
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise DocumentReadError(f"Không đọc được DOCX: {e}") from e

    # Duyệt theo thứ tự xuất hiện trong body (đoạn văn + bảng)
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(child, d).text)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and (not cells or cells[-1] != t):  # ô gộp bị lặp lại
                        cells.append(t)
                if cells:
                    parts.append(" | ".join(cells))
    return "\n".join(parts), {}


def read_doc(data: bytes) -> tuple[str, dict]:
    """.doc (Word 97-2003) cần công cụ ngoài."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.doc")
        with open(src, "wb") as f:
            f.write(data)
        if shutil.which("antiword"):
            out = subprocess.run(["antiword", "-w", "0", src], capture_output=True, timeout=60)
            if out.returncode == 0:
                return _decode(out.stdout), {}
        office = shutil.which("libreoffice") or shutil.which("soffice")
        if office:
            subprocess.run(
                [office, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, src],
                capture_output=True, timeout=120,
            )
            txt = os.path.join(tmp, "input.txt")
            if os.path.exists(txt):
                with open(txt, "rb") as f:
                    return _decode(f.read()), {}
    raise DocumentReadError("Không đọc được file .doc: cần cài 'antiword' hoặc LibreOffice trên máy chủ, hoặc lưu lại thành .docx.")


def read_odt(data: bytes) -> tuple[str, dict]:
    try:
        from odf import teletype, text as odf_text
        from odf.opendocument import load
    except ImportError as e:
        raise _missing("odfpy", ".odt") from e
    try:
        doc = load(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise DocumentReadError(f"Không đọc được ODT: {e}") from e
    parts = [teletype.extractText(el) for el in doc.getElementsByType(odf_text.P)]
    return "\n".join(parts), {}


def read_rtf(data: bytes) -> tuple[str, dict]:
    raw = _decode(data)
    try:
        from striprtf.striprtf import rtf_to_text
        return rtf_to_text(raw), {}
    except ImportError:
        pass
    # Fallback đơn giản: giải mã \'xx và \uN, bỏ control word / nhóm
    def _hex(m):
        return bytes([int(m.group(1), 16)]).decode("cp1252", errors="ignore")

    def _uni(m):
        n = int(m.group(1))
        return chr(n + 65536 if n < 0 else n)

    t = re.sub(r"\\par[d]?\b", "\n", raw)
    t = re.sub(r"\\u(-?\d+)\??", _uni, t)
    t = re.sub(r"\\'([0-9a-fA-F]{2})", _hex, t)
    t = re.sub(r"\{\\\*[^{}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", t)
    t = t.replace("{", "").replace("}", "")
    return t, {}


def _html_to_text(html: str | bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise _missing("beautifulsoup4", ".html") from e
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "head", "nav", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def read_html(data: bytes) -> tuple[str, dict]:
    return _html_to_text(_decode(data)), {}


def read_pptx(data: bytes) -> tuple[str, dict]:
    try:
        from pptx import Presentation
    except ImportError as e:
        raise _missing("python-pptx", ".pptx") from e
    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise DocumentReadError(f"Không đọc được PPTX: {e}") from e
    slides = []
    for slide in prs.slides:
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        slides.append("\n".join(texts))
    return "\n\n".join(slides), {"n_slides": len(slides)}


def read_xlsx(data: bytes) -> tuple[str, dict]:
    try:
        import openpyxl
    except ImportError as e:
        raise _missing("openpyxl", ".xlsx") from e
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        raise DocumentReadError(f"Không đọc được XLSX: {e}") from e
    parts = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(v) for v in row if v is not None and str(v).strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts), {"n_sheets": len(wb.worksheets)}


def read_epub(data: bytes) -> tuple[str, dict]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise DocumentReadError("File EPUB bị hỏng.") from e
    # Thứ tự chương lấy theo spine trong file .opf
    names = zf.namelist()
    order: list[str] = []
    opf = next((n for n in names if n.endswith(".opf")), None)
    if opf:
        opf_xml = zf.read(opf).decode("utf-8", errors="ignore")
        base = opf.rsplit("/", 1)[0] + "/" if "/" in opf else ""
        manifest = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf_xml))
        manifest.update({k: v for v, k in re.findall(r'<item[^>]*href="([^"]+)"[^>]*id="([^"]+)"', opf_xml)})
        for idref in re.findall(r'<itemref[^>]*idref="([^"]+)"', opf_xml):
            if idref in manifest and base + manifest[idref] in names:
                order.append(base + manifest[idref])
    if not order:
        order = sorted(n for n in names if n.endswith((".xhtml", ".html", ".htm")))
    chapters = [_html_to_text(zf.read(n)) for n in order]
    return "\n\n".join(chapters), {"n_chapters": len(chapters)}


READERS: dict[str, Callable[[bytes], tuple[str, dict]]] = {
    ".txt": read_plain,
    ".md": read_plain,
    ".markdown": read_plain,
    ".csv": read_plain,
    ".json": read_plain,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".doc": read_doc,
    ".odt": read_odt,
    ".rtf": read_rtf,
    ".html": read_html,
    ".htm": read_html,
    ".pptx": read_pptx,
    ".xlsx": read_xlsx,
    ".epub": read_epub,
}

SUPPORTED_EXTENSIONS = sorted(READERS)


def _sniff_ext(data: bytes) -> str | None:
    """Đoán định dạng từ nội dung khi tên file không có đuôi."""
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"{\\rtf"):
        return ".rtf"
    if data.startswith(b"\xd0\xcf\x11\xe0"):
        return ".doc"
    if data.startswith(b"PK"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            return None
        if "word/document.xml" in names:
            return ".docx"
        if "ppt/presentation.xml" in names:
            return ".pptx"
        if "xl/workbook.xml" in names:
            return ".xlsx"
        if "META-INF/container.xml" in names:
            return ".epub"
        if "content.xml" in names:
            return ".odt"
    return None


def read_document(data: bytes, filename: str = "") -> Document:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in READERS:
        ext = _sniff_ext(data) or (ext if ext else ".txt")
    reader = READERS.get(ext)
    if reader is None:
        raise DocumentReadError(
            f"Định dạng '{ext}' chưa được hỗ trợ. Hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    text, meta = reader(data)
    text = _normalize(text)
    if not text:
        raise DocumentReadError("Không tìm thấy nội dung văn bản trong file.")
    return Document(text=text, filename=filename, file_type=ext.lstrip("."), meta=meta)
