"""PDF uploads through pypdfium2: page text reaches the LLM, scanned pages reach
the vision model as PNGs, and an unreadable file becomes an error reply. PDFium
crashes the process on concurrent calls, so its calls hold PDFIUM_LOCK and the
LLM calls, which take up to 90 s, do not.

The PDFs are written here with the standard library, so no PDF library is
needed to make them.
"""

import base64
import hashlib
import io
import json
import logging
import struct
import sys

import pytest

from ingestion.file_conversion_handler import convert_file_to_diagram_spec


SPEC = json.dumps({"classes": [{"className": "Book", "attributes": [], "methods": []}], "relationships": []})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# US Letter at 200 dpi: 612 x 792 pt.
LETTER_AT_200_DPI = (1700, 2200)
RED_SQUARE = "1 0 0 rg 100 100 200 200 re f"
_PASSWORD_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _text(*lines: str) -> str:
    """A page content stream with one text line per argument."""
    shown = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    return f"BT /F1 12 Tf 72 720 Td {shown} ET"


def _rc4(key: bytes, data: bytes) -> bytes:
    state, j = list(range(256)), 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) % 256
        state[i], state[j] = state[j], state[i]
    out, i, j = bytearray(), 0, 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + state[i]) % 256
        state[i], state[j] = state[j], state[i]
        out.append(byte ^ state[(state[i] + state[j]) % 256])
    return bytes(out)


def _pdf(*pages: str, user_password: str = None) -> bytes:
    """A PDF of the given content streams; a password applies the 40-bit RC4
    standard security handler (PDF 1.7, 7.6), so the file is really encrypted."""
    file_id = b"0123456789abcdef"
    padded = (user_password or "").encode()[:32]
    padded += _PASSWORD_PAD[:32 - len(padded)]
    owner = _rc4(hashlib.md5(b"owner" + _PASSWORD_PAD).digest()[:5], padded)
    permissions = struct.pack("<i", -4)
    file_key = hashlib.md5(padded + owner + permissions + file_id).digest()[:5]

    first_content = 4
    kids = " ".join(f"{first_content + 2 * i + 1} 0 R" for i in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, content in enumerate(pages):
        number = first_content + 2 * index
        data = content.encode("latin-1")
        if user_password is not None:
            object_key = file_key + struct.pack("<I", number)[:3] + b"\x00\x00"
            data = _rc4(hashlib.md5(object_key).digest()[:10], data)
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(data), data))
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {number} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
    trailer = f"/Root 1 0 R /ID [<{file_id.hex()}> <{file_id.hex()}>]".encode()
    if user_password is not None:
        objects.append(
            b"<< /Filter /Standard /V 1 /R 2 /P -4 /O <%s> /U <%s> >>"
            % (owner.hex().encode(), _rc4(file_key, _PASSWORD_PAD).hex().encode())
        )
        trailer += b" /Encrypt %d 0 R" % len(objects)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d %s >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, trailer, xref)
    return bytes(out)


class _Models:
    """Fake llm_predict and vision call, recording what they were sent."""

    def __init__(self):
        self.prompts = []
        self.images = []
        self.timeout = None

    def predict(self, prompt):
        self.prompts.append(prompt)
        return SPEC

    def vision(self, prompt, images, timeout):
        self.images = images
        self.timeout = timeout
        return SPEC


def _convert(pdf: bytes, models: _Models, vision: bool = True) -> dict:
    return convert_file_to_diagram_spec(
        file_content_b64=base64.b64encode(pdf).decode(),
        filename="doc.pdf",
        llm_predict=models.predict,
        llm_vision_predict=models.vision if vision else None,
    )


def _png(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    return base64.b64decode(data_url[len(prefix):])


def _png_size(png: bytes) -> tuple:
    assert png[:8] == PNG_SIGNATURE
    return struct.unpack(">II", png[16:24])


# ── Text path ────────────────────────────────────────────────────────────────


def test_page_text_reaches_the_prompt_under_page_markers():
    models = _Models()

    result = _convert(_pdf(_text("A Library lends Books to Members."), _text("A Member has a name.")), models)

    assert result["action"] == "inject_complete_system"
    prompt = models.prompts[0]
    assert "--- Page 1 ---\nA Library lends Books to Members.\n\n--- Page 2 ---\nA Member has a name." in prompt
    assert "doc.pdf', 2 page(s))" in prompt
    assert models.images == []


def test_line_breaks_reach_the_prompt_as_lf_only():
    models = _Models()

    _convert(_pdf(_text("First line of the requirements text.", "Second line of the requirements text.")), models)

    assert "First line of the requirements text.\nSecond line of the requirements text." in models.prompts[0]
    assert "\r" not in models.prompts[0]


def test_only_the_first_20_pages_are_read_but_all_are_counted():
    models = _Models()

    _convert(_pdf(*[_text(f"Text of the page numbered {n} here.") for n in range(1, 26)]), models)

    prompt = models.prompts[0]
    assert "--- Page 20 ---\nText of the page numbered 20 here." in prompt
    assert "--- Page 21 ---" not in prompt
    assert "numbered 21" not in prompt
    assert "25 page(s)" in prompt


def test_a_page_without_text_gets_no_marker():
    models = _Models()

    _convert(_pdf(_text("Text on the first page of this file."), RED_SQUARE, _text("Text on the third page.")), models)

    assert "--- Page 1 ---" in models.prompts[0]
    assert "--- Page 2 ---" not in models.prompts[0]
    assert "--- Page 3 ---" in models.prompts[0]


def test_text_over_30_kb_is_truncated():
    models = _Models()
    line = "All work and no play makes a very long requirements document."
    page = _text(*[line] * 40)

    _convert(_pdf(*[page] * 20), models)

    assert "... [truncated — showing first 30KB]" in models.prompts[0]
    body = models.prompts[0].split("Extracted PDF Content:\n```\n")[1]
    assert body.index("\n\n... [truncated") == 30_000


@pytest.mark.parametrize("body_length, reaches_text_path", [(36, True), (35, False)])
def test_the_text_path_needs_more_than_50_characters(body_length, reaches_text_path):
    # "--- Page 1 ---\n" is 15 of the characters counted.
    models = _Models()

    _convert(_pdf(_text("x" * body_length)), models)

    assert bool(models.prompts) is reaches_text_path
    assert bool(models.images) is not reaches_text_path


# ── Vision path ──────────────────────────────────────────────────────────────


def test_a_scan_sends_its_first_three_pages_as_200_dpi_pngs():
    models = _Models()

    result = _convert(_pdf(*[RED_SQUARE] * 5), models)

    assert result["action"] == "inject_complete_system"
    assert models.prompts == []
    assert models.timeout == 90
    assert len(models.images) == 3
    assert [_png_size(_png(image)) for image in models.images] == [LETTER_AT_200_DPI] * 3


def test_a_rendered_page_keeps_its_colours():
    # A BGR bitmap written out as RGB would turn the red square blue.
    Image = pytest.importorskip("PIL.Image")
    models = _Models()

    _convert(_pdf(RED_SQUARE), models)

    image = Image.open(io.BytesIO(_png(models.images[0]))).convert("RGB")
    assert image.getpixel((550, 1800)) == (255, 0, 0)  # inside the square
    assert image.getpixel((50, 50)) == (255, 255, 255)  # the page around it


def test_a_scan_without_a_vision_model_asks_for_a_key():
    models = _Models()

    result = _convert(_pdf(RED_SQUARE), models, vision=False)

    assert result["action"] == "agent_error"
    assert "API key" in result["message"]
    assert models.prompts == []


def test_a_page_that_fails_to_render_is_skipped_with_a_warning(monkeypatch, caplog):
    pypdfium2 = pytest.importorskip("pypdfium2")
    real_render = pypdfium2.PdfPage.render
    calls = []

    def render(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 2:
            raise RuntimeError("cannot render")
        return real_render(self, *args, **kwargs)

    monkeypatch.setattr(pypdfium2.PdfPage, "render", render)
    models = _Models()

    with caplog.at_level(logging.WARNING):
        result = _convert(_pdf(*[RED_SQUARE] * 3), models)

    assert result["action"] == "inject_complete_system"
    assert len(models.images) == 2
    assert "Failed to render PDF page 2" in caplog.text


def test_no_page_rendering_at_all_is_an_error(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")

    def render(self, *args, **kwargs):
        raise RuntimeError("cannot render")

    monkeypatch.setattr(pypdfium2.PdfPage, "render", render)

    result = _convert(_pdf(RED_SQUARE), _Models())

    assert result["action"] == "agent_error"
    assert "Could not extract content" in result["message"]


# ── Unreadable files ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("content", [b"this is not a PDF at all", b"%PDF-1.4\ntruncated", b"\xff\xfe\x00\x01"])
def test_a_corrupt_file_is_reported_as_unreadable(content):
    models = _Models()

    result = _convert(content, models)

    assert result["action"] == "agent_error"
    assert "corrupted or password-protected" in result["message"]
    assert models.prompts == []


def test_a_password_protected_file_is_reported_as_unreadable():
    models = _Models()

    result = _convert(_pdf(_text("Confidential requirements of the library system."), user_password="secret"), models)

    assert result["action"] == "agent_error"
    assert "corrupted or password-protected" in result["message"]
    assert models.prompts == []


def test_the_encrypted_fixture_really_is_readable_with_its_password():
    # Otherwise the test above would only show that a broken file is rejected.
    pypdfium2 = pytest.importorskip("pypdfium2")
    pdf = _pdf(_text("Confidential requirements of the library system."), user_password="secret")

    doc = pypdfium2.PdfDocument(pdf, password="secret")
    page = doc[0]
    textpage = page.get_textpage()
    text = textpage.get_text_bounded()
    textpage.close()
    page.close()
    doc.close()

    assert "Confidential requirements" in text


def test_a_file_encrypted_with_an_empty_password_is_read():
    # Restriction-only PDFs: viewers open these without asking.
    models = _Models()

    _convert(_pdf(_text("Permissions are restricted but the text is readable."), user_password=""), models)

    assert "Permissions are restricted" in models.prompts[0]


def test_pdfium_refuses_a_file_without_pages_at_open():
    # PyMuPDF opened these and reported "has no pages"; PDFium's error code for
    # them is not reliable (0 or a stale value), so nothing can tell them apart.
    models = _Models()

    result = _convert(_pdf(), models)

    assert result["action"] == "agent_error"
    assert "corrupted or password-protected" in result["message"]


def test_a_document_reporting_no_pages_is_reported(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")
    monkeypatch.setattr(pypdfium2.PdfDocument, "__len__", lambda self: 0)

    result = _convert(_pdf(_text("Anything at all.")), _Models())

    assert result["action"] == "agent_error"
    assert "has no pages" in result["message"]


@pytest.mark.parametrize("vision", [True, False])
def test_pages_counted_but_impossible_to_load_are_an_error_reply(vision, caplog):
    # /Kids points at an object that does not exist: PDFium still counts the page.
    pdf = _pdf(_text("Text nobody will ever read.")).replace(b"/Kids [5 0 R]", b"/Kids [9 0 R]")
    models = _Models()

    with caplog.at_level(logging.WARNING):
        result = _convert(pdf, models, vision=vision)

    assert result["action"] == "agent_error"
    assert models.prompts == []
    assert models.images == []


def test_a_missing_library_names_pypdfium2(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdfium2", None)  # makes the import raise ImportError

    result = _convert(_pdf(_text("Anything at all.")), _Models())

    assert result["action"] == "agent_error"
    assert "pypdfium2" in result["message"]
    assert "pip install pypdfium2" in result["message"]
    assert "PyMuPDF" not in result["message"]


# ── PDFium specifics ─────────────────────────────────────────────────────────


def test_pdfiums_line_end_hyphen_marker_never_reaches_the_prompt():
    pypdfium2 = pytest.importorskip("pypdfium2")
    pdf = _pdf(_text("A sentence that ends with a word that is hy-", "phenated across two lines of the text."))
    doc = pypdfium2.PdfDocument(pdf)
    page = doc[0]
    textpage = page.get_textpage()
    raw = textpage.get_text_bounded()
    textpage.close()
    page.close()
    doc.close()
    assert "\x02" in raw  # what the normalisation exists for
    models = _Models()

    _convert(pdf, models)

    assert "\x02" not in models.prompts[0]
    assert "hy-phenated" in models.prompts[0]


class _Opened:
    """Records every PDFium object a conversion opens, and whether PDFIUM_LOCK
    was held for each call made on one."""

    def __init__(self, monkeypatch):
        self.pypdfium2 = pytest.importorskip("pypdfium2")
        from utilities.pdf import PDFIUM_LOCK

        self.objects = []
        self.calls_outside_lock = []
        self.open_when_parent_closed = []
        self.lock = PDFIUM_LOCK
        for owner, name in (
            (self.pypdfium2.PdfDocument, "__init__"),
            (self.pypdfium2.PdfDocument, "close"),
            (self.pypdfium2.PdfPage, "close"),
            (self.pypdfium2.PdfPage, "get_textpage"),
            (self.pypdfium2.PdfPage, "render"),
            (self.pypdfium2.PdfTextPage, "get_text_bounded"),
            (self.pypdfium2.PdfTextPage, "close"),
            (self.pypdfium2.PdfBitmap, "close"),
        ):
            monkeypatch.setattr(owner, name, self._spy(owner, name, getattr(owner, name)))
        real_get_page = self.pypdfium2.PdfDocument.get_page
        monkeypatch.setattr(self.pypdfium2.PdfDocument, "get_page", self._spy(None, "get_page", real_get_page))

    def _spy(self, owner, name, real):
        def wrapper(obj, *args, **kwargs):
            if not self.lock.locked():
                self.calls_outside_lock.append(name)
            if name == "__init__":
                self.objects.append(obj)
            # Closing a parent also closes its children, which would hide one left open.
            if owner is self.pypdfium2.PdfDocument and name == "close":
                self.open_when_parent_closed += [o for o in self.still_open if o is not obj]
            if owner is self.pypdfium2.PdfPage and name == "close":
                self.open_when_parent_closed += [o for o in self.still_open if getattr(o, "page", None) is obj]
            result = real(obj, *args, **kwargs)
            if name in ("get_page", "get_textpage", "render"):
                self.objects.append(result)
            return result
        return wrapper

    @property
    def still_open(self):
        return [obj for obj in self.objects if getattr(obj, "raw", None)]


@pytest.mark.parametrize("pages", [
    pytest.param([_text("Enough text on this page to take the text path.")], id="text"),
    pytest.param([RED_SQUARE] * 2, id="scan"),
    pytest.param([], id="no-pages"),
])
def test_everything_opened_is_closed_under_the_lock(monkeypatch, pages):
    opened = _Opened(monkeypatch)

    _convert(_pdf(*pages), _Models())

    assert opened.objects
    assert opened.open_when_parent_closed == []
    assert opened.still_open == []
    assert opened.calls_outside_lock == []
    assert not opened.lock.locked()


def test_a_failing_extraction_still_closes_everything(monkeypatch, caplog):
    opened = _Opened(monkeypatch)

    def fail(self, *args, **kwargs):
        raise RuntimeError("cannot extract")

    monkeypatch.setattr(opened.pypdfium2.PdfTextPage, "get_text_bounded", fail)

    with caplog.at_level(logging.WARNING):
        result = _convert(_pdf(_text("Some text.")), _Models(), vision=False)

    assert result["action"] == "agent_error"
    assert "Failed to read PDF page 1" in caplog.text
    assert opened.objects
    assert opened.open_when_parent_closed == []
    assert opened.still_open == []
    assert not opened.lock.locked()


def test_a_file_that_fails_to_open_releases_the_lock():
    from utilities.pdf import PDFIUM_LOCK

    _convert(b"not a PDF", _Models())

    assert not PDFIUM_LOCK.locked()


@pytest.mark.parametrize("content", [
    pytest.param(_text("Enough text on this page to take the text path."), id="text"),
    pytest.param(RED_SQUARE, id="scan"),
])
def test_the_model_is_called_outside_the_lock(content):
    from utilities.pdf import PDFIUM_LOCK

    held_during_call = []
    models = _Models()
    real_predict, real_vision = models.predict, models.vision

    def predict(prompt):
        held_during_call.append(PDFIUM_LOCK.locked())
        return real_predict(prompt)

    def vision(prompt, images, timeout):
        held_during_call.append(PDFIUM_LOCK.locked())
        return real_vision(prompt, images, timeout)

    convert_file_to_diagram_spec(
        file_content_b64=base64.b64encode(_pdf(content)).decode(),
        filename="doc.pdf",
        llm_predict=predict,
        llm_vision_predict=vision,
    )

    assert held_during_call == [False]
