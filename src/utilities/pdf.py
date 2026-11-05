"""What the code reading PDFs shares. pypdfium2 itself stays a lazy import there."""

import threading

# PDFium is not thread-safe, even across documents: concurrent calls crash the
# process. Hold this from opening a document until it is closed.
PDFIUM_LOCK = threading.Lock()


def page_text(page) -> str:
    """A page's text, with PDFium's CRLF and line-end hyphen marker normalised."""
    textpage = page.get_textpage()
    try:
        return textpage.get_text_bounded().replace("\r\n", "\n").replace("\x02", "-")
    finally:
        textpage.close()
