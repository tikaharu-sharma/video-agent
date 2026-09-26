import re

from fpdf import FPDF


def safe_filename(title: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", title).strip() or "meeting"

# fpdf2's core fonts (Helvetica) are latin-1 only; Claude's output routinely
# uses smart punctuation/bullets that aren't in that charset.
_PDF_CHAR_MAP = {
    "–": "-", "—": "--", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "-", "…": "...", "→": "->",
}


def _pdf_safe(text: str) -> str:
    for char, replacement in _PDF_CHAR_MAP.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def _sections(result: dict) -> list:
    return [
        ("Summary", result["summary"]),
        ("Action Items", result["action_items"]),
        ("Key Decisions", result["key_decisions"]),
    ]


def build_txt(result: dict) -> bytes:
    lines = [result["title"], "=" * len(result["title"])]
    for heading, body in _sections(result):
        lines += ["", heading, "-" * len(heading), body]
    return "\n".join(lines).encode("utf-8")


def build_pdf(result: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _pdf_safe(result["title"]))

    for heading, body in _sections(result):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading)
        # multi_cell() leaves x at the end of its last line, not back at the
        # left margin -- without this, the next multi_cell() starts with ~0
        # width left and raises FPDFException.
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _pdf_safe(body))

    return bytes(pdf.output())
