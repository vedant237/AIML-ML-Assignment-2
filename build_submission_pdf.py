"""Build BITS WILP ML Assignment 2 submission PDF (professional layout)."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"
OUT = ROOT / "2026aj05282_ML_Assignment_2.pdf"

# Source screenshots (copy your VM/browser captures here with these names)
SCREENSHOT_SOURCES: list[tuple[str, str]] = [
    ("submission_fig01_terminal_train_streamlit.png", "Virtual Lab terminal: train_models.py output and streamlit run app.py."),
    ("submission_fig02_logistic_regression.png", "Streamlit (localhost): Logistic Regression; test_data.csv (500 rows); metrics, confusion matrix, classification report."),
    ("submission_fig03_decision_tree.png", "Streamlit (localhost): Decision Tree; same test file and dashboard layout."),
    ("submission_fig04_knn.png", "Streamlit (localhost): K-Nearest Neighbor; same test file and dashboard layout."),
    ("submission_fig05_naive_bayes.png", "Streamlit (localhost): Naive Bayes (Gaussian); same test file and dashboard layout."),
    ("submission_fig06_svm.png", "Streamlit (localhost): Support Vector Machine; same test file and dashboard layout."),
    ("submission_fig07_all_models_cloud.png", "Streamlit Cloud: all models comparison table and bar chart on uploaded test data."),
]

# Target long edge (px) for upscale — larger sources embed with more detail at fixed PDF width
SCREENSHOT_MIN_LONG_EDGE = 2560
SCREENSHOT_MAX_LONG_EDGE = 3840

# Evidence layout: §3 intro + Figure 1 on one page (one large image, height capped to page); Figures 2-7 two per page
FIGURE_MAX_HEIGHT_MM = 104.0
# Upper bound (mm) for Figure 1 before page-fit clamping in add_image_fit — use full usable page height
FIGURE_MAX_HEIGHT_INTRO_PAGE_MM = 260.0
FIGURES_PER_PAGE = 2
# Frame around each figure (mm); gap after figure block before next caption
FIGURE_FRAME_PAD_MM = 1.2
FIGURE_GAP_AFTER_MM = 7.0

GITHUB = "https://github.com/vedant237/AIML-ML-Assignment-2"
STREAMLIT = "https://aiml-ml-assignment-2-7zp9qjtttffmzswfxtrdzb.streamlit.app/"
BITS_ID = "2026aj05282"
STUDENT_NAME = "Vedant Chaudhari"

# Lowercase in titles except when first word (Title Case style)
_TITLE_SMALL = frozenset(
    "a an the and or but nor in on at to for of as with by from into through vs per via".split()
)


def title_case_heading(phrase: str) -> str:
    """Title Case for headings: important words capitalized; minor words lower unless first."""
    phrase = sanitize_for_core_font(phrase.strip())
    out: list[str] = []
    for i, word in enumerate(phrase.split()):
        punct_tail = ""
        w = word
        while len(w) > 1 and w[-1] in ".,;:!?)]}":
            punct_tail = w[-1] + punct_tail
            w = w[:-1]
        punct_head = ""
        while len(w) > 1 and w[0] in "([{":
            punct_head += w[0]
            w = w[1:]
        if not w:
            out.append(word)
            continue
        if w.isalpha() and w.isupper() and 2 <= len(w) <= 10:
            core = w
        elif len(w) >= 2 and w[:-1].isalpha() and w[:-1].isupper() and w[-1] == "s":
            core = w
        else:
            lc = w.lower()
            if i > 0 and lc in _TITLE_SMALL:
                core = lc
            else:
                core = w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper()
        out.append(punct_head + core + punct_tail)
    return " ".join(out)


def polish_heading_title(phrase: str) -> str:
    """Apply title case and fix common acronyms for headings."""
    t = title_case_heading(phrase)
    t = re.sub(r"\bBits\b", "BITS", t)
    t = re.sub(r"\bAiml\b", "AIML", t)
    t = re.sub(r"\bGithub\b", "GitHub", t)
    t = re.sub(r"\bStreamlit\b", "Streamlit", t)
    t = re.sub(r"\bReadme\b", "README", t)
    return t


def toc_topic_display(phrase: str) -> str:
    """Table of contents topic: title polish, but README section letters stay lowercase in parentheses."""
    t = polish_heading_title(phrase)
    return re.sub(r"\(([A-Za-z])\)", lambda m: f"({m.group(1).lower()})", t)


_OBSERVATION_BOLD_TERMS = (
    "Confusion Matrix",
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes",
    "Support Vector Machine",
    "Streamlit",
    "random_state=42",
    "random_state",
    "Deployment",
    "Recommended Model",
    "MCC",
    "AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "hold-out",
    "Hold-out",
    "kNN",
    "SVM",
)


def _inside_markdown_bold(text: str, idx: int) -> bool:
    """True if idx lies inside a ** ... ** span (naive scan from start)."""
    depth = 0
    i = 0
    while i < idx and i + 1 < len(text):
        if text[i : i + 2] == "**":
            depth = 1 - depth
            i += 2
            continue
        i += 1
    return depth == 1


def _bold_phrase_first_occurrence(text: str, phrase: str) -> str:
    """Wrap first occurrence of phrase (case-insensitive, word-ish) if not already bold."""
    if not phrase.strip():
        return text
    low = text.lower()
    plow = phrase.lower()
    pos = 0
    while True:
        j = low.find(plow, pos)
        if j < 0:
            return text
        before = j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_")
        after = j + len(phrase) < len(text) and (
            text[j + len(phrase)].isalnum() or text[j + len(phrase)] in "_="
        )
        if before or after:
            pos = j + 1
            continue
        if _inside_markdown_bold(text, j):
            pos = j + 1
            continue
        chunk = text[j : j + len(phrase)]
        return text[:j] + f"**{chunk}**" + text[j + len(phrase) :]


def observation_text_to_markdown(obs: str) -> str:
    """Add **bold** for first sentence, decimals, and key terms (fpdf2 markdown)."""
    s = sanitize_for_core_font(obs.strip())
    if not s:
        return s
    out = s
    first_dot = out.find(". ")
    if 8 < first_dot < 250:
        first = out[: first_dot + 1].strip()
        rest = out[first_dot + 1 :].strip()
        out = f"**{first}** {rest}"
    out = re.sub(r"\b(\d+\.\d{3,4})\b", r"**\1**", out)
    for phrase in sorted(_OBSERVATION_BOLD_TERMS, key=len, reverse=True):
        out = _bold_phrase_first_occurrence(out, phrase)
    return out


def sanitize_for_core_font(text: str) -> str:
    """Helvetica core fonts are Latin-1; normalize common Unicode from README."""
    repl = (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2011", "-"),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u00b7", " / "),
        ("\u2212", "-"),
    )
    for a, b in repl:
        text = text.replace(a, b)
    return text


def strip_markdown_fences_and_headers(body: str) -> str:
    body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"```[\s\S]*?```", "[ code block omitted in PDF - see repository ]", body)
    return body.strip()


def extract_readme_abcd(text: str) -> tuple[str, str, str, str]:
    """Pull sections a-d from README.md (stop before first Appendix)."""

    def between(start: str, end: str) -> str:
        i = text.find(start)
        if i < 0:
            return ""
        nl = text.find("\n", i)
        start_content = nl + 1 if nl >= 0 else i
        j = text.find(end, start_content)
        if j < 0:
            j = len(text)
        return strip_markdown_fences_and_headers(text[start_content:j])

    a = between("## a. Problem statement", "## b.")
    b = between("## b. Dataset description", "## c.")
    c = between("## c.", "## d.")
    d = between("## d. Models used", "## Appendix")
    return a, b, c, d


def extract_metrics_table_rows(d_raw: str) -> list[list[str]] | None:
    """Parse GitHub markdown metrics table into rows including header."""
    m = re.search(
        r"\| ML Model Name \|[^\n]+\n\|[-\s|]+\|\n([\s\S]*?)(?=\n\nMetrics:|\Z)",
        d_raw,
    )
    if not m:
        return None
    rows: list[list[str]] = []
    header = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |"
    rows.append([c.strip() for c in header.strip("|").split("|")])
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 7:
            rows.append(cells[:7])
    return rows if len(rows) > 1 else None


def extract_observation_rows(d_raw: str) -> list[tuple[str, str]]:
    """Parse README observation table into (model label, observation) rows."""
    if "### Observations" not in d_raw:
        return []
    _pre, rest = d_raw.split("### Observations", 1)
    rows: list[tuple[str, str]] = []
    for line in rest.strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        model, obs = cells[0], cells[1]
        if model.lower() == "ml model name":
            continue
        rows.append((sanitize_for_core_font(model), sanitize_for_core_font(obs)))
    return rows


def polish_section_d(d_raw: str) -> str:
    """Format section d: inline code plain; drop comparison + observation tables (rendered separately)."""
    text = re.sub(r"`([^`]+)`", r"\1", d_raw)

    if "### Comparison table" in text and "\n\nMetrics:" in text:
        pre, rest = text.split("### Comparison table", 1)
        _table_blob, post = rest.split("\n\nMetrics:", 1)
        text = pre.rstrip() + "\n\nMetrics:" + post

    if "### Observations" in text:
        pre, _ = text.split("### Observations", 1)
        text = pre.rstrip()

    return text.strip()


def _clean_markdown_cell(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"\1", s.strip())
    link = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", s)
    if link:
        label = sanitize_for_core_font(link.group(1))
        return f"{label} ({link.group(2)})"
    return sanitize_for_core_font(s)


def extract_dataset_for_pdf(b_raw: str) -> tuple[str, list[tuple[str, str]], str]:
    """
    Split section (b) into intro text, key-value rows for a table, and trailing note.
    """
    if "| Item |" not in b_raw or "| Description |" not in b_raw:
        return b_raw.strip(), [], ""

    parts = b_raw.split("| Item |", 1)
    intro = parts[0].strip()
    rest = "| Item |" + parts[1]
    m = re.search(
        r"\| Item \| Description \|\s*\n\|[-\s:|]+\|\s*\n([\s\S]*?)(?=\n\n|\Z)",
        rest,
    )
    if not m:
        return b_raw.strip(), [], ""

    body_rows: list[tuple[str, str]] = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            body_rows.append((_clean_markdown_cell(cells[0]), _clean_markdown_cell(cells[1])))

    tail = rest[m.end() :].strip()
    return intro, body_rows, tail


class SubmissionPDF(FPDF):
    def at_margin(self) -> None:
        """Reset x to left content edge (avoids overflow after line() etc.)."""
        self.set_x(self.l_margin)

    def header(self) -> None:
        if self.page_no() <= 1:
            return
        self.at_margin()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(71, 85, 105)
        right = sanitize_for_core_font(f"{STUDENT_NAME}  |  BITS ID {BITS_ID}")
        self.cell(self.epw, 5, right, align="R")
        self.ln(1)
        self.at_margin()
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.35)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(9)

    def footer(self) -> None:
        self.set_y(-11)
        self.set_draw_color(230, 234, 242)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="C")

    def cover_block(self) -> None:
        """Title page: centered hierarchy, modern table of contents card, bottom accent bar."""
        # Subtle top accent
        self.set_fill_color(248, 250, 252)
        self.rect(0, 0, self.w, 26, style="F")

        y0 = 38.0
        self.set_y(y0)
        self.set_font("Helvetica", "B", 12.5)
        self.set_text_color(37, 99, 235)
        self.cell(self.epw, 7, polish_heading_title("Birla Institute of Technology and Science, Pilani"), align="C")
        self.ln(7)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(71, 85, 105)
        self.cell(
            self.epw,
            6.5,
            polish_heading_title("Work Integrated Learning Programme (WILP)"),
            align="C",
        )
        self.ln(10)
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.45)
        self.line(self.l_margin + 28, self.get_y(), self.w - self.r_margin - 28, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.multi_cell(self.epw, 9, polish_heading_title("Machine Learning - Assignment 2 (AIML)"), align="C")
        self.ln(3)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(51, 65, 85)
        self.multi_cell(
            self.epw,
            6.5,
            polish_heading_title("Mobile Price Classification - Multi-Class Models - Streamlit Deployment"),
            align="C",
        )
        self.ln(12)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(30, 41, 59)
        self.cell(self.epw, 7, f"{polish_heading_title('Student Name')}: {STUDENT_NAME}", align="C")
        self.ln(5)
        self.cell(self.epw, 7, f"{polish_heading_title('BITS Pilani Student ID')}: {BITS_ID}", align="C")
        self.ln(12)

        # Table of contents — card with header strip and zebra rows
        toc_title = polish_heading_title("Table of Contents")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 58, 138)
        self.cell(self.epw, 7, toc_title, align="C")
        self.ln(12)

        col_sec = 14.0
        col_pg = 18.0
        toc_side_pad = 4.5
        inner_w_tbl = self.epw - 2 * toc_side_pad
        col_topic = inner_w_tbl - col_sec - col_pg
        pad = 1.8
        lh = 4.2
        toc_rows: list[tuple[str, str, str]] = [
            ("1-2", "GitHub Repository and Live Streamlit Application", "2"),
            ("3", "Evidence Screenshots (Virtual Lab and Application)", "3-6"),
            ("4", "README Documentation (a) through (d)", "7+"),
        ]
        row_heights: list[float] = []
        for _sec, topic, _pg in toc_rows:
            nlines = self._lines_for_cell(toc_topic_display(topic), col_topic - 2 * pad, 8.2)
            row_heights.append(max(8.5, lh * float(nlines) + 2 * pad))
        header_row_h = 8.0
        card_vpad = 3.0
        card_inner_h = header_row_h + sum(row_heights) + card_vpad
        card_outer_pad = 5.0
        y_card = self.get_y()
        self.set_draw_color(148, 163, 184)
        self.set_fill_color(255, 255, 255)
        self.set_line_width(0.4)
        self.rect(
            self.l_margin,
            y_card,
            self.epw,
            card_inner_h + 2 * card_outer_pad,
            style="DF",
        )
        yh = y_card + card_outer_pad + card_vpad / 2
        inner_x = self.l_margin + toc_side_pad
        self.set_xy(inner_x, yh)
        self.set_fill_color(30, 58, 138)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8.5)
        self.set_draw_color(30, 58, 138)
        self.cell(col_sec, header_row_h, polish_heading_title("Sec."), border=1, align="C", fill=True)
        self.cell(col_topic, header_row_h, polish_heading_title("Topic"), border=1, align="C", fill=True)
        self.cell(col_pg, header_row_h, polish_heading_title("Pages"), border=1, align="C", fill=True)
        self.set_y(yh + header_row_h)
        self.set_font("Helvetica", "", 8.2)
        for ri, ((_sec, topic, pg), row_h) in enumerate(zip(toc_rows, row_heights)):
            y0r = self.get_y()
            bg = (248, 250, 252) if ri % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_text_color(30, 41, 59)
            self.set_draw_color(226, 232, 240)
            self.rect(inner_x, y0r, col_sec, row_h, style="FD")
            self.rect(inner_x + col_sec, y0r, col_topic, row_h, style="FD")
            self.rect(inner_x + col_sec + col_topic, y0r, col_pg, row_h, style="FD")
            self.set_font("Helvetica", "B", 8.2)
            self.set_xy(inner_x + pad, y0r + pad)
            self.multi_cell(col_sec - 2 * pad, lh, _sec, border=0, align="C")
            self.set_font("Helvetica", "B", 8.2)
            self.set_xy(inner_x + col_sec + pad, y0r + pad)
            self.multi_cell(col_topic - 2 * pad, lh, toc_topic_display(topic), border=0, align="L")
            self.set_font("Helvetica", "B", 8.0)
            self.set_text_color(71, 85, 105)
            self.set_xy(inner_x + col_sec + col_topic + pad, y0r + pad)
            self.multi_cell(col_pg - 2 * pad, lh, pg, border=0, align="C")
            self.set_y(y0r + row_h)
        self.ln(10)

        # Bottom brand bar
        bar_y = self.h - self.b_margin - 14
        self.set_fill_color(241, 245, 249)
        self.rect(0, bar_y, self.w, 14, style="F")
        self.set_y(bar_y + 4)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.set_x(0)
        self.cell(self.w, 5, sanitize_for_core_font("BITS Pilani WILP / AIML / Course Submission Document"), align="C")

    def section_heading(self, num: str, title: str) -> None:
        self.at_margin()
        self.ln(6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 58, 138)
        self.multi_cell(self.epw, 8, f"{num} {polish_heading_title(title)}")
        self.at_margin()
        y_rule = self.get_y()
        self.set_draw_color(191, 208, 232)
        self.set_line_width(0.35)
        self.line(self.l_margin, y_rule, self.w - self.r_margin, y_rule)
        self.set_line_width(0.2)
        self.at_margin()
        self.ln(3)

    def figure_caption_block_height_mm(self, figure_num: int, short_description: str) -> float:
        """Height (mm) consumed by figure_caption() including underline and trailing ln — no drawing."""
        cap = f"Figure {figure_num}. {sanitize_for_core_font(short_description)}"
        nlines = self._lines_for_cell(cap, self.epw, 8.5)
        line_h = 3.6
        after_underline_ln = 1.6
        return nlines * line_h + after_underline_ln

    def figure_caption(self, figure_num: int, short_description: str) -> None:
        """Consistent caption above each screenshot."""
        self.at_margin()
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(51, 65, 85)
        cap = f"Figure {figure_num}. {sanitize_for_core_font(short_description)}"
        self.multi_cell(self.epw, 3.6, cap)
        self.at_margin()
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.15)
        yln = self.get_y()
        self.line(self.l_margin, yln, self.l_margin + min(55, self.epw * 0.28), yln)
        self.ln(1.6)

    def body_paragraph(self, text: str, size: int = 10, leading: float = 5.2) -> None:
        self.at_margin()
        self.set_font("Helvetica", "", size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.epw, leading, sanitize_for_core_font(text))

    def link_resource_card(self, title: str, url: str) -> None:
        """Wide link card: accent strip, title, underlined URL (no badge or hint lines)."""
        self.at_margin()
        card_h = 22.0
        accent_w = 3.8
        x0 = self.l_margin
        y0 = self.get_y()
        self.set_fill_color(239, 246, 255)
        self.set_draw_color(147, 197, 253)
        self.set_line_width(0.35)
        self.rect(x0, y0, self.epw, card_h, style="DF")
        self.set_fill_color(37, 99, 235)
        self.rect(x0, y0, accent_w, card_h, style="F")

        inner_left = x0 + accent_w + 5.5
        tx_w = self.epw - accent_w - 11.0
        self.set_xy(inner_left, y0 + 5.5)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(15, 23, 42)
        self.multi_cell(tx_w, 6.0, polish_heading_title(title), align="L")
        self.set_x(inner_left)
        self.set_font("Helvetica", "U", 9.0)
        self.set_text_color(29, 78, 216)
        self.multi_cell(tx_w, 5.0, url, link=url, align="L")
        self.set_text_color(30, 30, 30)
        self.set_y(y0 + card_h + 8.0)

    def readme_heading(self, letter: str, subtitle: str) -> None:
        self.at_margin()
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(15, 23, 42)
        self.multi_cell(self.epw, 7, f"Section ({letter}). {polish_heading_title(subtitle)}")
        self.at_margin()
        self.ln(2)

    @staticmethod
    def _lines_for_cell(text: str, width_mm: float, font_size_pt: float) -> int:
        """Approximate wrapped line count for proportional font (Helvetica)."""
        if not text.strip():
            return 1
        chars_per_line = max(8, int(width_mm / (font_size_pt * 0.42)))
        total = 0
        for para in text.splitlines():
            if not para.strip():
                total += 1
                continue
            total += max(1, (len(para) + chars_per_line - 1) // chars_per_line)
        return max(1, total)

    def render_dataset_table(self, rows: list[tuple[str, str]]) -> None:
        """Two-column key-value table for dataset specification."""
        if not rows:
            return
        col1_w = 52.0
        col2_w = self.epw - col1_w
        pad = 1.5
        lh = 3.6
        self.at_margin()
        self.ln(2)
        yh = self.get_y()
        self.set_draw_color(30, 58, 138)
        self.set_fill_color(30, 58, 138)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(self.l_margin, yh)
        self.cell(col1_w, 8.5, polish_heading_title("Item"), border=1, align="C", fill=True)
        self.cell(col2_w, 8.5, polish_heading_title("Description"), border=1, align="C", fill=True)
        self.set_y(yh + 8.5)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(40, 40, 40)
        for ri, (key, val) in enumerate(rows):
            key_lines = self._lines_for_cell(key, col1_w - 2 * pad, 8.5)
            val_lines = self._lines_for_cell(val, col2_w - 2 * pad, 8.5)
            row_h = max(8.0, lh * max(key_lines, val_lines) + 2 * pad)
            y0 = self.get_y()
            bg = (248, 250, 252) if ri % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.rect(self.l_margin, y0, col1_w, row_h, style="F")
            self.rect(self.l_margin + col1_w, y0, col2_w, row_h, style="F")
            self.set_draw_color(186, 198, 218)
            self.set_line_width(0.25)
            self.rect(self.l_margin, y0, col1_w, row_h, style="D")
            self.rect(self.l_margin + col1_w, y0, col2_w, row_h, style="D")
            self.set_line_width(0.2)
            self.set_font("Helvetica", "B", 8.5)
            self.set_xy(self.l_margin + pad, y0 + pad)
            self.multi_cell(col1_w - 2 * pad, lh, key, border=0, align="L")
            self.set_font("Helvetica", "", 8.5)
            self.set_xy(self.l_margin + col1_w + pad, y0 + pad)
            self.multi_cell(col2_w - 2 * pad, lh, val, border=0, align="L")
            self.set_y(y0 + row_h)
        self.ln(4)
        self.at_margin()

    def section_c_links(self) -> None:
        """Explicit clickable links for section (c)."""
        self.at_margin()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.epw, 6, polish_heading_title("GitHub repository") + ":")
        self.at_margin()
        self.set_font("Helvetica", "U", 9)
        self.set_text_color(37, 99, 235)
        self.multi_cell(self.epw, 5, GITHUB, link=GITHUB)
        self.set_text_color(30, 30, 30)
        self.at_margin()
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(self.epw, 6, polish_heading_title("Live Streamlit application") + ":")
        self.at_margin()
        self.set_font("Helvetica", "U", 9)
        self.set_text_color(37, 99, 235)
        self.multi_cell(self.epw, 5, STREAMLIT, link=STREAMLIT)
        self.set_text_color(30, 30, 30)
        self.at_margin()
        self.ln(3)

    def render_metrics_table(self, rows: list[list[str]]) -> None:
        """Seven-column model comparison table (bordered grid)."""
        if not rows or len(rows) < 2:
            return
        hdr = ["Model", "Accuracy", "AUC", "Prec.", "Recall", "F1", "MCC"]
        rest = (self.epw - 42) / 6
        col_w = [42.0] + [rest] * 6
        self.at_margin()
        self.ln(2)
        h_head = 7.0
        y_head = self.get_y()
        x = self.l_margin

        self.set_draw_color(30, 58, 138)
        self.set_fill_color(30, 58, 138)
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(255, 255, 255)
        for i, lab in enumerate(hdr):
            self.set_xy(x, y_head)
            self.cell(col_w[i], h_head, sanitize_for_core_font(lab), border=1, align="C", fill=True)
            x += col_w[i]
        self.set_y(y_head + h_head)

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(40, 40, 40)
        for rj, row in enumerate(rows[1:], start=1):
            h_row = 8.0 if row and len(str(row[0])) > 22 else 7.0
            y_row = self.get_y()
            x = self.l_margin
            is_lr = row and "logistic" in str(row[0]).lower()
            zebra = (255, 251, 235) if is_lr else ((241, 245, 249) if rj % 2 == 1 else (255, 255, 255))
            for i in range(7):
                txt = sanitize_for_core_font(str(row[i])) if i < len(row) else ""
                align = "L" if i == 0 else "C"
                if len(txt) > 30 and i == 0:
                    txt = txt[:28] + ".."
                self.set_fill_color(*zebra)
                self.set_draw_color(186, 198, 218)
                self.set_line_width(0.25)
                self.set_xy(x, y_row)
                self.cell(col_w[i], h_row, txt, border=1, align=align, fill=True)
                self.set_line_width(0.2)
                x += col_w[i]
            self.set_y(y_row + h_row)

        self.ln(4)
        self.at_margin()

    def render_observations_table(self, rows: list[tuple[str, str]]) -> None:
        """Two-column table: model and qualitative observation (professional layout)."""
        if not rows:
            return
        col_model = 38.0
        col_obs = self.epw - col_model
        pad = 1.4
        lh = 3.4
        self.at_margin()
        self.ln(2)
        yh = self.get_y()
        self.set_draw_color(30, 58, 138)
        self.set_fill_color(30, 58, 138)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(255, 255, 255)
        self.set_xy(self.l_margin, yh)
        self.cell(col_model, 8, polish_heading_title("Model"), border=1, align="C", fill=True)
        self.cell(col_obs, 8, polish_heading_title("Observation"), border=1, align="C", fill=True)
        self.set_y(yh + 8)
        self.set_font("Helvetica", "", 8.0)
        self.set_text_color(40, 40, 40)
        for rk, (model, obs) in enumerate(rows):
            y0 = self.get_y()
            ml = self._lines_for_cell(model, col_model - 2 * pad, 8.0)
            ol = self._lines_for_cell(obs, col_obs - 2 * pad, 8.0)
            row_h = max(9.0, lh * max(ml, ol) + 2 * pad)
            bg = (248, 250, 252) if rk % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.rect(self.l_margin, y0, col_model, row_h, style="F")
            self.rect(self.l_margin + col_model, y0, col_obs, row_h, style="F")
            self.set_draw_color(186, 198, 218)
            self.set_line_width(0.25)
            self.rect(self.l_margin, y0, col_model, row_h, style="D")
            self.rect(self.l_margin + col_model, y0, col_obs, row_h, style="D")
            self.set_line_width(0.2)
            self.set_font("Helvetica", "B", 8.0)
            self.set_xy(self.l_margin + pad, y0 + pad)
            self.multi_cell(col_model - 2 * pad, lh, model, border=0, align="L")
            self.set_font("Helvetica", "", 8.0)
            self.set_xy(self.l_margin + col_model + pad, y0 + pad)
            self.multi_cell(
                col_obs - 2 * pad,
                lh,
                observation_text_to_markdown(obs),
                border=0,
                align="L",
                markdown=True,
            )
            self.set_y(y0 + row_h)
        self.ln(3)
        self.at_margin()


def hd_path_for(src: Path) -> Path:
    return src.parent / (src.stem + "_hd.png")


def enhance_screenshot(src: Path, dst: Path) -> Path:
    """
    Upscale (up to 4K-class long edge) and lightly sharpen for crisp PDF embedding.
    True 4K is only possible if the source PNG has enough pixels; upscaling improves
    sharpness when the embedded width is fixed in mm.
    """
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        target = min(SCREENSHOT_MAX_LONG_EDGE, max(SCREENSHOT_MIN_LONG_EDGE, long_edge))
        if long_edge < target:
            scale = target / float(long_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        im = ImageEnhance.Contrast(im).enhance(1.04)
        im = ImageEnhance.Sharpness(im).enhance(1.12)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=2))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG", optimize=True, compress_level=3)
    return dst


def _fitted_image_dims(
    pdf: FPDF,
    path: Path,
    max_h_mm: float,
    y_top_mm: float,
) -> tuple[float, float] | None:
    """Scaled (w_mm, h_mm) for image drawn with top at y_top_mm; respects content width and page bottom."""
    with Image.open(path) as im:
        iw_px, ih_px = im.size
    if not iw_px or not ih_px:
        return None
    w_avail = pdf.epw
    h_from_w = w_avail * (ih_px / iw_px)
    if h_from_w > max_h_mm:
        h = max_h_mm
        w = h * (iw_px / ih_px)
    else:
        w, h = w_avail, h_from_w

    pad = FIGURE_FRAME_PAD_MM
    bottom_limit = pdf.h - pdf.b_margin
    max_h_page = bottom_limit - y_top_mm - pad - FIGURE_GAP_AFTER_MM
    if h > max_h_page and max_h_page > 8:
        h = max_h_page
        w = h * (iw_px / ih_px)
        if w > w_avail:
            w = w_avail
            h = w_avail * (ih_px / iw_px)
    return (w, h)


def place_fitted_image(pdf: FPDF, path: Path, w: float, h: float) -> None:
    """Draw image at current y with given size, frame, advance y past block."""
    pad = FIGURE_FRAME_PAD_MM
    x = pdf.l_margin + (pdf.epw - w) / 2
    y = pdf.get_y()
    pdf.image(str(path), x=x, y=y, w=w, h=h)
    pdf.set_draw_color(59, 91, 130)
    pdf.set_line_width(0.55)
    pdf.rect(x - pad, y - pad, w + 2 * pad, h + 2 * pad, style="D")
    pdf.set_line_width(0.2)
    pdf.set_y(y + h + pad + FIGURE_GAP_AFTER_MM)


def add_image_fit(
    pdf: FPDF,
    path: Path,
    max_h_mm: float = FIGURE_MAX_HEIGHT_MM,
) -> None:
    """Insert image scaled to content width; cap height to page bottom below current y."""
    y_top = pdf.get_y()
    dims = _fitted_image_dims(pdf, path, max_h_mm, y_top)
    if dims is None:
        return
    w, h = dims
    place_fitted_image(pdf, path, w, h)


def main() -> None:
    if not README_PATH.is_file():
        raise SystemExit(f"Missing required file: {README_PATH}")

    fig_entries: list[tuple[Path, str]] = []
    for fname, desc in SCREENSHOT_SOURCES:
        p = ROOT / fname
        if not p.is_file():
            raise SystemExit(f"Missing screenshot: {p} (expected files: {[x[0] for x in SCREENSHOT_SOURCES]})")
        fig_entries.append((p, desc))

    for src, _ in fig_entries:
        enhance_screenshot(src, hd_path_for(src))

    readme_full = README_PATH.read_text(encoding="utf-8")
    raw_a, raw_b, raw_c, raw_d = extract_readme_abcd(readme_full)
    if not all([raw_a, raw_b, raw_c, raw_d]):
        raise SystemExit("Could not parse README.md sections a-d. Check headings.")

    intro_b, dataset_rows, tail_b = extract_dataset_for_pdf(raw_b)
    sec_a = raw_a
    metrics_rows = extract_metrics_table_rows(raw_d)
    observation_rows = extract_observation_rows(raw_d)
    sec_d = polish_section_d(raw_d)

    pdf = SubmissionPDF()
    pdf.alias_nb_pages()
    pdf.set_margins(16, 22, 16)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    pdf.cover_block()

    pdf.add_page()
    pdf.section_heading("1.", "GitHub Repository")
    pdf.link_resource_card("GitHub Repository", GITHUB)

    pdf.section_heading("2.", "Live Streamlit Application")
    pdf.link_resource_card("Hosted Streamlit Application", STREAMLIT)

    pdf.add_page()
    pdf.section_heading("3.", "Evidence Screenshots (Virtual Lab and Application)")
    pdf.body_paragraph(
        "Figures 1-7 show the Virtual Lab workflow, five single-model Streamlit evaluations on "
        "test_data.csv (500 records), and the Streamlit Cloud multi-model summary. They document "
        "command usage, UI configuration, and comparative results in one submission packet.",
        size=10,
        leading=4.8,
    )

    idx = 1
    first_src, first_desc = fig_entries[0]
    y_after_intro = pdf.get_y()
    bottom_limit = pdf.h - pdf.b_margin
    pad = FIGURE_FRAME_PAD_MM
    cap_h = pdf.figure_caption_block_height_mm(idx, first_desc)
    y_cap = y_after_intro
    w, h = 1.0, 1.0
    for _ in range(8):
        y_img_top = y_cap + cap_h
        dims = _fitted_image_dims(
            pdf,
            hd_path_for(first_src),
            FIGURE_MAX_HEIGHT_INTRO_PAGE_MM,
            y_img_top,
        )
        if dims is None:
            break
        w, h = dims
        img_bottom = y_img_top + h + pad + FIGURE_GAP_AFTER_MM
        slack = bottom_limit - img_bottom
        y_cap = y_after_intro + max(0.0, slack / 2)
    y_img_top = y_cap + cap_h
    dims_final = _fitted_image_dims(
        pdf,
        hd_path_for(first_src),
        FIGURE_MAX_HEIGHT_INTRO_PAGE_MM,
        y_img_top,
    )
    if dims_final is not None:
        w, h = dims_final
    pdf.set_xy(pdf.l_margin, y_cap)
    pdf.figure_caption(idx, first_desc)
    place_fitted_image(pdf, hd_path_for(first_src), w, h)
    idx += 1

    remaining = fig_entries[1:]
    for batch_start in range(0, len(remaining), FIGURES_PER_PAGE):
        pdf.add_page()
        chunk = remaining[batch_start : batch_start + FIGURES_PER_PAGE]
        for src, desc in chunk:
            pdf.figure_caption(idx, desc)
            add_image_fit(pdf, hd_path_for(src), max_h_mm=FIGURE_MAX_HEIGHT_MM)
            idx += 1

    pdf.add_page()
    pdf.section_heading("4.", "README Documentation (sections a through d)")
    pdf.body_paragraph(
        "Sections (a) through (d) below restate the course README committed to the repository: "
        "problem statement, dataset specification, repository and application links, and model comparison.",
        size=10,
        leading=5.0,
    )
    pdf.ln(1)

    pdf.readme_heading("a", "Problem Statement")
    pdf.body_paragraph(sec_a, size=10, leading=5.4)

    pdf.readme_heading("b", "Dataset Description")
    if dataset_rows:
        if intro_b.strip():
            pdf.body_paragraph(intro_b, size=10, leading=5.4)
        pdf.render_dataset_table(dataset_rows)
        if tail_b.strip():
            pdf.body_paragraph(tail_b, size=10, leading=5.4)
    else:
        pdf.body_paragraph(raw_b, size=10, leading=5.4)

    pdf.readme_heading("c", "Repository and Application Links")
    pdf.section_c_links()

    pdf.add_page()
    pdf.readme_heading("d", "Models Used and Evaluation")
    if metrics_rows and "\n\nMetrics:" in sec_d:
        before_metrics, after_metrics = sec_d.split("\n\nMetrics:", 1)
        pdf.body_paragraph(before_metrics, size=10, leading=5.4)
        pdf.render_metrics_table(metrics_rows)
        after_stripped = after_metrics.strip()
        if after_stripped.lower().startswith("metrics:"):
            after_stripped = after_stripped[8:].strip()
        if after_stripped:
            pdf.at_margin()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(pdf.epw, 6, polish_heading_title("Metrics note"))
            pdf.body_paragraph(after_stripped, size=10, leading=5.2)
        if observation_rows:
            pdf.add_page()
            pdf.at_margin()
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(pdf.epw, 7, polish_heading_title("Observations"))
            pdf.at_margin()
            pdf.ln(1)
            pdf.render_observations_table(observation_rows)
    else:
        pdf.body_paragraph(sec_d, size=10, leading=5.4)

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
