"""Zero-dependency standards-compliant PDF 1.4 multi-page report builder (`core/evolution/pdf_writer.py`)."""

from __future__ import annotations

import textwrap

PAGE_W = 612.0  # US Letter width (pt)
PAGE_H = 792.0  # US Letter height (pt)
MARGIN_X = 44.0
MARGIN_TOP = 48.0
MARGIN_BOTTOM = 48.0
CONTENT_W = PAGE_W - 2.0 * MARGIN_X

_ASCII_REPLACEMENTS = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "·": "-",
    "•": "*",
    "Δ": "Delta ",
    "≥": ">=",
    "≤": "<=",
    "±": "+/-",
    "—": "--",
    "–": "-",
    "“": '"',
    "”": '"',
    "’": "'",
}


def _pdf_escape(text: str) -> str:
    cleaned = text
    for k, v in _ASCII_REPLACEMENTS.items():
        cleaned = cleaned.replace(k, v)
    safe_chars: list[str] = []
    for ch in cleaned:
        code = ord(ch)
        if ch == "\\":
            safe_chars.append("\\\\")
        elif ch == "(":
            safe_chars.append("\\(")
        elif ch == ")":
            safe_chars.append("\\)")
        elif 32 <= code <= 126:
            safe_chars.append(ch)
        elif 160 <= code <= 255:
            safe_chars.append(f"\\{code:03o}")
        else:
            safe_chars.append("?")
    return "".join(safe_chars)


class MinimalPDFBuilder:
    """Builds a valid multi-page `%PDF-1.4` document with text, tables, and vector bar charts."""

    def __init__(self, title: str = "NEURAL DIFF Evolution Report") -> None:
        self.title = title
        self._pages: list[list[str]] = [[]]
        self._y = PAGE_H - MARGIN_TOP

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def _stream(self) -> list[str]:
        return self._pages[-1]

    def _ensure_space(self, needed_pt: float) -> None:
        if self._y - needed_pt < MARGIN_BOTTOM:
            self._pages.append([])
            self._y = PAGE_H - MARGIN_TOP

    def add_banner(self, title: str, subtitle: str) -> None:
        self._ensure_space(58.0)
        s = self._stream()
        box_y = self._y - 46.0
        s.append(f"0.08 0.08 0.10 rg {MARGIN_X:.1f} {box_y:.1f} {CONTENT_W:.1f} 46.0 re f")
        s.append(
            f"BT /F2 14 Tf 0.96 0.96 0.97 rg {MARGIN_X + 12:.1f} {self._y - 20:.1f} Td ({_pdf_escape(title)}) Tj ET"
        )
        s.append(
            f"BT /F1 8.5 Tf 0.72 0.72 0.76 rg {MARGIN_X + 12:.1f} {self._y - 35:.1f} Td ({_pdf_escape(subtitle)}) Tj ET"
        )
        self._y -= 58.0

    def add_section_heading(self, heading: str) -> None:
        self._ensure_space(28.0)
        self._y -= 6.0
        s = self._stream()
        s.append(
            f"BT /F2 11 Tf 0.10 0.10 0.12 rg {MARGIN_X:.1f} {self._y - 10:.1f} Td ({_pdf_escape(heading)}) Tj ET"
        )
        line_y = self._y - 14.0
        s.append(
            f"0.78 0.78 0.82 RG 0.6 w {MARGIN_X:.1f} {line_y:.1f} m {MARGIN_X + CONTENT_W:.1f} {line_y:.1f} l S"
        )
        self._y -= 22.0

    def add_paragraph(self, text: str, *, bold: bool = False, mono: bool = False, font_size: float = 9.0) -> None:
        font = "/F3" if mono else ("/F2" if bold else "/F1")
        wrap_width = int(CONTENT_W / (font_size * (0.61 if mono else 0.50)))
        lines = textwrap.wrap(text, width=max(40, wrap_width)) or [""]
        line_h = font_size + 3.5
        for ln in lines:
            self._ensure_space(line_h)
            self._stream().append(
                f"BT {font} {font_size:.1f} Tf 0.16 0.16 0.19 rg {MARGIN_X:.1f} {self._y - font_size:.1f} Td ({_pdf_escape(ln)}) Tj ET"
            )
            self._y -= line_h
        self._y -= 2.0

    def add_table(self, headers: list[str], rows: list[list[str]], col_weights: list[float] | None = None) -> None:
        if not headers:
            return
        weights = col_weights or [1.0] * len(headers)
        w_sum = sum(weights) or 1.0
        col_widths = [(w / w_sum) * CONTENT_W for w in weights]

        row_h = 15.0
        self._ensure_space(row_h * 2)
        s = self._stream()

        # Header background
        hy = self._y - row_h
        s.append(f"0.92 0.92 0.94 rg {MARGIN_X:.1f} {hy:.1f} {CONTENT_W:.1f} {row_h:.1f} re f")
        cx = MARGIN_X + 4.0
        for hdr, cw in zip(headers, col_widths, strict=False):
            max_chars = max(4, int((cw - 6.0) / 4.6))
            s.append(
                f"BT /F2 8.0 Tf 0.12 0.12 0.15 rg {cx:.1f} {hy + 4.2:.1f} Td ({_pdf_escape(hdr[:max_chars])}) Tj ET"
            )
            cx += cw
        self._y -= row_h

        for idx, row in enumerate(rows):
            self._ensure_space(row_h)
            ry = self._y - row_h
            if idx % 2 == 1:
                s.append(f"0.97 0.97 0.98 rg {MARGIN_X:.1f} {ry:.1f} {CONTENT_W:.1f} {row_h:.1f} re f")
            cx = MARGIN_X + 4.0
            for cell, cw in zip(row, col_widths, strict=False):
                max_chars = max(4, int((cw - 6.0) / 4.5))
                txt = str(cell)
                if len(txt) > max_chars:
                    txt = txt[: max_chars - 2] + ".."
                s.append(
                    f"BT /F3 7.6 Tf 0.18 0.18 0.22 rg {cx:.1f} {ry + 4.2:.1f} Td ({_pdf_escape(txt)}) Tj ET"
                )
                cx += cw
            self._y -= row_h
        self._y -= 6.0

    def add_horizontal_bar_chart(
        self,
        title: str,
        items: list[tuple[str, float, str]],
        *,
        max_value: float | None = None,
    ) -> None:
        """Render a vector horizontal bar chart (`(label, numeric_val, formatted_val)`)."""
        if not items:
            return
        bar_h = 13.0
        needed = 18.0 + len(items) * (bar_h + 3.0)
        self._ensure_space(needed)
        self._stream().append(
            f"BT /F2 8.8 Tf 0.18 0.18 0.22 rg {MARGIN_X:.1f} {self._y - 9.0:.1f} Td ({_pdf_escape(title)}) Tj ET"
        )
        self._y -= 14.0

        eff_max = max_value if (max_value is not None and max_value > 0) else max((abs(v) for _, v, _ in items), default=1.0)
        if eff_max <= 0:
            eff_max = 1.0

        label_w = 140.0
        track_w = CONTENT_W - label_w - 80.0

        for label, val, val_str in items:
            self._ensure_space(bar_h + 3.0)
            s = self._stream()
            by = self._y - bar_h
            s.append(
                f"BT /F3 7.6 Tf 0.20 0.20 0.24 rg {MARGIN_X:.1f} {by + 3.0:.1f} Td ({_pdf_escape(label[:26])}) Tj ET"
            )
            # Track background
            tx = MARGIN_X + label_w
            s.append(f"0.93 0.93 0.95 rg {tx:.1f} {by + 2.0:.1f} {track_w:.1f} 8.5 re f")
            fill_w = max(1.5, min(track_w, (max(0.0, val) / eff_max) * track_w))
            s.append(f"0.22 0.24 0.30 rg {tx:.1f} {by + 2.0:.1f} {fill_w:.1f} 8.5 re f")
            s.append(
                f"BT /F3 7.6 Tf 0.15 0.15 0.18 rg {tx + track_w + 8.0:.1f} {by + 3.0:.1f} Td ({_pdf_escape(val_str[:16])}) Tj ET"
            )
            self._y -= bar_h + 3.0
        self._y -= 4.0

    def to_bytes(self) -> bytes:
        """Assemble PDF object graph, exact byte-offset xref table, and trailer."""
        # Object IDs:
        # 1: Catalog
        # 2: Pages
        # 3: Font Helvetica (/F1)
        # 4: Font Helvetica-Bold (/F2)
        # 5: Font Courier (/F3)
        # 6..: Pairs of (Page obj, Content stream obj)
        objects: dict[int, bytes] = {}
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"

        page_obj_ids: list[int] = []
        next_id = 6
        total_pages = len(self._pages)

        for page_idx, cmds in enumerate(self._pages, start=1):
            page_id = next_id
            stream_id = next_id + 1
            next_id += 2
            page_obj_ids.append(page_id)

            footer_cmd = (
                f"BT /F1 7.5 Tf 0.45 0.45 0.50 rg {MARGIN_X:.1f} 24.0 Td "
                f"({_pdf_escape(self.title)}  |  Page {page_idx} of {total_pages}) Tj ET"
            )
            full_stream = ("\n".join([*cmds, footer_cmd]) + "\n").encode("latin-1", errors="replace")
            objects[stream_id] = (
                f"<< /Length {len(full_stream)} >>\nstream\n".encode("ascii")
                + full_stream
                + b"endstream"
            )
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode("ascii")

        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects[2] = f"<< /Type /Pages /Count {len(page_obj_ids)} /Kids [ {kids_str} ] >>".encode("ascii")
        objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

        info_id = next_id
        objects[info_id] = (
            f"<< /Title ({_pdf_escape(self.title)}) /Producer (NEURAL DIFF Evolution Engine) >>"
        ).encode("ascii")

        max_obj_id = info_id
        buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: dict[int, int] = {}

        for oid in range(1, max_obj_id + 1):
            offsets[oid] = len(buf)
            buf.extend(f"{oid} 0 obj\n".encode("ascii"))
            buf.extend(objects[oid])
            buf.extend(b"\nendobj\n")

        xref_offset = len(buf)
        buf.extend(f"xref\n0 {max_obj_id + 1}\n".encode("ascii"))
        buf.extend(b"0000000000 65535 f \n")
        for oid in range(1, max_obj_id + 1):
            buf.extend(f"{offsets[oid]:010d} 00000 n \n".encode("ascii"))

        buf.extend(
            (
                f"trailer\n<< /Size {max_obj_id + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(buf)
