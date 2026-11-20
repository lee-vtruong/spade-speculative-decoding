"""Patch Figure 1 (docs/paper/latex/figures/spade_overview.pdf): 'Upper 25% Layers' -> 'Upper 50% Layers'."""
from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


def main() -> None:
    pdf_path = Path("docs/paper/latex/figures/spade_overview.pdf")
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)

    txt_before = page.extract_text() or ""
    if "Upper 50% Layers" in txt_before and "Upper 25% Layers" not in txt_before:
        print("spade_overview.pdf already displays 'Upper 50% Layers'; nothing to patch.")
        return
    assert "Upper 25% Layers" in txt_before, "Expected 'Upper 25% Layers' in spade_overview.pdf"

    # Measured exact raster/PDF coordinates from 144 DPI check:
    # full pt box: x0=376.0, x1=431.5, y_bot=272.58, y_top=279.08; bg=#FFFFFF; text=#475569 (71, 85, 105)
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(w, h))
    c.setFillColor(Color(1.0, 1.0, 1.0))
    c.rect(373.0, 271.0, 62.0, 10.0, stroke=0, fill=1)
    c.setFillColor(Color(71 / 255.0, 85 / 255.0, 105 / 255.0))
    c.setFont("Helvetica", 7.1)
    c.drawRightString(431.5, 274.1, "Upper 50% Layers")
    c.save()
    packet.seek(0)

    overlay = PdfReader(packet).pages[0]
    page.merge_page(overlay)
    writer = PdfWriter()
    writer.add_page(page)
    tmp_path = pdf_path.with_suffix(".tmp.pdf")
    with tmp_path.open("wb") as f:
        writer.write(f)
    tmp_path.replace(pdf_path)

    chk_text = (PdfReader(str(pdf_path)).pages[0].extract_text() or "")
    print("Patched PDF text contains 'Upper 50% Layers':", "Upper 50% Layers" in chk_text)


if __name__ == "__main__":
    main()
