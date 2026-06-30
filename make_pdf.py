"""Render math_set_15.xlsx into a printable PDF: each question with its figure,
the four options, and the correct answer. Arabic text is RTL-shaped.

Usage:
    python make_pdf.py
    python make_pdf.py --xlsx questins_visual/math_set_15.xlsx --out questins_visual/math_set_15.pdf
"""

import argparse
from pathlib import Path

import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER

ROOT = Path(__file__).resolve().parent
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT = "ArabicUni"


def ar(text) -> str:
    """Shape + reorder Arabic so reportlab renders it correctly (RTL)."""
    s = "" if text is None else str(text)
    if s.strip() in ("", "nan", "None"):
        return ""
    try:
        return get_display(arabic_reshaper.reshape(s))
    except Exception:
        return s


def build(xlsx: Path, out: Path):
    pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
    df = pd.read_excel(xlsx)

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("title", parent=styles["Title"], fontName=FONT,
                             fontSize=20, alignment=TA_CENTER, spaceAfter=4)
    h_sub = ParagraphStyle("sub", parent=styles["Normal"], fontName=FONT,
                           fontSize=10, alignment=TA_CENTER, textColor=colors.grey,
                           spaceAfter=10)
    q_style = ParagraphStyle("q", parent=styles["Normal"], fontName=FONT,
                             fontSize=13, alignment=TA_RIGHT, leading=20, spaceAfter=6)
    meta_style = ParagraphStyle("meta", parent=styles["Normal"], fontName=FONT,
                                fontSize=9, alignment=TA_RIGHT, textColor=colors.grey)
    opt_style = ParagraphStyle("opt", parent=styles["Normal"], fontName=FONT,
                               fontSize=12, alignment=TA_RIGHT, leading=18)
    ans_style = ParagraphStyle("ans", parent=styles["Normal"], fontName=FONT,
                               fontSize=11, alignment=TA_RIGHT,
                               textColor=colors.HexColor("#1a7f37"))

    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            title="Math Set - 15 Questions")
    n_q = len(df)
    n_ar = str(n_q).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    flow = [
        Paragraph(ar(f"مجموعة الرياضيات — {n_ar} أسئلة مع الأشكال والجداول"), h_title),
        Paragraph("Quantitative section · figures, shapes & tables", h_sub),
        HRFlowable(width="100%", color=colors.lightgrey, spaceAfter=10),
    ]

    letters = ["A", "B", "C", "D"]
    for i, row in df.iterrows():
        n = i + 1
        cat = f"{row.get('Category','')} · {row.get('Sub-Category','')}"
        flow.append(Paragraph(ar(f"السؤال {n}"), meta_style))
        flow.append(Paragraph(ar(cat), meta_style))
        flow.append(Paragraph(ar(row.get("Question", "")), q_style))

        # Figure
        img_rel = str(row.get("ImagePath") or "").strip()
        img_path = (xlsx.parent / img_rel) if img_rel else None
        if img_path and img_path.exists():
            try:
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(str(img_path)).getSize()
                max_w = 95 * mm
                w = min(max_w, iw)
                h = w * ih / iw
                max_h = 90 * mm
                if h > max_h:
                    h = max_h
                    w = h * iw / ih
                flow.append(Spacer(1, 4))
                flow.append(Image(str(img_path), width=w, height=h, hAlign="CENTER"))
                flow.append(Spacer(1, 6))
            except Exception:
                pass

        # Options A-D (RTL: letter on the right)
        correct = str(row.get("CorrectOption", "")).strip().upper()
        data = []
        for L in letters:
            val = row.get(f"Option{L}", "")
            if str(val).strip() in ("", "nan", "None"):
                continue
            mark = " ✓" if L == correct else ""
            data.append([Paragraph(ar(f"{val}{mark}  ({L})"), opt_style)])
        if data:
            t = Table(data, colWidths=[doc.width])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(t)

        ans = str(row.get("Answer", "")).strip()
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(ar(f"الإجابة الصحيحة: ({correct}) {ans}"), ans_style))
        flow.append(HRFlowable(width="100%", color=colors.lightgrey,
                               spaceBefore=10, spaceAfter=12))

    doc.build(flow)
    print(f"Wrote {len(df)} questions -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ROOT / "questins_visual" / "math_set_15.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "questins_visual" / "math_set_15.pdf"))
    args = ap.parse_args()
    build(Path(args.xlsx), Path(args.out))


if __name__ == "__main__":
    main()
