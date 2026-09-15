"""Γράφημα χρονοσειράς (matplotlib) και αναφορά PDF (reportlab) στα ελληνικά."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .metrics import Result, Sample  # noqa: E402

# Χρώματα: μία σειρά (μπλε), κείμενο/γραμμές αναφοράς σε ουδέτερους τόνους.
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
FONT_BOLD_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


def _mmss(seconds: float, _pos=None) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def make_chart(samples: list[Sample], result: Result, out_png: Path, title: str) -> Path:
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "DejaVu Sans"

    t = np.array([s.t for s in samples])
    v = np.array([s.value if s.valid else np.nan for s in samples], dtype=float)
    invalid_t = np.array([s.t for s in samples if not s.valid])

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(t, v, color=SERIES, linewidth=1.6, solid_joinstyle="round", solid_capstyle="round",
            label="dB(A) ανά δευτερόλεπτο")
    y_lo = np.nanmin(v) - 3
    if invalid_t.size:
        ax.plot(invalid_t, np.full(invalid_t.shape, y_lo + 0.8), linestyle="none", marker="|",
                color=TEXT_SECONDARY, markersize=6, label="άκυρο δείγμα")

    refs = [("LAeq", result.laeq, TEXT_PRIMARY, 1.2), ("LAF10", result.laf10, TEXT_SECONDARY, 0.8),
            ("LAF90", result.laf90, TEXT_SECONDARY, 0.8)]
    for name, val, color, lw in refs:
        ax.axhline(val, color=color, linewidth=lw, zorder=1)
        ax.annotate(f"{name} {val:.1f}", xy=(t[-1], val), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=color)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y_lo, np.nanmax(v) + 3)
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_xlabel("Χρόνος (λεπτά:δευτερόλεπτα)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Στάθμη dB(A)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.9, bottom=0.14)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def make_pdf(samples: list[Sample], result: Result, video_name: str, chart_png: Path,
             out_pdf: Path, notes: list[str] | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_BOLD_PATH)))

    h1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11, leading=14, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=9.5, leading=13)
    small = ParagraphStyle("small", fontName="DejaVu", fontSize=8.5, leading=11.5, textColor=colors.HexColor(TEXT_SECONDARY))

    n_invalid = result.n_total - result.n_valid
    meta = [
        ["Αρχείο βίντεο", video_name],
        ["Ημερομηνία ανάλυσης", dt.datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Διάρκεια", f"{_mmss(result.duration)} ({result.duration:.0f} s)"],
        ["Δείγματα", f"{result.n_valid} έγκυρα από {result.n_total} (άκυρα: {n_invalid})"],
    ]
    results = [
        ["Μέγεθος", "Τιμή dB(A)", "Περιγραφή"],
        ["LAeq", f"{result.laeq:.1f}", "Ισοδύναμη συνεχής Α-σταθμισμένη στάθμη"],
        ["LAF10", f"{result.laf10:.1f}", "Στάθμη που ξεπερνιέται το 10% του χρόνου"],
        ["LAF90", f"{result.laf90:.1f}", "Στάθμη που ξεπερνιέται το 90% του χρόνου (θόρυβος βάθους)"],
        ["Lmin / Lmax", f"{result.lmin:.1f} / {result.lmax:.1f}", "Ελάχιστη / μέγιστη ένδειξη"],
    ]

    grid = colors.HexColor(GRID)
    meta_table = Table(meta, colWidths=[45 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(TEXT_SECONDARY)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, grid), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    res_table = Table(results, colWidths=[30 * mm, 30 * mm, 105 * mm])
    res_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"), ("FONTNAME", (0, 1), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(TEXT_PRIMARY)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, grid), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor(TEXT_SECONDARY)),
    ]))

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Ανάλυση στάθμης θορύβου", author="video2laeq")
    img_w = A4[0] - 36 * mm
    story = [
        Paragraph("Ανάλυση στάθμης θορύβου από βίντεο ηχομέτρου", h1),
        Paragraph("Αποτελέσματα LAeq, LAF10, LAF90 από ένα δείγμα dB(A) ανά δευτερόλεπτο.", small),
        Spacer(1, 6), meta_table,
        Paragraph("Αποτελέσματα", h2), res_table,
        Paragraph("Χρονοσειρά dB(A)", h2),
        Image(str(chart_png), width=img_w, height=img_w * 0.42),
        Paragraph("Μέθοδος", h2),
        Paragraph(
            "Από το βίντεο εξάγεται ένα καρέ ανά δευτερόλεπτο και διαβάζεται αυτόματα η ένδειξη της "
            "οθόνης του ηχομέτρου. Άκυρα δείγματα (μη αναγνώσιμη ένδειξη, τιμή εκτός 20–140 dB ή "
            "απομονωμένο άλμα άνω των 15 dB) απορρίπτονται. "
            "LAeq = 10·log10(μέσος όρος 10^(Li/10)). LAF10 και LAF90 είναι το 90ο και 10ο "
            "εκατοστημόριο των δειγμάτων. Με ένα δείγμα ανά δευτερόλεπτο, τα LAF10/LAF90 αποτελούν "
            "προσέγγιση των τιμών που θα προέκυπταν από συνεχή καταγραφή FAST (125 ms).", body),
    ]
    for note in notes or []:
        story.append(Paragraph(note, small))
    doc.build(story)
    return out_pdf
