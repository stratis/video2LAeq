"""Γραμμή εντολών: video2laeq INPUT.mov [--pdf out.pdf] [--csv out.csv] [--json] ..."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2

from . import __version__
from .frames import FfmpegMissing, check_ffmpeg, extract_frames, probe
from .metrics import Sample, aggregate, compute, validate
from .roi import Rect, find_lcd, parse_roi
from .sevenseg import read_panel


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _decimals(text: str) -> int | None:
    if text.lower() == "auto":
        return None
    n = int(text)
    if n < 0 or n > 3:
        raise argparse.ArgumentTypeError("--decimals: 0–3 ή auto")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video2laeq",
        description="Υπολογισμός LAeq / LAF10 / LAF90 από βίντεο ηχομέτρου (1 δείγμα dB(A) ανά δευτερόλεπτο).",
    )
    p.add_argument("input", type=Path, help="Αρχείο βίντεο (.mov, .mp4 κ.λπ.)")
    p.add_argument("--pdf", type=Path, metavar="FILE", help="Extended mode: αναφορά PDF με γράφημα χρονοσειράς")
    p.add_argument("--csv", type=Path, metavar="FILE", help="Εξαγωγή δειγμάτων (t, dBA, έγκυρο, αιτία)")
    p.add_argument("--json", action="store_true", help="Έξοδος αποτελεσμάτων σε JSON αντί για κείμενο")
    p.add_argument("--roi", type=parse_roi, metavar="X,Y,W,H",
                   help="Σταθερή περιοχή της οθόνης LCD σε pixels (παρακάμπτει την αυτόματη ανίχνευση)")
    p.add_argument("--decimals", type=_decimals, default=1, metavar="N|auto",
                   help="Δεκαδικά ψηφία της ένδειξης (default 1, π.χ. 68.5). 'auto': από την υποδιαστολή")
    p.add_argument("--fps", type=float, default=4.0, metavar="N",
                   help="Καρέ που αναλύονται ανά δευτερόλεπτο (default 4)· το δείγμα κάθε δευτερολέπτου "
                        "είναι η διάμεσος των έγκυρων αναγνώσεών του")
    p.add_argument("--debug", type=Path, metavar="DIR",
                   help="Αποθήκευση crops/δυαδικών εικόνων ανά δευτερόλεπτο για έλεγχο")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def analyse(video: Path, roi: Rect | None, decimals: int | None, fps: float,
            debug: Path | None) -> tuple[list[Sample], float]:
    info = probe(video)
    _log(f"Βίντεο: {video.name} — {info.width}x{info.height}, {_mmss(info.duration)}, {info.fps:.0f} fps")
    with tempfile.TemporaryDirectory(prefix="video2laeq_") as tmp:
        _log(f"Εξαγωγή καρέ ({fps:g}/sec)…")
        frames = extract_frames(video, Path(tmp), fps=fps)
        _log(f"Ανάγνωση ένδειξης σε {len(frames)} καρέ…")
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
        per_second: dict[int, list[float]] = {}
        reasons: dict[int, list[str]] = {}
        n_no_lcd = 0
        for i, (t, path) in enumerate(frames):
            sec = int(t)
            frame = cv2.imread(str(path))
            rect = roi or find_lcd(frame)
            if rect is None:
                n_no_lcd += 1
                reasons.setdefault(sec, []).append("δεν βρέθηκε η οθόνη")
                if debug:
                    cv2.imwrite(str(debug / f"{t:07.2f}_NOLCD.png"), frame)
                continue
            reading = read_panel(rect.crop(frame), decimals=decimals, want_debug=bool(debug))
            if reading.value is not None:
                per_second.setdefault(sec, []).append(reading.value)
            else:
                reasons.setdefault(sec, []).append(reading.reason)
            if debug:
                tag = f"{reading.value:.1f}" if reading.value is not None else "X"
                cv2.imwrite(str(debug / f"{t:07.2f}_{tag}.png"), reading.debug["binary"])
                cv2.imwrite(str(debug / f"{t:07.2f}_{tag}_crop.png"), rect.crop(frame))
                if i == 0:
                    vis = frame.copy()
                    cv2.rectangle(vis, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 0, 255), 2)
                    cv2.imwrite(str(debug / "frame0_roi.png"), vis)
            if (i + 1) % (30 * max(1, int(fps))) == 0:
                _log(f"  {i + 1}/{len(frames)}")
        if n_no_lcd:
            _log(f"Προσοχή: σε {n_no_lcd} καρέ δεν εντοπίστηκε η οθόνη (δοκίμασε --roi ή --debug).")
    samples: list[Sample] = []
    for sec in range(int(frames[-1][0]) + 1 if frames else 0):
        values = per_second.get(sec, [])
        agg = aggregate(values)
        if agg is not None:
            samples.append(Sample(float(sec), agg))
        elif values:
            samples.append(Sample(float(sec), None, False, f"ασυνεπείς αναγνώσεις {sorted(values)}"))
        else:
            why = reasons.get(sec, ["κανένα καρέ"])
            samples.append(Sample(float(sec), None, False, max(set(why), key=why.count)))
    return samples, info.duration


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_ffmpeg()
    except FfmpegMissing as e:
        _log(f"Σφάλμα: {e}")
        return 2
    if not args.input.exists():
        _log(f"Σφάλμα: δεν βρέθηκε το αρχείο {args.input}")
        return 2

    samples, duration = analyse(args.input, args.roi, args.decimals, args.fps, args.debug)
    validate(samples)
    n_valid = sum(1 for s in samples if s.valid)
    if n_valid < 10:
        _log(f"Σφάλμα: μόνο {n_valid} έγκυρα δείγματα από {len(samples)}. "
             "Έλεγξε την ανάγνωση με --debug DIR ή δώσε --roi X,Y,W,H.")
        return 1
    result = compute(samples, duration)
    invalid_frac = 1 - result.n_valid / result.n_total
    if invalid_frac > 0.10:
        _log(f"Προσοχή: {invalid_frac:.0%} των δειγμάτων απορρίφθηκαν ως άκυρα.")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "dBA", "valid", "reason"])
            for s in samples:
                w.writerow([f"{s.t:.0f}", "" if s.value is None else f"{s.value:.1f}", int(s.valid), s.reason])
        _log(f"CSV: {args.csv}")

    if args.pdf:
        from .report import make_chart, make_pdf
        with tempfile.TemporaryDirectory(prefix="video2laeq_") as tmp:
            png = make_chart(samples, result, Path(tmp) / "chart.png", args.input.name)
            make_pdf(samples, result, args.input.name, png, args.pdf)
        _log(f"PDF: {args.pdf}")

    if args.json:
        print(json.dumps({
            "file": str(args.input), "duration_sec": round(result.duration, 1),
            "n_valid": result.n_valid, "n_total": result.n_total,
            "LAeq": round(result.laeq, 1), "LAF10": round(result.laf10, 1), "LAF90": round(result.laf90, 1),
            "Lmin": round(result.lmin, 1), "Lmax": round(result.lmax, 1),
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Αρχείο:    {args.input.name}")
        print(f"Διάρκεια:  {_mmss(result.duration)}")
        print(f"Δείγματα:  {result.n_valid} έγκυρα / {result.n_total} σύνολο")
        print(f"LAeq   = {result.laeq:5.1f} dB(A)")
        print(f"LAF10  = {result.laf10:5.1f} dB(A)")
        print(f"LAF90  = {result.laf90:5.1f} dB(A)")
        print(f"Lmin   = {result.lmin:5.1f} dB(A)   Lmax = {result.lmax:5.1f} dB(A)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
