"""Εξαγωγή ενός frame ανά δευτερόλεπτο από βίντεο με ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FfmpegMissing(
                f"Δεν βρέθηκε το {tool}. Εγκατάστησέ το με: brew install ffmpeg"
            )


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    duration = float(stream.get("duration") or data["format"]["duration"])
    return VideoInfo(duration=duration, width=int(stream["width"]),
                     height=int(stream["height"]), fps=fps)


def extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[tuple[float, Path]]:
    """Γράφει PNG frames στο out_dir με ρυθμό `fps` και επιστρέφει (χρόνος_sec, αρχείο)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f%06d.png"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(path),
        "-vf", f"fps={fps}", "-start_number", "0", str(pattern),
    ]
    subprocess.run(cmd, check=True)
    files = sorted(out_dir.glob("f*.png"))
    return [(i / fps, f) for i, f in enumerate(files)]
