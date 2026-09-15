import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from synth import render_frame
from video2laeq.cli import main
from video2laeq.metrics import laeq

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="χρειάζεται ffmpeg")


def make_video(path: Path, values: list[float], fps: int = 10) -> None:
    frames = path.parent / "frames"
    frames.mkdir()
    k = 0
    for sec, v in enumerate(values):
        for j in range(fps):
            cv2.imwrite(str(frames / f"{k:05d}.png"), render_frame(v, seed=k, body_x=105 + (sec % 3)))
            k += 1
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(1)
    values = [round(float(v), 1) for v in rng.uniform(45, 70, 40)]
    video = tmp_path / "synthetic.mov"
    make_video(video, values)
    csv = tmp_path / "out.csv"
    pdf = tmp_path / "out.pdf"
    assert main([str(video), "--json", "--csv", str(csv), "--pdf", str(pdf)]) == 0
    import json
    out = json.loads(capsys.readouterr().out)
    arr = np.array(values)
    assert out["n_valid"] >= 0.9 * len(values)
    assert out["LAeq"] == pytest.approx(laeq(arr), abs=0.15)
    assert out["LAF10"] == pytest.approx(np.percentile(arr, 90), abs=0.6)
    assert out["LAF90"] == pytest.approx(np.percentile(arr, 10), abs=0.6)
    assert pdf.stat().st_size > 10_000
    assert csv.read_text(encoding="utf-8").count("\n") == len(values) + 1
