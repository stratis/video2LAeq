"""Συνθετικά καρέ ηχομέτρου για tests: πορτοκαλί σώμα, σκούρο πλαίσιο, LCD, ψηφία 7-segment."""

from __future__ import annotations

import cv2
import numpy as np

from video2laeq.sevenseg import PATTERNS

BG = (90, 110, 120)
ORANGE = (20, 140, 240)   # BGR
BEZEL = (25, 25, 25)
LCD = (165, 175, 170)
INK = (60, 62, 60)


def draw_digit(img: np.ndarray, ch: str, x: int, y: int, w: int, h: int, s: int, slant: float) -> None:
    """Σχεδιάζει ψηφίο 7-segment με κενά ανάμεσα στα segments και italic κλίση."""
    a, b, c, d, e, f, g = PATTERNS[ch]
    gap = max(1, s // 2)
    mid = y + h // 2

    def seg(x0: int, y0: int, x1: int, y1: int) -> None:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        pts[:, 0] += slant * (y + h - pts[:, 1])   # η κορυφή δεξιότερα από τη βάση
        cv2.fillPoly(img, [pts.astype(np.int32)], INK, lineType=cv2.LINE_AA)

    if a:
        seg(x + gap, y, x + w - gap, y + s)
    if d:
        seg(x + gap, y + h - s, x + w - gap, y + h)
    if g:
        seg(x + gap, mid - s // 2, x + w - gap, mid - s // 2 + s)
    if f:
        seg(x, y + gap, x + s, mid - gap)
    if e:
        seg(x, mid + gap, x + s, y + h - gap)
    if b:
        seg(x + w - s, y + gap, x + w, mid - gap)
    if c:
        seg(x + w - s, mid + gap, x + w, y + h - gap)


def render_frame(value: float, width: int = 360, height: int = 640, body_w: int = 140,
                 body_x: int = 110, body_y: int = 300, noise: float = 4.0,
                 seed: int = 0) -> np.ndarray:
    """Καρέ με το όργανο και την ένδειξη `value` (μορφή XX.X)."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), BG, np.uint8)
    body_h = int(body_w * 2.2)
    cv2.rectangle(img, (body_x, body_y), (body_x + body_w, min(height - 1, body_y + body_h)), ORANGE, -1)
    # Πρόσοψη και LCD στη γεωμετρία που περιμένει το roi.find_lcd.
    cx = body_x + body_w // 2
    lcd_w, lcd_h = int(0.55 * body_w), int(0.65 * body_w)
    lcd_x, lcd_y = cx - lcd_w // 2, body_y + int(0.50 * body_w)
    bez = int(0.06 * body_w)
    cv2.rectangle(img, (lcd_x - bez, lcd_y - bez), (lcd_x + lcd_w + bez, lcd_y + lcd_h + bez), BEZEL, -1)
    cv2.rectangle(img, (lcd_x, lcd_y), (lcd_x + lcd_w, lcd_y + lcd_h), LCD, -1)
    # Ψηφία: 3 ή 4, ύψος ~0.42 της LCD, στο κάτω μέρος της.
    text = f"{value:.1f}".replace(".", "")
    n = len(text)
    dh = int(0.42 * lcd_h)
    dw = int(dh * (0.36 if n <= 3 else 0.30))     # με 4 ψηφία η οθόνη τα στριμώχνει
    s = max(3, int(dw * 0.3))
    pitch = int(dw * (1.75 if n <= 3 else 1.5))
    total = pitch * (n - 1) + dw
    x0 = lcd_x + (lcd_w - total) // 2
    y0 = lcd_y + int(0.42 * lcd_h)
    for i, ch in enumerate(text):
        draw_digit(img, ch, x0 + i * pitch, y0, dw, dh, s, slant=0.12)
    # Υποδιαστολή πριν το τελευταίο ψηφίο.
    dot_x = x0 + (n - 1) * pitch - (pitch - dw) // 2
    cv2.circle(img, (dot_x, y0 + dh - 2), max(1, s // 2), INK, -1)
    # Ετικέτα «dBA» πάνω από τα ψηφία.
    cv2.putText(img, "dBA", (cx - 8, lcd_y + int(0.3 * lcd_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, INK, 1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
