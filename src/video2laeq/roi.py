"""Εντοπισμός της οθόνης LCD του ηχομέτρου σε κάθε frame.

Στρατηγική: το πορτοκαλί σώμα του οργάνου εντοπίζεται με κατώφλι HSV, μέσα στο
κυρτό περίβλημά του η LCD είναι η μεγαλύτερη «άχρωμη» (χαμηλού κορεσμού) περιοχή
που περιβάλλεται από σκούρο πλαίσιο. Η κάμερα είναι χειρός, οπότε η ανίχνευση
γίνεται ανά frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y + self.h, self.x:self.x + self.w]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def parse_roi(text: str) -> Rect:
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("Το --roi πρέπει να είναι X,Y,W,H με θετικά W,H")
    return Rect(*parts)


def _kernel(n: float) -> np.ndarray:
    n = max(3, int(n)) | 1
    return np.ones((n, n), np.uint8)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


# Πορτοκαλί σώμα (Cadrim και παρόμοια όργανα)
ORANGE_LO = (5, 120, 120)
ORANGE_HI = (25, 255, 255)

# Γεωμετρία της LCD σε σχέση με το πλάτος (bw) και την κορυφή του πορτοκαλί σώματος,
# μετρημένη σε πραγματικά πλάνα: κεντραρισμένη οριζόντια, κορυφή ~0.50·bw κάτω από
# την κορυφή του σώματος, μέγεθος ~0.55·bw x 0.65·bw. Το crop έχει περιθώριο γύρω της.
LCD_TOP = 0.42
LCD_HEIGHT = 0.82
LCD_WIDTH = 0.74


def find_lcd(frame_bgr: np.ndarray) -> Rect | None:
    """Εντοπίζει το πορτοκαλί σώμα και επιστρέφει το crop της LCD από τη γεωμετρία του."""
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, _kernel(9))
    body_contour = _largest_contour(orange)
    if body_contour is None:
        return None
    body_area = cv2.contourArea(body_contour)
    if body_area < 0.01 * H * W:
        return None
    # Ενώνουμε και τα υπόλοιπα πορτοκαλί κομμάτια που επικαλύπτονται οριζόντια με το
    # μεγαλύτερο (π.χ. αν το σκούρο μέτωπο ή ένα δάχτυλο κόβει το σώμα στα δύο).
    bx, by, bw, bh = cv2.boundingRect(body_contour)
    contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 0.05 * body_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x < bx + bw and x + w > bx:
            nx0, ny0 = min(bx, x), min(by, y)
            nx1, ny1 = max(bx + bw, x + w), max(by + bh, y + h)
            bx, by, bw, bh = nx0, ny0, nx1 - nx0, ny1 - ny0
    if bh < 0.8 * bw:            # το σώμα είναι ψηλότερο από πλατύ
        return None
    cx = bx + bw / 2
    x0 = int(cx - LCD_WIDTH * bw / 2)
    y0 = int(by + LCD_TOP * bw)
    x1 = int(cx + LCD_WIDTH * bw / 2)
    y1 = int(by + (LCD_TOP + LCD_HEIGHT) * bw)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)
