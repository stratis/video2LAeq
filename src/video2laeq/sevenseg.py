"""Ανάγνωση ψηφίων 7-segment από το crop της οθόνης LCD.

Βήματα: κλιμάκωση και κανονικοποίηση αντίθεσης, black-hat + Otsu για τα σκούρα
segments, αυτόματη διόρθωση της italic κλίσης, εντοπισμός ψηφίων ως στήλες της
κατακόρυφης προβολής, δειγματοληψία 7 ζωνών ανά ψηφίο και επιλογή του
πλησιέστερου προτύπου.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#                a  b  c  d  e  f  g
PATTERNS: dict[str, tuple[int, ...]] = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}

TARGET_HEIGHT = 320          # ύψος crop μετά την κλιμάκωση (px)
SLANT_CANDIDATES = np.arange(0.0, 0.46, 0.05)   # tan(γωνίας) της italic κλίσης
MIN_IOU = 0.42               # ελάχιστη ομοιότητα (IoU) με πρότυπο ψηφίου
MIN_MARGIN = 0.02            # ελάχιστη διαφορά από το δεύτερο καλύτερο πρότυπο
TIEBREAK_IOU = 0.6           # ελάχιστο IoU για να επιχειρηθεί διάκριση ισοπαλίας ανά segment

Box = tuple[int, int, int, int]


@dataclass
class Reading:
    value: float | None
    digits: str = ""
    reason: str = ""
    slant: float = 0.0
    debug: dict[str, np.ndarray] = field(default_factory=dict)


def _preprocess(panel_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = TARGET_HEIGHT / h
    gray = cv2.resize(gray, (max(8, int(w * scale)), TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(gray, (2, 98))
    if hi - lo < 5:
        hi = lo + 5
    gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _big_structures(binary: np.ndarray) -> np.ndarray:
    """Μάσκα με πλαίσιο/κουμπιά: blobs πολύ ψηλά ή πλατιά για ψηφία, ή μεγάλα στο όριο."""
    H, W = binary.shape
    mask = np.zeros_like(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        edge_strip = h > 0.35 * H and (x < 0.12 * W or x + w > 0.88 * W)
        if h > 0.55 * H or w > 0.5 * W or edge_strip or (touches and area > 0.02 * H * W):
            mask[labels == i] = 255
    return mask


def _binarize(gray: np.ndarray, min_area_frac: float = 0.001) -> np.ndarray:
    H, W = gray.shape
    # Black-hat: αναδεικνύει σκούρες γραμμές (segments) πάνω σε ανοιχτό φόντο,
    # ανεξάρτητα από προσανατολισμό και ανομοιόμορφο φωτισμό.
    k = int(H * 0.10) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Πέρασμα 1: υψηλό κατώφλι για να βρεθούν το πλαίσιο και τα κουμπιά, που έχουν
    # πολύ πιο έντονη απόκριση από τα ψηφία (σε χαμηλό κατώφλι θα ενώνονταν με αυτά).
    rough = (blackhat > 40).astype(np.uint8) * 255
    big = cv2.dilate(_big_structures(rough), _kernel(int(H * 0.03)))
    # Πέρασμα 2: κατώφλι από το κέντρο του crop και μόνο εκτός των μεγάλων δομών.
    centre = np.zeros_like(rough)
    centre[int(0.2 * H):int(0.85 * H), int(0.15 * W):int(0.85 * W)] = 255
    vals = blackhat[(centre > 0) & (big == 0)]
    if vals.size < 100:
        return np.zeros_like(rough)
    t_otsu, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = max(float(t_otsu), 0.5 * float(np.percentile(vals, 95)), 8.0)
    binary = ((blackhat > t).astype(np.uint8) * 255) & ~big
    # Άνοιγμα 3x3: κόβει λεπτές γέφυρες θορύβου πριν την ανάλυση συνιστωσών.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        if stats[i][4] < min_area_frac * H * W:
            binary[labels == i] = 0
    binary &= ~_big_structures(binary)
    return binary


def _kernel(n: int) -> np.ndarray:
    n = max(3, int(n)) | 1
    return np.ones((n, n), np.uint8)


def _shear(img: np.ndarray, slant: float) -> np.ndarray:
    """x' = x + slant * y: μετακινεί τη βάση δεξιά ώστε να ευθυγραμμιστεί με την κορυφή."""
    H, W = img.shape
    M = np.float32([[1, slant, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (int(W + slant * H) + 1, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Επιλέγει την κλίση που κάνει τα κατακόρυφα segments πιο «αιχμηρά» στην προβολή."""
    best, best_score = 0.0, -1.0
    for s in SLANT_CANDIDATES:
        profile = _shear(binary, float(s)).sum(axis=0, dtype=np.float64)
        score = float((profile ** 2).sum())
        if score > best_score:
            best, best_score = float(s), score
    return _shear(binary, best), best


def _digit_band(binary: np.ndarray) -> tuple[int, int] | None:
    """Ζώνη (top, bottom) των ψηφίων.

    Βάση: το κάτω άκρο των κύριων συνιστωσών (ψηφία ή κατακόρυφα segments τους).
    Κορυφή: ανεβαίνουμε από την κορυφή της ψηλότερης συνιστώσας όσο οι γραμμές έχουν
    pixels· η ετικέτα «dBA» χωρίζεται από τα ψηφία με κενές γραμμές και μένει εκτός.
    """
    H, W = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [stats[i] for i in range(1, n)
             if 0.07 * H <= stats[i][3] <= 0.55 * H and stats[i][2] < 1.2 * stats[i][3]
             and stats[i][0] >= 0.08 * W and stats[i][0] + stats[i][2] <= 0.92 * W]
    if not comps:
        return None
    hmax = max(c[3] for c in comps)
    main = [c for c in comps if c[3] >= 0.5 * hmax]
    bottom = int(np.percentile([c[1] + c[3] for c in main], 90))
    x0 = min(c[0] for c in main)
    x1 = max(c[0] + c[2] for c in main)
    rows = binary[:, x0:x1].any(axis=1)
    top = max(0, bottom - hmax)
    empty = 0
    while top > 0 and empty < 3 and bottom - top < 0.5 * H:
        top -= 1
        empty = empty + 1 if not rows[top] else 0
    top += empty
    if bottom - top < 0.15 * H:
        return None
    return top, bottom


def _clean_band(band: np.ndarray, band_h: int) -> np.ndarray:
    """Αφαιρεί μακριές λεπτές γραμμές (πλαίσιο/αντανακλάσεις) και μικροσκοπικά στίγματα."""
    out = band.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (h < 0.12 * band_h and w > 0.6 * band_h) or area < 0.0015 * band_h * band_h:
            out[labels == i] = 0
    return out


def _without_dots(band: np.ndarray, band_h: int) -> np.ndarray:
    """Αφαιρεί μικρά blobs (υποδιαστολή, στίγματα) ώστε να μη γεφυρώνουν γειτονικά ψηφία."""
    out = band.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < 0.15 * band_h and w < 0.15 * band_h and y + h > 0.8 * band.shape[0]:
            out[labels == i] = 0
    return out


def _columns(band: np.ndarray, band_h: int) -> list[Box]:
    """Στήλες ψηφίων από την κατακόρυφη προβολή: χωρίζουμε όπου η στήλη έχει
    λιγότερα από 5% του ύψους σε pixels (λεπτές γέφυρες θορύβου δεν ενώνουν ψηφία)."""
    band = _without_dots(band, band_h)
    counts = (band > 0).sum(axis=0)
    profile = counts >= max(2, int(0.05 * band_h))
    cols: list[Box] = []
    x = 0
    W = band.shape[1]
    while x < W:
        if not profile[x]:
            x += 1
            continue
        x0 = x
        gap = 0
        while x < W and gap <= 2:
            gap = 0 if profile[x] else gap + 1
            x += 1
        x1 = x - gap
        ys = np.nonzero(band[:, x0:x1].any(axis=1))[0]
        cols.append((x0, int(ys.min()), x1 - x0, int(ys.max() - ys.min()) + 1))
    return [c for c in cols if c[3] >= 0.6 * band_h and 0.08 * band_h <= c[2] <= 0.9 * band_h
            and c[0] >= 0.08 * W and c[0] + c[2] <= 0.92 * W]


def _tighten(work: np.ndarray, box: Box) -> Box:
    """Αφαιρεί από την κορυφή της στήλης ό,τι χωρίζεται από το ψηφίο με κενές γραμμές
    (π.χ. η ετικέτα «dBA»). Δέχεται το κόψιμο μόνο αν αφαιρεί μικρό μέρος (≤ 30%),
    ώστε ένα κατακερματισμένο ψηφίο να μην κοπεί στη μέση.
    """
    x, y, w, h = box
    rows = work[y:y + h, x:x + w].any(axis=1)
    i = h - 1
    gap = 0
    while i > 0 and gap < 2:
        i -= 1
        gap = gap + 1 if not rows[i] else 0
    new_top = i + gap
    if new_top <= 0 or new_top > 0.3 * h:
        return box
    return (x, y + new_top, w, h - new_top)


def render_digit(ch: str, w: int, h: int, stroke: int) -> np.ndarray:
    """Πρότυπο 7-segment ψηφίου (λευκά segments σε μαύρο) μεγέθους w x h με πάχος stroke."""
    a, b, c, d, e, f, g = PATTERNS[ch]
    img = np.zeros((h, w), np.uint8)
    t = max(1, stroke)
    mid = h // 2
    if a:
        img[0:t, :] = 255
    if d:
        img[h - t:h, :] = 255
    if g:
        img[mid - t // 2:mid - t // 2 + t, :] = 255
    if f:
        img[0:mid, 0:t] = 255
    if e:
        img[mid:h, 0:t] = 255
    if b:
        img[0:mid, w - t:w] = 255
    if c:
        img[mid:h, w - t:w] = 255
    return img


def _stroke_width(digit: np.ndarray) -> int:
    """Πάχος γραμμής: σε ομοιόμορφη γραμμή πάχους s η διάμεση απόσταση από το φόντο
    είναι s/4 (το μέγιστο παραπλανά στις γωνίες όπου ενώνονται segments)."""
    fg = digit > 0
    if not fg.any():
        return 1
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    return int(round(4 * float(np.median(dist[fg]))))


def segment_fills(digit: np.ndarray, stroke: int) -> list[float]:
    """Μέσο γέμισμα της «αποκλειστικής» περιοχής κάθε segment (a..g), δηλαδή χωρίς τις
    γωνίες όπου διασταυρώνεται με άλλα segments."""
    h, w = digit.shape
    obs = digit > 0
    s = max(1, stroke)
    mid = h // 2
    regions = (
        (0, s, s, w - s),                              # a
        (s, mid - s // 2, w - s, w),                   # b
        (mid + s // 2, h - s, w - s, w),               # c
        (h - s, h, s, w - s),                          # d
        (mid + s // 2, h - s, 0, s),                   # e
        (s, mid - s // 2, 0, s),                       # f
        (mid - s // 2, mid - s // 2 + s, s, w - s),    # g
    )
    fills = []
    for y0, y1, x0, x1 in regions:
        region = obs[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        fills.append(float(region.mean()) if region.size else 0.0)
    return fills


def _classify(digit: np.ndarray, narrow: bool) -> tuple[str | None, float]:
    """Επιστρέφει (ψηφίο, IoU) συγκρίνοντας με πρότυπα ίδιου πάχους γραμμής.

    Αν τα δύο καλύτερα πρότυπα είναι σχεδόν ισοδύναμα (π.χ. 8/0 διαφέρουν μόνο στο
    μεσαίο segment), αποφασίζει το γέμισμα των segments στα οποία διαφέρουν.
    """
    h, w = digit.shape
    if narrow:
        solid = float(digit.mean()) / 255.0
        return ("1", solid) if solid >= 0.3 else (None, solid)
    obs = digit > 0
    stroke = int(np.clip(_stroke_width(digit), 0.12 * w, 0.4 * w))
    scores: list[tuple[float, str]] = []
    for ch in PATTERNS:
        if ch == "1":
            continue
        tpl = render_digit(ch, w, h, stroke) > 0
        inter = np.logical_and(obs, tpl).sum()
        union = np.logical_or(obs, tpl).sum()
        scores.append((inter / union if union else 0.0, ch))
    scores.sort(reverse=True)
    (best_iou, best), (second_iou, second) = scores[0], scores[1]
    if best_iou < MIN_IOU:
        return None, best_iou
    fills = segment_fills(digit, stroke)
    if best_iou - second_iou >= MIN_MARGIN:
        return (best if _consistent(best, fills) else None), best_iou
    if best_iou < TIEBREAK_IOU:
        return None, best_iou              # κατακερματισμένο ψηφίο: δεν ρισκάρουμε
    votes: set[str] = set()
    for f, pb, ps in zip(fills, PATTERNS[best], PATTERNS[second]):
        if pb == ps:
            continue
        if 0.35 <= f <= 0.65:
            return None, best_iou          # αμφίβολο segment
        votes.add(best if (f > 0.5) == bool(pb) else second)
    if len(votes) != 1:
        return None, best_iou              # τα segments δεν συμφωνούν μεταξύ τους
    winner = votes.pop()
    return (winner if _consistent(winner, fills) else None), best_iou


def _consistent(ch: str, fills: list[float]) -> bool:
    """Το ψηφίο πρέπει να έχει ορατά όλα τα «αναμμένα» segments του, και τα σβηστά
    κατακόρυφα segments (b, c, e, f) άδεια — εκεί γίνονται οι συγχύσεις 3/5/9/6/0."""
    verticals = {1, 2, 4, 5}
    for i, (f, on) in enumerate(zip(fills, PATTERNS[ch])):
        if on and f < 0.3:
            return False
        if not on and i in verticals and f > 0.6:
            return False
    return True


def _find_dot(binary: np.ndarray, digits: list[Box], band_h: int) -> int | None:
    """Επιστρέφει πόσα ψηφία βρίσκονται δεξιά της υποδιαστολής, ή None."""
    bottom = max(d[1] + d[3] for d in digits)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (h < 0.12 * band_h and w < 0.12 * band_h and abs((y + h) - bottom) < 0.12 * band_h):
            continue
        cx = x + w / 2
        # Η υποδιαστολή βρίσκεται στο κενό ανάμεσα σε δύο ψηφία (όχι στο εσωτερικό τους).
        for k in range(len(digits) - 1):
            right_edge = digits[k][0] + digits[k][2]
            left_edge = digits[k + 1][0]
            if right_edge - 0.05 * band_h <= cx <= left_edge + 0.05 * band_h:
                return len(digits) - (k + 1)
    return None


def read_panel(panel_bgr: np.ndarray, decimals: int | None = 1,
               want_debug: bool = False) -> Reading:
    """Διαβάζει την ένδειξη. decimals=None: χρήση της υποδιαστολής που εντοπίζεται."""
    if panel_bgr.size == 0 or panel_bgr.shape[0] < 8 or panel_bgr.shape[1] < 8:
        return Reading(None, reason="κενό crop")
    gray = _preprocess(panel_bgr)
    binary, slant = _deskew(_binarize(gray))
    dbg: dict[str, np.ndarray] = {}

    def fail(reason: str, digits: str = "") -> Reading:
        if want_debug:
            dbg.update(gray=gray, binary=cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        return Reading(None, digits=digits, reason=reason, slant=slant, debug=dbg)

    band = _digit_band(binary)
    if band is None:
        return fail("δεν βρέθηκε ζώνη ψηφίων")
    top, bottom = band
    band_h = bottom - top
    y0 = max(0, top - int(0.03 * band_h))
    y1 = min(binary.shape[0], bottom + int(0.03 * band_h))
    cleaned = _clean_band(binary[y0:y1], band_h)
    digits = [(x, y + y0, w, h) for x, y, w, h in _columns(cleaned, band_h)]
    work = binary.copy()
    work[y0:y1] = cleaned

    if want_debug:
        vis = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in digits:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        dbg.update(gray=gray, binary=vis)
    if not 2 <= len(digits) <= 4:
        return fail(f"βρέθηκαν {len(digits)} ψηφία")

    typical_w = max(d[2] for d in digits)
    if typical_w < 0.2 * band_h:      # μόνο «1»
        typical_w = 0.4 * band_h
    digits = [_tighten(work, d) for d in digits]
    text = ""
    for x, y, w, h in digits:
        narrow = w < 0.55 * typical_w
        ch, _ = _classify(work[y:y + h, x:x + w], narrow)
        if ch is None and narrow:
            continue  # λεπτή γραμμή-θόρυβος (π.χ. άκρη πλαισίου), όχι «1»
        if ch is None:
            return fail("άγνωστο ψηφίο", text + "?")
        text += ch
    if not 2 <= len(text) <= 4:
        return fail(f"βρέθηκαν {len(text)} ψηφία", text)

    dec = decimals
    if dec is None:
        # Η υποδιαστολή είναι πολύ μικρή· ξαναφτιάχνουμε το δυαδικό χωρίς αφαίρεση μικρών blobs.
        fine = _shear(_binarize(gray, min_area_frac=0.0), slant)
        dec = _find_dot(fine, digits, band_h)
        if dec is None:
            return fail("δεν βρέθηκε υποδιαστολή", text)
    dec = min(dec, len(text) - 1)
    return Reading(float(text) / (10 ** dec), digits=text, slant=slant, debug=dbg)
