"""
Helpers de deteccion (sin SAM): manos, balon, equipo, geometria de campo,
y descubrimiento de robots por resta de fondo (para validar en CPU sin SAM).
"""
import cv2
import numpy as np
import config as C


# ---------- geometria de campo ----------
def field_polygon(calib: dict) -> np.ndarray:
    return np.array(calib["field_corners"], dtype=np.int32)


def make_field_mask(calib: dict, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [field_polygon(calib)], 255)
    return m


def on_field(cx: float, cy: float, field_mask: np.ndarray) -> bool:
    h, w = field_mask.shape
    xi, yi = int(round(cx)), int(round(cy))
    if not (0 <= xi < w and 0 <= yi < h):
        return False
    return field_mask[yi, xi] > 0


# ---------- manos (piel) ----------
def hand_blob_area(frame_bgr: np.ndarray) -> int:
    """Area (px) del mayor blob de piel conectado. Brazo cruzando -> grande."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, C.SKIN_LO, C.SKIN_HI)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(skin)
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


# ---------- balon ----------
def detect_ball(frame_bgr: np.ndarray, field_mask: np.ndarray) -> dict | None:
    """Devuelve {'mask','bbox','centroid'} del balon naranja, o None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, C.BALL_LO, C.BALL_HI)
    m = cv2.bitwise_and(m, field_mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    for c in cnts:
        a = cv2.contourArea(c)
        if not (C.BALL_AREA_MIN < a < C.BALL_AREA_MAX):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        circ = a / (np.pi * r * r + 1e-6)        # 1 = circulo perfecto
        if circ < 0.5:
            continue
        if a > best_score:                        # el blob naranja mas grande y circular
            best_score = a
            mask = np.zeros(m.shape, bool)
            cv2.drawContours(mask.view(np.uint8), [c], -1, 1, -1)
            xx, yy, ww, hh = cv2.boundingRect(c)
            best = {"mask": mask, "bbox": [xx, yy, xx + ww, yy + hh],
                    "centroid": [float(x), float(y)]}
    return best


# ---------- clasificacion de equipo ----------
def classify_team(frame_bgr: np.ndarray, mask: np.ndarray) -> tuple[int, dict]:
    """
    Sobre los pixeles del robot (excluyendo campo verde), decide equipo por
    el hue mediano. Devuelve (class_id, info_debug).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[mask.astype(bool)]
    if len(px) == 0:
        return C.CID["robot_oscuro"], {"reason": "mask vacia"}
    h, s, v = px[:, 0], px[:, 1], px[:, 2]
    # excluir pixeles de campo (verde teal saturado) y sombras muy oscuras
    is_field = (h >= C.FIELD_LO[0]) & (h <= C.FIELD_HI[0]) & (s > 150)
    keep = px[~is_field & (v > 40)]
    if len(keep) < 20:
        keep = px
    hue_med = float(np.median(keep[:, 0]))
    green_frac = float(((keep[:, 0] >= C.ROBOT_GREEN_LO[0]) &
                        (keep[:, 0] <= C.ROBOT_GREEN_HI[0]) &
                        (keep[:, 1] >= C.ROBOT_GREEN_LO[1])).mean())
    cid = C.CID["robot_verde"] if hue_med >= C.TEAM_HUE_THRESH else C.CID["robot_oscuro"]
    return cid, {"hue_med": round(hue_med, 1), "green_frac": round(green_frac, 2)}


# ---------- descubrimiento por resta de fondo (CPU, sin SAM) ----------
def build_background(video_path, n: int = 40) -> np.ndarray:
    """Fondo = mediana de N frames espaciados (robots en movimiento desaparecen)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def discover_robots_bgsub(frame_bgr, background, field_mask) -> list[dict]:
    """Robots = primer plano grande sobre el campo (excluye balon naranja)."""
    diff = cv2.absdiff(frame_bgr, background)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(g, 35, 255, cv2.THRESH_BINARY)
    fg = cv2.bitwise_and(fg, field_mask)
    # quitar balon naranja del primer plano
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ball = cv2.inRange(hsv, C.BALL_LO, C.BALL_HI)
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(ball))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (C.ROBOT_AREA_MIN < a < C.ROBOT_AREA_MAX):
            continue
        mask = np.zeros(fg.shape, bool)
        cv2.drawContours(mask.view(np.uint8), [c], -1, 1, -1)
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = M["m10"] / (M["m00"] + 1e-6)
        cy = M["m01"] / (M["m00"] + 1e-6)
        out.append({"mask": mask, "bbox": [x, y, x + w, y + h],
                    "centroid": [cx, cy], "area": a})
    return out
