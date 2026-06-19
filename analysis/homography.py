"""
Homografia y mapa tactico (adaptado de NB10-12).

Proyecta posiciones de camara -> campo canonico cenital y dibuja el mapa
tactico estilo transmision deportiva.
"""
import json
import cv2
import numpy as np
import config as C


def load_homography(calib_path=C.CALIB):
    """H (camara -> canonico) a partir de las 4 esquinas calibradas."""
    calib = json.loads(calib_path.read_text())
    src = np.float32(calib["field_corners"])              # TL,TR,BR,BL
    dst = np.float32([[0, 0], [C.CAMPO_W, 0],
                      [C.CAMPO_W, C.CAMPO_H], [0, C.CAMPO_H]])
    H = cv2.getPerspectiveTransform(src, dst)
    return H, calib


def field_mask_camera(calib, shape, pad=50):
    """Mascara (camara) del poligono del campo, para acotar el balon HSV.

    pad: dilata la mascara N px para tolerar el drift de camara handheld
    (el balon sigue dentro aunque la homografia este calibrada en un frame).
    """
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(calib["field_corners"])], 255)
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        m = cv2.dilate(m, k)
    return m


def ball_candidates(frame_bgr, field_mask):
    """Blobs naranjas circulares dentro del campo. Devuelve [(cx,cy,area), ...]."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.bitwise_and(cv2.inRange(hsv, C.BALL_LO, C.BALL_HI), field_mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # CLOSE reconecta el balon partido por motion-blur antes de medir el contorno
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (C.BALL_AREA_MIN < a < C.BALL_AREA_MAX):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if a / (np.pi * r * r + 1e-6) < C.BALL_CIRC_MIN:   # descarta blobs alargados
            continue
        cand.append((float(x), float(y), float(a)))
    return cand


def detect_ball_hsv(frame_bgr, field_mask, last=None, gate=120.0):
    """
    Balon naranja por HSV. Si `last`=(x,y) camara conocido, prefiere el candidato
    mas cercano dentro de `gate` px (rechaza saltos a muñequeras de borde);
    si no hay previo, toma el blob mas grande. Devuelve (cx,cy) o None.
    """
    cand = ball_candidates(frame_bgr, field_mask)
    if not cand:
        return None
    if last is not None:
        near = [(c, (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2) for c in cand]
        near = [(c, d) for c, d in near if d <= gate * gate]
        if near:
            best = min(near, key=lambda cd: cd[1])[0]
            return (best[0], best[1])
    best = max(cand, key=lambda c: c[2])           # sin previo: el mayor
    return (best[0], best[1])


def project_point(pt_cam, H):
    """(x,y) camara -> (x,y) canonico (int)."""
    p = np.float32([[[pt_cam[0], pt_cam[1]]]])
    q = cv2.perspectiveTransform(p, H)
    return int(q[0][0][0]), int(q[0][0][1])


def anchor_from_bbox(bbox, is_ball: bool):
    """Punto a proyectar: balon=centro; robot=bottom-center (toca el piso)."""
    x1, y1, x2, y2 = bbox
    if is_ball:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return ((x1 + x2) / 2, y2)


def in_zone(pt, zone):
    x, y = pt
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]


def draw_field(canvas):
    """Dibuja lineas del campo canonico sobre canvas BGR."""
    g = (120, 200, 116)
    cv2.rectangle(canvas, (0, 0), (C.CAMPO_W - 1, C.CAMPO_H - 1), g, 2)
    cv2.line(canvas, (0, C.CAMPO_H // 2), (C.CAMPO_W, C.CAMPO_H // 2), g, 1)
    cv2.circle(canvas, (C.CAMPO_W // 2, C.CAMPO_H // 2), int(30 * C.ESCALA_PX_CM), g, 1)
    pen_w, pen_h = int(80 * C.ESCALA_PX_CM), int(40 * C.ESCALA_PX_CM)
    px = (C.CAMPO_W - pen_w) // 2
    cv2.rectangle(canvas, (px, 0), (px + pen_w, pen_h), g, 1)
    cv2.rectangle(canvas, (px, C.CAMPO_H - pen_h), (px + pen_w, C.CAMPO_H - 1), g, 1)
    # porterias
    cv2.rectangle(canvas, tuple(C.GOAL_AMARILLA_ZONE[:2]), tuple(C.GOAL_AMARILLA_ZONE[2:]),
                  (0, 214, 255), -1)
    cv2.rectangle(canvas, tuple(C.GOAL_AZUL_ZONE[:2]), tuple(C.GOAL_AZUL_ZONE[2:]),
                  (216, 130, 0), -1)


def draw_tactical(objects, base=None):
    """
    objects: lista {pos_canon:(x,y), kind:'verde'|'oscuro'|'balon', tid:int}
    Devuelve canvas BGR del mapa tactico.
    """
    if base is None:
        canvas = np.full((C.CAMPO_H, C.CAMPO_W, 3), (50, 67, 27), np.uint8)
    else:
        canvas = base.copy()
    draw_field(canvas)
    M = 40   # margen: objetos un poco fuera del campo se dibujan en el borde
    for o in objects:
        x, y = o["pos_canon"]
        if not (-M <= x < C.CAMPO_W + M and -M <= y < C.CAMPO_H + M):
            continue
        x = int(np.clip(x, 4, C.CAMPO_W - 4))
        y = int(np.clip(y, 4, C.CAMPO_H - 4))
        if o["kind"] == "balon":
            cv2.circle(canvas, (x, y), 6, (0, 140, 255), -1)
            cv2.circle(canvas, (x, y), 6, (255, 255, 255), 1)
        else:
            col = C.COLOR_TEAM[o["kind"]]
            cv2.circle(canvas, (x, y), 12, col, -1)
            cv2.circle(canvas, (x, y), 12, (255, 255, 255), 1)
            lbl = "V" if o["kind"] == "verde" else "O"
            cv2.putText(canvas, f"{lbl}{o.get('tid','')}", (x - 9, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas
