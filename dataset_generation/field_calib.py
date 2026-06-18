"""
Calibracion de campo y porterias (cámara fija -> se calcula UNA vez).

Detecta automaticamente:
  - cuadrilatero del campo verde (4 esquinas) -> on-field filter + homografia
  - zona de porteria amarilla (arriba) y azul (abajo) -> deteccion de gol

Guarda output/field_calib.json y un overlay de control en output/vis/.

Uso:
    python field_calib.py [--frame N]
"""
import argparse
import json
import cv2
import numpy as np
import config as C


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como TL, TR, BR, BL."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL (x+y min)
        pts[np.argmin(d)],   # TR (x-y max -> -d min)
        pts[np.argmax(s)],   # BR (x+y max)
        pts[np.argmax(d)],   # BL (x-y min)
    ], dtype=np.float32)


def detect_field(hsv: np.ndarray) -> np.ndarray:
    """
    Cuadrilatero del campo (4 esquinas, orden TL,TR,BR,BL) por deteccion
    de lineas blancas dentro de la mayor region verde.
    Candidato automatico; refinar con --click si se necesita precision.
    """
    H, W = hsv.shape[:2]
    green = cv2.inRange(hsv, np.array([72, 90, 30]), np.array([95, 255, 230]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No se detecto campo verde")
    field_mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(field_mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

    white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 70, 255]))
    white = cv2.bitwise_and(white, field_mask)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ys, xs = np.where(white > 0)
    if len(xs) < 100:
        raise RuntimeError("Pocas lineas blancas detectadas")
    pts = np.column_stack([xs, ys]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return order_quad(box)


def click_corners(img: np.ndarray) -> np.ndarray:
    """Selector interactivo: clic en 4 esquinas del campo (TL,TR,BR,BL)."""
    import matplotlib.pyplot as plt
    pts = []
    fig, ax = plt.subplots(figsize=(7, 12))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Clic 4 esquinas linea blanca: TL -> TR -> BR -> BL")
    line, = ax.plot([], [], "o-", color="#ffd60a", ms=10, lw=2)

    def on_click(ev):
        if ev.inaxes != ax or len(pts) >= 4:
            return
        pts.append([ev.xdata, ev.ydata])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if len(pts) == 4:
            xs.append(pts[0][0]); ys.append(pts[0][1])
        line.set_data(xs, ys); fig.canvas.draw()
        if len(pts) == 4:
            print("4 esquinas capturadas:", [[int(a), int(b)] for a, b in pts])

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    if len(pts) != 4:
        raise SystemExit("Se necesitan 4 clics")
    return order_quad(np.array(pts, dtype=np.float32))


def detect_goal(hsv: np.ndarray, lo, hi, region: str) -> list | None:
    """Bounding box de una porteria por color, restringida a region top/bottom."""
    H, W = hsv.shape[:2]
    band = np.zeros((H, W), np.uint8)
    if region == "top":
        band[0:int(H * 0.30), :] = 255
    else:
        band[int(H * 0.70):, :] = 255
    m = cv2.bitwise_and(cv2.inRange(hsv, lo, hi), band)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 800]
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return [int(x), int(y), int(x + w), int(y + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=600,
                    help="frame de referencia (sin manos, robots en juego)")
    ap.add_argument("--click", action="store_true",
                    help="seleccionar esquinas del campo manualmente (preciso)")
    ap.add_argument("--video", choices=["ds", "an"], default="ds",
                    help="ds=IMG_9933 (dataset) | an=IMG_9938 (analisis)")
    args = ap.parse_args()

    video = C.VIDEO_DS if args.video == "ds" else C.VIDEO_AN
    out_json = C.FIELD_CALIB if args.video == "ds" else C.OUT / "field_calib_analisis.json"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("No se pudo leer el frame de referencia")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    field = click_corners(img) if args.click else detect_field(hsv)
    goal_y = detect_goal(hsv, C.GOAL_YELLOW_LO, C.GOAL_YELLOW_HI, "top")
    goal_b = detect_goal(hsv, C.GOAL_BLUE_LO, C.GOAL_BLUE_HI, "bottom")

    calib = {
        "video": video.name,
        "ref_frame": args.frame,
        "img_size": [int(img.shape[1]), int(img.shape[0])],
        "field_corners": field.tolist(),       # TL,TR,BR,BL en px camara
        "goal_amarilla": goal_y,               # xyxy o None
        "goal_azul": goal_b,
    }
    out_json.write_text(json.dumps(calib, indent=2))
    print("Guardado:", out_json)
    print(json.dumps(calib, indent=2))

    # Overlay de control
    vis = img.copy()
    fp = field.astype(int)
    cv2.polylines(vis, [fp], True, (0, 255, 255), 3)
    for i, p in enumerate(fp):
        cv2.circle(vis, tuple(p), 12, (0, 0, 255), -1)
        cv2.putText(vis, "TL TR BR BL".split()[i], tuple(p + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if goal_y:
        cv2.rectangle(vis, goal_y[:2], goal_y[2:], (0, 255, 255), 3)
        cv2.putText(vis, "GOAL_AMARILLA", (goal_y[0], goal_y[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if goal_b:
        cv2.rectangle(vis, goal_b[:2], goal_b[2:], (255, 120, 0), 3)
        cv2.putText(vis, "GOAL_AZUL", (goal_b[0], goal_b[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
    out = C.DIR_VIS / f"field_calib_{args.video}.jpg"
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("Overlay:", out)


if __name__ == "__main__":
    main()
