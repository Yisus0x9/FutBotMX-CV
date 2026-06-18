"""
Perfilado de colores reales del video cenital (IMG_9933).

Mide rangos HSV de: campo (verde), balon (naranja), porteria amarilla,
porteria azul. Sirve para calibrar los filtros HSV del generador de dataset
con numeros reales, no a ojo.

Uso:
    python tools/profile_colors.py
"""
import cv2
import numpy as np
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[3] / "Camara_superior" / "IMG_9933.MOV.mp4"
SAMPLE_FRAMES = [300, 900, 1500, 2400, 3000, 3600]


def hsv_stats(hsv_pixels: np.ndarray) -> dict:
    """Percentiles 5-95 por canal para definir rangos robustos."""
    if len(hsv_pixels) == 0:
        return {}
    lo = np.percentile(hsv_pixels, 5, axis=0).astype(int)
    hi = np.percentile(hsv_pixels, 95, axis=0).astype(int)
    med = np.median(hsv_pixels, axis=0).astype(int)
    return {"lo": lo.tolist(), "med": med.tolist(), "hi": hi.tolist(), "n": len(hsv_pixels)}


def dominant_region(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Devuelve los pixeles HSV donde mask es True."""
    return hsv[mask.astype(bool)]


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"No abre: {VIDEO}")

    acc = {"campo": [], "balon": [], "goal_amarilla": [], "goal_azul": []}

    for fr in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, W = img.shape[:2]

        # CAMPO: verde dominante. Buscar pixeles verdes amplios.
        green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([95, 255, 255]))
        # quedarse con la componente grande (centro de la imagen)
        cy0, cy1 = int(H * 0.35), int(H * 0.65)
        cx0, cx1 = int(W * 0.35), int(W * 0.65)
        center_green = np.zeros_like(green)
        center_green[cy0:cy1, cx0:cx1] = green[cy0:cy1, cx0:cx1]
        acc["campo"].append(dominant_region(hsv, center_green))

        # BALON: naranja saturado (pequeno). Rango amplio inicial.
        ball = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([30, 255, 255]))
        ball = cv2.morphologyEx(ball, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cnts, _ = cv2.findContours(ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # balon = blob pequeno y compacto (area 20-600 px), circular
        for c in cnts:
            a = cv2.contourArea(c)
            if 15 < a < 800:
                m = np.zeros_like(ball)
                cv2.drawContours(m, [c], -1, 255, -1)
                acc["balon"].append(dominant_region(hsv, m))

        # PORTERIA AMARILLA: arriba (primeras filas), amarillo
        top = np.zeros((H, W), np.uint8)
        top[0:int(H * 0.20), :] = 255
        yel = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([40, 255, 255]))
        yel = cv2.bitwise_and(yel, top)
        acc["goal_amarilla"].append(dominant_region(hsv, yel))

        # PORTERIA AZUL: abajo, azul
        bot = np.zeros((H, W), np.uint8)
        bot[int(H * 0.80):, :] = 255
        blu = cv2.inRange(hsv, np.array([95, 80, 60]), np.array([130, 255, 255]))
        blu = cv2.bitwise_and(blu, bot)
        acc["goal_azul"].append(dominant_region(hsv, blu))

    cap.release()

    print(f"Video: {VIDEO.name}  frames={SAMPLE_FRAMES}\n")
    print("Rangos HSV (OpenCV: H 0-179, S 0-255, V 0-255)")
    print("formato: [H,S,V]  lo=p5  med=mediana  hi=p95\n")
    for k, lst in acc.items():
        pix = np.vstack([a for a in lst if len(a)]) if any(len(a) for a in lst) else np.empty((0, 3))
        s = hsv_stats(pix)
        if s:
            print(f"  {k:16s} n={s['n']:7d}  lo={s['lo']}  med={s['med']}  hi={s['hi']}")
        else:
            print(f"  {k:16s} SIN PIXELES (ajustar rango inicial)")


if __name__ == "__main__":
    main()
