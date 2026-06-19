"""
Escanea IMG_9938 buscando eventos de gol (balon naranja dentro de zona de
porteria) y actividad de balon. Rapido: solo HSV + homografia (sin modelo).
Sirve para elegir el clip del video demo.

Uso: python scan_goals.py [--stride 3]
"""
import argparse
import cv2
import numpy as np
import config as C
import homography as Hm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    H, calib = Hm.load_homography()
    cap = cv2.VideoCapture(str(C.VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    fmask = Hm.field_mask_camera(calib, f0.shape)

    goals, ball_frames = [], []
    prev_zone = None
    fi = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while fi < total:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.stride == 0:
            bc = Hm.detect_ball_hsv(frame, fmask)
            if bc is not None:
                pc = Hm.project_point(bc, H)
                ball_frames.append(fi)
                z = ("amarilla" if Hm.in_zone(pc, C.GOAL_AMARILLA_ZONE)
                     else "azul" if Hm.in_zone(pc, C.GOAL_AZUL_ZONE) else None)
                if z and z != prev_zone:
                    goals.append((fi, z, round(fi / fps, 1)))
                prev_zone = z
            else:
                prev_zone = None
        fi += 1
    cap.release()

    print(f"Total frames: {total}  ({total/fps:.0f}s)  | frames con balon: {len(ball_frames)}")
    print(f"\nEVENTOS DE GOL ({len(goals)}):")
    for fr, z, t in goals:
        print(f"  frame {fr:6d}  t={t:6.1f}s  porteria {z}")

    # segmentos con mas actividad de balon (ventanas de 600 frames)
    print("\nACTIVIDAD de balon por ventana de 20s (frames con balon):")
    win = 600
    bf = np.array(ball_frames)
    for w0 in range(0, total, win):
        c = int(((bf >= w0) & (bf < w0 + win)).sum())
        if c > 20:
            print(f"  {w0:6d}-{w0+win:6d}  (t {w0/fps:.0f}-{(w0+win)/fps:.0f}s): {c} frames balon")


if __name__ == "__main__":
    main()
