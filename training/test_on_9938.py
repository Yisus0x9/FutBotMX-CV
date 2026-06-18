"""
Prueba un detector entrenado sobre el video de ANALISIS (9938).
Lee nombres de clase del modelo (robusto a cambios de orden Roboflow).

Uso:
    python test_on_9938.py --weights runs/qa_s/weights/best.pt --name qa_s
    python test_on_9938.py --weights runs/baseline/weights/best.pt --name baseline
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "Camara_superior" / "IMG_9938.MOV.mp4"

# color por NOMBRE de clase (no por indice)
COLORS = {"robot_verde": (0, 200, 0), "robot_oscuro": (40, 40, 40),
          "balon": (0, 140, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "runs" / f"test_9938_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    names = model.names
    print("clases del modelo:", names)

    cap = cv2.VideoCapture(str(VIDEO))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.linspace(2000, N - 2000, 6).astype(int)
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, conf=args.conf, imgsz=960, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        vis = frame.copy()
        labs = []
        if det.mask is not None:
            for i in range(len(det)):
                nm = names[int(det.class_id[i])]
                labs.append(nm)
                col = COLORS.get(nm, (255, 255, 255))
                m = det.mask[i]
                ov = vis.copy(); ov[m] = col
                vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
                x1, y1, x2, y2 = det.xyxy[i].astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis, f"{nm} {det.confidence[i]:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(str(out / f"test_{int(fi):06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"frame {fi}: {len(det)} -> {labs}")
    cap.release()
    print("Overlays:", out)


if __name__ == "__main__":
    main()
