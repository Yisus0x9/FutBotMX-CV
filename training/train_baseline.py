"""
Entrenamiento baseline YOLOv8-seg sobre el dataset auto-etiquetado (9933).

Detector de segmentacion: robot_verde, robot_oscuro, balon.
Valida la Fase B end-to-end. Reentrenar luego con dataset corregido en Roboflow.

Uso:
    python train_baseline.py
    python train_baseline.py --model yolov8s-seg.pt --epochs 120 --imgsz 1024
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
from pathlib import Path
from ultralytics import YOLO

DATA = Path(__file__).resolve().parents[1] / "dataset_generation" / "output" / "dataset" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--data", default=str(DATA), help="ruta a data.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--plots", action="store_true", help="generar plots (puede crashear con poca RAM)")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"No existe {data}.")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=0,
        project=str(Path(__file__).resolve().parent / "runs"),
        name=args.name,
        exist_ok=True,
        plots=args.plots,
        workers=4,
        # robots/balon son pequenos -> aug moderada, sin volteo vertical
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        verbose=True,
    )
    metrics = model.val(data=str(data), device=0)
    print("\n=== METRICAS (val) ===")
    print(f"  mask mAP50    : {metrics.seg.map50:.3f}")
    print(f"  mask mAP50-95 : {metrics.seg.map:.3f}")
    print(f"  box  mAP50    : {metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
