"""
Exporta las anotaciones crudas (output/labels_raw/*.json) a un dataset
YOLOv8-seg listo para subir a Roboflow (QA/correccion) y entrenar.

Estructura generada:
    output/dataset/
        data.yaml
        images/{train,val}/frame_XXXXXX.jpg
        labels/{train,val}/frame_XXXXXX.txt   (poligonos normalizados)

Formato de etiqueta YOLOv8-seg por linea:
    class_id x1 y1 x2 y2 ... xn yn      (coords normalizadas 0-1)

Uso:
    python export_roboflow.py --val 0.2
"""
import argparse
import json
import random
import shutil
import cv2
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.2, help="fraccion validacion")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anns = sorted(C.DIR_LABELS.glob("*frame_*.json"))
    if not anns:
        raise SystemExit("No hay anotaciones en labels_raw/. Corre autolabel.py primero.")

    # reset dataset dir
    if C.DIR_DATASET.exists():
        shutil.rmtree(C.DIR_DATASET)
    for split in ("train", "val"):
        (C.DIR_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (C.DIR_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(anns)
    n_val = int(len(anns) * args.val)
    val_set = set(anns[:n_val])

    caps = {}   # cache de VideoCapture por ruta de video
    n_train = n_kept = 0
    for ann_path in anns:
        ann = json.loads(ann_path.read_text())
        objs = ann.get("objects", [])
        if not objs:
            continue
        fi = ann["frame"]
        W, H = ann["img_size"]
        vpath = ann.get("video", str(C.VIDEO_DS))       # compat anotaciones viejas
        vstem = ann.get("vstem", "IMG_9933")
        split = "val" if ann_path in val_set else "train"

        if vpath not in caps:
            caps[vpath] = cv2.VideoCapture(vpath)
        cap = caps[vpath]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        stem = f"{vstem}_frame_{fi:06d}"
        cv2.imwrite(str(C.DIR_DATASET / "images" / split / f"{stem}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        lines = []
        for o in objs:
            poly = o["polygon"]
            if not poly or len(poly) < 6:        # min 3 puntos
                continue
            norm = []
            for j in range(0, len(poly), 2):
                norm.append(f"{poly[j] / W:.6f}")
                norm.append(f"{poly[j + 1] / H:.6f}")
            lines.append(f"{o['class_id']} " + " ".join(norm))
        (C.DIR_DATASET / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n_kept += 1
        n_train += (split == "train")
    for c in caps.values():
        c.release()

    data_yaml = (
        f"path: {C.DIR_DATASET.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(C.CLASSES)}\n"
        f"names: {C.CLASSES}\n"
    )
    (C.DIR_DATASET / "data.yaml").write_text(data_yaml)

    print(f"Dataset exportado: {C.DIR_DATASET}")
    print(f"  total={n_kept}  train={n_train}  val={n_kept - n_train}")
    print(f"  clases={C.CLASSES}")
    print("Sube la carpeta a Roboflow (formato YOLOv8) para QA/correccion.")


if __name__ == "__main__":
    main()
