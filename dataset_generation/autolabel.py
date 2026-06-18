"""
Auto-etiquetado de frames para el dataset (Copa FutBotMX).

Por cada frame muestreado:
  1. filtra frames con manos (piel)
  2. descubre robots  (--discover sam | bgsub)
  3. filtra: dentro del campo + rango de area
  4. clasifica equipo por color (verde / oscuro)
  5. detecta balon (HSV naranja)
  6. ByteTrack -> IDs persistentes entre frames
  7. guarda anotacion (json con poligonos) + overlay de control

Salida cruda en output/labels_raw/ y output/vis/.
Luego export_roboflow.py convierte a YOLOv8-seg.

Uso:
    python autolabel.py --discover bgsub --max 20        # validar en CPU
    python autolabel.py --discover sam                   # real (requiere GPU)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # evita OMP Error #15
import argparse
import json
import cv2
import numpy as np
import supervision as sv
import config as C
import detect as D


def mask_to_polygon(mask: np.ndarray) -> list | None:
    """Contorno exterior mayor como lista [x1,y1,x2,y2,...]."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    eps = 0.01 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)
    return c.reshape(-1).astype(int).tolist()


def build_detections(objs: list[dict], shape) -> sv.Detections:
    """objs: lista {mask,bbox,class_id}. Crea sv.Detections con conf=1."""
    if not objs:
        return sv.Detections.empty()
    xyxy = np.array([o["bbox"] for o in objs], dtype=np.float32)
    masks = np.stack([o["mask"] for o in objs]).astype(bool)
    cid = np.array([o["class_id"] for o in objs], dtype=int)
    conf = np.ones(len(objs), dtype=np.float32)
    return sv.Detections(xyxy=xyxy, mask=masks, class_id=cid, confidence=conf)


# ---- descubrimiento SAM 3 (texto) ----
class SamDiscoverer:
    def __init__(self):
        from ultralytics.models.sam import SAM3SemanticPredictor
        import torch
        ov = dict(conf=C.SAM_CONF, task="segment", mode="predict",
                  model=str(C.SAM3_PT))
        if torch.cuda.is_available():
            ov["device"] = 0
            ov["half"] = True
        self.predictor = SAM3SemanticPredictor(overrides=ov)

    def __call__(self, frame_bgr) -> list[dict]:
        self.predictor.set_image(frame_bgr)
        res = self.predictor(text=C.SAM_TEXT_PROMPTS)[0]
        det = sv.Detections.from_ultralytics(res)
        out = []
        if det.mask is None or len(det) == 0:
            return out
        # dedup: varios prompts ("robot","small robot") detectan el mismo robot.
        # class_agnostic porque cada prompt asigna class_id distinto.
        det = det.with_nms(threshold=0.5, class_agnostic=True)
        for i in range(len(det)):
            m = det.mask[i].astype(bool)
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            out.append({"mask": m,
                        "bbox": [int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())],
                        "centroid": [float(xs.mean()), float(ys.mean())]})
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", choices=["sam", "bgsub"], default="bgsub")
    ap.add_argument("--max", type=int, default=None, help="limitar #frames (test)")
    ap.add_argument("--stride", type=int, default=C.SAMPLE_STRIDE)
    ap.add_argument("--video", choices=["ds", "an"], default="ds",
                    help="ds=IMG_9933 | an=IMG_9938")
    args = ap.parse_args()

    video = C.VIDEO_DS if args.video == "ds" else C.VIDEO_AN
    calib_path = C.FIELD_CALIB if args.video == "ds" else C.OUT / "field_calib_analisis.json"
    vstem = video.stem.split(".")[0]      # IMG_9933 / IMG_9938
    calib = json.loads(calib_path.read_text())
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    field_mask = D.make_field_mask(calib, frame0.shape)

    discoverer = None
    background = None
    if args.discover == "sam":
        discoverer = SamDiscoverer()
    else:
        print("Construyendo fondo (mediana)...")
        background = D.build_background(video, n=40)

    tracker = sv.ByteTrack()
    idxs = list(range(C.FRAME_START, C.FRAME_END or total, args.stride))
    if args.max:
        idxs = idxs[:args.max]

    n_saved, n_skip_hands = 0, 0
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        if D.hand_blob_area(frame) > C.HAND_BLOB_REJECT:
            n_skip_hands += 1
            continue

        # 1. descubrir robots
        if args.discover == "sam":
            cands = discoverer(frame)
        else:
            cands = D.discover_robots_bgsub(frame, background, field_mask)

        # 2. filtrar + clasificar equipo
        objs = []
        for r in cands:
            cx, cy = r["centroid"]
            if not D.on_field(cx, cy, field_mask):
                continue
            area = int(r["mask"].sum())
            if not (C.ROBOT_AREA_MIN < area < C.ROBOT_AREA_MAX):
                continue
            cid, _ = D.classify_team(frame, r["mask"])
            objs.append({**r, "class_id": cid})

        # 3. balon
        ball = D.detect_ball(frame, field_mask)
        if ball:
            objs.append({**ball, "class_id": C.CID["balon"]})

        # 4. tracking
        dets = build_detections(objs, frame.shape)
        dets = tracker.update_with_detections(dets)

        # 5. guardar anotacion
        ann = {"frame": int(fi), "video": str(video), "vstem": vstem,
               "img_size": [frame.shape[1], frame.shape[0]], "objects": []}
        for i in range(len(dets)):
            poly = mask_to_polygon(dets.mask[i]) if dets.mask is not None else None
            if poly is None:
                continue
            ann["objects"].append({
                "class_id": int(dets.class_id[i]),
                "class_name": C.CLASSES[int(dets.class_id[i])],
                "tracker_id": int(dets.tracker_id[i]) if dets.tracker_id is not None else -1,
                "bbox": [float(x) for x in dets.xyxy[i]],
                "polygon": poly,
            })
        (C.DIR_LABELS / f"{vstem}_frame_{fi:06d}.json").write_text(json.dumps(ann))
        save_overlay(frame, dets, f"{vstem}_{fi:06d}")
        n_saved += 1

    cap.release()
    print(f"Frames guardados: {n_saved} | saltados por manos: {n_skip_hands}")
    print(f"Anotaciones: {C.DIR_LABELS}")
    print(f"Overlays:    {C.DIR_VIS}")


def save_overlay(frame, dets, tag):
    colors = {0: (0, 200, 0), 1: (40, 40, 40), 2: (0, 140, 255)}
    vis = frame.copy()
    if dets.mask is not None:
        for i in range(len(dets)):
            cid = int(dets.class_id[i])
            col = colors.get(cid, (255, 255, 255))
            m = dets.mask[i]
            overlay = vis.copy()
            overlay[m] = col
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            x1, y1, x2, y2 = dets.xyxy[i].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            tid = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
            cv2.putText(vis, f"{C.CLASSES[cid]}#{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(str(C.DIR_VIS / f"label_{tag}.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
