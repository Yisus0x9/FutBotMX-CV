"""
Pipeline de analisis Copa FutBotMX sobre IMG_9938.

Por frame: YOLOv8-seg (qa_s) detecta+clasifica equipo -> [SAM3 refina mascara]
-> ByteTrack -> homografia -> mapa tactico. Acumula heatmap, posesion y goles.

Salidas en output/<name>/:
  - analisis.mp4   (camara con mascaras | mapa tactico)
  - heatmap.png    (control por equipo, acumulado)
  - stats.json     (posesion, goles, frames)
  - tracks.csv     (posiciones canonicas por-frame: frame,t_s,kind,tid,x_cm,y_cm)

Uso:
  python run_analysis.py --start 3000 --end 6000              # clip rapido (YOLO-seg)
  python run_analysis.py --start 3000 --end 4000 --sam        # refina con SAM3 (lento)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import config as C
import homography as Hm


def gaussian_splat(grid, x, y, sigma=12, amp=1.0):
    """Suma un blob gaussiano en (x,y) sobre grid (in-place)."""
    h, w = grid.shape
    if not (0 <= x < w and 0 <= y < h):
        return
    r = int(3 * sigma)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    xs = np.arange(x0, x1)[None, :]
    ys = np.arange(y0, y1)[:, None]
    grid[y0:y1, x0:x1] += amp * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--end", type=int, default=6000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sam", action="store_true", help="refinar mascaras con SAM3 (lento)")
    ap.add_argument("--name", default="clip")
    ap.add_argument("--video", default=None, help="ruta de video (default: config.VIDEO)")
    ap.add_argument("--calib", default=None, help="ruta de field_calib json (default: config.CALIB)")
    args = ap.parse_args()

    video_path = Path(args.video) if args.video else C.VIDEO
    calib_path = Path(args.calib) if args.calib else C.CALIB

    out = C.OUT / args.name
    out.mkdir(parents=True, exist_ok=True)

    H, calib = Hm.load_homography(calib_path)
    # H de alta-res para warpear la imagen real a vista cenital (display)
    Hd = C.DISPLAY_H
    Wd = int(C.CAMPO_W / C.CAMPO_H * Hd)
    sx, sy = Wd / C.CAMPO_W, Hd / C.CAMPO_H
    H_warp = cv2.getPerspectiveTransform(
        np.float32(calib["field_corners"]),
        np.float32([[0, 0], [Wd, 0], [Wd, Hd], [0, Hd]]))
    model = YOLO(str(C.WEIGHTS))
    names = model.names
    tracker = sv.ByteTrack()

    sam = None
    if args.sam:
        from ultralytics import SAM
        sam = SAM(str(C.SAM3_PT))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end = min(args.end, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # colores por clase (orden model.names: 0=balon,1=robot_oscuro,2=robot_verde)
    pal = sv.ColorPalette.from_hex(["#ff8c00", "#323232", "#00c800"])
    mask_ann = sv.MaskAnnotator(color=pal)
    box_ann = sv.BoxAnnotator(color=pal)

    # acumuladores
    heat = {"verde": np.zeros((C.CAMPO_H, C.CAMPO_W), np.float32),
            "oscuro": np.zeros((C.CAMPO_H, C.CAMPO_W), np.float32)}
    possession = {"verde": 0, "oscuro": 0}
    goals = []
    in_goal_prev = None
    records = []   # data por-frame para analitica/visualizacion
    last_ball_cam = None       # ultima posicion del balon (camara) para gating
    ball_lost = 0

    writer = None
    fmask = None
    n = 0
    fi = args.start
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    while fi < end:
        ok, frame = cap.read()
        if not ok:
            break
        if fmask is None:
            fmask = Hm.field_mask_camera(calib, frame.shape, pad=30)
        if (fi - args.start) % args.stride != 0:
            fi += 1
            continue

        res = model(frame, conf=C.CONF, imgsz=C.IMGSZ, agnostic_nms=True, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        det = tracker.update_with_detections(det)

        # SAM refina mascaras (opcional)
        if sam is not None and len(det) > 0:
            sres = sam(frame, bboxes=det.xyxy.tolist(), verbose=False)[0]
            sdet = sv.Detections.from_ultralytics(sres)
            if sdet.mask is not None and len(sdet) == len(det):
                det.mask = sdet.mask

        # proyectar + clasificar (robots por YOLO); guarda balones YOLO como fallback
        robots, ball = [], None
        yolo_balls = []
        for i in range(len(det)):
            nm = names[int(det.class_id[i])]
            if nm == "balon":
                yolo_balls.append(Hm.anchor_from_bbox(det.xyxy[i], True))  # centro
                continue   # el balon se detecta por HSV (mas fiable)
            pc = Hm.project_point(Hm.anchor_from_bbox(det.xyxy[i], False), H)
            tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
            kind = C.TEAM_OF.get(nm, "oscuro")
            robots.append({"pos_canon": pc, "kind": kind, "tid": tid})
            gaussian_splat(heat[kind], pc[0], pc[1])

        # balon por HSV (primario) con gating temporal (rechaza FP de borde)
        bc = Hm.detect_ball_hsv(frame, fmask, last=last_ball_cam, gate=120.0)
        # fallback: si HSV falla, usar balon de YOLO (gated al ultimo conocido)
        if bc is None and yolo_balls:
            if last_ball_cam is not None:
                cand = [(b, (b[0] - last_ball_cam[0]) ** 2 + (b[1] - last_ball_cam[1]) ** 2)
                        for b in yolo_balls]
                cand = [b for b, d in cand if d <= 120.0 ** 2]
                bc = min(cand, key=lambda b: (b[0] - last_ball_cam[0]) ** 2
                         + (b[1] - last_ball_cam[1]) ** 2) if cand else None
            else:
                bc = yolo_balls[0]
        if bc is not None:
            ball = {"pos_canon": Hm.project_point(bc, H), "kind": "balon"}
            last_ball_cam, ball_lost = bc, 0
        else:
            ball_lost += 1
            if ball_lost > 15:        # reacquire: permite nuevo blob mayor
                last_ball_cam = None

        # log por-frame (posiciones canonicas -> cm) para analitica
        t_s = round(fi / fps, 3)
        for r in robots:
            records.append((fi, t_s, r["kind"], r["tid"],
                            round(r["pos_canon"][0] / C.ESCALA_PX_CM, 2),
                            round(r["pos_canon"][1] / C.ESCALA_PX_CM, 2)))
        if ball is not None:
            records.append((fi, t_s, "balon", -1,
                            round(ball["pos_canon"][0] / C.ESCALA_PX_CM, 2),
                            round(ball["pos_canon"][1] / C.ESCALA_PX_CM, 2)))

        # posesion: equipo del robot mas cercano al balon
        if ball is not None and robots:
            bx, by = ball["pos_canon"]
            nearest = min(robots, key=lambda r: (r["pos_canon"][0] - bx) ** 2 + (r["pos_canon"][1] - by) ** 2)
            possession[nearest["kind"]] += 1

        # gol: balon dentro de zona de porteria (con debounce)
        cur = None
        if ball is not None:
            if Hm.in_zone(ball["pos_canon"], C.GOAL_AMARILLA_ZONE):
                cur = "amarilla"
            elif Hm.in_zone(ball["pos_canon"], C.GOAL_AZUL_ZONE):
                cur = "azul"
        if cur and cur != in_goal_prev:
            goals.append({"frame": fi, "porteria": cur, "t_s": round(fi / fps, 1)})
        in_goal_prev = cur

        # render
        objs = robots + ([ball] if ball else [])
        # panel 1: camara con mascaras/boxes por equipo
        cam = frame.copy()
        if det.mask is not None:
            cam = mask_ann.annotate(cam, det)
        cam = box_ann.annotate(cam, det)
        cam_r = cv2.resize(cam, (int(cam.shape[1] * Hd / cam.shape[0]), Hd))
        # panel 2: cenital REAL (warp) + marcadores de tracking encima
        warp = cv2.warpPerspective(frame, H_warp, (Wd, Hd))
        for o in objs:
            x, y = int(o["pos_canon"][0] * sx), int(o["pos_canon"][1] * sy)
            if o["kind"] == "balon":
                cv2.circle(warp, (x, y), 8, (0, 140, 255), 2)
            else:
                col = C.COLOR_TEAM[o["kind"]]
                cv2.circle(warp, (x, y), 16, col, 2)
        # panel 3: mapa tactico esquematico
        tac = cv2.resize(Hm.draw_tactical(objs), (Wd, Hd), interpolation=cv2.INTER_NEAREST)
        combo = np.hstack([cam_r, warp, tac])

        if writer is None:
            writer = cv2.VideoWriter(str(out / "analisis.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        n += 1
        fi += 1

    cap.release()
    if writer:
        writer.release()

    # heatmap PNG (verde vs oscuro sobre campo)
    base = np.full((C.CAMPO_H, C.CAMPO_W, 3), (50, 67, 27), np.uint8)
    Hm.draw_field(base)
    hm_img = base.copy()
    for kind, cmap in [("verde", cv2.COLORMAP_SUMMER), ("oscuro", cv2.COLORMAP_HOT)]:
        g = heat[kind]
        if g.max() > 0:
            gn = (g / g.max() * 255).astype(np.uint8)
            col = cv2.applyColorMap(gn, cmap)
            mask = gn > 10
            hm_img[mask] = cv2.addWeighted(hm_img, 0.3, col, 0.7, 0)[mask]
    cv2.imwrite(str(out / "heatmap.png"), hm_img)

    tot = possession["verde"] + possession["oscuro"]
    stats = {
        "frames": n, "rango": [args.start, end], "sam": args.sam,
        "posesion_pct": {k: round(100 * v / tot, 1) if tot else 0 for k, v in possession.items()},
        "goles": goals,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # tracks.csv: posiciones canonicas por-frame (cm) para visualize.py
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "kind", "tid", "x_cm", "y_cm"])
        w.writerows(records)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"tracks.csv: {len(records)} filas")
    print("Salidas en:", out)


if __name__ == "__main__":
    main()
