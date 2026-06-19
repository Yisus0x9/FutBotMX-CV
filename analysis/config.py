"""
Config del pipeline de analisis (Copa FutBotMX) sobre IMG_9938.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipelines"
SAM3_PT = PIPE / "sam3.pt"
WEIGHTS = PIPE / "training" / "runs" / "qa_s" / "weights" / "best.pt"
VIDEO = ROOT / "Camara_superior" / "IMG_9938.MOV.mp4"
CALIB = PIPE / "dataset_generation" / "output" / "field_calib_analisis.json"

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Campo canonico — RCJ Soccer Field 2023: 182 x 243 cm a 2 px/cm
ESCALA_PX_CM = 2.0
CAMPO_W, CAMPO_H = 364, 486

# Clases (se leen de model.names en runtime; este mapeo es de respaldo)
# Roboflow order: 0=balon, 1=robot_oscuro, 2=robot_verde
TEAM_OF = {"robot_verde": "verde", "robot_oscuro": "oscuro"}
COLOR_BGR = {"robot_verde": (0, 200, 0), "robot_oscuro": (50, 50, 50),
             "balon": (0, 140, 255)}
COLOR_TEAM = {"verde": (0, 200, 0), "oscuro": (50, 50, 50)}

# Zonas de gol en el campo canonico (franjas en cada porteria)
# amarilla = arriba (y~0), azul = abajo (y~CAMPO_H). Ancho de boca ~60cm.
GOAL_W_PX = int(60 * ESCALA_PX_CM)
GOAL_DEPTH_PX = int(12 * ESCALA_PX_CM)
GOAL_X0 = (CAMPO_W - GOAL_W_PX) // 2
GOAL_AMARILLA_ZONE = [GOAL_X0, 0, GOAL_X0 + GOAL_W_PX, GOAL_DEPTH_PX]
GOAL_AZUL_ZONE = [GOAL_X0, CAMPO_H - GOAL_DEPTH_PX, GOAL_X0 + GOAL_W_PX, CAMPO_H]

CONF = 0.15          # umbral deteccion (bajo para no perder oscuros)
IMGSZ = 960
DISPLAY_H = 720      # alto de los paneles de salida

# Balon por HSV (naranja) — mas fiable que YOLO para objeto pequeno
BALL_LO = np.array([0, 90, 80])
BALL_HI = np.array([18, 255, 255])   # hue<18: naranja, evita amarillo porteria (~22-26)
BALL_AREA_MIN = 12
BALL_AREA_MAX = 1600                  # alto: tolera motion-blur (balon se estira al moverse)
BALL_CIRC_MIN = 0.35                  # circularidad min (baja para aceptar balon borroso)
