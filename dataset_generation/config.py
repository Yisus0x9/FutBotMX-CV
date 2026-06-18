"""
Configuracion central del generador de dataset (Copa FutBotMX - vision).

Fuente: video cenital IMG_9933 (1080x1920, 30 fps).
Rangos HSV medidos con tools/profile_colors.py sobre frames reales.
OpenCV HSV: H 0-179, S 0-255, V 0-255.
"""
from pathlib import Path
import numpy as np

# ---- Rutas ----
ROOT      = Path(__file__).resolve().parents[2]          # C:/Files/Cursos/ROBOT-VISION
PIPE      = ROOT / "pipelines"
SAM3_PT   = PIPE / "sam3.pt"
VIDEO_DS  = ROOT / "Camara_superior" / "IMG_9933.MOV.mp4"  # video para DATASET
VIDEO_AN  = ROOT / "Camara_superior" / "IMG_9938.MOV.mp4"  # video para ANALISIS/demo

OUT       = Path(__file__).resolve().parent / "output"
DIR_FRAMES = OUT / "frames"          # frames muestreados
DIR_VIS    = OUT / "vis"             # overlays de control
DIR_LABELS = OUT / "labels_raw"      # anotaciones intermedias (json por frame)
DIR_DATASET = OUT / "dataset"        # export final YOLOv8-seg / Roboflow
for d in (OUT, DIR_FRAMES, DIR_VIS, DIR_LABELS, DIR_DATASET):
    d.mkdir(parents=True, exist_ok=True)

FIELD_CALIB = OUT / "field_calib.json"   # esquinas campo + zonas porteria

# ---- Muestreo de frames ----
SAMPLE_STRIDE = 10        # 1 de cada 10 frames (~3 fps a 30fps)
FRAME_START   = 0
FRAME_END     = None      # None = hasta el final

# ---- Clases del dataset (instance segmentation) ----
CLASSES = ["robot_verde", "robot_oscuro", "balon"]
CID = {n: i for i, n in enumerate(CLASSES)}

# ---- HSV: CAMPO (verde teal, muy saturado) ----
# Se usa para enmascarar el campo dentro de las mascaras de robot.
FIELD_LO = np.array([78, 150,  40])
FIELD_HI = np.array([90, 255, 200])

# ---- HSV: BALON (naranja) ----
BALL_LO = np.array([0, 90, 80])
BALL_HI = np.array([25, 255, 255])
BALL_AREA_MIN = 15        # px
BALL_AREA_MAX = 900

# ---- HSV: PORTERIAS (estaticas) ----
GOAL_YELLOW_LO = np.array([18, 100, 100])   # amarilla (arriba)
GOAL_YELLOW_HI = np.array([30, 255, 255])
GOAL_BLUE_LO   = np.array([95,  70,  50])   # azul (abajo)
GOAL_BLUE_HI   = np.array([115, 230, 170])

# ---- HSV: PIEL (manos, para filtrar frames de colocacion) ----
# Gate suave: solo se rechaza un frame si hay un BLOB de piel grande (brazo
# cruzando el campo). Robot/sombra producen piel-ruido pequeno -> no rechaza.
SKIN_LO = np.array([0,  40,  90])
SKIN_HI = np.array([20, 140, 235])
HAND_BLOB_REJECT = 40000   # px del mayor blob de piel -> frame con manos

# ---- Clasificacion de equipo (sobre pixeles del robot, excluyendo campo) ----
# verde:  hue mediano alto (~55-80) | oscuro: hue mediano bajo (~15-35)
TEAM_HUE_THRESH   = 45    # median_hue >= -> robot_verde ; < -> robot_oscuro
ROBOT_GREEN_LO = np.array([35,  40,  50])   # verde-PCB dentro del robot
ROBOT_GREEN_HI = np.array([85, 255, 255])

# ---- Filtro de detecciones de robot ----
ROBOT_AREA_MIN = 1500     # px (mascara) - ajustar tras ver tamano real
ROBOT_AREA_MAX = 60000

# ---- Descubrimiento SAM 3 (texto) ----
SAM_TEXT_PROMPTS = ["robot", "small robot", "circuit board robot"]
SAM_CONF = 0.25
