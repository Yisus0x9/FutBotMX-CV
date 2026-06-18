# Generación de dataset — Copa FutBotMX (Visión por Computadora)

Auto-etiquetado de un video cenital de fútbol robótico para producir un dataset
de segmentación (robots por equipo + balón) listo para QA en Roboflow y entrenar.

## Idea

Cámara fija cenital (`Camara_superior/IMG_9933.MOV`). En vez de etiquetar a mano,
se auto-etiqueta:

```
frame → descubrir robots (SAM 3 texto) → máscara → filtrar (campo + área)
      → clasificar equipo por color (verde-PCB vs oscuro) → balón (HSV naranja)
      → ByteTrack (IDs persistentes) → polígonos → export YOLOv8-seg → Roboflow QA
```

- **9933 → dataset** | **9938 → análisis/demo** (sin fuga train/test).
- Equipos: un equipo muestra **PCB verde** arriba, el otro es **oscuro**.
  Clasificación por *hue mediano* de los píxeles del robot (excluyendo campo).
- Campo, líneas y porterías son **estáticos** (cámara fija) → se calibran una vez,
  no van en el modelo entrenado. Porterías → zonas para lógica de gol.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas, rangos HSV (medidos), parámetros, clases |
| `tools/profile_colors.py` | Mide rangos HSV reales (campo/balón/porterías) |
| `field_calib.py` | Detecta campo (4 esquinas) + porterías → `output/field_calib.json` |
| `detect.py` | Helpers: manos, balón, equipo, geometría, resta de fondo |
| `autolabel.py` | Orquestador: descubre→filtra→clasifica→trackea→guarda |
| `export_roboflow.py` | Convierte a dataset YOLOv8-seg (train/val + data.yaml) |

Clases: `0 robot_verde`, `1 robot_oscuro`, `2 balon`.

## Uso

Requisitos (GPU NVIDIA muy recomendada — SAM 3 en CPU ~30-60 s/frame):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics supervision
# sam3.pt en ../sam3.pt (acceso aprobado en huggingface.co/facebook/sam3)
```

Pasos:

```bash
# 1. (opcional) medir colores reales
python tools/profile_colors.py

# 2. calibrar campo + porterías (auto, o --click para precisión)
python field_calib.py --frame 600
python field_calib.py --click          # refinar esquinas a mano (1 vez)

# 3. auto-etiquetar
python autolabel.py --discover bgsub --stride 60 --max 20   # prueba CPU (sin SAM)
python autolabel.py --discover sam                          # real (GPU)

# 4. exportar a YOLOv8-seg / Roboflow
python export_roboflow.py --val 0.2
```

Salida en `output/`: `frames/`, `vis/` (overlays de control), `labels_raw/` (json),
`dataset/` (YOLOv8-seg).

## QA en Roboflow

El auto-etiquetado **no es perfecto** (robots pegados se fusionan, sombras, oclusión).
Sube `output/dataset/` (formato YOLOv8) a Roboflow, corrige máscaras y clases,
documenta los errores encontrados (parte del entregable Amateur).

## Notas / pendientes

- `--discover bgsub` es solo para validar en CPU; produce máscaras rústicas y
  falsos positivos (sombras/líneas). El dataset real usa `--discover sam`.
- Afinar `TEAM_HUE_THRESH` y `ROBOT_AREA_*` sobre máscaras SAM reales.
- Tramos de colocación/retiro de robots: el gate de manos descarta los obvios;
  el resto se filtra en QA.
