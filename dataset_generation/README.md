# Generación del dataset — `pipelines/dataset_generation`

Esta etapa **prepara los ejemplos** con los que después aprende el detector. En vez de etiquetar
miles de imágenes a mano, la computadora se **auto-etiqueta** con ayuda de SAM 3.

## Objetivo

Convertir un video del partido en un **conjunto de imágenes con su "respuesta correcta"**: dónde
está cada robot, a qué equipo pertenece y dónde está el balón — con su silueta recortada. Ese
material es la materia prima para entrenar.

## Qué se usó

- **SAM 3 (Segment Anything Model 3, Meta)** — recorta los robots con solo decirle *"robot"* en
  texto, sin necesidad de un detector previo. Hace el trabajo pesado de marcar siluetas.
- **Color (HSV)** — para dos cosas: distinguir el **equipo** (placa verde vs robot oscuro,
  mirando el tono dominante del robot) y encontrar el **balón** naranja.
- **ByteTrack** — sigue a cada robot entre frames para darle un **identificador estable**.
- **Roboflow** — donde después se revisan y corrigen a mano las etiquetas automáticas (QA).

> Idea clave: **SAM 3 marca, el color clasifica el equipo, el seguimiento ordena.** Así se obtiene
> un dataset bueno sin etiquetar imagen por imagen.

## Entradas

| Entrada | Detalle |
|---|---|
| Video cenital del partido | `Camara_superior/IMG_9933.MOV` (cámara fija) |
| Modelo `sam3.pt` | SAM 3 (acceso aprobado en Hugging Face) |
| Calibración del campo | 4 esquinas + porterías (se calcula una vez, cámara fija) |

> Se usa **un video para el dataset (9933)** y **otro distinto para el análisis (9938)**, para no
> "hacer trampa" entrenando y probando con lo mismo.

## Salidas

| Salida | Qué es |
|---|---|
| `output/dataset/` (formato YOLOv8-seg) | imágenes + etiquetas listas para entrenar |
| `output/field_calib.json` | esquinas del campo y porterías (para homografía y goles) |
| Subida a Roboflow | versión revisada/corregida a mano |

Clases: `robot_verde`, `robot_oscuro`, `balon`.

## Cómo funciona

```
frame → SAM 3 recorta robots → filtra (dentro del campo) → clasifica equipo por color
      → balón por color naranja → ByteTrack (IDs) → guarda en formato YOLOv8-seg → Roboflow (QA)
```

El auto-etiquetado **no es perfecto** (robots pegados se fusionan, sombras, oclusión); por eso el
paso de **revisión en Roboflow** es importante antes de entrenar.

## Cómo se ejecuta

```bash
# 1. Calibrar el campo (clic en las 4 esquinas, una vez por cámara)
python pipelines/dataset_generation/field_calib.py --frame 600 --click

# 2. Auto-etiquetar con SAM 3 y exportar el dataset
python pipelines/dataset_generation/autolabel.py --discover sam
python pipelines/dataset_generation/export_roboflow.py --val 0.2
```

## Archivos

```
config.py             rutas, colores (HSV) medidos, parámetros y clases
field_calib.py        detecta el campo (4 esquinas) y porterías
autolabel.py          orquestador: recorta → filtra → clasifica → sigue → guarda
detect.py             utilidades (manos, balón, equipo, geometría)
export_roboflow.py    arma el dataset YOLOv8-seg (train/val + data.yaml)
tools/profile_colors.py   mide los rangos de color reales del campo/balón/porterías
```

---

Siguiente etapa: **`pipelines/training`** toma este dataset (revisado en Roboflow) y entrena el
detector.
