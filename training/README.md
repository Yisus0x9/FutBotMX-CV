# Entrenamiento del detector — `pipelines/training`

Esta etapa **enseña a la computadora a reconocer los robots y el balón** en la cancha.
Es el "cerebro" que luego usa el análisis del partido.

## Objetivo

Tomar las imágenes ya etiquetadas (generadas con SAM 3 en la etapa anterior) y **entrenar un
modelo de visión** que, ante cualquier frame del partido, dibuje y clasifique:

- 🟢 **robot_verde** — equipo con placa (PCB) verde
- ⚫ **robot_oscuro** — equipo oscuro
- 🟠 **balon** — pelota naranja

El modelo no solo dice *dónde* está cada objeto, sino *qué es* (a qué equipo pertenece) y su
**silueta exacta** (segmentación), no solo una caja.

## Qué se usó

- **YOLOv8-seg** (Ultralytics) — modelo de detección + segmentación, rápido y ligero. Se partió
  del modelo pre-entrenado `yolov8s-seg.pt` y se **afinó (fine-tuning)** con nuestras imágenes.
- **Roboflow** — para revisar y corregir a mano las etiquetas automáticas antes de entrenar
  (control de calidad / QA).
- **GPU NVIDIA RTX 4060** — para entrenar en minutos en lugar de horas.

> ¿Por qué afinar YOLO y no SAM 3? SAM 3 segmenta pero **no distingue equipos** y es pesado para
> video en tiempo real. YOLO sí clasifica, es veloz y barato de afinar. Estrategia: **SAM 3
> etiqueta → YOLO aprende y corre**.

## Entradas

| Entrada | De dónde viene |
|---|---|
| Imágenes + máscaras etiquetadas (`data.yaml`) | `pipelines/dataset_generation` (auto-etiquetado SAM 3) |
| Versión revisada en Roboflow (`ROVOT-VISION.yolov8/`) | corrección manual de las etiquetas |
| Pesos base `yolov8s-seg.pt` | modelo pre-entrenado de Ultralytics |

## Salidas

| Salida | Qué es |
|---|---|
| `runs/qa_s/weights/best.pt` | **el detector entrenado** (lo que usa el análisis del partido) |
| `runs/qa_s/results.csv` + gráficas | curvas de aprendizaje y métricas por época |
| `runs/test_9938_*/` | imágenes de prueba con las detecciones dibujadas (verificación visual) |

## Resultado

El detector afinado (`qa_s`, sobre `yolov8s-seg`) logró frente a las etiquetas corregidas:

- **Caja (box) mAP@50: 0.91** — encuentra muy bien dónde está cada objeto.
- **Silueta (máscara) mAP@50: 0.81** — recorta bien la forma de robots y balón.

Y **generaliza**: aunque se entrenó con un video, funciona en otros partidos y cámaras distintas.

## Cómo se ejecuta

```bash
# Entrenar el detector con el dataset revisado en Roboflow
python pipelines/training/train_baseline.py \
    --model yolov8s-seg.pt \
    --data ROVOT-VISION.yolov8/data_local.yaml \
    --name qa_s --epochs 100 --imgsz 960

# Probar el detector sobre un video real (genera imágenes con las detecciones)
python pipelines/training/test_on_9938.py --weights runs/qa_s/weights/best.pt --name qa_s
```

## Archivos

```
train_baseline.py     entrena (y valida) el detector YOLOv8-seg
test_on_9938.py       corre el detector sobre el video y dibuja resultados para revisar
yolov8s-seg.pt        pesos base pre-entrenados (punto de partida)
runs/                 resultados de cada entrenamiento/prueba (qa_s = el bueno)
```

---

Siguiente etapa: **`pipelines/analysis`** usa `runs/qa_s/weights/best.pt` para analizar el partido
(posesión, mapas de calor, recorrido del balón, mapa táctico).
