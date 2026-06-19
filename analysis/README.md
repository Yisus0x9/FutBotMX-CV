# Análisis del partido — `pipelines/analysis`

Esta es la etapa final: con el detector ya entrenado, **se analiza un partido completo** y se
generan las estadísticas y los gráficos estilo transmisión deportiva.

## Objetivo

Ver un video de fútbol robótico y responder, automáticamente: **¿quién tuvo el balón?, ¿por dónde
se jugó?, cuánto corrió cada equipo, qué tan rápido, y hubo gol?** Todo proyectado a una **vista
cenital** de la cancha, como en un análisis táctico real.

## Qué se usó

- **El detector entrenado** (`pipelines/training/runs/qa_s/weights/best.pt`) — encuentra y
  clasifica robots y balón en cada frame.
- **SAM 3** (opcional, modo `--sam`) — afina las siluetas de los robots (cumple el reto de
  "aplicar SAM 3" en el análisis).
- **ByteTrack** — sigue a cada robot para medir distancia y velocidad.
- **Homografía (OpenCV)** — convierte la cámara inclinada en una **vista cenital a escala real**
  (cancha RCJ de 182 × 243 cm) para medir posiciones de verdad.
- **Color (HSV)** — para localizar el balón; cuando un robot lo tapa, se infiere del robot que lo
  está dribleando.
- **Matplotlib** — para el reporte gráfico estilo broadcast (diseño guiado por *The Art of
  Insight*: cada gráfico titula la conclusión, color = identidad de equipo, lectura rápida).

## Entradas

| Entrada | Detalle |
|---|---|
| Video del partido | cualquier `.mp4` cenital (p. ej. `IMG_9938` o `video_test/Video2minutos.mp4`) |
| Detector entrenado | `best.pt` de la etapa de training |
| Calibración del campo | `field_calib_*.json` (4 esquinas de la cancha para ese video) |

Funciona con **cualquier video + su calibración** gracias a los parámetros `--video` y `--calib`.

## Salidas

En `output/<nombre>/`:

| Salida | Qué es |
|---|---|
| `analisis.mp4` | video de 3 paneles: cámara + máscaras · vista cenital real · mapa táctico |
| `heatmap.png` | mapa de calor de control por equipo |
| `stats.json` | posesión por equipo, goles, número de frames |
| `tracks.csv` | posiciones de robots y balón por frame (la "materia prima" de los gráficos) |
| `viz/` | **reporte estilo transmisión** (ver abajo) |

El **reporte gráfico** (`viz/`) incluye:

- `match_report.png` — panel maestro con KPIs (posesión, distancia, velocidad máx, goles).
- `posesion_momentum.png` — posesión en el tiempo (quién domina y cuándo).
- `heatmaps.png` — mapas de calor de ocupación por equipo.
- `trayectoria_balon.png` — recorrido del balón (color = avance del partido).
- `fisico.png` — distancia recorrida y velocidad por equipo.
- `territorio.png` — dominio del balón por tercio de la cancha.

## Cómo funciona (en una línea)

```
frame → detector (robots+equipo) → [SAM 3 afina] → ByteTrack (IDs) → homografía a cenital
      → balón por color → posesión / distancia / velocidad / territorio / gol → video + reporte
```

## Cómo se ejecuta

```bash
# 1. Analizar un partido (genera video + heatmap + stats + tracks.csv)
python pipelines/analysis/run_analysis.py \
    --video Pipelines/video_test/Video2minutos.mp4 \
    --calib pipelines/analysis/output/field_calib_video2.json \
    --start 120 --end 3600 --name partido2min

# 2. Generar el reporte gráfico estilo broadcast
python pipelines/analysis/visualize.py --name partido2min
```

> Para usar el video por defecto (9938) basta con omitir `--video` y `--calib`.
> Añadir `--sam` afina las siluetas con SAM 3 (más lento).

## Archivos

```
config.py        rutas, escala de la cancha, colores del balón, zonas de gol
homography.py    proyección a vista cenital, mapa táctico, detección del balón
run_analysis.py  pipeline principal (detecta → trackea → proyecta → mide → graba)
visualize.py     genera el reporte gráfico a partir de tracks.csv
scan_goals.py    utilidad para buscar posibles goles en el video
```

## Notas / limitaciones

- La cámara del clip de prueba es **de mano** (no fija): la vista cenital queda **aproximada**.
- El balón es pequeño y a menudo lo **tapa un robot**; se rellena interpolando trayectos cortos y,
  en oclusiones largas, asumiendo que va con el robot que lo dribla.
- La detección de gol marca cuando el balón entra a la zona de portería; puede dar algún falso
  positivo y se revisa a mano.
