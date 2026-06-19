"""
Visualizacion analitica Copa FutBotMX — estilo transmision deportiva (broadcast).

Disenado siguiendo principios de *The Art of Insight* (Alberto Cairo): cada grafico
lleva un titulo que enuncia la CONCLUSION (no solo describe el eje), el color codifica
identidad de equipo de forma consistente, se reduce el "chart-junk", se etiqueta en
directo (sin leyendas redundantes) y se compone una jerarquia visual que guia la lectura.

Lee output/<name>/tracks.csv (posiciones cenitales por-frame, cm) + stats.json y produce
en output/<name>/viz/:
  match_report.png      panel maestro estilo broadcast (KPIs + cancha + tendencias)
  posesion_momentum.png "momentum" de posesion (vaiven en torno al 50%)
  heatmaps.png          mapas de calor suavizados por equipo sobre la cancha
  trayectoria_balon.png recorrido del balon con estela
  fisico.png            distancia recorrida + velocidad por equipo
  territorio.png        dominio territorial por tercio de cancha

Uso:
  python visualize.py --name partido2min
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe

import config as C

# ------------------------------------------------------------------ geometria
ESC = C.ESCALA_PX_CM
W_CM = C.CAMPO_W / ESC          # 182 (ancho)
H_CM = C.CAMPO_H / ESC          # 243 (alto)
GOAL_X0_CM, GOAL_W_CM, GOAL_D_CM = C.GOAL_X0 / ESC, C.GOAL_W_PX / ESC, C.GOAL_DEPTH_PX / ESC
TEAMS = ["verde", "oscuro"]

# ------------------------------------------------------------------ tema broadcast
BG = "#0d1117"          # fondo (casi negro)
PANEL = "#161b22"       # paneles
PITCH = "#16352a"       # cesped oscuro
LINE = "#3f6b57"        # lineas de cancha
INK = "#e6edf3"         # texto principal
MUTED = "#8b949e"       # texto secundario
TEAM = {"verde": "#2ee06a", "oscuro": "#3aa0ff"}   # identidad de equipo
TEAM_LBL = {"verde": "VERDE", "oscuro": "OSCURO"}
BALL = "#ff7a18"
AMARILLA, AZUL = "#ffd400", "#1f7ad8"

# velocidad fisica plausible de robot RCJ -> filtra teleports de id-switch
MAX_STEP_CM, BALL_MAX_STEP_CM = 12.0, 30.0


def _stroke(c="#000000", lw=2):
    return [pe.withStroke(linewidth=lw, foreground=c)]


def team_cmap(hex_color):
    """Colormap transparente -> color de equipo -> blanco (para heatmaps)."""
    r, g, b, _ = to_rgba(hex_color)
    return LinearSegmentedColormap.from_list(
        "t", [(r, g, b, 0.0), (r, g, b, 0.75), (1, 1, 1, 0.95)])


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.edgecolor": "#30363d",
        "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "font.family": "DejaVu Sans", "axes.titlecolor": INK,
        "figure.dpi": 130, "savefig.bbox": "tight",
    })


# ------------------------------------------------------------------ datos
def load(name):
    base = C.OUT / name
    df = pd.read_csv(base / "tracks.csv")
    stats = json.loads((base / "stats.json").read_text(encoding="utf-8"))
    return df, stats, base


def est_dt(df):
    ts = np.sort(df["t_s"].unique())
    return float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30.0


def tid_team_map(df):
    rob = df[df["kind"].isin(TEAMS)]
    if rob.empty:
        return {}
    return rob.groupby("tid")["kind"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def track_steps(df, kinds, max_step):
    sub = df[df["kind"].isin(kinds)].sort_values(["tid", "frame"])
    out = {}
    for tid, g in sub.groupby("tid"):
        dx, dy, dt = g["x_cm"].diff(), g["y_cm"].diff(), g["t_s"].diff()
        step = np.sqrt(dx ** 2 + dy ** 2)
        ok = (step <= max_step) & (dt > 0)
        out[tid] = pd.DataFrame({"t_s": g["t_s"].values,
                                 "step": np.where(ok, step, np.nan),
                                 "v": np.where(ok, step / dt, np.nan)})
    return out


def clean_ball(df, mx=20, my=16):
    """Quita detecciones de balon en margenes de borde (FP: muñequeras/reflejos)."""
    is_ball = df["kind"] == "balon"
    edge = is_ball & ((df["x_cm"] < mx) | (df["x_cm"] > W_CM - mx) |
                      (df["y_cm"] < my) | (df["y_cm"] > H_CM - my))
    return df[~edge]


def ball_series(df, max_gap=12):
    """Balon limpio + interpolado en gaps cortos (oclusion: el balon sigue su curso).
    Devuelve DataFrame(frame, t_s, x_cm, y_cm) continuo donde se pudo rellenar."""
    b = clean_ball(df)
    b = b[b["kind"] == "balon"][["frame", "t_s", "x_cm", "y_cm"]].sort_values("frame")
    if b.empty:
        return b
    frames = np.arange(int(b["frame"].min()), int(b["frame"].max()) + 1)
    s = b.drop_duplicates("frame").set_index("frame").reindex(frames)
    s[["x_cm", "y_cm"]] = s[["x_cm", "y_cm"]].interpolate(limit=max_gap, limit_area="inside")
    s["t_s"] = s["t_s"].interpolate()
    return s.dropna(subset=["x_cm", "y_cm"]).reset_index(names="frame")


def ball_track_full(df, max_gap=15, attach_dist=55):
    """Balon continuo: detecciones + interpolacion + 'dribbling' (durante oclusion
    larga, el balon se asume en el robot mas cercano a su ultima posicion).
    Para trayectoria/territorio; NO usar en posesion (seria circular)."""
    b = ball_series(df, max_gap)
    if b.empty:
        return b
    bm = {int(r.frame): (r.x_cm, r.y_cm) for r in b.itertuples()}
    rob = df[df["kind"].isin(TEAMS)]
    rob_by_f = {f: g[["x_cm", "y_cm"]].values for f, g in rob.groupby("frame")}
    rows, last = [], None
    for f in range(int(df["frame"].min()), int(df["frame"].max()) + 1):
        if f in bm:
            last = bm[f]; rows.append((f, last[0], last[1])); continue
        if last is None or f not in rob_by_f:
            continue
        P = rob_by_f[f]
        d2 = ((P[:, 0] - last[0]) ** 2 + (P[:, 1] - last[1]) ** 2)
        if d2.min() <= attach_dist ** 2:        # robot driblando -> balon ahi
            last = tuple(P[int(d2.argmin())]); rows.append((f, last[0], last[1]))
    return pd.DataFrame(rows, columns=["frame", "x_cm", "y_cm"])


def possession_series(df):
    ball_xy = {int(r.frame): (r.x_cm, r.y_cm) for r in ball_series(df).itertuples()}
    rows = []
    for fr, g in df.groupby("frame"):
        rob = g[g["kind"].isin(TEAMS)]
        if fr not in ball_xy or rob.empty:
            continue
        bx, by = ball_xy[fr]
        d = (rob["x_cm"] - bx) ** 2 + (rob["y_cm"] - by) ** 2
        rows.append((g["t_s"].iloc[0], rob.iloc[int(np.argmin(d.values))]["kind"]))
    return pd.DataFrame(rows, columns=["t_s", "team"])


def team_distance_m(df):
    steps, tmap = track_steps(df, TEAMS, MAX_STEP_CM), tid_team_map(df)
    dist = {t: 0.0 for t in TEAMS}
    for tid, s in steps.items():
        if tmap.get(tid) in dist:
            dist[tmap[tid]] += np.nansum(s["step"].values)
    return {t: dist[t] / 100.0 for t in TEAMS}


def team_speeds(df):
    steps, tmap = track_steps(df, TEAMS, MAX_STEP_CM), tid_team_map(df)
    out = {t: [] for t in TEAMS}
    for tid, s in steps.items():
        if tmap.get(tid) in out:
            out[tmap[tid]].extend(s["v"].dropna().tolist())
    return out


def heat_grid(x, y, bins=(26, 34), sigma=1.4):
    Hh, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, W_CM], [0, H_CM]])
    return gaussian_filter(Hh.T, sigma)        # (ny, nx)


# ------------------------------------------------------------------ cancha
def draw_pitch(ax, lw=1.4):
    ax.set_facecolor(PITCH)
    ax.add_patch(Rectangle((0, 0), W_CM, H_CM, fill=False, ec=LINE, lw=lw + 0.4))
    ax.plot([0, W_CM], [H_CM / 2, H_CM / 2], color=LINE, lw=lw)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(W_CM / 2 + 30 * np.cos(th), H_CM / 2 + 30 * np.sin(th), color=LINE, lw=lw)
    pw, ph, pxx = 80, 40, (W_CM - 80) / 2
    ax.add_patch(Rectangle((pxx, 0), pw, ph, fill=False, ec=LINE, lw=lw))
    ax.add_patch(Rectangle((pxx, H_CM - ph), pw, ph, fill=False, ec=LINE, lw=lw))
    ax.add_patch(Rectangle((GOAL_X0_CM, -2), GOAL_W_CM, GOAL_D_CM + 2, color=AMARILLA))
    ax.add_patch(Rectangle((GOAL_X0_CM, H_CM - GOAL_D_CM), GOAL_W_CM, GOAL_D_CM + 2, color=AZUL))
    ax.set_xlim(-6, W_CM + 6)
    ax.set_ylim(H_CM + 6, -6)          # amarilla arriba
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ 1. momentum
def plot_momentum(ax, df, stats):
    """Area apilada 0-100%: altura verde = % posesion verde en el tiempo.
    Lectura instantanea: cuando el verde cruza la linea 50% -> verde domina."""
    ps = possession_series(df)
    ax.set_facecolor(PANEL)
    if ps.empty:
        ax.text(0.5, 0.5, "sin datos", color=MUTED, ha="center"); return
    dt = est_dt(df)
    win = max(5, int(round(6.0 / dt)))
    gv = (ps["team"] == "verde").astype(float).rolling(win, min_periods=1, center=True).mean().values * 100
    t = ps["t_s"].values - ps["t_s"].values.min()
    # verde desde abajo, oscuro desde arriba
    ax.fill_between(t, 0, gv, color=TEAM["verde"], alpha=0.92, zorder=2)
    ax.fill_between(t, gv, 100, color=TEAM["oscuro"], alpha=0.92, zorder=2)
    ax.plot(t, gv, color="white", lw=1.1, alpha=0.9, zorder=3)        # frontera nitida
    ax.axhline(50, color="white", lw=1.0, ls=(0, (4, 3)), alpha=0.85, zorder=3)
    ax.set_ylim(0, 100); ax.set_xlim(t.min(), t.max())
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0", "50", "100%"])
    ax.set_xlabel("tiempo (s)")

    gp = stats.get("posesion_pct", {})
    lead = "VERDE" if gp.get("verde", 0) >= gp.get("oscuro", 0) else "OSCURO"
    pct = max(gp.get("verde", 0), gp.get("oscuro", 0))
    ax.set_title(f"Posesion en el tiempo — domina {lead} ({pct:.0f}%)",
                 loc="left", fontsize=12, fontweight="bold", pad=8)

    # etiquetas en BLANCO con contorno negro -> legibles sobre verde y sobre azul
    ax.text(0.012, 0.10, "VERDE", transform=ax.transAxes, color="white",
            fontsize=10, fontweight="bold", va="center",
            path_effects=_stroke("#000000", 3))
    ax.text(0.012, 0.90, "OSCURO", transform=ax.transAxes, color="white",
            fontsize=10, fontweight="bold", va="center",
            path_effects=_stroke("#000000", 3))
    # chips a la derecha con el % global (lectura rapida del marcador)
    ax.text(0.995, 0.10, f"{gp.get('verde',0):.0f}%", transform=ax.transAxes,
            color="white", fontsize=13, fontweight="bold", ha="right", va="center",
            path_effects=_stroke("#000000", 3))
    ax.text(0.995, 0.90, f"{gp.get('oscuro',0):.0f}%", transform=ax.transAxes,
            color="white", fontsize=13, fontweight="bold", ha="right", va="center",
            path_effects=_stroke("#000000", 3))
    for sp in ax.spines.values():
        sp.set_visible(False)


# ------------------------------------------------------------------ 2. heatmap
def plot_heat(ax, df, team, title=True):
    draw_pitch(ax)
    sub = df[df["kind"] == team]
    if len(sub) > 5:
        g = heat_grid(sub["x_cm"].values, sub["y_cm"].values)
        if g.max() > 0:
            ax.imshow(g, extent=[0, W_CM, H_CM, 0], origin="upper",
                      cmap=team_cmap(TEAM[team]), interpolation="bilinear",
                      alpha=0.92, zorder=1.5, vmax=g.max())
    if title:
        ax.set_title(TEAM_LBL[team], color=TEAM[team], fontsize=11,
                     fontweight="bold", pad=6)


# ------------------------------------------------------------------ 3. trayectoria
def plot_ball(ax, df):
    draw_pitch(ax)
    b = ball_track_full(df)
    if len(b) < 2:
        ax.set_title("Trayectoria del balon", color=INK); return
    # suaviza jitter con mediana movil corta
    x = b["x_cm"].rolling(5, min_periods=1, center=True).median().values
    y = b["y_cm"].rolling(5, min_periods=1, center=True).median().values
    t = (b["frame"].values - b["frame"].values.min()) / 30.0    # segundos
    seg_d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    keep = seg_d <= BALL_MAX_STEP_CM
    # "turbo": azul(inicio) -> verde -> rojo(fin); orden temporal legible
    lc = LineCollection(segs[keep], cmap="turbo", linewidths=1.6, alpha=0.9, zorder=2)
    lc.set_array(t[:-1][keep]); ax.add_collection(lc)
    ax.scatter(x[0], y[0], s=70, c="#2bd2ff", ec="white", lw=1.4, zorder=5, label="inicio")
    ax.scatter(x[-1], y[-1], s=85, marker="*", c="#ff3b3b", ec="white", lw=1.2,
               zorder=5, label="fin")
    ax.set_title("Recorrido del balon", color=INK, fontsize=11, fontweight="bold", pad=6)
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                    facecolor=PANEL, edgecolor="#30363d", labelcolor=INK)
    # barra de color = tiempo, con extremos etiquetados (explica el degradado)
    cb = ax.figure.colorbar(lc, ax=ax, orientation="horizontal",
                            fraction=0.045, pad=0.03)
    cb.set_ticks([t.min(), t.max()])
    cb.set_ticklabels(["inicio (0 s)", f"fin ({t.max():.0f} s)"])
    cb.ax.tick_params(colors=INK, labelsize=8, length=0)
    cb.set_label("avance del partido", color=MUTED, fontsize=8)
    cb.outline.set_edgecolor("#30363d")


# ------------------------------------------------------------------ 4. fisico
def plot_distance(ax, df):
    d = team_distance_m(df)
    ax.set_facecolor(PANEL)
    order = sorted(TEAMS, key=lambda t: d[t])
    y = np.arange(len(order))
    ax.barh(y, [d[t] for t in order], color=[TEAM[t] for t in order], height=0.55)
    ax.set_yticks(y); ax.set_yticklabels([TEAM_LBL[t] for t in order], fontweight="bold")
    for i, t in enumerate(order):
        ax.text(d[t], i, f"  {d[t]:.0f} m", va="center", color=INK, fontweight="bold")
    ax.set_title("Distancia recorrida", loc="left", color=INK, fontsize=11,
                 fontweight="bold", pad=6)
    ax.set_xlabel("metros"); ax.margins(x=0.18)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False)


def plot_speed(ax, df):
    sp = team_speeds(df)
    ax.set_facecolor(PANEL)
    data = [np.array(sp[t]) for t in TEAMS]
    data = [d[(d > 0) & (d < 120)] for d in data]
    # violinplot crashea con arrays vacios -> placeholder de 1 muestra
    safe = [d if len(d) > 1 else np.array([0.0, 0.0]) for d in data]
    parts = ax.violinplot(safe, showextrema=False, widths=0.8)
    for pc, t in zip(parts["bodies"], TEAMS):
        pc.set_facecolor(TEAM[t]); pc.set_alpha(0.7)
    for i, t in enumerate(TEAMS):
        if len(data[i]):
            mx = np.percentile(data[i], 95)
            ax.scatter(i + 1, np.median(data[i]), c="white", s=25, zorder=4)
            ax.text(i + 1, mx, f"max {mx:.0f}", color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.set_xticks([1, 2]); ax.set_xticklabels([TEAM_LBL[t] for t in TEAMS], fontweight="bold")
    ax.set_title("Velocidad (cm/s)", loc="left", color=INK, fontsize=11,
                 fontweight="bold", pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color="#30363d", lw=0.5)


# ------------------------------------------------------------------ 5. territorio
def territory_pct(df):
    b = ball_track_full(df)
    if b.empty:
        return [0, 0, 0]
    y = b["y_cm"].values
    return [np.mean(y < H_CM / 3) * 100,
            np.mean((y >= H_CM / 3) & (y < 2 * H_CM / 3)) * 100,
            np.mean(y >= 2 * H_CM / 3) * 100]


def plot_territory(ax, df):
    t1, t2, t3 = territory_pct(df)
    draw_pitch(ax)
    # franjas de tercio con intensidad ~ % de tiempo del balon
    vals = [t1, t2, t3]
    mx = max(vals) or 1
    for i, v in enumerate(vals):
        y0 = i * H_CM / 3
        ax.add_patch(Rectangle((0, y0), W_CM, H_CM / 3, color=BALL,
                               alpha=0.10 + 0.45 * v / mx, zorder=1.2))
        ax.text(W_CM / 2, y0 + H_CM / 6, f"{v:.0f}%", ha="center", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=3,
                path_effects=_stroke("#000000", 3))
    ax.set_title("Dominio territorial del balon", color=INK, fontsize=11,
                 fontweight="bold", pad=6)


# ------------------------------------------------------------------ KPIs + report
def kpi_card(fig, x, y, w, h, value, label, color=INK):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=PANEL, ec="#30363d", lw=1, transform=ax.transAxes))
    ax.text(0.5, 0.62, value, ha="center", va="center", color=color,
            fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.22, label, ha="center", va="center", color=MUTED,
            fontsize=9, transform=ax.transAxes)


def match_report(df, stats, viz):
    apply_theme()
    fig = plt.figure(figsize=(15, 10)); fig.patch.set_facecolor(BG)
    gp = stats.get("posesion_pct", {})
    dist = team_distance_m(df); sp = team_speeds(df)
    dur = df["t_s"].max() - df["t_s"].min()

    # encabezado
    fig.text(0.04, 0.955, "PERROS DEL ROUTING", color=INK, fontsize=22, fontweight="bold")
    fig.text(0.04, 0.925, "Reporte analitico del partido — Copa FutBotMX",
             color=MUTED, fontsize=12)
    fig.text(0.96, 0.95, f"{dur:.0f}s  ·  {stats.get('frames','?')} frames",
             color=MUTED, fontsize=11, ha="right")

    # franja KPI
    def topspeed(t):
        a = np.array(sp[t]); a = a[(a > 0) & (a < 120)]
        return np.percentile(a, 95) if len(a) else 0
    kpis = [
        (f"{gp.get('verde',0):.0f}%", "POSESION VERDE", TEAM["verde"]),
        (f"{gp.get('oscuro',0):.0f}%", "POSESION OSCURO", TEAM["oscuro"]),
        (f"{dist['verde']+dist['oscuro']:.0f} m", "DISTANCIA TOTAL", INK),
        (f"{max(topspeed('verde'),topspeed('oscuro')):.0f}", "VEL. MAX (cm/s)", BALL),
        (f"{len(stats.get('goles',[]))}", "EVENTOS GOL", AMARILLA),
    ]
    for i, (v, l, c) in enumerate(kpis):
        kpi_card(fig, 0.04 + i * 0.187, 0.80, 0.175, 0.085, v, l, c)

    gs = fig.add_gridspec(2, 4, left=0.04, right=0.97, top=0.74, bottom=0.05,
                          hspace=0.28, wspace=0.25, height_ratios=[1.15, 1])
    plot_momentum(fig.add_subplot(gs[0, 0:2]), df, stats)
    plot_ball(fig.add_subplot(gs[0:2, 2]), df)
    plot_territory(fig.add_subplot(gs[0:2, 3]), df)
    plot_heat(fig.add_subplot(gs[1, 0]), df, "verde")
    plot_heat(fig.add_subplot(gs[1, 1]), df, "oscuro")

    fig.text(0.04, 0.02, "Deteccion YOLOv8-seg + SAM3 · homografia a cancha RCJ (182×243 cm) · "
             "balon HSV+YOLO; en oclusion se infiere del robot que lo dribla · "
             "camara handheld (posiciones aproximadas)",
             color="#586069", fontsize=8)
    fig.savefig(viz / "match_report.png", dpi=130)
    plt.close(fig)


def save_all(name):
    df, stats, base = load(name)
    viz = base / "viz"; viz.mkdir(parents=True, exist_ok=True)
    apply_theme()

    fig, ax = plt.subplots(figsize=(10, 4)); plot_momentum(ax, df, stats)
    fig.savefig(viz / "posesion_momentum.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6.5)); fig.patch.set_facecolor(BG)
    for ax, t in zip(axs, TEAMS):
        plot_heat(ax, df, t)
    fig.suptitle("Mapas de calor de ocupacion", color=INK, fontweight="bold")
    fig.savefig(viz / "heatmaps.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 6.5)); plot_ball(ax, df)
    fig.savefig(viz / "trayectoria_balon.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4)); fig.patch.set_facecolor(BG)
    plot_distance(axs[0], df); plot_speed(axs[1], df)
    fig.savefig(viz / "fisico.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 6.5)); plot_territory(ax, df)
    fig.savefig(viz / "territorio.png"); plt.close(fig)

    match_report(df, stats, viz)
    print("Figuras en:", viz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="partido2min")
    args = ap.parse_args()
    save_all(args.name)
